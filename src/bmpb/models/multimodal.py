"""Vision-language models: CLIP, BLIP, ALIGN, FLAVA, ViLT.

A note on Bangla
----------------
None of these encoders were pretrained on Bangla, and CLIP/ALIGN in particular
tokenize Bangla script into near-unusable fragments. That is the finding the
comparison exists to document, not a bug to paper over — so each family is
wired up faithfully, and `params.translate_text` is offered for the ablation
that asks how much of the gap is a tokenizer problem rather than a grounding
problem.
"""

from __future__ import annotations

from bmpb.config import ExperimentConfig
from bmpb.models.registry import register


def _pooled(out):
    """Pooled vector from a transformers ModelOutput, falling back to the [CLS] state."""
    pooled = getattr(out, "pooler_output", None)
    return pooled if pooled is not None else out.last_hidden_state[:, 0]


@register("clip_like")
def clip_like(cfg: ExperimentConfig):
    """Dual-encoder models with a joint processor: CLIP and ALIGN.

    Both expose `get_text_features` / `get_image_features`, so one wrapper with
    a fusion head covers them.
    """
    import torch
    from transformers import AutoModel, AutoProcessor

    from bmpb.models.heads import FusionClassifier

    processor = AutoProcessor.from_pretrained(cfg.pretrained)
    backbone = AutoModel.from_pretrained(cfg.pretrained)

    class DualEncoderClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            if cfg.freeze_encoder:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            dim = (
                getattr(backbone.config, "projection_dim", None)
                or backbone.config.text_config.hidden_size
            )
            self.head = FusionClassifier(
                text_dim=dim,
                image_dim=dim,
                num_labels=cfg.num_labels,
                mode=cfg.params.get("fusion", "concat"),
            )

        def forward(
            self, input_ids=None, attention_mask=None, pixel_values=None, labels=None, **kw
        ):
            text_out = self.backbone.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )
            image_out = self.backbone.get_image_features(pixel_values=pixel_values)
            # Newer transformers may return ModelOutput instead of a plain tensor.
            if not isinstance(text_out, torch.Tensor):
                text_out = _pooled(text_out)
            if not isinstance(image_out, torch.Tensor):
                image_out = _pooled(image_out)
            return self.head(text_out, image_out)

    return DualEncoderClassifier(), processor


@register("fusion_encoder")
def fusion_encoder(cfg: ExperimentConfig):
    """Single-stream models that fuse inside the encoder: FLAVA, ViLT, BLIP.

    These take text and image together and expose a pooled multimodal state.
    ViLT caps text at 40 tokens, which the config must respect; the YAML sets
    `max_length` accordingly rather than the code guessing.
    """
    import torch
    from transformers import AutoConfig, AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(cfg.pretrained)
    # AutoModel maps a BLIP checkpoint to BlipModel, a dual encoder whose text
    # tower never attends to the image; its text [CLS] is image-blind. The
    # fused state is the image-text matching encoder's [CLS], which
    # cross-attends to the image embeddings.
    is_blip = AutoConfig.from_pretrained(cfg.pretrained).model_type == "blip"
    if is_blip:
        from transformers import BlipForImageTextRetrieval

        backbone = BlipForImageTextRetrieval.from_pretrained(cfg.pretrained)
    else:
        backbone = AutoModel.from_pretrained(cfg.pretrained)

    class FusedClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            if cfg.freeze_encoder:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            hidden = cfg.params.get("hidden_size") or _hidden_size(backbone.config)
            self.dropout = torch.nn.Dropout(0.1)
            self.classifier = torch.nn.Linear(hidden, cfg.num_labels)

        def forward(self, labels=None, **inputs):
            if is_blip:
                outputs = self.backbone(**inputs, use_itm_head=True)
                pooled = outputs.question_embeds[:, 0]
            else:
                outputs = self.backbone(**inputs)
                pooled = getattr(outputs, "pooler_output", None)
                if pooled is None:
                    state = getattr(outputs, "multimodal_embeddings", None)
                    if state is None:
                        state = outputs.last_hidden_state
                    pooled = state[:, 0]
            return {"logits": self.classifier(self.dropout(pooled))}

    return FusedClassifier(), processor


def _hidden_size(config) -> int:
    for attr in ("hidden_size", "multimodal_hidden_size", "d_model"):
        value = getattr(config, attr, None)
        if value:
            return value
    text_config = getattr(config, "text_config", None)
    if text_config is not None and getattr(text_config, "hidden_size", None):
        return text_config.hidden_size
    raise ValueError(
        f"could not infer a hidden size from {type(config).__name__}; "
        "set params.hidden_size in the config"
    )
