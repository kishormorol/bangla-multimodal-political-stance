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

    revision = cfg.params.get("revision")
    processor = AutoProcessor.from_pretrained(cfg.pretrained, revision=revision)
    backbone = AutoModel.from_pretrained(cfg.pretrained, revision=revision)

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

    revision = cfg.params.get("revision")
    model_config = AutoConfig.from_pretrained(cfg.pretrained, revision=revision)
    is_blip = model_config.model_type == "blip"
    processor = AutoProcessor.from_pretrained(cfg.pretrained, revision=revision)
    if is_blip:
        from transformers import BlipForImageTextRetrieval

        if (
            model_config.architectures
            and "BlipForImageTextRetrieval" not in model_config.architectures
        ):
            raise ValueError("BLIP fusion requires a BlipForImageTextRetrieval checkpoint")
        backbone = BlipForImageTextRetrieval.from_pretrained(
            cfg.pretrained, config=model_config, revision=revision
        )
    else:
        backbone = AutoModel.from_pretrained(cfg.pretrained, config=model_config, revision=revision)

    class FusedClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            if cfg.freeze_encoder:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            if is_blip:
                hidden = backbone.config.text_config.hidden_size
            else:
                hidden = cfg.params.get("hidden_size") or _hidden_size(backbone.config)
            self.dropout = torch.nn.Dropout(0.1)
            self.classifier = torch.nn.Linear(hidden, cfg.num_labels)

        def forward(self, labels=None, **inputs):
            if is_blip:
                pooled = _blip_multimodal_features(self.backbone, **inputs)
                return {"logits": self.classifier(self.dropout(pooled))}

            outputs = self.backbone(**inputs)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                # Some outputs expose the text-model state separately.
                text_out = getattr(outputs, "text_model_output", None)
                if text_out is not None:
                    pooled = text_out.last_hidden_state[:, 0]
                else:
                    state = getattr(outputs, "multimodal_embeddings", None)
                    if state is None:
                        state = outputs.last_hidden_state
                    pooled = state[:, 0]
            return {"logits": self.classifier(self.dropout(pooled))}

    return FusedClassifier(), processor


def _blip_multimodal_features(backbone, input_ids, attention_mask, pixel_values, **_):
    """Pool BLIP text after its image cross-attention layers."""
    import torch

    image_embeds = backbone.vision_model(pixel_values=pixel_values)[0]
    image_mask = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=image_embeds.device)
    text_outputs = backbone.text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_mask,
    )
    return text_outputs[0][:, 0]


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
