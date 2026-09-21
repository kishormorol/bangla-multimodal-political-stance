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
            text_features = self.backbone.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )
            image_features = self.backbone.get_image_features(pixel_values=pixel_values)
            return self.head(text_features, image_features)

    return DualEncoderClassifier(), processor


@register("fusion_encoder")
def fusion_encoder(cfg: ExperimentConfig):
    """Single-stream models that fuse inside the encoder: FLAVA, ViLT, BLIP.

    These take text and image together and expose a pooled multimodal state.
    ViLT caps text at 40 tokens, which the config must respect; the YAML sets
    `max_length` accordingly rather than the code guessing.
    """
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(cfg.pretrained)
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
