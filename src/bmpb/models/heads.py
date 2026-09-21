"""Classification heads used to put encoders of different shapes on equal footing.

Every model in the comparison ends in the same place — three logits — so that a
difference in the results table is a difference in representation, not in how
generously each model was wired up.
"""

from __future__ import annotations

import torch
from torch import nn


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return hidden.mean(dim=1)
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class EncoderClassifier(nn.Module):
    """Pooled encoder states -> dropout -> linear. Used for T5-style encoders."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        num_labels: int = 3,
        pooling: str = "mean",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = outputs.last_hidden_state
        pooled = hidden[:, 0] if self.pooling == "cls" else masked_mean(hidden, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        return {"logits": logits}


class FusionClassifier(nn.Module):
    """Late fusion of a text vector and an image vector.

    `mode` controls how much the two streams interact:
      concat — [t; i]                      (the usual baseline)
      gated  — a learned per-dimension gate over the two projections, which is
               the setting that lets the model ignore a decorative photo.
    """

    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        num_labels: int = 3,
        hidden: int = 512,
        mode: str = "concat",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.text_proj = nn.Linear(text_dim, hidden)
        self.image_proj = nn.Linear(image_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        if mode == "gated":
            self.gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Sigmoid())
            head_in = hidden
        else:
            head_in = 2 * hidden
        self.classifier = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor):
        t = self.dropout(torch.relu(self.text_proj(text_features)))
        i = self.dropout(torch.relu(self.image_proj(image_features)))
        if self.mode == "gated":
            gate = self.gate(torch.cat([t, i], dim=-1))
            fused = gate * t + (1 - gate) * i
        else:
            fused = torch.cat([t, i], dim=-1)
        return {"logits": self.classifier(fused)}
