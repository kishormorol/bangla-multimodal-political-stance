"""YAML-backed configuration.

Two kinds of config live in `configs/`:

* `configs/data.yaml`   — where the raw files are, how their columns map onto
  the canonical schema, and how splits are built. Loaded as `DataConfig`.
* `configs/{text,multimodal}/*.yaml` — one file per model in the comparison.
  Loaded as `ExperimentConfig`.

Anything a reviewer would need in order to rerun a number belongs in one of
these files rather than in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from bmpb.paths import CONFIGS


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIGS / path
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _from_dict(cls, payload: dict[str, Any]):
    """Build a dataclass from a dict, ignoring keys the dataclass doesn't know."""
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unrecognized keys {sorted(unknown)}")
    return cls(**payload)


@dataclass
class DataConfig:
    """Describes the raw dataset and how it becomes the canonical corpus."""

    drive_folder_id: str
    source_table: str  # file under data/raw/ that defines the corpus
    # The image subfolders are addressed directly: a recursive walk of the
    # parent folder dies on the Google Doc it contains. See data/download.py.
    drive_image_folder_id: str | None = None
    drive_processed_image_folder_id: str | None = None
    image_table: str | None = None  # file that maps items to image filenames
    annotation_table: str | None = None  # file carrying the per-annotator labels
    columns: dict[str, list[str]] = field(default_factory=dict)  # canonical -> aliases
    label_map: dict[str, str] = field(default_factory=dict)  # raw label -> canonical
    image_dir: str = "images"
    processed_image_dir: str = "processed_images"
    image_extensions: list[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp", ".avif"]
    )
    split: dict[str, Any] = field(default_factory=dict)
    text_max_length: int = 256

    @classmethod
    def load(cls, path: str | Path = "data.yaml") -> DataConfig:
        return _from_dict(cls, load_yaml(path))


@dataclass
class ExperimentConfig:
    """One row of the results table: a model, its inputs, and its training recipe."""

    name: str
    modality: str  # "text" | "image" | "multimodal"
    family: str  # registry key, e.g. "hf_text_classifier"
    pretrained: str | None = None  # HF model id, when the family needs one
    num_labels: int = 3
    seed: int = 42
    epochs: int = 5
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 256
    image_size: int = 224
    freeze_encoder: bool = False
    fp16: bool = False
    early_stopping_patience: int | None = 2
    class_weights: bool = True
    params: dict[str, Any] = field(default_factory=dict)  # family-specific extras

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        return _from_dict(cls, load_yaml(path))

    @property
    def uses_text(self) -> bool:
        return self.modality in {"text", "multimodal"}

    @property
    def uses_image(self) -> bool:
        return self.modality in {"image", "multimodal"}
