"""Torch datasets and collators for the text, image and multimodal settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from bmpb.paths import RAW

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - torch is optional for data-only work
    torch = None
    Dataset = object  # type: ignore[assignment]


def load_image(relative_path: str | None, size: int = 224) -> Image.Image | None:
    """Open an image stored under data/raw/. Returns None when it is unusable.

    Images come from news CDNs, so a handful are AVIF/WebP or truncated. A
    caller that needs a guaranteed image should filter on `has_image` first and
    still be ready for None.
    """
    if not relative_path or (isinstance(relative_path, float) and pd.isna(relative_path)):
        return None
    path = Path(relative_path)
    if not path.is_absolute():
        path = RAW / path
    if not path.exists():
        return None
    try:
        with Image.open(path) as handle:
            return handle.convert("RGB").resize((size, size), Image.BICUBIC)
    except Exception:  # noqa: BLE001 - a broken file should skip, not crash a run
        return None


@dataclass
class Example:
    item_id: str
    text: str
    label: int
    image_path: str | None


class BiasDataset(Dataset):
    """One row per article, exposing whichever modalities the model asked for.

    `require_image=True` drops rows whose image is missing, which is what the
    vision-language comparison runs on; the text-only models see every row.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        use_text: bool = True,
        use_image: bool = False,
        require_image: bool = False,
        image_size: int = 224,
        include_title: bool = True,
    ) -> None:
        if require_image:
            frame = frame[frame.get("has_image", False).astype(bool)].reset_index(drop=True)
        self.frame = frame.reset_index(drop=True)
        self.use_text = use_text
        self.use_image = use_image
        self.image_size = image_size
        self.include_title = include_title

    def __len__(self) -> int:
        return len(self.frame)

    def text_for(self, row: pd.Series) -> str:
        title = str(row.get("title") or "").strip()
        body = str(row.get("text") or "").strip()
        if self.include_title and title and not body.startswith(title):
            return f"{title} {body}".strip()
        return body

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        item: dict = {"item_id": row["item_id"], "labels": int(row["label"])}
        if self.use_text:
            item["text"] = self.text_for(row)
        if self.use_image:
            item["image"] = load_image(row.get("image_path"), size=self.image_size)
        return item

    @property
    def labels(self) -> list[int]:
        return self.frame["label"].astype(int).tolist()


@dataclass
class Collator:
    """Batches raw examples with a HuggingFace processor or tokenizer.

    One collator covers all three modalities because the HF processors share a
    call signature: text-only models get `text=`, vision-language models get
    `text=` and `images=`, and image-only models get `images=`.
    """

    processor: object
    max_length: int = 256
    use_text: bool = True
    use_image: bool = False

    def __call__(self, batch: list[dict]) -> dict:
        if torch is None:  # pragma: no cover - dependency guard
            raise RuntimeError("torch is required to collate batches")

        kwargs: dict = {"return_tensors": "pt", "padding": True, "truncation": True}
        if self.use_text:
            kwargs["text"] = [b["text"] for b in batch]
            kwargs["max_length"] = self.max_length
        if self.use_image:
            images = [b["image"] for b in batch]
            if any(image is None for image in images):
                raise ValueError(
                    "a batch contained a missing image; build the dataset with "
                    "require_image=True for image and multimodal models"
                )
            kwargs["images"] = images

        encoded = self.processor(**kwargs)
        encoded = dict(encoded)
        encoded["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        return encoded
