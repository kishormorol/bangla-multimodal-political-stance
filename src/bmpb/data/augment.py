"""Reattach the published augmentations to the split their parent article is in.

`Balanced_Augmented_Dataset.csv` holds 198 original rows plus 114 augmented ones
(59 `swap`, 52 `all`, 3 `synonym`), each carrying a `source_index` that is the
positional index of its parent in `Final_Dataset.csv`. The published pipeline
balanced the classes first and split afterwards, which is what put variants of
the same article on both sides of the train/test boundary.

The augmentations themselves are fine — the ordering was the problem. This
module maps each augmented row back to its parent `item_id` so the splitter can
place it in whatever split the parent landed in, which by construction is the
training split only.
"""

from __future__ import annotations

import pandas as pd

from bmpb.paths import RAW
from bmpb.utils.logging import get_logger

log = get_logger(__name__)

AUGMENTED_TABLE = "Balanced_Augmented_Dataset.csv"
SOURCE_TABLE = "Final_Dataset.csv"


def load_published_augmentations(corpus: pd.DataFrame) -> pd.DataFrame:
    """Augmented rows, keyed to their parent's item_id.

    Returns an empty frame (not an error) when the augmented table is absent, so
    the pipeline still runs on an un-augmented corpus.
    """
    path = RAW / AUGMENTED_TABLE
    source_path = RAW / SOURCE_TABLE
    if not path.exists() or not source_path.exists():
        log.info("no augmented table in data/raw/; training on original rows only")
        return pd.DataFrame()

    augmented = pd.read_csv(path)
    augmented = augmented[augmented["is_augmented"].astype(bool)].copy()
    if augmented.empty:
        return pd.DataFrame()

    # source_index is positional into Final_Dataset.csv, so resolve it there
    # rather than against the corpus, whose rows may have been filtered.
    source = pd.read_csv(source_path)
    id_column = next((c for c in ("Image_id",) if c in source), None)
    if id_column is None:
        log.warning("%s has no Image_id column; cannot attach augmentations", SOURCE_TABLE)
        return pd.DataFrame()

    from bmpb.data.ingest import normalize_item_id

    positions = augmented["source_index"].astype(int)
    valid = positions.between(0, len(source) - 1)
    if not valid.all():
        log.warning("%d augmented rows have an out-of-range source_index", int((~valid).sum()))
        augmented, positions = augmented[valid], positions[valid]

    parent_ids = source.iloc[positions][id_column].map(normalize_item_id).to_numpy()

    known = set(corpus["item_id"])
    out = pd.DataFrame(
        {
            "item_id": [f"{parent}__aug{i}" for i, parent in enumerate(parent_ids)],
            "parent_item_id": parent_ids,
            "title": augmented["Title"].to_numpy(),
            "text": augmented["Preprocessed_Text"].to_numpy(),
            "label": augmented["FINAL LABEL"].astype(int).to_numpy(),
            "is_augmented": True,
            "augmentation_strategy": augmented["augmentation_strategy"].to_numpy(),
        }
    )
    out = out[out["parent_item_id"].isin(known)].reset_index(drop=True)

    labels = corpus.set_index("item_id")["label_name"]
    out["label_name"] = out["parent_item_id"].map(labels)
    # An augmented row inherits its parent's image, if the parent has one.
    for column in ("image_path", "image_kind", "has_image", "outlet", "outlet_key"):
        if column in corpus:
            out[column] = out["parent_item_id"].map(corpus.set_index("item_id")[column])
    out["has_image"] = out.get("has_image", False).fillna(False)

    log.info(
        "%d augmented rows attached to %d parent articles (%s)",
        len(out),
        out["parent_item_id"].nunique(),
        out["augmentation_strategy"].value_counts().to_dict(),
    )
    return out


def attach(
    splits: dict[str, pd.DataFrame], augmentations: pd.DataFrame, target: str = "train"
) -> dict:
    """Add augmented rows to one split, following their parent article.

    Only rows whose parent is already in `target` are added, so this cannot
    reintroduce leakage no matter what the augmented table contains.
    """
    if augmentations.empty:
        return splits

    parents = set(splits[target]["item_id"])
    keep = augmentations[augmentations["parent_item_id"].isin(parents)].copy()
    dropped = len(augmentations) - len(keep)
    if dropped:
        log.info("dropped %d augmented rows whose parent is not in %s", dropped, target)

    keep["group_id"] = keep["parent_item_id"]
    keep["split"] = target
    merged = pd.concat([splits[target], keep], ignore_index=True)
    log.info("%s: %d -> %d rows after augmentation", target, len(splits[target]), len(merged))
    return {**splits, target: merged}
