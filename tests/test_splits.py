"""The split guarantee: augmentation must not straddle a split boundary."""

from __future__ import annotations

import pandas as pd
import pytest

from bmpb.config import DataConfig
from bmpb.data.splits import make_splits, verify_no_leakage


def _corpus(n_groups: int = 60) -> pd.DataFrame:
    """A corpus where every source article has one original and one augmented row."""
    rows = []
    for group in range(n_groups):
        label = group % 3
        for variant in range(2):
            rows.append(
                {
                    "item_id": f"Image_{group}_{variant}",
                    "title": "",
                    "text": f"article {group} variant {variant}",
                    "label": label,
                    "label_name": ["govt_critique", "neutral", "govt_leaning"][label],
                    "source_index": group,
                    "is_augmented": bool(variant),
                    "has_image": True,
                }
            )
    return pd.DataFrame(rows)


def test_grouped_split_has_no_source_overlap():
    splits = make_splits(_corpus(), DataConfig.load(), write=False)
    report = verify_no_leakage(splits["train"], splits["val"], splits["test"])
    assert report.clean, report.describe()


def test_every_row_lands_in_exactly_one_split():
    corpus = _corpus()
    splits = make_splits(corpus, DataConfig.load(), write=False)
    total = sum(len(frame) for frame in splits.values())
    assert total == len(corpus)
    ids = pd.concat([frame["item_id"] for frame in splits.values()])
    assert ids.is_unique


def test_all_three_labels_survive_in_every_split():
    splits = make_splits(_corpus(90), DataConfig.load(), write=False)
    for name, frame in splits.items():
        assert frame["label"].nunique() == 3, f"{name} lost a class"


def test_split_is_deterministic_for_a_seed():
    corpus = _corpus()
    cfg = DataConfig.load()
    first = make_splits(corpus, cfg, write=False)
    second = make_splits(corpus, cfg, write=False)
    for name in first:
        assert first[name]["item_id"].tolist() == second[name]["item_id"].tolist()


def test_fractions_must_sum_to_one():
    cfg = DataConfig.load()
    cfg.split = dict(cfg.split, train=0.8, val=0.3, test=0.15)
    with pytest.raises(ValueError, match="sum to 1.0"):
        make_splits(_corpus(), cfg, write=False)
