"""Ingest must survive the specific messiness of the raw sheets."""

from __future__ import annotations

import pandas as pd
import pytest

from bmpb.config import DataConfig
from bmpb.data.ingest import (
    LABEL_TO_ID,
    canonical_label,
    normalize_item_id,
    resolve_columns,
)


@pytest.fixture
def label_map() -> dict[str, str]:
    return DataConfig.load().label_map


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Image_01", "Image_1"),  # zero-padded in the first rows only
        ("Image_2", "Image_2"),
        ("image 3", "Image_3"),
        ("Image_107 ", "Image_107"),
        (None, None),
    ],
)
def test_item_ids_are_normalized(raw, expected):
    assert normalize_item_id(raw) == expected


def test_label_accepts_strings_ids_and_drifting_case(label_map):
    assert canonical_label("Govt Critique", label_map) == "govt_critique"
    assert canonical_label("Govt critique", label_map) == "govt_critique"  # raw sheet casing
    assert canonical_label(0, label_map) == "govt_critique"
    assert canonical_label("0.0", label_map) == "govt_critique"  # pandas float coercion
    assert canonical_label("Neutral", label_map) == "neutral"
    assert canonical_label(2, label_map) == "govt_leaning"
    assert canonical_label(None, label_map) is None


def test_label_ids_match_the_published_predictions():
    # The prediction CSVs in data/raw/ are keyed to these ids; renumbering them
    # would silently invalidate every published number.
    assert LABEL_TO_ID == {"govt_critique": 0, "neutral": 1, "govt_leaning": 2}


def test_column_aliases_resolve_across_sheets():
    aliases = {"label": ["FINAL LABEL", "Final_Label"], "text": ["Preprocessed_Text"]}
    article_sheet = pd.DataFrame(columns=["FINAL LABEL", "Preprocessed_Text"])
    image_sheet = pd.DataFrame(columns=["Final_Label", "Preprocessed_Text"])
    assert resolve_columns(article_sheet, aliases)["label"] == "FINAL LABEL"
    assert resolve_columns(image_sheet, aliases)["label"] == "Final_Label"
