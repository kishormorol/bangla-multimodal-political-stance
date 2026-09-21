"""Data health check.

Run this before trusting any number. It answers the questions a reviewer asks
first: how many items are there really, how balanced are they, how many actually
have an image, how much do the annotators agree, and do the splits leak.
"""

from __future__ import annotations

import pandas as pd

from bmpb.data.ingest import LABEL_NAMES
from bmpb.data.splits import load_original_splits, verify_no_leakage
from bmpb.metrics import annotator_agreement
from bmpb.paths import CORPUS, RAW, TEST, TRAIN, VAL
from bmpb.utils.logging import get_logger

log = get_logger(__name__)


def run_audit() -> dict:
    report: dict = {}

    if not CORPUS.exists():
        raise FileNotFoundError(f"{CORPUS} not found. Run `bmpb ingest` first.")
    corpus = pd.read_csv(CORPUS)

    report["corpus"] = {
        "items": len(corpus),
        "labels": corpus["label_name"].value_counts().to_dict(),
        "majority_class_share": round(
            float(corpus["label_name"].value_counts(normalize=True).max()), 3
        ),
        "with_image": int(corpus["has_image"].astype(bool).sum()),
        "without_image": int((~corpus["has_image"].astype(bool)).sum()),
        "outlets": int(corpus["outlet_key"].nunique()) if "outlet_key" in corpus else None,
        "median_text_chars": int(corpus["text"].astype(str).str.len().median()),
        "median_text_words": int(corpus["text"].astype(str).str.split().str.len().median()),
    }

    # The `Preprocessed_Text` column turned out to hold the headline, not the
    # article body (median ~8 words). Every "text model" number in the study is
    # therefore headline classification, which is worth stating out loud rather
    # than discovering during review.
    median_words = report["corpus"]["median_text_words"]
    if median_words < 30:
        report["corpus"]["warning"] = (
            f"median text is {median_words} words — this is headline-length, not article "
            "body. Text-model scores should be described as headline classification."
        )
    report["corpus"]["text_equals_title_share"] = (
        round(
            float(
                (
                    corpus["text"].astype(str).str.strip()
                    == corpus.get("title", pd.Series(dtype=str)).astype(str).str.strip()
                ).mean()
            ),
            3,
        )
        if "title" in corpus
        else None
    )

    # Label balance restricted to the multimodal subset: the vision models are
    # evaluated on a different, smaller population than the text models, which
    # is the single biggest reason their numbers are not directly comparable.
    with_image = corpus[corpus["has_image"].astype(bool)]
    report["multimodal_subset"] = {
        "items": len(with_image),
        "labels": with_image["label_name"].value_counts().to_dict(),
    }

    annotator_columns = [c for c in ("annotator_1", "annotator_2", "annotator_3") if c in corpus]
    if annotator_columns:
        coverage = {c: int(corpus[c].notna().sum()) for c in annotator_columns}
        report["annotators"] = {
            "coverage": coverage,
            "pairwise_cohen_kappa": annotator_agreement(corpus, annotator_columns),
        }

    # Article-level vs image-level labels: how often does the photo carry a
    # different stance from the text?
    if {"article_label_name", "image_label_name"}.issubset(corpus.columns):
        both = corpus[["article_label_name", "image_label_name"]].dropna()
        if len(both):
            agree = int((both["article_label_name"] == both["image_label_name"]).sum())
            report["modality_label_agreement"] = {
                "items_labelled_both_ways": len(both),
                "agree": agree,
                "agreement_rate": round(agree / len(both), 3),
            }

    report["splits"] = {}
    if all(p.exists() for p in (TRAIN, VAL, TEST)):
        train, val, test = (pd.read_csv(p) for p in (TRAIN, VAL, TEST))
        leakage = verify_no_leakage(train, val, test)
        report["splits"]["grouped"] = {
            "sizes": {"train": len(train), "val": len(val), "test": len(test)},
            "labels": {
                "train": train["label_name"].value_counts().to_dict(),
                "val": val["label_name"].value_counts().to_dict(),
                "test": test["label_name"].value_counts().to_dict(),
            },
            "leakage": leakage.describe(),
            "clean": leakage.clean,
        }
    else:
        report["splits"]["grouped"] = "not built yet — run `bmpb splits`"

    try:
        original = load_original_splits()
        leakage = verify_no_leakage(original["train"], original["val"], original["test"])
        report["splits"]["as_published"] = {
            "sizes": {k: len(v) for k, v in original.items()},
            "leakage": leakage.describe(),
            "clean": leakage.clean,
            "warning": (
                (
                    "augmented variants of the same source article appear in more than one "
                    "split; scores computed on this test set are optimistic"
                )
                if not leakage.clean
                else None
            ),
        }
    except FileNotFoundError:
        report["splits"]["as_published"] = "original split CSVs not in data/raw/"

    images_dir = RAW / "images"
    processed_dir = RAW / "processed_images"
    report["images"] = {
        "raw_files": len(list(images_dir.glob("*"))) if images_dir.exists() else 0,
        "processed_files": len(list(processed_dir.glob("*"))) if processed_dir.exists() else 0,
        "raw_extensions": (
            sorted({p.suffix.lower() for p in images_dir.glob("*") if p.is_file()})
            if images_dir.exists()
            else []
        ),
    }

    report["labels_reference"] = {str(i): name for i, name in enumerate(LABEL_NAMES)}
    return report


if __name__ == "__main__":
    import json

    print(json.dumps(run_audit(), indent=2, default=str))
