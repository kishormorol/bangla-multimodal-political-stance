"""Every filesystem location the project uses, resolved from the repo root.

Import paths from here instead of hardcoding strings, so a run from a notebook,
a test, or the CLI all land in the same place.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"  # untouched Drive mirror, never committed
INTERIM = DATA / "interim"  # intermediate artifacts, never committed
PROCESSED = DATA / "processed"  # canonical tables the models read

IMAGES_RAW = RAW / "images"
IMAGES_PROCESSED = RAW / "processed_images"

CONFIGS = ROOT / "configs"
EXPERIMENTS = ROOT / "experiments"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

# Canonical tables produced by `bmpb ingest` / `bmpb splits`.
CORPUS = PROCESSED / "corpus.csv"  # one row per annotated item
TRAIN = PROCESSED / "train.csv"
VAL = PROCESSED / "val.csv"
TEST = PROCESSED / "test.csv"
IMAGE_INDEX = PROCESSED / "image_index.csv"  # item id -> resolved image file


def ensure_dirs() -> None:
    """Create the writable directories a run needs."""
    for path in (RAW, INTERIM, PROCESSED, EXPERIMENTS, FIGURES, TABLES):
        path.mkdir(parents=True, exist_ok=True)
