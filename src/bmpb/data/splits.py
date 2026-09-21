"""Train/val/test splits, grouped so augmentation cannot leak across them.

Why this module exists
----------------------
The splits shipped in the Drive folder (`train_set.csv`, `val_set.csv`,
`test_set.csv`, 218/47/47 rows) were cut from `Balanced_Augmented_Dataset.csv`
*after* augmentation. Because an augmented row keeps the `source_index` of the
article it was derived from, the same article ends up on both sides of the
boundary:

    source_index overlap train/val:  17
    source_index overlap train/test: 25
    source_index overlap val/test:    3

A model can therefore see a swap-augmented variant of a test article during
training, and any score computed on that test set is optimistic.

`make splits` replaces them with a grouped, stratified split: articles are
assigned to a split first, augmented variants follow their parent, and only then
is the training half balanced. `verify_no_leakage` is asserted in the test suite
so the problem cannot come back silently.

The original splits stay in data/raw/ and can still be loaded with
`load_original_splits()` for a like-for-like comparison against the numbers
already in the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bmpb.config import DataConfig
from bmpb.data.augment import attach, load_published_augmentations
from bmpb.paths import CORPUS, PROCESSED, RAW, TEST, TRAIN, VAL
from bmpb.utils.logging import get_logger

log = get_logger(__name__)

SPLIT_FILES = {"train": TRAIN, "val": VAL, "test": TEST}


@dataclass
class LeakageReport:
    """Group ids shared between two splits. Empty means the split is clean."""

    train_val: set
    train_test: set
    val_test: set

    @property
    def clean(self) -> bool:
        return not (self.train_val or self.train_test or self.val_test)

    def describe(self) -> str:
        if self.clean:
            return "no group overlap between splits"
        return (
            f"train/val={len(self.train_val)} "
            f"train/test={len(self.train_test)} "
            f"val/test={len(self.val_test)}"
        )


def verify_no_leakage(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, group_key: str = "group_id"
) -> LeakageReport:
    def groups(df: pd.DataFrame) -> set:
        return set(df[group_key].dropna()) if group_key in df else set()

    g_train, g_val, g_test = groups(train), groups(val), groups(test)
    return LeakageReport(
        train_val=g_train & g_val,
        train_test=g_train & g_test,
        val_test=g_val & g_test,
    )


def _assign_groups(
    groups: pd.DataFrame, fractions: dict[str, float], rng: np.random.Generator
) -> dict[object, str]:
    """Greedy stratified assignment of whole groups to splits.

    Groups are walked label by label, shuffled, and handed to whichever split is
    furthest below its quota for that label. With ~200 articles across three
    classes this tracks the target proportions closely while keeping every group
    intact.
    """
    assignment: dict[object, str] = {}
    for label, label_groups in groups.groupby("label"):
        order = label_groups.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        quota = {name: frac * len(order) for name, frac in fractions.items()}
        taken = dict.fromkeys(fractions, 0.0)
        for group_id, size in zip(order["group_id"], order["size"], strict=True):
            target = max(fractions, key=lambda name: quota[name] - taken[name])
            assignment[group_id] = target
            taken[target] += size
        log.debug("label %s: %s", label, taken)
    return assignment


def make_splits(
    corpus: pd.DataFrame | None = None, cfg: DataConfig | None = None, write: bool = True
) -> dict[str, pd.DataFrame]:
    cfg = cfg or DataConfig.load()
    if corpus is None:
        if not CORPUS.exists():
            raise FileNotFoundError(f"{CORPUS} not found. Run `make ingest` first.")
        corpus = pd.read_csv(CORPUS)

    spec = cfg.split
    fractions = {
        "train": spec.get("train", 0.7),
        "val": spec.get("val", 0.15),
        "test": spec.get("test", 0.15),
    }
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total}")

    df = corpus.copy()
    # One group per source article. Un-augmented rows are their own group.
    group_key = spec.get("group_key", "source_index")
    if group_key in df and df[group_key].notna().any():
        df["group_id"] = df[group_key].fillna(pd.Series(df["item_id"], index=df.index))
    else:
        df["group_id"] = df["item_id"]

    groups = (
        df.groupby("group_id")
        .agg(label=("label", lambda s: s.mode().iat[0]), size=("label", "size"))
        .reset_index()
    )

    rng = np.random.default_rng(spec.get("seed", 42))
    assignment = _assign_groups(groups, fractions, rng)
    df["split"] = df["group_id"].map(assignment)

    out = {name: df[df["split"] == name].reset_index(drop=True) for name in fractions}

    # Augmentation comes after the split, never before: this is the ordering
    # that keeps a swap-augmented variant of a test article out of training.
    if spec.get("augment_train", True):
        out = attach(out, load_published_augmentations(df), target="train")

    report = verify_no_leakage(out["train"], out["val"], out["test"])
    if not report.clean:  # pragma: no cover - guarded by construction
        raise AssertionError(f"grouped split still leaks: {report.describe()}")
    log.info(
        "split sizes train=%d val=%d test=%d (%s)",
        len(out["train"]),
        len(out["val"]),
        len(out["test"]),
        report.describe(),
    )
    for name, frame in out.items():
        counts = frame["label_name"].value_counts().to_dict()
        log.info("  %-5s %s", name, counts)

    if write:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        for name, frame in out.items():
            frame.to_csv(SPLIT_FILES[name], index=False)
            log.info("wrote %s", SPLIT_FILES[name])
    return out


def load_original_splits() -> dict[str, pd.DataFrame]:
    """The as-published splits from the Drive folder, leakage and all.

    Useful only for reproducing the numbers already reported; do not use these
    for a new claim.
    """
    files = {"train": "train_set.csv", "val": "val_set.csv", "test": "test_set.csv"}
    out = {}
    for name, filename in files.items():
        path = RAW / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `make data` first.")
        frame = pd.read_csv(path)
        frame["group_id"] = frame.get("source_index")
        out[name] = frame
    return out


def load_split(name: str) -> pd.DataFrame:
    path = SPLIT_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make splits` first.")
    return pd.read_csv(path)


if __name__ == "__main__":
    make_splits()
