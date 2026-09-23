"""Grouped stratified k-fold cross-validation — the one protocol for every model.

Why this replaces the per-model protocols
-----------------------------------------
The first submission scored its four result tables three different ways: the
text models on a fixed 70/30 split of the augmented rows, the baseline
multimodal models on a single 80/20 split of 30 items, and the proposed models
on a genuine 5-fold CV over 149. A number from one table cannot be compared with
a number from another, which is exactly what the headline claim did.

Here every model sees the same folds over the same items. Three properties make
that comparison hold:

* **Grouped.** Folds are cut on the source article, so an augmented variant
  never lands in the fold that evaluates its parent.
* **Augmented inside the fold.** Augmentation is applied to each training half
  after the split, never before it.
* **Pooled out-of-fold predictions.** Every item is predicted exactly once, by
  the fold that did not train on it, so the headline score is computed over the
  whole corpus rather than averaged over five small test sets. The per-fold
  spread is reported alongside it, because a mean that hides a 15-point fold
  range is not a result.

Reported per model: pooled macro-F1 with a bootstrap interval, and the per-fold
mean ± standard deviation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bmpb.config import DataConfig, ExperimentConfig
from bmpb.data.augment import attach, load_published_augmentations
from bmpb.metrics import score
from bmpb.paths import CORPUS, EXPERIMENTS
from bmpb.utils.logging import get_logger
from bmpb.utils.seed import set_seed

log = get_logger(__name__)


def make_folds(
    corpus: pd.DataFrame, n_splits: int = 5, group_key: str = "source_index", seed: int = 42
) -> list[np.ndarray]:
    """Fold assignment over groups, balancing labels across folds.

    sklearn's StratifiedGroupKFold does this, but it is only stratified in
    expectation and with ~200 items in three classes the folds drift. Groups are
    instead walked label by label and dealt to whichever fold is currently
    smallest for that label, which keeps every fold's distribution close to the
    corpus.
    """
    df = corpus.copy()
    if group_key in df and df[group_key].notna().any():
        df["group_id"] = df[group_key].fillna(pd.Series(df["item_id"], index=df.index))
    else:
        df["group_id"] = df["item_id"]

    groups = (
        df.groupby("group_id")
        .agg(label=("label", lambda s: s.mode().iat[0]), size=("label", "size"))
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    assignment: dict[object, int] = {}
    for _, label_groups in groups.groupby("label"):
        order = label_groups.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        load = np.zeros(n_splits)
        for group_id, size in zip(order["group_id"], order["size"], strict=True):
            fold = int(np.argmin(load))
            assignment[group_id] = fold
            load[fold] += size

    fold_of = df["group_id"].map(assignment).to_numpy()
    return [np.where(fold_of == f)[0] for f in range(n_splits)]


def _fit_predict(cfg: ExperimentConfig, train: pd.DataFrame, test: pd.DataFrame):
    """Train on one fold's training half and predict its held-out half."""
    from bmpb.train import SKLEARN_FAMILIES, _train_sklearn, _train_torch

    splits = {"train": train, "val": test, "test": test}
    if cfg.family in SKLEARN_FAMILIES:
        frame, predicted, probabilities = _train_sklearn(cfg, splits, Path("."))
    else:
        frame, predicted, probabilities = _train_torch(cfg, splits, Path("."))
    return frame, predicted, probabilities


def cross_validate(
    config: str | Path,
    n_splits: int = 5,
    out: Path | str = EXPERIMENTS,
    corpus: pd.DataFrame | None = None,
    require_image: bool | None = None,
) -> dict:
    cfg = ExperimentConfig.load(config)
    data_cfg = DataConfig.load()
    set_seed(cfg.seed)

    if corpus is None:
        if not CORPUS.exists():
            raise FileNotFoundError(f"{CORPUS} not found. Run `bmpb ingest` first.")
        corpus = pd.read_csv(CORPUS)

    # A model that needs an image can only be scored on items that have one.
    # That is a smaller population, so the run records it explicitly rather than
    # letting the difference hide inside the metric.
    needs_image = cfg.uses_image if require_image is None else require_image
    population = "all items"
    if needs_image:
        corpus = corpus[corpus["has_image"].astype(bool)].reset_index(drop=True)
        population = "items with an image"
    log.info("%s: %d %s, %d folds", cfg.name, len(corpus), population, n_splits)

    folds = make_folds(
        corpus,
        n_splits=n_splits,
        group_key=data_cfg.split.get("group_key", "source_index"),
        seed=cfg.seed,
    )
    augmentations = load_published_augmentations(corpus) if cfg.augment else pd.DataFrame()

    started = time.time()
    per_fold: list[dict] = []
    pooled: list[pd.DataFrame] = []

    for index, test_idx in enumerate(folds, start=1):
        test = corpus.iloc[test_idx].reset_index(drop=True)
        train = corpus.drop(index=test_idx).reset_index(drop=True)
        if test.empty or train.empty:
            log.warning("fold %d is empty; skipping", index)
            continue

        # Augment the training half only, after the cut.
        grouped = attach({"train": train, "val": test, "test": test}, augmentations, target="train")
        train = grouped["train"]

        frame, predicted, _ = _fit_predict(cfg, train, test)
        truth = frame["label"].to_numpy()
        fold_scores = score(truth, predicted, bootstrap=False)
        per_fold.append(
            {
                "fold": index,
                "n_train": len(train),
                "n_test": len(frame),
                "accuracy": fold_scores.accuracy,
                "macro_f1": fold_scores.macro_f1,
            }
        )
        pooled.append(
            pd.DataFrame(
                {
                    "item_id": frame["item_id"].to_numpy(),
                    "label": truth,
                    "predicted": predicted,
                    "fold": index,
                }
            )
        )
        log.info(
            "  fold %d/%d  n=%d  macro-F1=%.3f",
            index,
            len(folds),
            len(frame),
            fold_scores.macro_f1,
        )

    if not pooled:
        raise RuntimeError(f"{cfg.name}: no fold produced predictions")

    predictions = pd.concat(pooled, ignore_index=True)
    overall = score(predictions["label"], predictions["predicted"], seed=cfg.seed)
    fold_f1 = np.array([f["macro_f1"] for f in per_fold])

    log.info(
        "%s  pooled macro-F1=%.3f [%.3f-%.3f]  per-fold %.3f ± %.3f",
        cfg.name,
        overall.macro_f1,
        overall.macro_f1_ci95[0],
        overall.macro_f1_ci95[1],
        fold_f1.mean(),
        fold_f1.std(ddof=1) if len(fold_f1) > 1 else 0.0,
    )
    for note in overall.notes:
        log.warning("  note: %s", note)

    directory = Path(out) / f"cv-{cfg.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    directory.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(directory / "predictions.csv", index=False)
    (directory / "config.yaml").write_text(
        yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8"
    )

    summary = {
        "name": cfg.name,
        "modality": cfg.modality,
        "family": cfg.family,
        "pretrained": cfg.pretrained,
        "protocol": f"grouped stratified {n_splits}-fold CV, {'augmentation inside training folds' if cfg.augment else 'no augmentation'}",
        "population": population,
        "items_scored": int(len(predictions)),
        "seed": cfg.seed,
        "accuracy": overall.accuracy,
        "macro_f1": overall.macro_f1,
        "macro_f1_ci95": list(overall.macro_f1_ci95) if overall.macro_f1_ci95 else None,
        "fold_macro_f1_mean": float(fold_f1.mean()),
        "fold_macro_f1_sd": float(fold_f1.std(ddof=1)) if len(fold_f1) > 1 else 0.0,
        "fold_macro_f1_min": float(fold_f1.min()),
        "fold_macro_f1_max": float(fold_f1.max()),
        "per_class_f1": overall.per_class_f1,
        "majority_baseline_accuracy": overall.majority_baseline_accuracy,
        "confusion": overall.confusion,
        "notes": overall.notes,
        "folds": per_fold,
        "seconds": round(time.time() - started, 1),
    }
    (directory / "cv.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("wrote %s", directory)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", default=str(EXPERIMENTS))
    args = parser.parse_args()
    cross_validate(args.config, n_splits=args.folds, out=args.out)
