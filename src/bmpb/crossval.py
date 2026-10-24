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

import hashlib
import importlib.metadata
import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bmpb.config import DataConfig, ExperimentConfig, pin_model_revisions
from bmpb.data.augment import attach, load_published_augmentations
from bmpb.data.groups import source_groups
from bmpb.metrics import score
from bmpb.paths import CORPUS, EXPERIMENTS, RAW
from bmpb.utils.logging import get_logger
from bmpb.utils.seed import set_seed

log = get_logger(__name__)


def execution_identity(
    config: str | Path | ExperimentConfig, corpus: pd.DataFrame, n_splits: int = 5
) -> dict:
    """Identity of the inputs and implementation that determine a CV run."""
    cfg = (
        config
        if isinstance(config, ExperimentConfig)
        else pin_model_revisions(ExperimentConfig.load(config))
    )
    data_cfg = DataConfig.load()
    frame = corpus.reset_index(drop=True)
    groups = source_groups(frame, data_cfg.split.get("group_key", "source_index"))
    folds = make_folds(
        frame,
        n_splits=n_splits,
        group_key=data_cfg.split.get("group_key", "source_index"),
        seed=cfg.seed,
    )
    rows = []
    group_rows = []
    for index, row in frame.iterrows():
        image_hash = None
        image_path = row.get("image_path")
        if isinstance(image_path, str) and image_path:
            path = Path(image_path)
            if not path.is_absolute():
                path = RAW / path
            if path.is_file():
                image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "item_id": str(row.item_id),
                "title": str(row.get("title", "")),
                "text": str(row.get("text", "")),
                "label": int(row.label),
                "source_url": str(row.get("source_url", "")),
                "group": str(groups.iloc[index]),
                "has_image": bool(row.get("has_image", False)),
                "image_sha256": image_hash,
            }
        )
        group_rows.append((str(row.item_id), str(groups.iloc[index])))
    input_sha = hashlib.sha256(
        json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    augmentation_requested = cfg.augment and cfg.params.get(
        "augment_train", data_cfg.split.get("augment_train", True)
    )
    augmentation_rows = []
    if augmentation_requested:
        augmentation_rows = load_published_augmentations(frame).to_dict("records")
    augmentation_sha = hashlib.sha256(
        json.dumps(augmentation_rows, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()
    package_root = Path(__file__).parent
    code_files = sorted(package_root.rglob("*.py")) + [
        Path(__file__).parents[2] / "scripts/run_matched_sweep.py"
    ]
    code_hash = hashlib.sha256()
    for path in code_files:
        if path.is_file():
            code_hash.update(
                str(
                    path.relative_to(package_root.parent.parent)
                    if path.is_relative_to(package_root.parent.parent)
                    else path.name
                ).encode()
            )
            code_hash.update(path.read_bytes())
    packages = {}
    for name in (
        "bmpb",
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "torchvision",
        "transformers",
        "tokenizers",
        "datasets",
        "safetensors",
        "Pillow",
        "huggingface-hub",
        "scipy",
        "normalizer",
        "accelerate",
        "sentencepiece",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "config": asdict(cfg),
        "seed": cfg.seed,
        "n_splits": n_splits,
        "fold_item_ids": [[str(frame.iloc[i]["item_id"]) for i in fold] for fold in folds],
        "input_sha256": input_sha,
        "group_membership_sha256": hashlib.sha256(
            json.dumps(group_rows, ensure_ascii=False).encode()
        ).hexdigest(),
        "augmentation_requested": bool(augmentation_requested),
        "augmentation_enabled": bool(augmentation_rows),
        "augmentation_count": len(augmentation_rows),
        "augmentation_sha256": augmentation_sha,
        "code_sha256": code_hash.hexdigest(),
        "software": {"python": sys.version, **packages},
    }


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
    if n_splits < 2:
        raise ValueError("Cross-validation requires at least two folds")
    df["group_id"] = source_groups(df, group_key)

    groups = (
        df.groupby("group_id")
        .agg(label=("label", lambda s: s.mode().iat[0]), size=("label", "size"))
        .reset_index()
    )
    if len(groups) < n_splits:
        raise ValueError("Cross-validation requires at least one source group per fold")

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


def _fit_predict(
    cfg: ExperimentConfig,
    train: pd.DataFrame,
    test: pd.DataFrame,
    val: pd.DataFrame | None = None,
):
    """Train on one fold's training half and predict its held-out half."""
    from bmpb.train import SKLEARN_FAMILIES, _train_sklearn, _train_torch

    if cfg.family not in SKLEARN_FAMILIES and val is None:
        raise ValueError("Neural CV requires validation data from the outer training fold")
    splits = {"train": train, "val": val, "test": test}
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
    cfg = pin_model_revisions(ExperimentConfig.load(config))
    from bmpb.train import SKLEARN_FAMILIES, git_commit

    data_cfg = DataConfig.load()
    set_seed(cfg.seed)

    if corpus is None:
        if not CORPUS.exists():
            raise FileNotFoundError(f"{CORPUS} not found. Run `bmpb ingest` first.")
        corpus = pd.read_csv(CORPUS)
    corpus = corpus.reset_index(drop=True)
    if corpus["item_id"].duplicated().any():
        raise ValueError("CV requires unique original item_id values")
    if "is_augmented" in corpus and corpus["is_augmented"].fillna(False).astype(bool).any():
        raise ValueError("CV corpus must contain original items only")

    # A model that needs an image can only be scored on items that have one.
    # That is a smaller population, so the run records it explicitly rather than
    # letting the difference hide inside the metric.
    needs_image = cfg.uses_image if require_image is None else require_image
    population = "all items"
    if needs_image:
        corpus = corpus[corpus["has_image"].astype(bool)].reset_index(drop=True)
        population = "items with an image"
    identity = execution_identity(cfg, corpus, n_splits)
    frozen_groups = source_groups(corpus, data_cfg.split.get("group_key", "source_index"))
    log.info("%s: %d %s, %d folds", cfg.name, len(corpus), population, n_splits)

    folds = make_folds(
        corpus,
        n_splits=n_splits,
        group_key=data_cfg.split.get("group_key", "source_index"),
        seed=cfg.seed,
    )
    if any(len(fold) == 0 for fold in folds):
        raise ValueError("CV requires a nonempty test set in every fold")
    augment_train = cfg.augment and cfg.params.get(
        "augment_train", data_cfg.split.get("augment_train", True)
    )
    augmentations = load_published_augmentations(corpus) if augment_train else pd.DataFrame()

    started = time.time()
    per_fold: list[dict] = []
    pooled: list[pd.DataFrame] = []

    for index, test_idx in enumerate(folds, start=1):
        test = corpus.iloc[test_idx].reset_index(drop=True)
        train = corpus.drop(index=test_idx).reset_index(drop=True)
        val = None
        if cfg.family not in SKLEARN_FAMILIES:
            # Select checkpoints using only a grouped split of outer training data.
            inner_folds = make_folds(
                train,
                n_splits=5,
                group_key=data_cfg.split.get("group_key", "source_index"),
                seed=cfg.seed + index,
            )
            val_idx = inner_folds[0]
            val = train.iloc[val_idx].reset_index(drop=True)
            train = train.drop(index=val_idx).reset_index(drop=True)
            if val.empty or train.empty:
                raise ValueError("Not enough groups for independent inner validation")
        if test.empty or train.empty:
            log.warning("fold %d is empty; skipping", index)
            continue

        # Augment the training half only, after the cut.
        grouped = attach({"train": train, "val": val, "test": test}, augmentations, target="train")
        train = grouped["train"]

        frame, predicted, _ = _fit_predict(cfg, train, test, val)
        truth = frame["label"].to_numpy()
        fold_scores = score(truth, predicted, bootstrap=False)
        augmentation_item_ids = (
            train.loc[train["is_augmented"].fillna(False).astype(bool), "item_id"]
            .astype(str)
            .tolist()
            if "is_augmented" in train
            else []
        )
        per_fold.append(
            {
                "fold": index,
                "n_train": len(train),
                "n_val": len(val) if val is not None else 0,
                "n_test": len(frame),
                "augmentation_item_ids": augmentation_item_ids,
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
                    "group": frozen_groups.iloc[test_idx].to_numpy(),
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
    if predictions["item_id"].duplicated().any() or set(predictions["item_id"]) != set(
        corpus["item_id"]
    ):
        raise AssertionError("Every eligible original item must be predicted exactly once")
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

    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(
        tempfile.mkdtemp(
            prefix=f"cv-{cfg.name}-{time.strftime('%Y%m%d-%H%M%S')}-",
            dir=output_dir,
        )
    )
    predictions.to_csv(directory / "predictions.csv", index=False)
    (directory / "config.yaml").write_text(
        yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8"
    )

    summary = {
        "run_dir": str(directory),
        "identity": identity,
        "name": cfg.name,
        "modality": cfg.modality,
        "family": cfg.family,
        "pretrained": cfg.pretrained,
        "protocol": f"grouped stratified {n_splits}-fold CV",
        "population": population,
        "items_scored": int(len(predictions)),
        "seed": cfg.seed,
        "git_commit": git_commit(),
        "augment_train": augment_train,
        "checkpoint_selection": (
            "grouped inner validation from outer training data"
            if cfg.family not in SKLEARN_FAMILIES
            else "none"
        ),
        "fold_item_ids": [[str(corpus.iloc[i]["item_id"]) for i in fold] for fold in folds],
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
    artifacts = {}
    for name in ("config.yaml", "predictions.csv", "cv.json"):
        artifacts[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    manifest = directory / "run.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"identity": identity, "artifacts": artifacts}, indent=2),
        encoding="utf-8",
    )
    temporary.replace(manifest)
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
