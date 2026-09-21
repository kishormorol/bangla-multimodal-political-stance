"""Train one model from one config, and write a self-describing run directory.

Every run writes to experiments/<name>-<timestamp>/ containing:

    config.yaml        the exact config used
    run.json           git commit, seed, split sizes, device, wall time
    predictions.csv    item_id, label, predicted, plus per-class probabilities
    metrics.json       scores on the test split
    model/             weights, when params.save_model is true

That directory is the unit the results table is built from, so a number in the
paper can always be traced back to the run that produced it.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bmpb.config import ExperimentConfig
from bmpb.data.dataset import BiasDataset, Collator, load_image
from bmpb.data.splits import load_split
from bmpb.metrics import score
from bmpb.models.registry import build
from bmpb.paths import EXPERIMENTS
from bmpb.utils.logging import get_logger
from bmpb.utils.seed import set_seed

log = get_logger(__name__)

SKLEARN_FAMILIES = {"sklearn_text", "majority", "countvec_vit"}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - a run outside a checkout is fine
        return None


def run_dir(cfg: ExperimentConfig, out: Path | str = EXPERIMENTS) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = Path(out) / f"{cfg.name}-{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def pick_device(cfg: ExperimentConfig) -> str:
    import torch

    requested = cfg.params.get("device")
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    # Apple silicon: MPS is materially faster than CPU for these encoders.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def class_weights_for(labels: list[int], num_labels: int):
    import torch

    counts = np.bincount(np.asarray(labels), minlength=num_labels).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float)


def _train_sklearn(cfg: ExperimentConfig, splits: dict[str, pd.DataFrame], out: Path):
    model, _ = build(cfg)
    train, test = splits["train"], splits["test"]

    if cfg.family == "countvec_vit":
        train = train[train["has_image"].astype(bool)]
        test = test[test["has_image"].astype(bool)]
        train_images = [load_image(p, cfg.image_size) for p in train["image_path"]]
        test_images = [load_image(p, cfg.image_size) for p in test["image_path"]]
        model.fit(train["text"].tolist(), train_images, train["label"].tolist())
        predicted = model.predict(test["text"].tolist(), test_images)
    else:
        model.fit(train["text"].tolist(), train["label"].tolist())
        predicted = model.predict(test["text"].tolist())

    return test, np.asarray(predicted), None


def _train_torch(cfg: ExperimentConfig, splits: dict[str, pd.DataFrame], out: Path):
    import torch
    from torch.utils.data import DataLoader

    device = pick_device(cfg)
    log.info("device: %s", device)

    model, processor = build(cfg)
    model.to(device)

    datasets = {
        name: BiasDataset(
            frame,
            use_text=cfg.uses_text,
            use_image=cfg.uses_image,
            require_image=cfg.uses_image,
            image_size=cfg.image_size,
        )
        for name, frame in splits.items()
    }
    collate = Collator(
        processor=processor,
        max_length=cfg.max_length,
        use_text=cfg.uses_text,
        use_image=cfg.uses_image,
    )
    loaders = {
        name: DataLoader(
            ds,
            batch_size=cfg.batch_size if name == "train" else cfg.eval_batch_size,
            shuffle=name == "train",
            collate_fn=collate,
        )
        for name, ds in datasets.items()
    }

    weights = (
        class_weights_for(datasets["train"].labels, cfg.num_labels).to(device)
        if cfg.class_weights
        else None
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    steps = max(1, len(loaders["train"])) * cfg.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.learning_rate, total_steps=steps, pct_start=cfg.warmup_ratio
    )

    best = {"macro_f1": -1.0, "state": None, "epoch": -1}
    patience = cfg.early_stopping_patience
    stale = 0

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for batch in loaders["train"]:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}
            outputs = model(**batch)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()

        val_true, val_pred, _ = _predict(model, loaders["val"], device)
        val = score(val_true, val_pred, bootstrap=False)
        log.info(
            "epoch %d/%d  loss=%.4f  val macro-F1=%.3f",
            epoch + 1,
            cfg.epochs,
            running / max(1, len(loaders["train"])),
            val.macro_f1,
        )

        if val.macro_f1 > best["macro_f1"]:
            best = {
                "macro_f1": val.macro_f1,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
            }
            stale = 0
        else:
            stale += 1
            if patience and stale >= patience:
                log.info("early stop at epoch %d (best was epoch %d)", epoch + 1, best["epoch"] + 1)
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    if cfg.params.get("save_model"):
        torch.save(model.state_dict(), out / "model.pt")

    true, predicted, probabilities = _predict(model, loaders["test"], device)
    test_frame = datasets["test"].frame
    return test_frame, predicted, probabilities


def _predict(model, loader, device):
    import torch

    model.eval()
    all_true, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}
            outputs = model(**batch)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            all_prob.append(probabilities)
            all_pred.append(probabilities.argmax(axis=-1))
            all_true.append(labels.numpy())
    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.vstack(all_prob),
    )


def train(config: str | Path, out: Path | str = EXPERIMENTS) -> Path:
    cfg = ExperimentConfig.load(config)
    set_seed(cfg.seed)
    directory = run_dir(cfg, out)
    log.info("run: %s", directory)

    splits = {name: load_split(name) for name in ("train", "val", "test")}
    started = time.time()

    if cfg.family in SKLEARN_FAMILIES:
        test_frame, predicted, probabilities = _train_sklearn(cfg, splits, directory)
    else:
        test_frame, predicted, probabilities = _train_torch(cfg, splits, directory)

    y_true = test_frame["label"].to_numpy()
    scores = score(y_true, predicted, seed=cfg.seed)
    log.info("test  %s", scores.summary())
    for note in scores.notes:
        log.warning("note: %s", note)

    predictions = pd.DataFrame(
        {
            "item_id": test_frame["item_id"].to_numpy(),
            "label": y_true,
            "predicted": predicted,
        }
    )
    if probabilities is not None:
        for i in range(probabilities.shape[1]):
            predictions[f"prob_{i}"] = probabilities[:, i]
    predictions.to_csv(directory / "predictions.csv", index=False)

    (directory / "config.yaml").write_text(
        yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8"
    )
    (directory / "metrics.json").write_text(
        json.dumps(scores.to_dict(), indent=2), encoding="utf-8"
    )
    (directory / "run.json").write_text(
        json.dumps(
            {
                "name": cfg.name,
                "family": cfg.family,
                "modality": cfg.modality,
                "pretrained": cfg.pretrained,
                "seed": cfg.seed,
                "git_commit": git_commit(),
                "split_sizes": {k: len(v) for k, v in splits.items()},
                "test_rows_scored": int(len(test_frame)),
                "seconds": round(time.time() - started, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return directory


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=str(EXPERIMENTS))
    args = parser.parse_args()
    train(args.config, args.out)
