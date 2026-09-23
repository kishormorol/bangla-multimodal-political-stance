"""Score text models on the 86-item image subset for apples-to-apples comparison."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# ── paths ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "processed" / "corpus.csv"
EXPERIMENTS = ROOT / "experiments"
OUT_DIR = ROOT / "reports" / "tables"

# ── label names ────────────────────────────────────────────────────────
LABEL_NAMES = ("govt_critique", "neutral", "govt_leaning")

# Text-only model prefixes (exclude multimodal)
TEXT_MODELS = {"banglabert", "bangla_electra", "mbert", "mt5", "xlmr",
               "tfidf_logreg", "majority"}


def bootstrap_macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, rounds: int = 2000, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"))
    samples = np.empty(rounds)
    for i in range(rounds):
        idx = rng.integers(0, n, n)
        samples[i] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
    return (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


def latest_dir(model: str) -> pathlib.Path | None:
    """Find the latest experiment directory for a given model."""
    dirs = sorted(EXPERIMENTS.glob(f"cv-{model}-*"))
    if not dirs:
        return None
    # pick last by sorted name (timestamp suffix)
    return dirs[-1]


def main() -> None:
    # 1. Get the 86 item_ids with images
    corpus = pd.read_csv(CORPUS)
    image_items = corpus.loc[corpus["has_image"] == True, "item_id"].tolist()
    print(f"Image subset: {len(image_items)} items")
    assert len(image_items) in (86, 87), f"Expected ~86, got {len(image_items)}"
    image_set = set(image_items)

    rows = []
    for model in sorted(TEXT_MODELS):
        d = latest_dir(model)
        if d is None:
            print(f"  {model}: no experiment directory found, skipping")
            continue
        pred_path = d / "predictions.csv"
        if not pred_path.exists():
            print(f"  {model}: no predictions.csv in {d.name}, skipping")
            continue

        preds = pd.read_csv(pred_path)
        # filter to image subset
        subset = preds[preds["item_id"].isin(image_set)].copy()
        n = len(subset)
        if n == 0:
            print(f"  {model}: 0 items overlap with image subset, skipping")
            continue

        y_true = subset["label"].values
        y_pred = subset["predicted"].values

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # per-class F1
        _, _, per_class_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], zero_division=0
        )

        # bootstrap CI
        ci_lo, ci_hi = bootstrap_macro_f1(y_true, y_pred, rounds=2000)

        row = {
            "Model": model,
            "n": n,
            "Accuracy": round(acc, 3),
            "Macro-F1": round(macro_f1, 3),
            "95% CI": f"{ci_lo:.3f}\u2013{ci_hi:.3f}",
            "F1_gc": round(float(per_class_f1[0]), 3),
            "F1_n": round(float(per_class_f1[1]), 3),
            "F1_gl": round(float(per_class_f1[2]), 3),
            "dir": d.name,
        }
        rows.append(row)
        print(f"  {model}: n={n}  acc={acc:.3f}  macro-F1={macro_f1:.3f}  CI=[{ci_lo:.3f}, {ci_hi:.3f}]")

    if not rows:
        print("No text model results found.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # ── save CSV (without dir column) ──────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "text_on_image_subset.csv"
    df.drop(columns=["dir"]).to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")

    # ── save Markdown ──────────────────────────────────────────────────
    md_lines = [
        f"# Text models scored on the {len(image_items)}-item image subset",
        "",
        f"Apples-to-apples comparison surface: only the {len(image_items)} items that have an image.",
        "Predictions come from each text model's latest 5-fold CV run, filtered to this subset.",
        "Bootstrap CI is 2000 rounds on macro-F1.",
        "",
        "| Model | n | Accuracy | Macro-F1 | 95% CI | F1_gc | F1_n | F1_gl |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for _, r in df.iterrows():
        md_lines.append(
            f"| {r['Model']} | {r['n']} | {r['Accuracy']:.3f} | {r['Macro-F1']:.3f} "
            f"| {r['95% CI']} | {r['F1_gc']:.3f} | {r['F1_n']:.3f} | {r['F1_gl']:.3f} |"
        )
    md_lines.append("")

    md_path = OUT_DIR / "text_on_image_subset.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown saved to {md_path}")


if __name__ == "__main__":
    main()
