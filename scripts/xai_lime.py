"""LIME explanations for BanglaBERT headline stance predictions.

Generates per-item token importance scores showing which words in each
headline drive the model's prediction. Requires GPU (run on Colab).

Usage:
    python scripts/xai_lime.py --config configs/text/banglabert.yaml --n 30

Outputs:
    reports/tables/lime_explanations.csv   — per-token weights for top-n items
    reports/figures/lime_top10.png         — visual explanation for 10 items
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bmpb.config import ExperimentConfig
from bmpb.data.ingest import LABEL_NAMES
from bmpb.paths import CORPUS, EXPERIMENTS, FIGURES, TABLES


def load_model_and_processor(config_path: str):
    """Load a trained model from its config."""
    cfg = ExperimentConfig.from_yaml(config_path)
    from bmpb.models.registry import build
    model, processor = build(cfg)

    # Find latest checkpoint
    latest = sorted(EXPERIMENTS.glob(f"cv-{cfg.name}-*"))[-1]
    ckpt = latest / "best_model.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model, processor, cfg, device


def make_predict_fn(model, processor, device, max_length=128):
    """Return a function that takes a list of texts and returns probabilities."""
    def predict(texts):
        probs_list = []
        for text in texts:
            inputs = processor(
                text, return_tensors="pt", truncation=True,
                max_length=max_length, padding="max_length"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()
                      if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            probs_list.append(probs)
        return np.array(probs_list)
    return predict


def run_lime(config_path: str, n_items: int = 30, n_features: int = 10):
    """Run LIME on the hardest items and save explanations."""
    model, processor, cfg, device = load_model_and_processor(config_path)
    predict_fn = make_predict_fn(model, processor, device, cfg.max_length)

    corpus = pd.read_csv(CORPUS)

    # Load predictions to find misclassified items (more interesting)
    latest = sorted(EXPERIMENTS.glob(f"cv-{cfg.name}-*"))[-1]
    preds = pd.read_csv(latest / "predictions.csv")
    preds_merged = preds.merge(corpus[["item_id", "title"]], on="item_id")

    # Prioritize misclassified items, then correct ones
    wrong = preds_merged[preds_merged["label"] != preds_merged["predicted"]]
    right = preds_merged[preds_merged["label"] == preds_merged["predicted"]]
    items = pd.concat([wrong.head(n_items), right.head(max(0, n_items - len(wrong)))])
    items = items.head(n_items)

    print(f"Running LIME on {len(items)} items ({len(wrong)} misclassified)...")

    explainer = LimeTextExplainer(class_names=list(LABEL_NAMES))
    all_explanations = []

    for i, row in items.iterrows():
        headline = row["title"]
        true_label = int(row["label"])
        pred_label = int(row["predicted"])

        exp = explainer.explain_instance(
            headline, predict_fn,
            num_features=n_features, num_samples=500,
            labels=[0, 1, 2]
        )

        # Get feature weights for predicted class
        weights = exp.as_list(label=pred_label)
        for word, weight in weights:
            all_explanations.append({
                "item_id": row["item_id"],
                "headline": headline,
                "true_label": true_label,
                "true_name": LABEL_NAMES[true_label],
                "predicted": pred_label,
                "pred_name": LABEL_NAMES[pred_label],
                "correct": true_label == pred_label,
                "word": word,
                "weight": round(weight, 4),
            })

        if (i + 1) % 5 == 0:
            print(f"  [{len(all_explanations) // n_features}/{len(items)}] done")

    # Save
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_explanations)
    df.to_csv(TABLES / "lime_explanations.csv", index=False)
    print(f"Saved {len(df)} token weights to {TABLES / 'lime_explanations.csv'}")

    # Generate visualization for top 10
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        shown = 0

        for item_id in items["item_id"].unique()[:10]:
            item_df = df[df["item_id"] == item_id].sort_values("weight")
            ax = axes[shown]
            colors = ["#d32f2f" if w < 0 else "#388e3c" for w in item_df["weight"]]
            ax.barh(item_df["word"], item_df["weight"], color=colors)
            headline = item_df["headline"].iloc[0][:30]
            correct = item_df["correct"].iloc[0]
            ax.set_title(f"{'✓' if correct else '✗'} {headline}...", fontsize=8)
            ax.tick_params(labelsize=7)
            shown += 1

        for j in range(shown, 10):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(FIGURES / "lime_top10.png", dpi=150, bbox_inches="tight")
        print(f"Saved figure to {FIGURES / 'lime_top10.png'}")
    except Exception as e:
        print(f"Figure generation failed: {e}")

    # Summary stats
    print("\nTop positive/negative words across all items:")
    top_pos = df.groupby("word")["weight"].mean().nlargest(10)
    top_neg = df.groupby("word")["weight"].mean().nsmallest(10)
    print("\nMost govt_critique-indicating words:")
    for w, v in top_neg.items():
        print(f"  {w}: {v:.4f}")
    print("\nMost govt_leaning-indicating words:")
    for w, v in top_pos.items():
        print(f"  {w}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/text/banglabert.yaml")
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()
    run_lime(args.config, args.n)
