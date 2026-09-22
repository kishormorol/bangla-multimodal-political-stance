"""Zero-shot LLM baselines for BanglaPoliticalStance.

Runs Claude and/or GPT-4 on the 198 annotated headlines with no fine-tuning.
Outputs predictions compatible with `bmpb evaluate` and the leaderboard.

Usage:
    python scripts/llm_baseline.py --model claude       # Claude only
    python scripts/llm_baseline.py --model gpt4         # GPT-4 only
    python scripts/llm_baseline.py --model both         # both

Requires:
    ANTHROPIC_API_KEY  for Claude
    OPENAI_API_KEY     for GPT-4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "processed" / "corpus.csv"
EXPERIMENTS = ROOT / "experiments"

LABEL_NAMES = ["govt_critique", "neutral", "govt_leaning"]

SYSTEM_PROMPT = """You are a political stance classifier for Bangladeshi news headlines written in Bangla.

Given a Bangla news headline, classify its political stance toward the current Bangladesh government into exactly one of three categories:

0 = govt_critique: The headline frames the government, ruling party, or its policies negatively — criticism, protest coverage, accountability demands, or negative consequences of government action.
1 = neutral: Factual reporting with no clear positive or negative framing toward the government.
2 = govt_leaning: The headline frames the government or its actions positively — praise, defense, development achievements, or favorable comparisons.

Respond with ONLY the number (0, 1, or 2). Nothing else."""


def classify_claude(headline: str) -> int:
    """Classify a single headline using Claude."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=5,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": headline}],
    )
    text = response.content[0].text.strip()
    for char in text:
        if char in "012":
            return int(char)
    return -1


def classify_gpt4(headline: str) -> int:
    """Classify a single headline using GPT-4."""
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=5,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": headline},
        ],
    )
    text = response.choices[0].message.content.strip()
    for char in text:
        if char in "012":
            return int(char)
    return -1


def run_baseline(model_name: str, classify_fn, corpus: pd.DataFrame) -> pd.DataFrame:
    """Run a model on all corpus items and return predictions."""
    print(f"\n{'='*60}")
    print(f"Running {model_name} on {len(corpus)} items...")
    print(f"{'='*60}")

    predictions = []
    errors = 0

    for i, row in corpus.iterrows():
        headline = row["text"]
        true_label = int(row["label"])

        try:
            pred = classify_fn(headline)
        except Exception as e:
            print(f"  [{i+1}] Error: {e}")
            pred = -1
            errors += 1

        predictions.append({
            "item_id": row["item_id"],
            "headline": headline,
            "true_label": true_label,
            "predicted_label": pred,
            "true_name": LABEL_NAMES[true_label] if 0 <= true_label <= 2 else "unknown",
            "predicted_name": LABEL_NAMES[pred] if 0 <= pred <= 2 else "error",
        })

        if (i + 1) % 20 == 0:
            correct = sum(1 for p in predictions if p["true_label"] == p["predicted_label"])
            print(f"  [{i+1}/{len(corpus)}] accuracy so far: {correct}/{len(predictions)} ({100*correct/len(predictions):.1f}%)")

        # Rate limiting
        time.sleep(0.5)

    df = pd.DataFrame(predictions)

    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    valid = df[df["predicted_label"] >= 0]
    if len(valid) < len(df):
        print(f"\n  {len(df) - len(valid)} items had errors and were excluded")

    acc = accuracy_score(valid["true_label"], valid["predicted_label"])
    macro_f1 = f1_score(valid["true_label"], valid["predicted_label"], average="macro")

    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Macro-F1:  {macro_f1:.3f}")
    print(f"  Errors:    {errors}")
    print(f"\n{classification_report(valid['true_label'], valid['predicted_label'], target_names=LABEL_NAMES)}")

    return df


def save_results(model_name: str, df: pd.DataFrame):
    """Save predictions in the bmpb experiment format."""
    out_dir = EXPERIMENTS / f"llm-{model_name}-zeroshot"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    df.to_csv(out_dir / "predictions.csv", index=False)

    # Compute and save metrics
    from sklearn.metrics import accuracy_score, f1_score

    valid = df[df["predicted_label"] >= 0]
    acc = accuracy_score(valid["true_label"], valid["predicted_label"])
    macro_f1 = f1_score(valid["true_label"], valid["predicted_label"], average="macro")

    metrics = {
        "name": f"{model_name}_zeroshot",
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "n": len(valid),
        "errors": len(df) - len(valid),
        "protocol": "zero-shot (no training)",
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="LLM zero-shot baselines")
    parser.add_argument("--model", choices=["claude", "gpt4", "both"], default="claude")
    args = parser.parse_args()

    if not CORPUS.exists():
        sys.exit(f"{CORPUS} not found. Run `bmpb ingest` first.")

    corpus = pd.read_csv(CORPUS)
    print(f"Loaded {len(corpus)} items from corpus")

    if args.model in ("claude", "both"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY not set, skipping Claude")
        else:
            df = run_baseline("claude", classify_claude, corpus)
            save_results("claude", df)

    if args.model in ("gpt4", "both"):
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set, skipping GPT-4")
        else:
            df = run_baseline("gpt4", classify_gpt4, corpus)
            save_results("gpt4", df)


if __name__ == "__main__":
    main()
