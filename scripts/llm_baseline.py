"""Zero-shot LLM baselines for BanglaPoliticalStance.

Runs LLMs on the 198 annotated headlines with no fine-tuning.
Outputs predictions compatible with `bmpb evaluate` and the leaderboard.

Usage:
    python scripts/llm_baseline.py --model gemini          # Gemini Flash (free)
    python scripts/llm_baseline.py --model llama           # Llama 3.1 70B via Groq (free)
    python scripts/llm_baseline.py --model claude          # Claude Sonnet (~$0.06)
    python scripts/llm_baseline.py --model gpt4            # GPT-4o (~$0.03)
    python scripts/llm_baseline.py --model all             # all four

Requires (set the relevant env var):
    GOOGLE_API_KEY     for Gemini (free: https://aistudio.google.com/apikey)
    GROQ_API_KEY       for Llama via Groq (free: https://console.groq.com/keys)
    ANTHROPIC_API_KEY  for Claude
    OPENAI_API_KEY     for GPT-4
"""

from __future__ import annotations

import argparse
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


def _parse_label(text: str) -> int:
    """Extract 0/1/2 from model response."""
    for char in text.strip():
        if char in "012":
            return int(char)
    return -1


def classify_gemini(headline: str) -> int:
    """Classify using Google Gemini Flash (free tier)."""
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=headline,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=5,
            safety_settings=[
                types.SafetySetting(category=c, threshold="BLOCK_NONE")
                for c in [
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                ]
            ],
        ),
    )
    text = response.text
    if text is None:
        return -1
    return _parse_label(text)


def classify_groq(headline: str) -> int:
    """Classify using Qwen 3.8 27B via Groq (free tier)."""
    from groq import Groq

    client = Groq()
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        max_tokens=16,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": headline},
        ],
    )
    return _parse_label(response.choices[0].message.content)


def classify_claude(headline: str) -> int:
    """Classify using Claude Sonnet."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=5,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": headline}],
    )
    return _parse_label(response.content[0].text)


def classify_gpt4(headline: str) -> int:
    """Classify using GPT-4o."""
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
    return _parse_label(response.choices[0].message.content)


MODELS = {
    "gemini": ("GOOGLE_API_KEY", classify_gemini, 13),  # 5 RPM free tier
    "qwen": ("GROQ_API_KEY", classify_groq, 0.5),
    "claude": ("ANTHROPIC_API_KEY", classify_claude, 0.5),
    "gpt4": ("OPENAI_API_KEY", classify_gpt4, 0.5),
}


def run_baseline(model_name: str, classify_fn, corpus: pd.DataFrame, delay: float = 0.5) -> pd.DataFrame:
    """Run a model on all corpus items and return predictions."""
    print(f"\n{'='*60}")
    print(f"Running {model_name} on {len(corpus)} items...")
    print(f"{'='*60}")

    # Resume from partial results if they exist
    out_dir = EXPERIMENTS / f"llm-{model_name}-zeroshot"
    partial_path = out_dir / "predictions_partial.csv"
    done_ids = set()
    predictions = []
    if partial_path.exists():
        partial = pd.read_csv(partial_path)
        done_ids = set(partial["item_id"])
        predictions = partial.to_dict("records")
        print(f"  Resuming: {len(done_ids)} items already done")

    errors = 0
    for i, row in corpus.iterrows():
        if row["item_id"] in done_ids:
            continue

        headline = row["text"]
        true_label = int(row["label"])

        try:
            pred = classify_fn(headline)
        except Exception as e:
            print(f"  [{i+1}] Error: {e}")
            pred = -1
            errors += 1
            # Back off on errors
            time.sleep(5)

        predictions.append({
            "item_id": row["item_id"],
            "label": true_label,
            "predicted": pred,
            "fold": 0,  # compatible with CV prediction format
        })

        done_count = len(predictions)
        if done_count % 20 == 0:
            correct = sum(1 for p in predictions if p["label"] == p["predicted"])
            print(f"  [{done_count}/{len(corpus)}] accuracy so far: {correct}/{done_count} ({100*correct/done_count:.1f}%)")
            # Save partial results
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(predictions).to_csv(partial_path, index=False)

        time.sleep(delay)

    df = pd.DataFrame(predictions)
    return df


def save_results(model_name: str, df: pd.DataFrame):
    """Save predictions in the bmpb-compatible format."""
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    out_dir = EXPERIMENTS / f"llm-{model_name}-zeroshot"
    out_dir.mkdir(parents=True, exist_ok=True)

    valid = df[df["predicted"] >= 0].copy()
    if len(valid) < len(df):
        print(f"  {len(df) - len(valid)} items had errors and were excluded")

    # Save predictions.csv (compatible with bmpb evaluate)
    valid[["item_id", "label", "predicted", "fold"]].to_csv(
        out_dir / "predictions.csv", index=False
    )

    acc = float(accuracy_score(valid["label"], valid["predicted"]))
    macro_f1 = float(f1_score(valid["label"], valid["predicted"], average="macro", zero_division=0))

    # Bootstrap CI
    sys.path.insert(0, str(ROOT / "src"))
    from bmpb.metrics import bootstrap_macro_f1
    import numpy as np
    ci = bootstrap_macro_f1(np.array(valid["label"]), np.array(valid["predicted"]))

    metrics = {
        "name": f"{model_name}_zeroshot",
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_f1_ci95": [round(ci[0], 4), round(ci[1], 4)],
        "n": len(valid),
        "errors": len(df) - len(valid),
        "protocol": "zero-shot (no training)",
        "notes": ["zero-shot, no fine-tuning or examples provided"],
    }

    # Also save as cv.json so collect_cv_runs can't pick it up accidentally,
    # and as metrics.json so collect_runs can.
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "run.json").write_text(json.dumps({
        "name": f"{model_name}_zeroshot",
        "modality": "text",
    }, indent=2))

    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Macro-F1:  {macro_f1:.3f}  [{ci[0]:.3f}-{ci[1]:.3f}]")
    print(f"\n{classification_report(valid['label'], valid['predicted'], target_names=LABEL_NAMES, zero_division=0)}")
    print(f"Saved to {out_dir}/")

    # Clean up partial file
    partial = out_dir / "predictions_partial.csv"
    if partial.exists():
        partial.unlink()


def main():
    parser = argparse.ArgumentParser(description="LLM zero-shot baselines")
    parser.add_argument("--model", choices=["gemini", "qwen", "claude", "gpt4", "all"], default="gemini")
    args = parser.parse_args()

    if not CORPUS.exists():
        sys.exit(f"{CORPUS} not found. Run `bmpb ingest` first.")

    corpus = pd.read_csv(CORPUS)
    print(f"Loaded {len(corpus)} items from corpus")

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_name in targets:
        env_var, classify_fn, delay = MODELS[model_name]
        key = os.environ.get(env_var)
        if not key:
            print(f"\n{env_var} not set, skipping {model_name}")
            continue

        # Configure SDK if needed
        if model_name == "gemini":
            os.environ["GOOGLE_API_KEY"] = key  # google-genai reads from env

        df = run_baseline(model_name, classify_fn, corpus, delay=delay)
        save_results(model_name, df)


if __name__ == "__main__":
    main()
