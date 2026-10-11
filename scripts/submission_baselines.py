"""Rebuild headline baselines on the full and readable-image populations.

Run with `.venv/bin/python scripts/submission_baselines.py` from the repository.
The majority floor uses original training frequencies, without augmentation.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from bmpb.crossval import cross_validate
from bmpb.data.dataset import load_image
from bmpb.paths import CORPUS


def main():
    corpus = pd.read_csv(CORPUS)
    corpus["text"] = corpus["title"]
    corpus["has_image"] = [load_image(path) is not None for path in corpus.image_path]
    rows = []
    populations = {
        "all_headlines": corpus,
        "paired_headlines": corpus[corpus.has_image].copy(),
    }
    for population, frame in populations.items():
        digest = hashlib.sha256(
            frame[["item_id", "text", "label", "source_url"]].to_csv(index=False).encode()
        ).hexdigest()
        for model in ("majority", "tfidf_logreg"):
            result = cross_validate(
                f"configs/text/{model}.yaml",
                corpus=frame,
                out=f"experiments/evaluation/{population}",
            )
            rows.append({**result, "population": population, "input_sha256": digest})
    destination = Path("reports/submission")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "baselines.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "| Population | Model | n | Accuracy | Pooled macro-F1 | 95% CI | Fold mean ± SD |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lo, hi = row["macro_f1_ci95"]
        lines.append(
            f"| {row['population']} | {row['name']} | {row['items_scored']} | "
            f"{row['accuracy']:.3f} | {row['macro_f1']:.3f} | {lo:.3f}–{hi:.3f} | "
            f"{row['fold_macro_f1_mean']:.3f} ± {row['fold_macro_f1_sd']:.3f} |"
        )
    (destination / "baselines.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
