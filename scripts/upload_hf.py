"""Build and upload the BanglaPoliticalStance dataset to Hugging Face.

    python scripts/upload_hf.py                    # dry run (build locally)
    python scripts/upload_hf.py --push             # upload to HF
    python scripts/upload_hf.py --push --repo ORG/NAME  # custom repo
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, Image

ROOT = Path(__file__).resolve().parent.parent
SCRAPED_CSV = ROOT / "data" / "raw" / "scraped" / "scraped_corpus_clean.csv"
SCRAPED_IMAGES = ROOT / "data" / "raw" / "scraped" / "images"
ORIGINAL_CORPUS = ROOT / "data" / "processed" / "corpus.csv"
ORIGINAL_IMAGES = ROOT / "data" / "raw" / "processed_images"
RAW_IMAGES = ROOT / "data" / "raw" / "images"
README = ROOT / "hf_dataset" / "README.md"

LABEL_NAMES = ["govt_critique", "neutral", "govt_leaning"]


def load_annotated() -> list[dict]:
    """Load the original 198 annotated items."""
    rows = list(csv.DictReader(open(ORIGINAL_CORPUS, encoding="utf-8")))
    items = []
    for r in rows:
        # Find image
        img_path = None
        if r.get("image_path"):
            candidate = ROOT / "data" / "raw" / r["image_path"]
            if candidate.exists():
                img_path = str(candidate)
        if not img_path and r.get("has_image") == "True":
            # Try processed_images
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                candidate = ORIGINAL_IMAGES / f"Image_{r.get('item_id', '')}{ext}"
                if candidate.exists():
                    img_path = str(candidate)
                    break

        items.append({
            "item_id": r.get("item_id", ""),
            "headline": r.get("text", r.get("title", "")),
            "source_url": r.get("source_url", ""),
            "outlet": r.get("outlet", r.get("outlet_key", "")),
            "image": img_path,
            "date": r.get("date", ""),
            "section": "",
            "label": int(r.get("label", -1)),
        })
    return items


def load_unannotated() -> list[dict]:
    """Load the scraped unannotated items."""
    rows = list(csv.DictReader(open(SCRAPED_CSV, encoding="utf-8")))
    items = []
    for r in rows:
        img_path = None
        if r.get("image_path"):
            candidate = SCRAPED_IMAGES.parent / r["image_path"]
            if candidate.exists():
                img_path = str(candidate)

        items.append({
            "item_id": r.get("item_id", ""),
            "headline": r.get("headline", ""),
            "source_url": r.get("source_url", ""),
            "outlet": r.get("outlet", ""),
            "image": img_path,
            "date": r.get("date", ""),
            "section": r.get("section", ""),
            "label": -1,  # unannotated
        })
    return items


def build_dataset() -> DatasetDict:
    print("Loading annotated split...")
    annotated = load_annotated()
    print(f"  {len(annotated)} items, {sum(1 for a in annotated if a['image'])} with images")

    print("Loading unannotated split...")
    unannotated = load_unannotated()
    print(f"  {len(unannotated)} items, {sum(1 for a in unannotated if a['image'])} with images")

    features = Features({
        "item_id": Value("string"),
        "headline": Value("string"),
        "source_url": Value("string"),
        "outlet": Value("string"),
        "image": Image(),
        "date": Value("string"),
        "section": Value("string"),
        "label": Value("int32"),
    })

    ds_annotated = Dataset.from_list(annotated, features=features)
    ds_unannotated = Dataset.from_list(unannotated, features=features)

    return DatasetDict({
        "annotated": ds_annotated,
        "unannotated": ds_unannotated,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="Push to HF Hub")
    parser.add_argument("--repo", default="kishormorol/BanglaPoliticalStance")
    args = parser.parse_args()

    ds = build_dataset()
    print(f"\nDataset built:")
    print(ds)

    if args.push:
        print(f"\nPushing to {args.repo}...")
        ds.push_to_hub(
            args.repo,
            private=False,
        )
        # Upload README
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(README),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
        )
        print(f"Done! https://huggingface.co/datasets/{args.repo}")
    else:
        print("\nDry run complete. Use --push to upload.")


if __name__ == "__main__":
    main()
