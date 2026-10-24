"""Build the BanglaPoliticalStance dataset locally and optionally publish it.

python scripts/upload_hf.py                         # dry run
python scripts/upload_hf.py --push                  # upload to HF Hub
python scripts/upload_hf.py --push --repo ORG/NAME  # upload to another repo
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parent.parent
SCRAPED_CSV = ROOT / "data" / "raw" / "scraped" / "scraped_corpus_clean.csv"
SCRAPED_IMAGES = ROOT / "data" / "raw" / "scraped" / "images"
ORIGINAL_CORPUS = ROOT / "data" / "processed" / "corpus.csv"
ORIGINAL_IMAGES = ROOT / "data" / "raw" / "processed_images"
README = ROOT / "hf_dataset" / "README.md"
LABEL_NAMES = ["govt_critique", "neutral", "govt_leaning"]
ANNOTATED_FIELDS = (
    "item_id",
    "title",
    "text",
    "source_url",
    "outlet",
    "date",
    "label",
    "image_path",
)
UNANNOTATED_FIELDS = (
    "item_id",
    "headline",
    "source_url",
    "outlet",
    "date",
    "section",
    "image_path",
)


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")
        return list(reader)


def _validate_rows(
    rows: list[dict], fields: tuple[str, ...], source: Path, annotated: bool
) -> None:
    ids = [str(row.get("item_id") or "").strip() for row in rows]
    if any(not item_id for item_id in ids):
        raise ValueError(f"{source} contains a row with a missing item_id")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source} contains duplicate item_id values")
    if annotated:
        for row in rows:
            try:
                label = int(row.get("label", ""))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{source}: invalid label for item {row['item_id']!r}") from exc
            if label not in range(len(LABEL_NAMES)):
                raise ValueError(f"{source}: invalid label {label} for item {row['item_id']!r}")
    missing = {
        field: sum(not str(row.get(field) or "").strip() for row in rows) for field in fields
    }
    print(f"  missing fields: {missing}")


def _validate_split_ids(annotated: list[dict], unannotated: list[dict]) -> None:
    annotated_ids = {row["item_id"] for row in annotated}
    overlap = annotated_ids.intersection(row["item_id"] for row in unannotated)
    if overlap:
        raise ValueError(f"item_id values occur in both splits: {sorted(overlap)[:5]}")


def _safe_image(path_value: str, base: Path) -> tuple[str | None, str]:
    if not path_value:
        return None, "missing"
    path = (base / path_value).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        return None, "missing"
    try:
        with PILImage.open(path) as image:
            image.load()
    except (OSError, ValueError):
        return None, "decode_failed"
    return str(path), "included"


def load_annotated() -> list[dict]:
    """Load annotated items, exporting the headline and numeric label as stored."""
    rows = _rows(ORIGINAL_CORPUS)
    _validate_rows(rows, ANNOTATED_FIELDS, ORIGINAL_CORPUS, annotated=True)
    items = []
    image_counts = {"included": 0, "missing": 0, "decode_failed": 0}
    for r in rows:
        img_path, status = (None, "missing")
        decode_failed = False
        if r.get("image_path"):
            img_path, status = _safe_image(r["image_path"], ROOT / "data" / "raw")
            decode_failed = status == "decode_failed"
        if not img_path and r.get("has_image") == "True":
            item_id = str(r["item_id"]).strip()
            image_id = item_id if item_id.startswith("Image_") else f"Image_{item_id}"
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                img_path, status = _safe_image(f"{image_id}{ext}", ORIGINAL_IMAGES)
                if img_path:
                    break
                decode_failed |= status == "decode_failed"
        if not img_path and decode_failed:
            status = "decode_failed"
        image_counts[status] += 1

        items.append(
            {
                "item_id": str(r["item_id"]).strip(),
                "headline": r["title"] if "title" in r else r.get("text", ""),
                "source_url": r.get("source_url", ""),
                "outlet": r.get("outlet", r.get("outlet_key", "")),
                "image": img_path,
                "date": r.get("date", ""),
                "section": "",
                "label": int(r.get("label", -1)),
            }
        )
    print(f"  image inventory: {image_counts}")
    return items


def load_unannotated() -> list[dict]:
    """Load the scraped unannotated items and report their actual image coverage."""
    rows = _rows(SCRAPED_CSV)
    _validate_rows(rows, UNANNOTATED_FIELDS, SCRAPED_CSV, annotated=False)
    items = []
    image_counts = {"included": 0, "missing": 0, "decode_failed": 0}
    for r in rows:
        img_path, status = _safe_image(r.get("image_path", ""), SCRAPED_IMAGES.parent)
        image_counts[status] += 1

        items.append(
            {
                "item_id": str(r["item_id"]).strip(),
                "headline": r.get("headline", ""),
                "source_url": r.get("source_url", ""),
                "outlet": r.get("outlet", ""),
                "image": img_path,
                "date": r.get("date", ""),
                "section": r.get("section", ""),
                "label": -1,  # unannotated
            }
        )
    print(f"  image inventory: {image_counts}")
    return items


def build_dataset() -> DatasetDict:
    print("Loading annotated split...")
    annotated = load_annotated()
    print(f"  {len(annotated)} items, {sum(1 for a in annotated if a['image'])} with images")

    print("Loading unannotated split...")
    unannotated = load_unannotated()
    print(f"  {len(unannotated)} items, {sum(1 for a in unannotated if a['image'])} with images")
    _validate_split_ids(annotated, unannotated)

    features = Features(
        {
            "item_id": Value("string"),
            "headline": Value("string"),
            "source_url": Value("string"),
            "outlet": Value("string"),
            "image": Image(),
            "date": Value("string"),
            "section": Value("string"),
            "label": Value("int32"),
        }
    )

    ds_annotated = Dataset.from_list(annotated, features=features)
    ds_unannotated = Dataset.from_list(unannotated, features=features)

    return DatasetDict(
        {
            "annotated": ds_annotated,
            "unannotated": ds_unannotated,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="Push publicly to the HF Hub")
    parser.add_argument("--repo", default="kishormorol/BanglaPoliticalStance")
    args = parser.parse_args()

    ds = build_dataset()
    print("\nDataset built:")
    print(ds)

    if args.push:
        print(f"\nPushing to {args.repo}...")
        ds.push_to_hub(args.repo, private=False)
        from huggingface_hub import HfApi

        HfApi().upload_file(
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
