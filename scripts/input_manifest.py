"""Freeze local input bytes, or check a previously frozen snapshot without training."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from bmpb.paths import CONFIGS, CORPUS, RAW, ROOT


def capture():
    paths = {CORPUS, CONFIGS / "data.yaml"}
    paths.update(
        RAW / name
        for name in (
            "Dataset.csv",
            "Final_Dataset.csv",
            "Preprocesing.csv",
            "Final_Image_Dataset_WithText.csv",
            "Balanced_Augmented_Dataset.csv",
        )
    )
    with CORPUS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("image_path"):
            image = (RAW / row["image_path"]).resolve()
            if not image.is_relative_to(RAW.resolve()):
                raise ValueError("input image path is outside data/raw")
            paths.add(image)
    return {
        "items": len(rows),
        "files": {
            str(path.relative_to(ROOT)): (
                hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            )
            for path in sorted(paths)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CORPUS.parent / "input_manifest.json")
    parser.add_argument(
        "--write", action="store_true", help="create a new manifest without overwrite"
    )
    args = parser.parse_args()
    try:
        current = capture()
        if args.write:
            with args.manifest.open("x", encoding="utf-8") as handle:
                json.dump(current, handle, indent=2)
                handle.write("\n")
        else:
            expected = json.loads(args.manifest.read_text(encoding="utf-8"))
            if current != expected:
                raise ValueError("input snapshot differs from its manifest; do not resume it")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Input verification failed: {error}\n")
    print(f"Verified input snapshot: {current['items']} items, {len(current['files'])} file states")


if __name__ == "__main__":
    main()
