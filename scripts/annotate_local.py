"""Lightweight local annotator — no server required.

Streams through the HF dataset's 'unannotated' split, shows one headline at a
time in the terminal, and writes labels to data/interim/local_annotations.csv.
Resumes from where it left off if interrupted.

Usage:
    python scripts/annotate_local.py
    python scripts/annotate_local.py --annotator alice
    python scripts/annotate_local.py --batch-size 50   # do 50 items then stop

Keys: 0 = govt_critique, 1 = neutral, 2 = govt_leaning
      s = skip, q = quit, ? = show guidelines

The output CSV has the same columns as bmpb's annotations.py expects, so you
can feed it into `bmpb merge-annotations` after collecting runs from multiple
annotators.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABEL_NAMES = {0: "govt_critique", 1: "neutral", 2: "govt_leaning"}
HF_REPO_ID  = "kishormorol/BanglaPoliticalStance"
HF_SPLIT    = "unannotated"


GUIDELINES = """
Political stance labels (toward current Bangladesh government):
  0 – govt_critique : negative framing, criticism, protest coverage
  1 – neutral       : factual, no clear slant
  2 – govt_leaning  : positive framing, government support/defence
"""


def load_done(out_path: Path) -> set[str]:
    """Return item IDs already labelled in a prior session."""
    if not out_path.exists():
        return set()
    with out_path.open(encoding="utf-8") as f:
        return {row["item_id"] for row in csv.DictReader(f) if row.get("label") not in ("", None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator", default=os.environ.get("USER", "annotator1"))
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Stop after N labels (0 = no limit)")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: data/interim/local_<annotator>.csv)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else (
        ROOT / "data" / "interim" / f"local_{args.annotator}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets not installed: pip install datasets")

    hf_token = os.environ.get("HF_TOKEN")
    print(f"Loading {HF_REPO_ID}[{HF_SPLIT}] …")
    ds = load_dataset(HF_REPO_ID, split=HF_SPLIT, token=hf_token)
    print(f"  {len(ds):,} items in split")

    done = load_done(out_path)
    print(f"  {len(done)} already labelled in {out_path.name}")
    print(GUIDELINES)
    print("Keys: 0/1/2 = label, s = skip, q = quit, ? = guidelines\n")

    fieldnames = ["item_id", "headline", "outlet", "date", "source_url", "label", "label_name", "annotator"]
    write_header = not out_path.exists()
    count = 0

    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for row in ds:
            item_id = str(row.get("item_id") or "")
            if item_id in done:
                continue

            headline = str(row.get("headline") or "")
            outlet   = str(row.get("outlet")   or "")
            date     = str(row.get("date")     or "")
            url      = str(row.get("source_url") or "")

            print(f"\n[{item_id}]  {outlet}  {date}")
            print(f"  {headline}")
            if url:
                print(f"  {url}")

            while True:
                try:
                    key = input("  Label: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nInterrupted — progress saved.")
                    return

                if key == "q":
                    print("Quit — progress saved.")
                    return
                if key == "?":
                    print(GUIDELINES)
                    continue
                if key == "s":
                    break  # skip without saving
                if key in ("0", "1", "2"):
                    label = int(key)
                    writer.writerow({
                        "item_id":    item_id,
                        "headline":   headline,
                        "outlet":     outlet,
                        "date":       date,
                        "source_url": url,
                        "label":      label,
                        "label_name": LABEL_NAMES[label],
                        "annotator":  args.annotator,
                    })
                    f.flush()
                    done.add(item_id)
                    count += 1
                    break
                print("  Invalid key. Use 0, 1, 2, s, q, or ?")

            if args.batch_size and count >= args.batch_size:
                print(f"\nBatch of {args.batch_size} complete — stopping.")
                break

    print(f"\nSession done: {count} new labels written to {out_path}")


if __name__ == "__main__":
    main()
