"""Argilla 2.x annotation pipeline for the unannotated BanglaPoliticalStance split.

Three sub-commands:
  push    -- load the HF dataset's 'unannotated' split into an Argilla Space
  status  -- count how many records have been annotated
  export  -- pull completed annotations back to a CSV, ready for bmpb ingest

Usage:
  # 1. Create an Argilla Space on HF (one-time, in the browser — see README below)
  # 2. Set env vars:
  #      HF_TOKEN       – your HF write token
  #      ARGILLA_URL    – your Space URL, e.g. https://kishormorol-bpsd-argilla.hf.space
  #      ARGILLA_APIKEY – "owner" API key shown in the Space settings
  # 3. Push the unannotated items:
  #      python scripts/annotate_argilla.py push
  # 4. Open the Space, annotate, then:
  #      python scripts/annotate_argilla.py export

## One-time Space setup (free tier)
# 1. Go to https://huggingface.co/new-space
# 2. Choose SDK = "Docker", then search "Argilla" in the template gallery.
#    Or use the direct link: https://huggingface.co/spaces/argilla/argilla-template
#    Click "Duplicate this Space".
# 3. Space name e.g.  bpsd-argilla  (URL becomes  <user>-bpsd-argilla.hf.space)
# 4. Visibility = Private (keeps your data private; share with annotators via HF OAuth).
# 5. IMPORTANT — add Persistent Storage before the Space first boots:
#    Space settings → Storage → Attach a (free Small) persistent volume.
#    Without this, every cold start (after 48 h inactivity) wipes annotations.
# 6. After the Space starts (1-2 min), open it and note the "owner" API key
#    in Settings → API keys.
#
## Cost note
# A free-tier CPU Space is enough — Argilla is a labelling UI, not a GPU workload.
# Persistent storage is ~$0 on the free "small" tier (up to 20 GB).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LABEL_NAMES = ["govt_critique", "neutral", "govt_leaning"]   # ids 0, 1, 2
DATASET_NAME = "bangla_political_stance_unannotated"
HF_REPO_ID   = "kishormorol/BanglaPoliticalStance"
HF_SPLIT     = "unannotated"
OUT_CSV      = ROOT / "data" / "interim" / "argilla_annotations.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client():
    """Return an authenticated Argilla client from env vars."""
    try:
        import argilla as rg
    except ImportError:
        sys.exit(
            "argilla not installed. Run:\n"
            "  .venv/Scripts/pip install argilla>=2.0\n"
            "or add it to pyproject.toml / requirements."
        )
    url    = os.environ.get("ARGILLA_URL", "").rstrip("/")
    apikey = os.environ.get("ARGILLA_APIKEY", "")
    if not url or not apikey:
        sys.exit(
            "Set ARGILLA_URL and ARGILLA_APIKEY env vars before running.\n"
            "  export ARGILLA_URL=https://<user>-bpsd-argilla.hf.space\n"
            "  export ARGILLA_APIKEY=<owner-key-from-space-settings>"
        )
    client = rg.Argilla(api_url=url, api_key=apikey)
    return rg, client


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------

def cmd_push(args):
    """Push the HF unannotated split into Argilla as a labelling task."""
    rg, client = _get_client()
    from datasets import load_dataset

    hf_token = os.environ.get("HF_TOKEN")
    print(f"Loading {HF_REPO_ID}[{HF_SPLIT}] from HF …")
    ds = load_dataset(HF_REPO_ID, split=HF_SPLIT, token=hf_token)
    print(f"  {len(ds):,} records loaded")

    # --- Dataset settings ---
    settings = rg.Settings(
        guidelines=(
            "Classify each Bangla news headline by its political stance toward the "
            "current Bangladesh government.\n\n"
            "govt_critique (0) — headline criticises, questions, or reports negatively "
            "on the government / ruling party.\n"
            "neutral (1) — factual reporting; no clear positive or negative slant.\n"
            "govt_leaning (2) — headline supports, defends, or portrays the government "
            "/ ruling party positively.\n\n"
            "If an image is attached, use it as context but base the label primarily on "
            "the headline text. Flag the record if you are unsure."
        ),
        fields=[
            rg.TextField(name="headline",   title="Headline (Bangla)",  required=True),
            rg.TextField(name="outlet",     title="News outlet",        required=False),
            rg.TextField(name="date",       title="Date",               required=False),
            rg.TextField(name="source_url", title="Source URL",         required=False),
        ],
        questions=[
            rg.LabelQuestion(
                name="stance",
                title="Political stance",
                labels={
                    "govt_critique": "Govt Critique (0)",
                    "neutral":       "Neutral (1)",
                    "govt_leaning":  "Govt Leaning (2)",
                },
                required=True,
            ),
            rg.TextQuestion(
                name="note",
                title="Optional note (e.g. if flagging)",
                required=False,
            ),
        ],
        metadata=[
            rg.TermsMetadataProperty(name="outlet",   title="Outlet"),
            rg.TermsMetadataProperty(name="item_id",  title="Item ID"),
        ],
        distribution=rg.TaskDistribution(min_submitted=args.min_submitted),
        allow_extra_metadata=False,
    )

    # Delete old dataset of same name if asked
    existing = client.datasets(name=DATASET_NAME)
    if existing:
        if args.overwrite:
            print(f"  Deleting existing dataset '{DATASET_NAME}' …")
            existing[0].delete()
        else:
            sys.exit(
                f"Dataset '{DATASET_NAME}' already exists in Argilla.\n"
                "Use --overwrite to replace it (you will lose any partial annotations)."
            )

    dataset = rg.Dataset(name=DATASET_NAME, settings=settings)
    dataset.create()
    print(f"  Created Argilla dataset '{DATASET_NAME}'")

    # Build records
    records = []
    for row in ds:
        # Argilla TextField only supports plain text (no image embed in free tier).
        # Annotators can click the source_url to view the original page + image.
        records.append(
            rg.Record(
                fields={
                    "headline":   str(row.get("headline") or ""),
                    "outlet":     str(row.get("outlet")   or ""),
                    "date":       str(row.get("date")     or ""),
                    "source_url": str(row.get("source_url") or ""),
                },
                metadata={
                    "outlet":  str(row.get("outlet") or ""),
                    "item_id": str(row.get("item_id") or ""),
                },
                # Store item_id so we can merge back later
                external_id=str(row.get("item_id") or ""),
            )
        )

    print(f"  Uploading {len(records):,} records …")
    dataset.records.log(records, batch_size=200)
    print(f"Done. Open your Argilla Space to start annotating:")
    print(f"  {os.environ['ARGILLA_URL']}")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def cmd_status(args):
    rg, client = _get_client()
    dataset = client.datasets(name=DATASET_NAME)
    if not dataset:
        sys.exit(f"Dataset '{DATASET_NAME}' not found in Argilla.")
    dataset = dataset[0]

    total      = dataset.records(with_responses=False).total
    annotated  = dataset.records(
        query=rg.Query(filter=rg.Filter([("status", "==", "submitted")])),
        with_responses=False,
    ).total
    print(f"Total records : {total:,}")
    print(f"Annotated     : {annotated:,}  ({100*annotated/max(total,1):.1f}%)")
    print(f"Remaining     : {total - annotated:,}")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def cmd_export(args):
    """Pull completed annotations from Argilla and write data/interim/argilla_annotations.csv."""
    rg, client = _get_client()
    import pandas as pd

    dataset = client.datasets(name=DATASET_NAME)
    if not dataset:
        sys.exit(f"Dataset '{DATASET_NAME}' not found in Argilla.")
    dataset = dataset[0]

    print("Fetching annotated records …")
    rows = []
    for record in dataset.records(
        query=rg.Query(filter=rg.Filter([("status", "==", "submitted")])),
        with_responses=True,
        batch_size=200,
    ):
        # Collect all submitted responses for majority vote
        votes: list[str] = []
        notes: list[str] = []
        annotator_cols: dict[str, str] = {}
        for i, resp in enumerate(record.responses, start=1):
            stance = resp.answers.get("stance")
            if stance and stance.value:
                votes.append(stance.value)
                annotator_cols[f"annotator_{i}"] = stance.value
            note = resp.answers.get("note")
            if note and note.value:
                notes.append(note.value)

        if not votes:
            continue

        from collections import Counter
        counts   = Counter(votes)
        top      = counts.most_common()
        majority = top[0][0] if top[0][1] >= 2 else None
        unanimous= (top[0][1] == len(votes) and len(votes) >= 2)

        rows.append({
            "item_id":             record.external_id or "",
            "headline":            record.fields.get("headline", ""),
            "outlet":              record.fields.get("outlet", ""),
            "date":                record.fields.get("date", ""),
            "source_url":          record.fields.get("source_url", ""),
            "label_name":          majority,
            "label":               LABEL_NAMES.index(majority) if majority in LABEL_NAMES else -1,
            "votes":               len(votes),
            "unanimous":           unanimous,
            "needs_adjudication":  majority is None,
            "notes":               " | ".join(notes),
            **annotator_cols,
        })

    if not rows:
        print("No completed annotations found yet.")
        return

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Exported {len(df):,} annotated records to {OUT_CSV}")
    adjudicate = df[df["needs_adjudication"]]
    if len(adjudicate):
        adj_path = OUT_CSV.parent / "adjudicate.csv"
        adjudicate.to_csv(adj_path, index=False)
        print(f"  {len(adjudicate)} items need adjudication → {adj_path}")
    print(f"  Label balance:\n{df['label_name'].value_counts().to_string()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Argilla annotation pipeline for BanglaPoliticalStance"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_push = sub.add_parser("push", help="Push unannotated HF split into Argilla")
    p_push.add_argument(
        "--min-submitted", type=int, default=3,
        help="Responses required per record (default 3 for 3-annotator IAA)"
    )
    p_push.add_argument(
        "--overwrite", action="store_true",
        help="Delete and recreate the dataset if it already exists"
    )
    p_push.set_defaults(func=cmd_push)

    p_status = sub.add_parser("status", help="Show annotation progress")
    p_status.set_defaults(func=cmd_status)

    p_export = sub.add_parser("export", help="Export completed annotations to CSV")
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
