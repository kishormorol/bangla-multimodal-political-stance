"""Turn the annotation desk's output into labelled rows, with agreement.

The desk (a published artifact) stores each annotator's work at
`annotations/<uid>/batches/<batchId>` and the item pool at `items/<batchId>`.
Claude exports those documents to JSON with the ArtifactData tool:

    action="list", collection="items",            out_dir=<dir>
    action="list", collection="annotations",      out_dir=<dir>   # per annotator
    action="list", collection="annotations/<uid>/batches", out_dir=<dir>

then this module merges them:

    bmpb merge-annotations --export <dir>

Majority of three is the label. Items with no majority, or that any annotator
flagged, are written to `data/interim/adjudicate.csv` rather than silently
resolved — the guidelines promise they are discussed, and a paper that reports
κ has to say what happened to the disagreements.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from bmpb.paths import INTERIM, ensure_dirs
from bmpb.utils.logging import get_logger

log = get_logger(__name__)

LABEL_NAMES = {0: "govt_critique", 1: "neutral", 2: "govt_leaning"}
ADJUDICATE = INTERIM / "adjudicate.csv"
ANNOTATED = INTERIM / "annotated.csv"


def _load_items(export: Path) -> dict[str, dict]:
    """Every pooled item, keyed by id, from the exported `items/` documents."""
    items: dict[str, dict] = {}
    for path in sorted((export / "items").glob("*.json")):
        if path.stem == "index":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("items", []):
            items[row["id"]] = row
    return items


def _load_labels(export: Path) -> dict[str, dict[str, dict]]:
    """annotator id -> {item id -> record}, from the exported annotation docs."""
    by_annotator: dict[str, dict[str, dict]] = {}
    root = export / "annotations"
    if not root.exists():
        return by_annotator

    # Exports land either as annotations/<uid>/batches/<bid>.json or flattened;
    # walk the tree rather than assuming one shape.
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        labels = payload.get("labels")
        if not isinstance(labels, dict):
            continue
        parts = path.relative_to(root).parts
        uid = parts[0] if parts else path.stem
        bucket = by_annotator.setdefault(uid, {})
        for item_id, record in labels.items():
            if isinstance(record, dict):
                bucket[item_id] = record
    return by_annotator


def _story_label(record: dict):
    """The story field, or None when the annotator marked it non-political."""
    value = record.get("story")
    if value in (None, "np"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def merge(export: Path, write: bool = True) -> pd.DataFrame:
    ensure_dirs()
    export = Path(export)
    items = _load_items(export)
    by_annotator = _load_labels(export)
    if not items:
        raise FileNotFoundError(f"no item documents under {export / 'items'}")
    if not by_annotator:
        raise FileNotFoundError(f"no annotation documents under {export / 'annotations'}")

    annotators = sorted(by_annotator)
    log.info("%d annotators, %d pooled items", len(annotators), len(items))
    for uid in annotators:
        log.info("  %s: %d items labelled", uid[:12], len(by_annotator[uid]))

    rows = []
    for item_id, item in items.items():
        votes, flags, notes, non_political = [], 0, [], 0
        per_annotator = {}
        for position, uid in enumerate(annotators, start=1):
            record = by_annotator[uid].get(item_id)
            if not record:
                continue
            if record.get("story") == "np":
                non_political += 1
            label = _story_label(record)
            per_annotator[f"annotator_{position}"] = (
                LABEL_NAMES.get(label) if label is not None else "not_political"
            )
            per_annotator[f"annotator_{position}_image"] = record.get("image")
            if label is not None:
                votes.append(label)
            if record.get("flag"):
                flags += 1
                if record.get("note"):
                    notes.append(f"{uid[:6]}: {record['note']}")

        coverage = len(votes) + non_political
        if coverage == 0:
            continue

        counts = Counter(votes)
        top = counts.most_common()
        majority = top[0][0] if top and top[0][1] >= 2 else None
        unanimous = bool(top) and top[0][1] == len(votes) and len(votes) >= 2

        needs_adjudication = majority is None or flags > 0 or (0 < non_political < coverage)

        rows.append(
            {
                "item_id": item_id,
                "title": item.get("title", ""),
                "text": item.get("body") or item.get("summary") or "",
                "image_url": item.get("image_url", ""),
                "source_url": item.get("source_url", ""),
                "outlet": item.get("outlet", ""),
                "date": item.get("published", ""),
                "label": majority,
                "label_name": LABEL_NAMES.get(majority) if majority is not None else None,
                "votes": len(votes),
                "unanimous": unanimous,
                "not_political_votes": non_political,
                "flags": flags,
                "notes": " | ".join(notes),
                "needs_adjudication": needs_adjudication,
                **per_annotator,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        log.warning("no item received a label yet")
        return frame

    resolved = frame[~frame.needs_adjudication & frame.label.notna()]
    pending = frame[frame.needs_adjudication]

    log.info("%d items have at least one label", len(frame))
    log.info("  resolved by majority: %d", len(resolved))
    log.info("  unanimous: %d", int(frame.unanimous.sum()))
    log.info("  awaiting adjudication: %d", len(pending))
    if len(resolved):
        log.info("  label balance: %s", resolved.label_name.value_counts().to_dict())

    agreement = pairwise_kappa(frame, len(annotators))
    for pair, value in agreement.items():
        log.info("  Cohen's kappa %s: %.3f", pair, value)

    if write:
        frame.to_csv(ANNOTATED, index=False)
        pending.to_csv(ADJUDICATE, index=False)
        log.info("wrote %s and %s", ANNOTATED, ADJUDICATE)
    return frame


def pairwise_kappa(frame: pd.DataFrame, n_annotators: int) -> dict[str, float]:
    """Cohen's κ for each annotator pair, on the items both of them labelled."""
    from sklearn.metrics import cohen_kappa_score

    out: dict[str, float] = {}
    columns = [f"annotator_{i}" for i in range(1, n_annotators + 1)]
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            if left not in frame or right not in frame:
                continue
            both = frame[[left, right]].dropna()
            if len(both) < 2 or both[left].nunique() < 2:
                continue
            out[f"{left}|{right}"] = float(
                cohen_kappa_score(both[left].astype(str), both[right].astype(str))
            )
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, help="directory holding the exported JSON")
    merge(Path(parser.parse_args().export))
