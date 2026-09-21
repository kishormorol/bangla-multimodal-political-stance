"""Fetch the full article text for items that only have a headline.

The original 198 items carry a `source_url` but only a headline in
`Preprocessed_Text` (median 8 words). The labels were assigned to the story, not
to the headline string, so they still hold once the body is attached — this is a
recovery of text that was always in scope, not a re-annotation.

    bmpb backfill                 # every corpus item lacking a body
    bmpb backfill --limit 50      # a sample first

Writes `data/interim/bodies.csv` (item_id, body, body_chars, status). `ingest`
picks it up and prefers the body over the headline when one is present, so the
corpus becomes article-level without anything else changing.

Expect misses. These URLs are one to two years old, several outlets answer
scripted requests with 403, and news sites reorganize. Every outcome is recorded
with a status so the paper can state precisely how much of the corpus is
article-level and how much stayed headline-only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bmpb.data.collect import Fetcher
from bmpb.data.sources import parse_article
from bmpb.paths import CORPUS, INTERIM, ensure_dirs
from bmpb.utils.logging import get_logger

log = get_logger(__name__)

BODIES = INTERIM / "bodies.csv"


def backfill(limit: int | None = None, delay: float = 1.5, out: Path = BODIES) -> pd.DataFrame:
    ensure_dirs()
    if not CORPUS.exists():
        raise FileNotFoundError(f"{CORPUS} not found. Run `bmpb ingest` first.")

    corpus = pd.read_csv(CORPUS)
    if "source_url" not in corpus:
        raise ValueError("the corpus has no source_url column to fetch from")

    done: dict[str, dict] = {}
    if out.exists():
        previous = pd.read_csv(out)
        # Retry the failures on a later run; keep what already succeeded.
        for row in previous.itertuples():
            if getattr(row, "status", "") == "ok":
                done[row.item_id] = {"body": row.body, "body_chars": row.body_chars, "status": "ok"}
        log.info("resuming: %d bodies already recovered", len(done))

    todo = corpus[corpus["source_url"].notna() & ~corpus["item_id"].isin(done)]
    if limit:
        todo = todo.head(limit)
    log.info(
        "%d items to try (%d have no source_url)", len(todo), int(corpus["source_url"].isna().sum())
    )

    fetcher = Fetcher(delay=delay)
    results = dict(done)

    for n, row in enumerate(todo.itertuples(), start=1):
        url = str(row.source_url).strip()
        record = {"body": "", "body_chars": 0, "status": "unreachable"}
        page = fetcher.get(url)
        if page:
            parsed = parse_article(page, url, str(getattr(row, "outlet", "") or ""))
            if parsed is None:
                record["status"] = "no_article_markup"
            elif not parsed["body"]:
                record["status"] = "no_body_in_markup"
            else:
                record = {
                    "body": parsed["body"],
                    "body_chars": parsed["body_chars"],
                    "status": "ok",
                }
        results[row.item_id] = record
        if n % 20 == 0:
            got = sum(1 for r in results.values() if r["status"] == "ok")
            log.info("  tried %d/%d, recovered %d", n, len(todo), got)

    frame = (
        pd.DataFrame([{"item_id": k, **v} for k, v in results.items()])
        .sort_values("item_id")
        .reset_index(drop=True)
    )
    frame.to_csv(out, index=False)

    got = frame[frame.status == "ok"]
    log.info(
        "recovered %d of %d corpus items (%.0f%%)",
        len(got),
        len(corpus),
        100 * len(got) / max(1, len(corpus)),
    )
    if len(got):
        log.info("  median body: %d chars", int(got.body_chars.median()))
    log.info("  outcomes: %s", frame.status.value_counts().to_dict())
    log.info("wrote %s", out)
    return frame


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=1.5)
    backfill(**vars(parser.parse_args()))
