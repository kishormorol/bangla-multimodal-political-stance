"""Compare local items with an exported BanglaBias JSON table.

Input is a JSON array of upstream records, not copied into the tracked report.
No fuzzy matching: NFC/whitespace-normalized headlines and decoded URLs only.
"""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

from bmpb.data.ingest import normalize_item_id
from bmpb.paths import CORPUS, RAW


def normalize(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value))).strip()


def normalize_url(value):
    return unquote(normalize(value)).rstrip("/")


def compare(local, upstream, title_column, url_column, id_column):
    title_index, url_index = defaultdict(set), defaultdict(set)
    for _, row in upstream.iterrows():
        if normalize(row.news_headline):
            title_index[normalize(row.news_headline)].add(int(row.id))
        if normalize_url(row.source_link):
            url_index[normalize_url(row.source_link)].add(int(row.id))
    result = []
    for position, row in local.iterrows():
        titles = title_index.get(normalize(row[title_column]), set())
        urls = url_index.get(normalize_url(row[url_column]), set())
        shared = titles & urls
        candidates = shared or (titles | urls)
        method = (
            "headline_and_url"
            if shared
            else (
                "conflicting_headline_and_url"
                if titles and urls
                else "url_only" if urls else "headline_only" if titles else "unmatched"
            )
        )
        result.append(
            {
                "local_position": int(position),
                "item_id": normalize_item_id(row[id_column]),
                "match_method": method,
                "candidate_upstream_ids": sorted(candidates),
                "declared_upstream_id": normalize(row.get("id from bangla bias")),
            }
        )
    return result


def summarize(records):
    counts = Counter(row["match_method"] for row in records)
    return {
        "items": len(records),
        "matched_items": sum(bool(row["candidate_upstream_ids"]) for row in records),
        "match_counts": dict(counts),
        "ambiguous_items": [
            row["item_id"] for row in records if len(row["candidate_upstream_ids"]) > 1
        ],
        "unmatched_items": [row["item_id"] for row in records if not row["candidate_upstream_ids"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/submission/provenance.json"))
    args = parser.parse_args()
    upstream = pd.read_json(args.upstream)
    raw = pd.read_csv(RAW / "Dataset.csv")
    corpus = pd.read_csv(CORPUS)
    raw_records = compare(raw, upstream, "Title", "source link", "Image_id")
    records = compare(corpus, upstream, "title", "source_url", "item_id")
    exceptions = []
    for record in raw_records:
        declared = record["declared_upstream_id"]
        if declared.isdigit() and int(declared) > 0 and record["candidate_upstream_ids"]:
            if int(declared) not in record["candidate_upstream_ids"]:
                exceptions.append(record)
    links = corpus["source_url"].map(normalize_url)
    duplicates = [
        {"item_ids": group["item_id"].tolist(), "n": len(group)}
        for _, group in corpus[links.ne("") & links.duplicated(keep=False)].groupby(links)
    ]
    payload = {
        "upstream": "https://huggingface.co/datasets/dipta007/BanglaBias",
        "paper": "https://aclanthology.org/2025.banglalp-1.5/",
        "upstream_export_sha256": hashlib.sha256(args.upstream.read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256((RAW / "Dataset.csv").read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        "upstream_items": len(upstream),
        "raw": summarize(raw_records),
        "processed": summarize(records),
        "removed_raw_item_ids": sorted(
            set(r["item_id"] for r in raw_records) - set(corpus["item_id"])
        ),
        "declared_id_conflicts": exceptions,
        "metadata_conflicts": [
            record for record in records if record["match_method"] == "conflicting_headline_and_url"
        ],
        "duplicate_source_groups": duplicates,
        "processed_matches": records,
        "raw_matches": raw_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in {"processed_matches", "raw_matches"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
