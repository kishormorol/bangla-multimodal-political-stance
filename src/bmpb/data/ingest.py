"""Turn the raw Drive sheets into one canonical table.

Raw state of the world, and what this module does about it:

* Column names drift between sheets ("FINAL LABEL" vs "Final_Label").
  -> resolved through the alias lists in configs/data.yaml.
* Labels appear both as strings ("Govt critique", with drifting case) and as
  integer ids. -> both map onto the same canonical name + id.
* `Image_id` is zero-padded in some rows ("Image_01") and not in others
  ("Image_2"). -> normalized to `Image_<n>` without padding.
* `Image_Path` points at the original Colab mount
  (/content/drive/MyDrive/Political_Bias/processed_images/Image_2.jpg).
  -> rewritten to a path under data/raw/, and only kept when the file is there.
* The annotation sheet carries three annotator columns plus trailing unnamed
  spreadsheet columns. -> annotators are preserved (they support the agreement
  numbers); the unnamed columns are dropped.

Output: data/processed/corpus.csv, one row per article, plus
data/processed/image_index.csv mapping item ids to resolved image files.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from bmpb.config import DataConfig
from bmpb.paths import CORPUS, IMAGE_INDEX, PROCESSED, RAW, ensure_dirs
from bmpb.utils.logging import get_logger

log = get_logger(__name__)

LABEL_NAMES = ("govt_critique", "neutral", "govt_leaning")
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}

_IMAGE_ID = re.compile(r"^\s*image[_\s-]*0*(\d+)\s*$", re.IGNORECASE)


def normalize_item_id(value: object) -> str | None:
    """`Image_01`, `image 2`, `Image_3 ` -> `Image_1`, `Image_2`, `Image_3`."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    match = _IMAGE_ID.match(str(value))
    if not match:
        return str(value).strip() or None
    return f"Image_{int(match.group(1))}"


def resolve_columns(df: pd.DataFrame, aliases: dict[str, list[str]]) -> dict[str, str]:
    """Map canonical field -> the actual column name present in this frame."""
    present = {c.strip().lower(): c for c in df.columns}
    resolved: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            actual = present.get(candidate.strip().lower())
            if actual is not None:
                resolved[canonical] = actual
                break
    return resolved


def canonical_label(value: object, label_map: dict[str, str]) -> str | None:
    """Accept either the string label or the integer id; return the canonical name."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    key = str(value).strip()
    if key in label_map:
        return label_map[key]
    # Integers arrive as "0.0" once pandas has seen a NaN in the column.
    try:
        return label_map[str(int(float(key)))]
    except (ValueError, KeyError):
        pass
    lowered = {k.lower(): v for k, v in label_map.items()}
    return lowered.get(key.lower())


def build_image_index(cfg: DataConfig, item_ids: list[str]) -> pd.DataFrame:
    """Find an on-disk image for each item id, preferring the processed copy.

    The raw `images/` folder was downloaded straight from the news sites, so its
    files carry whatever extension the CDN served (`Image_01.jpg.webp`,
    `Image_102.webp`, `Image_107.avif`). `processed_images/` is the uniform
    re-encoded set the vision models actually consumed.
    """
    processed_dir = RAW / cfg.processed_image_dir
    raw_dir = RAW / cfg.image_dir

    by_id: dict[str, dict[str, Path]] = {}
    for directory, kind in ((processed_dir, "processed"), (raw_dir, "raw")):
        if not directory.exists():
            log.warning("%s is missing — run `make data` first", directory)
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            # Strip every suffix: Image_01.jpg.webp -> Image_01
            stem = path.name.split(".", 1)[0]
            item_id = normalize_item_id(stem)
            if item_id:
                by_id.setdefault(item_id, {}).setdefault(kind, path)

    rows = []
    for item_id in item_ids:
        found = by_id.get(item_id, {})
        chosen = found.get("processed") or found.get("raw")
        rows.append(
            {
                "item_id": item_id,
                "image_path": str(chosen.relative_to(RAW)) if chosen else None,
                "image_kind": "processed" if "processed" in found else ("raw" if found else None),
                "has_image": chosen is not None,
            }
        )
    return pd.DataFrame(rows)


def _attach_bodies(df: pd.DataFrame) -> pd.DataFrame:
    """Use the recovered article body as `text` wherever `bmpb backfill` found one.

    The headline stays in `title`, and `text_level` records which items are
    article-level and which remained headline-only — the paper has to state that
    split rather than average over it.
    """
    from bmpb.paths import INTERIM

    bodies_file = INTERIM / "bodies.csv"
    df["text_level"] = "headline"
    if not bodies_file.exists():
        return df

    bodies = pd.read_csv(bodies_file)
    recovered = bodies[(bodies["status"] == "ok") & bodies["body"].notna()]
    if recovered.empty:
        return df

    lookup = recovered.set_index("item_id")["body"].to_dict()
    hits = df["item_id"].isin(lookup)
    df.loc[hits, "text"] = df.loc[hits, "item_id"].map(lookup)
    df.loc[hits, "text_level"] = "article"
    log.info(
        "attached article bodies to %d of %d items (%d remain headline-only)",
        int(hits.sum()),
        len(df),
        int((~hits).sum()),
    )
    return df


def _merge_annotators(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Pull the three annotator columns back in from the annotation sheet.

    Final_Dataset.csv drops them, but they are what the agreement numbers are
    computed from, so the canonical corpus should carry them.
    """
    if not cfg.annotation_table:
        return df
    path = RAW / cfg.annotation_table
    if not path.exists():
        log.info("%s not present; corpus will have no annotator columns", cfg.annotation_table)
        return df

    sheet = pd.read_csv(path)
    columns = resolve_columns(sheet, cfg.columns)
    annotator_columns = {k: v for k, v in columns.items() if k.startswith("annotator_")}
    if "item_id" not in columns or not annotator_columns:
        return df

    merged = pd.DataFrame({"item_id": sheet[columns["item_id"]].map(normalize_item_id)})
    for canonical, actual in annotator_columns.items():
        # Casing drifts between sheets ("Govt critique" vs "Govt Critique").
        merged[canonical] = sheet[actual].map(lambda v: canonical_label(v, cfg.label_map))
    merged = merged.dropna(subset=["item_id"]).drop_duplicates(subset="item_id")

    df = df.drop(columns=[c for c in annotator_columns if c in df], errors="ignore")
    out = df.merge(merged, on="item_id", how="left")
    coverage = {c: int(out[c].notna().sum()) for c in annotator_columns}
    log.info("annotator coverage: %s of %d items", coverage, len(out))
    return out


def ingest(cfg: DataConfig | None = None, write: bool = True) -> pd.DataFrame:
    cfg = cfg or DataConfig.load()
    ensure_dirs()

    source = RAW / cfg.source_table
    if not source.exists():
        raise FileNotFoundError(
            f"{source} not found. Run `make data` to mirror the Drive folder first."
        )

    raw = pd.read_csv(source)
    columns = resolve_columns(raw, cfg.columns)
    missing = {"item_id", "text", "label"} - set(columns)
    if missing:
        raise ValueError(f"{cfg.source_table} is missing required fields: {sorted(missing)}")

    df = pd.DataFrame(index=raw.index)
    for canonical, actual in columns.items():
        df[canonical] = raw[actual]

    df["item_id"] = df["item_id"].map(normalize_item_id)
    df["label_name"] = df["label"].map(lambda v: canonical_label(v, cfg.label_map))
    df["label"] = df["label_name"].map(LABEL_TO_ID)

    for side in ("article_label", "image_label"):
        if side in df:
            df[f"{side}_name"] = df[side].map(lambda v: canonical_label(v, cfg.label_map))

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    if "title" in df:
        df["title"] = df["title"].fillna("").astype(str).str.strip()
    if "outlet" in df:
        # "BBC Bangla", "\t\nBBC Bangla" and " BBC bangla" are one outlet.
        df["outlet"] = (
            df["outlet"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        )
        df["outlet_key"] = df["outlet"].str.lower()

    before = len(df)
    df = df[df["item_id"].notna() & df["label"].notna() & (df["text"] != "")]
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset="item_id", keep="first").reset_index(drop=True)
    log.info("kept %d of %d rows after dropping unlabeled/empty/duplicate items", len(df), before)

    df = _merge_annotators(df, cfg)
    df = _attach_bodies(df)

    images = build_image_index(cfg, df["item_id"].tolist())
    df = df.merge(images, on="item_id", how="left", suffixes=("_url_only", ""))
    df["has_image"] = df["has_image"].fillna(False)
    log.info("%d of %d items have an image on disk", int(df["has_image"].sum()), len(df))

    ordered = [
        c
        for c in (
            "item_id",
            "title",
            "text",
            "label",
            "label_name",
            "article_label_name",
            "image_label_name",
            "outlet",
            "outlet_key",
            "date",
            "source_url",
            "image_url",
            "image_path",
            "image_kind",
            "has_image",
            "annotator_1",
            "annotator_2",
            "annotator_3",
        )
        if c in df
    ]
    df = df[ordered + [c for c in df.columns if c not in ordered]]

    if write:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_csv(CORPUS, index=False)
        images.to_csv(IMAGE_INDEX, index=False)
        log.info("wrote %s (%d rows) and %s", CORPUS, len(df), IMAGE_INDEX)
    return df


if __name__ == "__main__":
    ingest()
