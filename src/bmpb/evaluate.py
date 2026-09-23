"""Score saved predictions and build the cross-model results table.

Two entry points:

* `evaluate_run` rescoring one run directory, so metrics can be recomputed
  without retraining after a scoring change.
* `evaluate_published` scoring the prediction CSVs that shipped in the Drive
  folder (BanglaBERT.csv, CLIP_Predictions.csv, …). These are the numbers in the
  current manuscript draft; having them in the same table as fresh runs is what
  makes the comparison honest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from bmpb.metrics import Scores, score
from bmpb.paths import EXPERIMENTS, RAW, TABLES
from bmpb.utils.logging import get_logger

log = get_logger(__name__)

# The prediction column each published CSV uses, keyed by file name.
PUBLISHED_PREDICTIONS = {
    "BanglaBERT.csv": ("FINAL LABEL", "banglabert"),
    "BanglaELECTRA_Predictions.csv": ("FINAL LABEL", "bangla_electra"),
    "mBERT_Predictions.csv": ("FINAL LABEL", "mbert"),
    "XLMRoBERTa_Predictions.csv": ("FINAL LABEL", "xlmroberta"),
    "mt5_predictions.csv": ("FINAL LABEL", "mt5_pred"),
    "CLIP_Predictions.csv": ("Final_Label", "Predicted_Label"),
    "BLIP_Predictions.csv": ("Final_Label", "Predicted_Label"),
    "ALIGN_Predictions.csv": ("Final_Label", "Predicted_Label"),
    "FLAVA_Predictions.csv": ("Final_Label", "Predicted_Label"),
    "ViLT_Predictions.csv": ("Final_Label", "Predicted_Label"),
    "CountVec_ViT_Predictions.csv": ("Final_Label", "Predicted_Label"),
}

# mt5 writes label strings rather than ids.
LABEL_TEXT_TO_ID = {"govt critique": 0, "neutral": 1, "govt leaning": 2}


def _as_ids(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "if":
        return series.astype("Int64")
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.map(LABEL_TEXT_TO_ID).astype("Int64")


def evaluate_run(run: Path | str, bootstrap: bool = True) -> Scores:
    run = Path(run)
    predictions = pd.read_csv(run / "predictions.csv")
    scores = score(predictions["label"], predictions["predicted"], bootstrap=bootstrap)
    (run / "metrics.json").write_text(json.dumps(scores.to_dict(), indent=2), encoding="utf-8")
    log.info("%s  %s", run.name, scores.summary())
    return scores


def evaluate_published(raw: Path = RAW) -> pd.DataFrame:
    """Score every published prediction CSV that is present in data/raw/."""
    rows = []
    for filename, (truth_col, pred_col) in PUBLISHED_PREDICTIONS.items():
        path = raw / filename
        if not path.exists():
            log.debug("skipping %s (not downloaded)", filename)
            continue
        frame = pd.read_csv(path)
        if truth_col not in frame or pred_col not in frame:
            log.warning("%s: expected columns %s/%s not found", filename, truth_col, pred_col)
            continue
        pair = pd.DataFrame(
            {"y": _as_ids(frame[truth_col]), "p": _as_ids(frame[pred_col])}
        ).dropna()
        if pair.empty:
            continue
        scores = score(pair["y"].astype(int), pair["p"].astype(int))
        rows.append(
            {
                "model": re.sub(r"(_Predictions)?\.csv$", "", filename),
                "source": "published",
                "n": scores.n,
                "accuracy": scores.accuracy,
                "macro_f1": scores.macro_f1,
                "ci_low": scores.macro_f1_ci95[0] if scores.macro_f1_ci95 else None,
                "ci_high": scores.macro_f1_ci95[1] if scores.macro_f1_ci95 else None,
                "notes": "; ".join(scores.notes),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def collect_cv_runs(runs_dir: Path | str = EXPERIMENTS) -> pd.DataFrame:
    """Cross-validated runs — the protocol the paper should report."""
    rows = []
    for run in sorted(Path(runs_dir).glob("cv-*/")):
        summary_file = run / "cv.json"
        if not summary_file.exists():
            continue
        s = json.loads(summary_file.read_text())
        ci = s.get("macro_f1_ci95") or [None, None]
        rows.append(
            {
                "model": s.get("name", run.name),
                "source": "5-fold CV",
                "modality": s.get("modality"),
                "population": s.get("population"),
                "run": run.name,
                "n": s.get("items_scored"),
                "accuracy": s.get("accuracy"),
                "macro_f1": s.get("macro_f1"),
                "ci_low": ci[0],
                "ci_high": ci[1],
                "fold_mean": s.get("fold_macro_f1_mean"),
                "fold_sd": s.get("fold_macro_f1_sd"),
                "notes": "; ".join(s.get("notes", [])),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame):
        # Keep the newest run per model so a re-run supersedes its predecessor.
        frame = (
            frame.sort_values("run")
            .drop_duplicates(subset="model", keep="last")
            .sort_values("macro_f1", ascending=False)
            .reset_index(drop=True)
        )
    return frame


def collect_runs(runs_dir: Path | str = EXPERIMENTS) -> pd.DataFrame:
    rows = []
    for run in sorted(Path(runs_dir).glob("*/")):
        if run.name.startswith("cv-"):
            continue  # cross-validated runs are collected separately
        metrics_file = run / "metrics.json"
        meta_file = run / "run.json"
        if not metrics_file.exists():
            continue
        metrics = json.loads(metrics_file.read_text())
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        ci = metrics.get("macro_f1_ci95") or [None, None]
        rows.append(
            {
                "model": meta.get("name", run.name),
                "source": "this repo",
                "modality": meta.get("modality"),
                "run": run.name,
                "n": metrics.get("n"),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "ci_low": ci[0],
                "ci_high": ci[1],
                "notes": "; ".join(metrics.get("notes", [])),
            }
        )
    frame = pd.DataFrame(rows)
    return (
        frame.sort_values("macro_f1", ascending=False).reset_index(drop=True)
        if len(frame)
        else frame
    )


def leaderboard(runs_dir: Path | str = EXPERIMENTS, out: Path | str = TABLES) -> Path:
    """Write reports/tables/leaderboard.md from local runs plus published CSVs."""
    frames = [
        f
        for f in (collect_cv_runs(runs_dir), collect_runs(runs_dir), evaluate_published())
        if len(f)
    ]
    if not frames:
        raise RuntimeError(
            "nothing to report: no runs in experiments/ and no prediction CSVs in data/raw/"
        )
    table = pd.concat(frames, ignore_index=True).sort_values("macro_f1", ascending=False)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "leaderboard.csv", index=False)

    lines = [
        "# Results",
        "",
        "Macro-F1 is the headline metric; the 95% interval is a 2000-round item bootstrap",
        "over pooled out-of-fold predictions. **Only rows marked `5-fold CV` are mutually",
        "comparable** — they share one grouped, stratified protocol with augmentation applied",
        "inside the training folds.",
        "",
        "Rows marked `published` are scored from the prediction files in the Drive folder.",
        "They come from three different protocols (see `notebooks/original/README.md`) and the",
        "text-model rows are computed on a test set containing augmented variants of training",
        "articles, so they are optimistic. They are kept here for traceability to the first",
        "submission, not for comparison.",
        "",
        "`per-fold` is the mean ± standard deviation across folds; a wide spread means the",
        "pooled figure rests on folds that disagree.",
        "",
        "| Model | Protocol | Population | n | Accuracy | Macro-F1 | 95% CI | Per-fold | Notes |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in table.itertuples():
        ci = (
            f"{row.ci_low:.3f}–{row.ci_high:.3f}"
            if pd.notna(row.ci_low) and pd.notna(row.ci_high)
            else "—"
        )
        fold = (
            f"{row.fold_mean:.3f} ± {row.fold_sd:.3f}"
            if pd.notna(getattr(row, "fold_mean", None)) and pd.notna(getattr(row, "fold_sd", None))
            else "—"
        )
        population = getattr(row, "population", None)
        lines.append(
            f"| {row.model} | {row.source} | {population if pd.notna(population) else '—'} | "
            f"{row.n} | {row.accuracy:.3f} | {row.macro_f1:.3f} | {ci} | {fold} | "
            f"{row.notes or ''} |"
        )
    path = out / "leaderboard.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s (%d rows)", path, len(table))
    return path
