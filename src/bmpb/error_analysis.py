"""Error analysis across CV model predictions.

Produces confusion matrices, per-class error rates, hardest-item analysis,
outlet bias breakdown, and headline-length effects. All outputs go to
reports/tables/ and reports/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

from bmpb.paths import CORPUS, EXPERIMENTS, FIGURES, TABLES
from bmpb.significance import _latest_predictions
from bmpb.utils.logging import get_logger

log = get_logger(__name__)

LABEL_NAMES = {0: "govt_critique", 1: "neutral", 2: "govt_leaning"}
LABEL_ORDER = [0, 1, 2]


# ── helpers ──────────────────────────────────────────────────────────────────


def _load_corpus() -> pd.DataFrame:
    """Load corpus with item_id, title, outlet, has_image."""
    df = pd.read_csv(CORPUS)
    return df[["item_id", "title", "label", "label_name", "outlet", "has_image"]].copy()


def _pick_top_models(all_preds: dict[str, pd.DataFrame]) -> list[str]:
    """Return display-friendly names for BanglaBERT, CLIP, TF-IDF if present."""
    targets = {"banglabert", "clip", "tfidf_logreg"}
    found = []
    for name in all_preds:
        norm = name.lower().replace("-", "_").replace(" ", "_")
        if norm in targets:
            found.append(name)
    return sorted(found) if found else sorted(all_preds.keys())[:3]


# ── confusion matrix heatmaps ───────────────────────────────────────────────


def _save_confusion_matrices(
    all_preds: dict[str, pd.DataFrame], fig_dir: Path
) -> list[Path]:
    """Save confusion matrix heatmaps for top models. Returns saved paths."""
    top = _pick_top_models(all_preds)
    log.info("Confusion matrices for: %s", top)
    paths = []
    for model_name in top:
        df = all_preds[model_name]
        y_true = df["label"].values
        y_pred = df["predicted"].values
        cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

        fig, ax = plt.subplots(figsize=(5, 4))
        display_labels = [LABEL_NAMES[i] for i in LABEL_ORDER]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=display_labels,
            yticklabels=display_labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix: {model_name} (n={len(df)})")
        fig.tight_layout()

        safe = model_name.lower().replace(" ", "_")
        path = fig_dir / f"confusion_{safe}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
        log.info("  wrote %s", path)
    return paths


# ── per-class error rates ────────────────────────────────────────────────────


def _per_class_error_rates(all_preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-class accuracy and error rate for every model."""
    rows = []
    for model_name, df in sorted(all_preds.items()):
        for cls in LABEL_ORDER:
            mask = df["label"] == cls
            if mask.sum() == 0:
                continue
            correct = (df.loc[mask, "predicted"] == cls).sum()
            total = int(mask.sum())
            acc = correct / total
            rows.append({
                "model": model_name,
                "class": LABEL_NAMES[cls],
                "class_id": cls,
                "n": total,
                "correct": int(correct),
                "accuracy": round(float(acc), 4),
                "error_rate": round(1 - float(acc), 4),
            })
    return pd.DataFrame(rows)


# ── hardest items ────────────────────────────────────────────────────────────


def _hardest_items(
    all_preds: dict[str, pd.DataFrame], corpus: pd.DataFrame, top_n: int = 20
) -> pd.DataFrame:
    """Items most models get wrong, with headline text and model predictions."""
    # Collect error counts per item across all models
    error_counts: dict[str, int] = {}
    model_predictions: dict[str, dict[str, int]] = {}  # item -> {model: pred}
    total_models_per_item: dict[str, int] = {}

    for model_name, df in all_preds.items():
        for _, row in df.iterrows():
            iid = row["item_id"]
            total_models_per_item[iid] = total_models_per_item.get(iid, 0) + 1
            model_predictions.setdefault(iid, {})[model_name] = int(row["predicted"])
            if row["label"] != row["predicted"]:
                error_counts[iid] = error_counts.get(iid, 0) + 1

    # Build dataframe of items with errors
    rows = []
    for iid, err_count in error_counts.items():
        total = total_models_per_item[iid]
        rows.append({
            "item_id": iid,
            "errors": err_count,
            "models_tested": total,
            "error_frac": round(err_count / total, 3),
        })
    hard = pd.DataFrame(rows).sort_values("error_frac", ascending=False).head(top_n)

    # Join with corpus for headline and label
    hard = hard.merge(
        corpus[["item_id", "title", "label", "label_name", "outlet"]],
        on="item_id",
        how="left",
    )

    # Add model predictions as a string column
    def _pred_summary(iid: str) -> str:
        preds = model_predictions.get(iid, {})
        parts = []
        for m, p in sorted(preds.items()):
            parts.append(f"{m}={LABEL_NAMES.get(p, str(p))}")
        return "; ".join(parts)

    hard["model_predictions"] = hard["item_id"].apply(_pred_summary)
    return hard


# ── outlet bias ──────────────────────────────────────────────────────────────


def _outlet_accuracy(
    all_preds: dict[str, pd.DataFrame], corpus: pd.DataFrame
) -> pd.DataFrame:
    """Accuracy broken down by news outlet, averaged across models."""
    rows = []
    for model_name, df in sorted(all_preds.items()):
        merged = df.merge(corpus[["item_id", "outlet"]], on="item_id", how="left")
        for outlet, grp in merged.groupby("outlet"):
            acc = accuracy_score(grp["label"], grp["predicted"])
            rows.append({
                "model": model_name,
                "outlet": outlet,
                "n": len(grp),
                "accuracy": round(float(acc), 4),
            })
    return pd.DataFrame(rows)


def _save_outlet_chart(outlet_df: pd.DataFrame, fig_dir: Path) -> Path:
    """Bar chart of accuracy by outlet, averaged across models."""
    avg = outlet_df.groupby("outlet").agg(
        accuracy=("accuracy", "mean"),
        n=("n", "first"),
    ).reset_index().sort_values("accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(avg)), avg["accuracy"], color="steelblue")
    ax.set_xticks(range(len(avg)))
    ax.set_xticklabels(avg["outlet"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean Accuracy (across models)")
    ax.set_title("Accuracy by News Outlet")
    ax.set_ylim(0, 1)
    for bar, n in zip(bars, avg["n"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.tight_layout()
    path = fig_dir / "error_by_outlet.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── headline length effect ───────────────────────────────────────────────────


def _length_accuracy(
    all_preds: dict[str, pd.DataFrame], corpus: pd.DataFrame
) -> pd.DataFrame:
    """Accuracy by headline length bin."""
    corpus = corpus.copy()
    corpus["title_len"] = corpus["title"].fillna("").str.len()
    bins = [0, 30, 60, 100, 200, 9999]
    labels_bin = ["<30", "30-60", "60-100", "100-200", "200+"]
    corpus["len_bin"] = pd.cut(corpus["title_len"], bins=bins, labels=labels_bin, right=False)

    rows = []
    for model_name, df in sorted(all_preds.items()):
        merged = df.merge(corpus[["item_id", "len_bin"]], on="item_id", how="left")
        for lbin, grp in merged.groupby("len_bin", observed=True):
            if len(grp) == 0:
                continue
            acc = accuracy_score(grp["label"], grp["predicted"])
            rows.append({
                "model": model_name,
                "len_bin": str(lbin),
                "n": len(grp),
                "accuracy": round(float(acc), 4),
            })
    return pd.DataFrame(rows)


def _save_length_chart(length_df: pd.DataFrame, fig_dir: Path) -> Path:
    """Grouped bar chart of accuracy by headline length."""
    avg = length_df.groupby("len_bin").agg(
        accuracy=("accuracy", "mean"),
        n=("n", "sum"),
    ).reset_index()

    # Preserve bin order
    bin_order = ["<30", "30-60", "60-100", "100-200", "200+"]
    avg["len_bin"] = pd.Categorical(avg["len_bin"], categories=bin_order, ordered=True)
    avg = avg.sort_values("len_bin")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(avg)), avg["accuracy"], color="coral")
    ax.set_xticks(range(len(avg)))
    ax.set_xticklabels(avg["len_bin"])
    ax.set_xlabel("Headline Length (chars)")
    ax.set_ylabel("Mean Accuracy (across models)")
    ax.set_title("Accuracy by Headline Length")
    ax.set_ylim(0, 1)
    for bar, n in zip(bars, avg["n"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    path = fig_dir / "error_by_length.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── markdown report ──────────────────────────────────────────────────────────


def _df_to_md_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a pipe-delimited markdown table (no tabulate)."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def _write_markdown(
    per_class: pd.DataFrame,
    hardest: pd.DataFrame,
    outlet_df: pd.DataFrame,
    length_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Write a readable error_analysis.md summary."""
    lines = ["# Error Analysis\n"]

    # Per-class error rates
    lines.append("## Per-Class Error Rates\n")
    pivot = per_class.pivot_table(
        index="model", columns="class", values="error_rate"
    ).reset_index()
    lines.append(_df_to_md_table(pivot))
    lines.append("")

    # Hardest items
    lines.append("\n## Hardest Items (most models wrong)\n")
    for _, row in hardest.iterrows():
        title = str(row.get("title", ""))[:80]
        lines.append(
            f"- **{row['item_id']}** ({row['error_frac']:.0%} wrong, "
            f"{row['errors']}/{row['models_tested']} models) "
            f"true={row.get('label_name', '?')}, outlet={row.get('outlet', '?')}"
        )
        lines.append(f"  - Headline: {title}")
        preds = str(row.get("model_predictions", ""))
        if len(preds) > 200:
            preds = preds[:200] + "..."
        lines.append(f"  - Predictions: {preds}")
    lines.append("")

    # Outlet accuracy
    lines.append("\n## Accuracy by Outlet\n")
    outlet_avg = outlet_df.groupby("outlet").agg(
        mean_acc=("accuracy", "mean"),
        n=("n", "first"),
    ).sort_values("mean_acc", ascending=False).reset_index()
    lines.append(_df_to_md_table(outlet_avg))
    lines.append("")

    # Length accuracy
    lines.append("\n## Accuracy by Headline Length\n")
    len_avg = length_df.groupby("len_bin").agg(
        mean_acc=("accuracy", "mean"),
        total_items=("n", "sum"),
    ).reset_index()
    lines.append(_df_to_md_table(len_avg))
    lines.append("")

    path = out_dir / "error_analysis.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ── public entry point ───────────────────────────────────────────────────────


def run_error_analysis(
    runs_dir: Path = EXPERIMENTS,
    out_dir: Path = TABLES,
    fig_dir: Path = FIGURES,
) -> dict:
    """Run the full error analysis pipeline.

    Returns a summary dict with counts and file paths.
    """
    out_dir = Path(out_dir)
    fig_dir = Path(fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load predictions and corpus
    all_preds = _latest_predictions(runs_dir)
    if not all_preds:
        raise RuntimeError("No CV predictions found in " + str(runs_dir))
    log.info("Loaded predictions for %d models", len(all_preds))

    corpus = _load_corpus()
    log.info("Corpus: %d items", len(corpus))

    # 2a. Confusion matrix heatmaps
    cm_paths = _save_confusion_matrices(all_preds, fig_dir)

    # 2b. Per-class error rates
    per_class = _per_class_error_rates(all_preds)
    log.info("Per-class error rates computed for %d models", per_class["model"].nunique())

    # 2c. Hardest items
    hardest = _hardest_items(all_preds, corpus, top_n=20)
    log.info("Top %d hardest items identified", len(hardest))

    # 2d. Outlet bias
    outlet_df = _outlet_accuracy(all_preds, corpus)
    outlet_path = _save_outlet_chart(outlet_df, fig_dir)
    log.info("Outlet chart: %s", outlet_path)

    # 2e. Headline length effect
    length_df = _length_accuracy(all_preds, corpus)
    length_path = _save_length_chart(length_df, fig_dir)
    log.info("Length chart: %s", length_path)

    # 3. Write outputs
    md_path = _write_markdown(per_class, hardest, outlet_df, length_df, out_dir)
    log.info("Markdown report: %s", md_path)

    # Raw CSV with per-class data
    csv_path = out_dir / "error_analysis.csv"
    per_class.to_csv(csv_path, index=False)
    log.info("CSV: %s", csv_path)

    return {
        "models": len(all_preds),
        "confusion_matrices": [str(p) for p in cm_paths],
        "hardest_items": len(hardest),
        "md_report": str(md_path),
        "csv": str(csv_path),
        "outlet_chart": str(outlet_path),
        "length_chart": str(length_path),
    }
