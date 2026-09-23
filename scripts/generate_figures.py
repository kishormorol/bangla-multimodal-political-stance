"""Generate publication-quality bar charts for 5-fold CV results.

Outputs:
    paper/latex/TL.png    — Text model comparison
    paper/latex/Multi_L.png — Multimodal model comparison
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD = ROOT / "reports" / "tables" / "leaderboard.csv"
OUT_DIR = ROOT / "paper" / "latex"

# Display names for models
DISPLAY = {
    "banglabert": "BanglaBERT",
    "tfidf_logreg": "TF-IDF+LogReg",
    "mbert": "mBERT",
    "mt5": "mT5",
    "xlmr": "XLM-RoBERTa",
    "bangla_electra": "Bangla-ELECTRA",
    "majority": "Majority",
    "clip": "CLIP",
    "countvec_vit": "CountVec+ViT",
    "align": "ALIGN",
    "blip": "BLIP",
    "vilt": "ViLT",
    "flava": "FLAVA",
}

TEXT_MODELS = ["banglabert", "tfidf_logreg", "mbert", "mt5", "xlmr", "bangla_electra", "majority"]
MULTI_MODELS = ["clip", "countvec_vit", "align", "blip", "vilt", "flava"]


def load_leaderboard():
    rows = {}
    with open(LEADERBOARD, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["model"]] = row
    return rows


def make_bar_chart(models, data, out_path, title=None):
    """Create a horizontal bar chart sorted by Macro-F1 descending."""
    entries = []
    for m in models:
        r = data[m]
        f1 = float(r["macro_f1"])
        ci_lo = float(r["ci_low"])
        ci_hi = float(r["ci_high"])
        entries.append((DISPLAY[m], f1, ci_lo, ci_hi))

    # Sort descending by F1
    entries.sort(key=lambda x: x[1], reverse=True)

    names = [e[0] for e in entries]
    f1s = np.array([e[1] for e in entries])
    ci_los = np.array([e[2] for e in entries])
    ci_his = np.array([e[3] for e in entries])

    # Error bars: distance from the mean
    err_lo = f1s - ci_los
    err_hi = ci_his - f1s

    fig, ax = plt.subplots(figsize=(3.5, 2.2))

    y_pos = np.arange(len(names))
    bars = ax.barh(
        y_pos, f1s,
        xerr=[err_lo, err_hi],
        height=0.55,
        color="#4878CF",
        edgecolor="none",
        capsize=2.5,
        error_kw={"linewidth": 0.8, "color": "#333333"},
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("Macro-F1", fontsize=8)
    ax.set_xlim(0, 0.75)
    ax.tick_params(axis="x", labelsize=7)
    ax.invert_yaxis()  # best on top

    # Remove spines except bottom
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Light horizontal grid behind bars
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    # Add value labels
    for i, (v, lo, hi) in enumerate(zip(f1s, ci_los, ci_his)):
        ax.text(v + (ci_his[i] - v) + 0.012, i, f"{v:.3f}", va="center", fontsize=6.5, color="#333333")

    if title:
        ax.set_title(title, fontsize=9, pad=6)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    data = load_leaderboard()

    make_bar_chart(
        TEXT_MODELS, data,
        OUT_DIR / "TL.png",
        title="Text Models — 5-Fold CV (n = 198)",
    )

    make_bar_chart(
        MULTI_MODELS, data,
        OUT_DIR / "Multi_L.png",
        title="Multimodal Models — 5-Fold CV (n = 86)",
    )


if __name__ == "__main__":
    main()
