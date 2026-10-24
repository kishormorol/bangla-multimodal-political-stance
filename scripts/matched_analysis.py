"""Tabulate the matched sweep and estimate paired macro-F1 differences.

Comparisons are fixed here before results are seen: every vision-language model
against each of two text references, TF-IDF/logistic regression and BanglaBERT,
on the same readable-image items and outer folds. The interval resamples source
groups (not items) 2,000 times, because items sharing a source are not
independent.

    python scripts/matched_analysis.py experiments/matched
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

TEXT_REFERENCES = ["tfidf_logreg", "banglabert"]
MULTIMODAL = ["countvec_vit", "clip", "align", "blip", "vilt", "flava"]
RESAMPLES = 2000
SEED = 42


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)


def predictions(root: Path, population: str, name: str) -> pd.DataFrame:
    summary = json.loads((root / "sweep_summary.json").read_text(encoding="utf-8"))
    rows = [r for r in summary if r.get("population") == population and r.get("name") == name]
    if len(rows) != 1:
        raise ValueError(f"expected one summary row for {population}/{name}, found {len(rows)}")
    row = rows[0]
    if row.get("status") != "ok" or not row.get("identity") or not row.get("run_dir"):
        raise ValueError(
            f"{population}/{name} lacks current run identity metadata; historical inspection only"
        )
    run = (root / row["run_dir"]).resolve()
    if not run.is_relative_to(root.resolve()):
        raise ValueError(f"run path escapes sweep root for {population}/{name}")
    manifest_path = run / "run.json"
    prediction_path = run / "predictions.csv"
    cv_path = run / "cv.json"
    if not all(path.is_file() for path in (manifest_path, prediction_path, cv_path)):
        raise FileNotFoundError(
            f"required exact run artifacts missing for {population}/{name}: {run}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    config_path = run / "config.yaml"
    artifacts = manifest.get("artifacts", {})
    if not config_path.is_file() or any(
        artifacts.get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (config_path, prediction_path, cv_path)
    ):
        raise ValueError(f"run artifact checksum mismatch for {population}/{name}")
    if (
        manifest.get("identity") != row["identity"]
        or cv.get("identity") != row["identity"]
        or cv.get("name") != name
        or cv.get("seed") != row.get("seed")
    ):
        raise ValueError(f"run manifest mismatch for {population}/{name}")
    for field, value in cv.items():
        if field not in {"population", "run_dir"} and row.get(field) != value:
            raise ValueError(
                f"summary disagrees with verified run field {field}: {population}/{name}"
            )
    frame = pd.read_csv(prediction_path)
    if frame.item_id.duplicated().any() or len(frame) != row.get("items_scored"):
        raise ValueError(f"prediction artifact does not match run summary for {population}/{name}")
    return frame


def paired_difference(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    merged = a.merge(b, on=["item_id", "label"], suffixes=("_a", "_b"))
    if len(merged) != len(a) or len(merged) != len(b):
        raise AssertionError("paired comparison requires identical item populations")
    if not (merged["fold_a"] == merged["fold_b"]).all():
        raise AssertionError("paired comparison requires identical outer folds")
    if "group_a" not in merged or not (merged["group_a"] == merged["group_b"]).all():
        raise AssertionError("paired comparison requires identical frozen groups")
    merged["group"] = merged["group_a"]
    observed = macro_f1(merged.label, merged.predicted_a) - macro_f1(
        merged.label, merged.predicted_b
    )
    members = [g.index.to_numpy() for _, g in merged.groupby("group")]
    rng = np.random.default_rng(SEED)
    y = merged["label"].to_numpy()
    pa, pb = merged["predicted_a"].to_numpy(), merged["predicted_b"].to_numpy()
    deltas = np.empty(RESAMPLES)
    for r in range(RESAMPLES):
        index = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        deltas[r] = macro_f1(y[index], pa[index]) - macro_f1(y[index], pb[index])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "n": int(len(merged)),
        "groups": len(members),
        "delta_macro_f1": float(observed),
        "ci95": [float(lo), float(hi)],
        "share_resamples_above_zero": float((deltas > 0).mean()),
    }


LABELS = {
    "majority": "Majority",
    "tfidf_logreg": "TF-IDF + LogReg",
    "banglabert": "BanglaBERT",
    "bangla_electra": "Bangla-ELECTRA",
    "mbert": "mBERT",
    "xlmr": "XLM-RoBERTa",
    "mt5": "mT5 (encoder)",
    "countvec_vit": "CountVec + ViT",
    "clip": "CLIP",
    "align": "ALIGN",
    "blip": "BLIP",
    "vilt": "ViLT",
    "flava": "FLAVA",
}


def latex(summary: list[dict], comparisons: list[dict]) -> str:
    """Table bodies for the manuscript; numbers are never retyped by hand."""
    out = ["% Generated by scripts/matched_analysis.py from experiments/matched/; do not edit."]
    for population in ("paired_headlines", "all_headlines"):
        out.append(f"\\newcommand{{\\Rows{population.split('_')[0].capitalize()}}}{{%")
        for row in summary:
            if row["population"] != population or row["status"] != "ok":
                continue
            lo, hi = row["macro_f1_ci95"]
            f1 = row["per_class_f1"]
            modality = "text+image" if row["name"] in MULTIMODAL else "text"
            out.append(
                f"{LABELS[row['name']]} & {modality} & {row['accuracy']:.3f} & "
                f"{row['macro_f1']:.3f} & {lo:.3f}--{hi:.3f} & "
                f"{row['fold_macro_f1_mean']:.3f} $\\pm$ {row['fold_macro_f1_sd']:.3f} & "
                f"{f1['govt_critique']:.3f} & {f1['neutral']:.3f} & {f1['govt_leaning']:.3f} \\\\"
            )
        out.append("}")
    out.append("\\newcommand{\\RowsPairedDiff}{%")
    for c in comparisons:
        if c.get("status") == "missing":
            continue
        lo, hi = c["ci95"]
        out.append(
            f"{LABELS[c['model']]} & {LABELS[c['reference']]} & "
            f"${c['delta_macro_f1']:+.3f}$ & ${lo:+.3f}$ to ${hi:+.3f}$ & "
            f"{c['share_resamples_above_zero']:.3f} \\\\"
        )
    out.append("}")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dest", type=Path, default=Path("reports/submission"))
    args = parser.parse_args()

    summary = json.loads((args.root / "sweep_summary.json").read_text())
    for row in summary:
        if row.get("status") == "ok":
            predictions(args.root, row["population"], row["name"])
    for population in {r.get("population") for r in summary}:
        rows = [r for r in summary if r.get("population") == population and r.get("status") == "ok"]
        contracts = {
            json.dumps(
                {
                    key: r.get("identity", {}).get(key)
                    for key in ("input_sha256", "seed", "n_splits", "fold_item_ids")
                },
                sort_keys=True,
            )
            for r in rows
        }
        if rows and (
            any(not r.get("identity", {}).get("input_sha256") for r in rows) or len(contracts) > 1
        ):
            raise ValueError(f"mixed or missing run populations in {population}")
    lines = [
        "| Population | Model | Status | n | Accuracy | Pooled macro-F1 | 95% CI | "
        "Fold mean ± SD | F1 gc | F1 neu | F1 gl | Notes |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary:
        if row["status"] != "ok":
            lines.append(
                f"| {row['population']} | {row['name']} | failed: {row['error'][:60]} "
                "|  |  |  |  |  |  |  |  |  |"
            )
            continue
        lo, hi = row["macro_f1_ci95"]
        f1 = row["per_class_f1"]
        lines.append(
            f"| {row['population']} | {row['name']} | ok | {row['items_scored']} | "
            f"{row['accuracy']:.3f} | {row['macro_f1']:.3f} | {lo:.3f}–{hi:.3f} | "
            f"{row['fold_macro_f1_mean']:.3f} ± {row['fold_macro_f1_sd']:.3f} | "
            f"{f1['govt_critique']:.3f} | {f1['neutral']:.3f} | {f1['govt_leaning']:.3f} | "
            f"{'; '.join(row['notes'])} |"
        )

    ok = {(r["population"], r["name"]) for r in summary if r["status"] == "ok"}
    comparisons = []
    for model in MULTIMODAL:
        for reference in TEXT_REFERENCES:
            if {("paired_headlines", model), ("paired_headlines", reference)} - ok:
                comparisons.append({"model": model, "reference": reference, "status": "missing"})
                continue
            result = paired_difference(
                predictions(args.root, "paired_headlines", model),
                predictions(args.root, "paired_headlines", reference),
            )
            comparisons.append({"model": model, "reference": reference, **result})

    lines += [
        "",
        "Paired macro-F1 difference (vision-language minus text), readable-image items, "
        f"source-group bootstrap with {RESAMPLES} resamples:",
        "",
        "| Model | Text reference | n | Groups | Δ macro-F1 | 95% CI | Share Δ>0 |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for c in comparisons:
        if c.get("status") == "missing":
            lines.append(f"| {c['model']} | {c['reference']} | missing |  |  |  |  |")
            continue
        lo, hi = c["ci95"]
        lines.append(
            f"| {c['model']} | {c['reference']} | {c['n']} | {c['groups']} | "
            f"{c['delta_macro_f1']:+.3f} | {lo:+.3f} to {hi:+.3f} | "
            f"{c['share_resamples_above_zero']:.3f} |"
        )

    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "matched_tables.tex").write_text(latex(summary, comparisons))
    (args.dest / "matched.json").write_text(
        json.dumps({"runs": summary, "paired_differences": comparisons}, indent=2) + "\n"
    )
    (args.dest / "matched.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
