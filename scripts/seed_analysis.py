"""Aggregate the matched sweep over seeds.

Fixed before the extra seeds were run: for every model, the mean, sample SD,
minimum and maximum of pooled macro-F1 across seeds; for every pre-specified
pair in matched_analysis (each VL model vs TF-IDF and vs BanglaBERT), the
per-seed paired difference with its source-group interval, the mean difference,
and how many seeds give an interval entirely above or below zero.

    python scripts/seed_analysis.py experiments/matched experiments/seeds/seed-*
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matched_analysis import LABELS, MULTIMODAL, TEXT_REFERENCES, paired_difference, predictions


def comparable_run_contract(identity: dict) -> str:
    config = dict(identity.get("config", {}))
    config.pop("seed", None)
    return json.dumps(
        {
            "config": config,
            "n_splits": identity.get("n_splits"),
            "augmentation_requested": identity.get("augmentation_requested"),
            "augmentation_enabled": identity.get("augmentation_enabled"),
            "augmentation_count": identity.get("augmentation_count"),
            "augmentation_sha256": identity.get("augmentation_sha256"),
            "code_sha256": identity.get("code_sha256"),
            "software": identity.get("software"),
            "group_membership_sha256": identity.get("group_membership_sha256"),
        },
        sort_keys=True,
    )


def format_sd(value):
    return f"{value:.3f}" if pd.notna(value) else "NA"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--dest", type=Path, default=Path("reports/submission"))
    args = parser.parse_args()

    runs, diffs = [], []
    seen_seeds = set()
    population_inputs = {}
    population_groups = {}
    model_contracts = {}
    for root in args.roots:
        summary = json.loads((root / "sweep_summary.json").read_text())
        root_contracts = {}
        for row in summary:
            if row.get("status") != "ok":
                continue
            predictions(root, row["population"], row["name"])
            identity = row.get("identity")
            if not identity:
                raise ValueError(
                    f"{root}: legacy summary lacks run identity; historical inspection only"
                )
            if identity.get("seed") != row.get("seed"):
                raise ValueError(f"{root}: seed differs from run identity for {row['name']}")
            prior = population_inputs.setdefault(row["population"], identity.get("input_sha256"))
            if prior != identity.get("input_sha256"):
                raise ValueError(f"mixed input populations for {row['population']}")
            group_hash = identity.get("group_membership_sha256")
            prior_groups = population_groups.setdefault(row["population"], group_hash)
            if not group_hash or prior_groups != group_hash:
                raise ValueError(f"mixed group membership for {row['population']}")
            model_key = (row["population"], row["name"])
            prior_model = model_contracts.setdefault(model_key, comparable_run_contract(identity))
            if prior_model != comparable_run_contract(identity):
                raise ValueError(f"incomparable run contract across roots for {model_key}")
            contract = (
                identity.get("seed"),
                identity.get("n_splits"),
                json.dumps(identity.get("fold_item_ids"), sort_keys=True),
            )
            previous = root_contracts.setdefault(row["population"], contract)
            if previous != contract:
                raise ValueError(f"mixed seeds or folds within {root}/{row['population']}")
            seed_key = (row["population"], row["name"], row.get("seed"))
            if seed_key in seen_seeds:
                raise ValueError(f"duplicate seed run for {seed_key}")
            seen_seeds.add(seed_key)
        ok = {(r["population"], r["name"]) for r in summary if r["status"] == "ok"}
        for row in summary:
            if row["status"] != "ok":
                runs.append(
                    {
                        "root": str(root),
                        "population": row["population"],
                        "name": row["name"],
                        "status": "failed",
                        "error": row.get("error"),
                    }
                )
                continue
            runs.append(
                {
                    "root": str(root),
                    "population": row["population"],
                    "name": row["name"],
                    "seed": row["seed"],
                    "status": "ok",
                    "macro_f1": row["macro_f1"],
                    "run_dir": row["run_dir"],
                    "items_scored": row["items_scored"],
                    "input_items": row["input_items"],
                    "git_commit": row.get("git_commit"),
                    "identity": row["identity"],
                }
            )
        for model in MULTIMODAL:
            for reference in TEXT_REFERENCES:
                if {("paired_headlines", model), ("paired_headlines", reference)} - ok:
                    continue
                result = paired_difference(
                    predictions(root, "paired_headlines", model),
                    predictions(root, "paired_headlines", reference),
                )
                seed = next(
                    r["seed"]
                    for r in summary
                    if r["name"] == model and r["population"] == "paired_headlines"
                )
                diffs.append({"seed": seed, "model": model, "reference": reference, **result})

    frame = pd.DataFrame([r for r in runs if r["status"] == "ok"])
    if frame.empty:
        raise ValueError("no successful runs to aggregate")
    failed = [r for r in runs if r["status"] != "ok"]
    models = (
        frame.groupby(["population", "name"])["macro_f1"]
        .agg(seeds="count", mean="mean", sd=lambda s: s.std(ddof=1), min="min", max="max")
        .reset_index()
    )
    deltas = pd.DataFrame(diffs, columns=["seed", "model", "reference", "delta_macro_f1", "ci95"])
    deltas["above"] = deltas["ci95"].map(lambda ci: ci[0] > 0)
    deltas["below"] = deltas["ci95"].map(lambda ci: ci[1] < 0)
    pairs = (
        deltas.groupby(["model", "reference"])
        .agg(
            seeds=("seed", "count"),
            mean=("delta_macro_f1", "mean"),
            sd=("delta_macro_f1", lambda s: s.std(ddof=1)),
            min=("delta_macro_f1", "min"),
            max=("delta_macro_f1", "max"),
            ci_above_zero=("above", "sum"),
            ci_below_zero=("below", "sum"),
        )
        .reset_index()
    )

    lines = [
        "| Population | Model | Seeds | Mean macro-F1 | SD | Min | Max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in models.itertuples():
        lines.append(
            f"| {r.population} | {r.name} | {r.seeds} | {r.mean:.3f} | {format_sd(r.sd)} | "
            f"{r.min:.3f} | {r.max:.3f} |"
        )
    lines += [
        "",
        "| VL model | Reference | Seeds | Mean Δ | SD | Min | Max | CI>0 | CI<0 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in pairs.itertuples():
        lines.append(
            f"| {r.model} | {r.reference} | {r.seeds} | {r.mean:+.3f} | {format_sd(r.sd)} | "
            f"{r.min:+.3f} | {r.max:+.3f} | {r.ci_above_zero} | {r.ci_below_zero} |"
        )
    if failed:
        lines += ["", f"Failed runs: {failed}"]

    tex = ["% Generated by scripts/seed_analysis.py; do not edit."]
    for population in ("paired_headlines", "all_headlines"):
        tex.append(f"\\newcommand{{\\SeedRows{population.split('_')[0].capitalize()}}}{{%")
        for r in models[models.population == population].itertuples():
            modality = "text+image" if r.name in MULTIMODAL else "text"
            tex.append(
                f"{LABELS[r.name]} & {modality} & {r.seeds} & {r.mean:.3f} & {format_sd(r.sd)} & "
                f"{r.min:.3f}--{r.max:.3f} \\\\"
            )
        tex.append("}")
    tex.append("\\newcommand{\\SeedRowsDiff}{%")
    for r in pairs.itertuples():
        tex.append(
            f"{LABELS[r.model]} & {LABELS[r.reference]} & ${r.mean:+.3f}$ & {format_sd(r.sd)} & "
            f"${r.min:+.3f}$ to ${r.max:+.3f}$ & {r.ci_above_zero}/{r.seeds} & "
            f"{r.ci_below_zero}/{r.seeds} \\\\"
        )
    tex.append("}")

    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "seeds.json").write_text(
        json.dumps(
            {
                "roots": [str(r) for r in args.roots],
                "runs": runs,
                "paired_differences": diffs,
                "models": models.astype(object).where(models.notna(), None).to_dict("records"),
                "pairs": pairs.astype(object).where(pairs.notna(), None).to_dict("records"),
            },
            indent=2,
            default=lambda v: v.item() if isinstance(v, np.generic) else str(v),
        )
        + "\n"
    )
    (args.dest / "seeds.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.dest / "seeds_tables.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
