"""Recalculate committed summary arithmetic without data or model dependencies.

This verifies saved summaries, not predictions, confidence intervals, annotation
semantics, or training. Original run artifacts are required for those checks.
"""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def equal(actual, expected, context):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"{context}: saved {actual}, recalculated {expected}")


def distribution(row, values, context):
    if not values:
        raise ValueError(f"{context}: no successful runs")
    if row["seeds"] != len(values):
        raise ValueError(f"{context}: inconsistent seed count")
    for field, expected in (
        ("mean", mean(values)),
        ("sd", stdev(values) if len(values) > 1 else float("nan")),
        ("min", min(values)),
        ("max", max(values)),
    ):
        if field == "sd" and len(values) == 1 and row[field] is None:
            continue
        equal(row[field], expected, f"{context} {field}")


def verify(directory):
    matched = json.loads((directory / "matched.json").read_text())
    seeds = json.loads((directory / "seeds.json").read_text())
    early = json.loads((directory / "early_stopping.json").read_text())
    matched_counts = defaultdict(set)
    checked = 0
    for row in matched["runs"]:
        if row["status"] != "ok":
            continue
        context = f"{row['population']}/{row['name']}"
        matrix = row["confusion"]
        if len(matrix) != 3 or any(len(cells) != 3 for cells in matrix):
            raise ValueError(f"{context}: expected three class confusion matrix")
        if any(type(v) is not int or v < 0 for cells in matrix for v in cells):
            raise ValueError(f"{context}: invalid confusion counts")
        n = sum(map(sum, matrix))
        if not n or n != row["items_scored"] or n != row["input_items"]:
            raise ValueError(f"{context}: inconsistent item count")
        f1 = []
        for i in range(3):
            denominator = sum(matrix[i]) + sum(cells[i] for cells in matrix)
            f1.append(2 * matrix[i][i] / denominator if denominator else 0.0)
        equal(row["macro_f1"], mean(f1), f"{context} macro F1")
        equal(row["accuracy"], sum(matrix[i][i] for i in range(3)) / n, context)
        ids = [item for fold in row["fold_item_ids"] for item in fold]
        if len(ids) != n or len(set(ids)) != n:
            raise ValueError(f"{context}: repeated or missing fold identifiers")
        matched_counts[row["population"]].add(n)
        checked += 1

    runs = {}
    by_model = defaultdict(list)
    for row in seeds["runs"]:
        if row["status"] != "ok":
            continue
        key = row["population"], row["name"], row["seed"]
        if key in runs:
            raise ValueError(f"duplicate seed result: {key}")
        runs[key] = row["macro_f1"]
        by_model[key[:2]].append(row["macro_f1"])
    model_keys = [(row["population"], row["name"]) for row in seeds["models"]]
    if len(set(model_keys)) != len(model_keys) or set(model_keys) != set(by_model):
        raise ValueError("model summaries do not cover successful seed runs exactly")
    for row in seeds["models"]:
        key = row["population"], row["name"]
        distribution(row, by_model[key], str(key))

    pairs = defaultdict(list)
    pair_keys = set()
    populations = set()
    for row in seeds["paired_differences"]:
        key = row["model"], row["reference"], row["seed"]
        if key in pair_keys:
            raise ValueError(f"duplicate seed comparison: {key}")
        pair_keys.add(key)
        expected = (
            runs["paired_headlines", row["model"], row["seed"]]
            - runs["paired_headlines", row["reference"], row["seed"]]
        )
        equal(row["delta_macro_f1"], expected, str(key))
        if not 0 < row["groups"] <= row["n"]:
            raise ValueError(f"{key}: invalid group or item count")
        pairs[key[:2]].append(row)
        populations.add((row["n"], row["groups"]))
    summary_keys = [(row["model"], row["reference"]) for row in seeds["pairs"]]
    if len(set(summary_keys)) != len(summary_keys) or set(summary_keys) != set(pairs):
        raise ValueError("pair summaries do not cover seed comparisons exactly")
    for row in seeds["pairs"]:
        key = row["model"], row["reference"]
        values = pairs[key]
        distribution(row, [v["delta_macro_f1"] for v in values], str(key))
        for field, expected in (
            ("ci_above_zero", sum(v["ci95"][0] > 0 for v in values)),
            ("ci_below_zero", sum(v["ci95"][1] < 0 for v in values)),
        ):
            if row[field] != expected:
                raise ValueError(f"{key}: inconsistent {field}")

    selected = early["selected_epoch_when_early_stopped"]
    if sum(selected.values()) != early["early_stopped"]:
        raise ValueError("selected epoch counts do not sum to early stopped runs")
    if early["early_stopped"] > early["neural_fold_runs"]:
        raise ValueError("early stopped count exceeds neural fold runs")
    if (
        sum(selected.get(str(i), 0) for i in (1, 2))
        != early["early_stopped_selecting_epoch_1_or_2"]
    ):
        raise ValueError("inconsistent count of selected epochs 1 and 2")
    reproduced = 0
    reproduction = directory.parent / "verification/original_baselines.json"
    if reproduction.exists():
        saved = json.loads(reproduction.read_text())
        for row in saved["results"]:
            source = next(
                r
                for r in matched["runs"]
                if r["population"] == row["population"] and r["name"] == row["model"]
            )
            equal(row["macro_f1"], source["macro_f1"], f"reproduction {row['model']}")
            if (
                row["input_sha256"] != source["input_sha256"]
                or row["confusion"] != source["confusion"]
                or row["ci95"] != source["macro_f1_ci95"]
            ):
                raise ValueError(f"reproduction mismatch: {row['population']}/{row['model']}")
            reproduced += 1
    recovery_path = directory.parent / "verification/input_recovery.json"
    if recovery_path.exists():
        recovery = json.loads(recovery_path.read_text())
        paired = [r for r in matched["runs"] if r["population"] == "paired_headlines"]
        ids = {item for fold in paired[0]["fold_item_ids"] for item in fold}
        removed, added = map(set, (recovery["saved_membership_only"], recovery["image_table_only"]))
        if (
            recovery["matched_report_sha256"]
            != hashlib.sha256((directory / "matched.json").read_bytes()).hexdigest()
            or recovery["saved_models_with_identical_membership"] != len(paired)
            or any(r["fold_item_ids"] != paired[0]["fold_item_ids"] for r in paired)
            or not removed <= ids
            or bool(added & ids)
            or recovery["saved_paired_items"] != len(ids)
            or recovery["membership_overlap"] != len(ids - removed)
            or recovery["image_table_items"] != len((ids - removed) | added)
            or recovery["verified_selected_images"] + recovery["missing_selected_images"]
            != len(ids)
            or sum(recovery["image_sources"].values()) != recovery["verified_selected_images"]
            or sum(recovery["label_counts"].values()) != len(ids)
            or recovery["headline_input_sha256"] != paired[0]["input_sha256"]
        ):
            raise ValueError("input recovery counts or source identity disagree")
    return {
        "matched_scores_recalculated": checked,
        "matched_item_counts": {key: sorted(v) for key, v in matched_counts.items()},
        "seed_runs_checked": len(runs),
        "seed_comparisons_checked": len(pair_keys),
        "independent_baseline_summaries_matched": reproduced,
        "recovered_input_summary_checked": recovery_path.exists(),
        "seed_comparison_populations": [
            {"items": n, "source_groups": groups} for n, groups in sorted(populations)
        ],
        "limits": (
            "This command checks summaries only. It does not load predictions or train models."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports/submission"))
    args = parser.parse_args()
    try:
        result = verify(args.reports)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"Report verification failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
