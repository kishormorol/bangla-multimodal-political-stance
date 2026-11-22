"""Paired comparisons for the image ablation, across seeds.

Fixed before any ablation run (author's decision, HANDOFF.md, 2026-11-18), on
the frozen 149 paired items and each seed's shared outer folds:

* every VL model with its real photos vs the same model with a blank photo,
  and vs the same model with shuffled photos (does the model use the image?);
* tfidf_vit vs tfidf_logreg (does the photo add to the best text features?)
  and tfidf_vit vs tfidf_vit_shuffled;
* vit_only vs majority (do the photos carry label signal on their own?).

Each comparison is a source-group bootstrap of the macro-F1 difference (A minus
B), as in matched_analysis. Seeds are summarised by the mean, sample SD and
range of the difference, and by how many seeds give an interval entirely above
or below zero.

    python scripts/ablation_analysis.py experiments/seeds149/seed-*
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from matched_analysis import MULTIMODAL, paired_difference, predictions

from bmpb.data.groups import source_groups
from bmpb.paths import CORPUS

POPULATION = "paired_headlines"
FROZEN_N = 149
COMPARISONS = (
    [(m, f"{m}_blank") for m in MULTIMODAL]
    + [(m, f"{m}_shuffled") for m in MULTIMODAL]
    + [("tfidf_vit", "tfidf_logreg"), ("tfidf_vit", "tfidf_vit_shuffled"),
       ("vit_only", "majority")]
)
# Follow-up set, fixed before its runs (author's decision, HANDOFF.md,
# 2026-11-22), after the primary set's results were seen:
# * tfidf_vit_l2 (image vectors L2-normalised) vs tfidf_logreg, vs
#   tfidf_vit_l2_shuffled, and vs tfidf_vit: is tfidf_vit's deficit a scale
#   artefact, and does the photo then add anything?
# * clip_en (English machine-translated headlines) vs clip, and vs
#   clip_en_blank: is CLIP held back by its tokenizer, and does it still use
#   the photo with readable text?
FOLLOWUP = [
    ("tfidf_vit_l2", "tfidf_logreg"), ("tfidf_vit_l2", "tfidf_vit_l2_shuffled"),
    ("tfidf_vit_l2", "tfidf_vit"),
    ("clip_en", "clip"), ("clip_en", "clip_en_blank"),
]
SETS = {"primary": (COMPARISONS, "ablation", "AblationRows"),
        "followup": (FOLLOWUP, "ablation_followup", "AblationFollowupRows")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--dest", type=Path, default=Path("reports/submission"))
    parser.add_argument("--set", choices=sorted(SETS), default="primary")
    args = parser.parse_args()
    comparisons, stem, macro = SETS[args.set]

    corpus = pd.read_csv(CORPUS)
    groups = pd.Series(source_groups(corpus).to_numpy(), index=corpus["item_id"])

    rows, missing = [], []
    for root in args.roots:
        summary = json.loads((root / "sweep_summary.json").read_text())
        ok = {r["name"]: r for r in summary
              if r["population"] == POPULATION and r["status"] == "ok"}
        for a, b in comparisons:
            if a not in ok or b not in ok:
                missing.append({"root": str(root), "a": a, "b": b})
                continue
            result = paired_difference(
                predictions(root, POPULATION, a), predictions(root, POPULATION, b), groups
            )
            if result["n"] != FROZEN_N:
                raise AssertionError(
                    f"{root}: {a} vs {b} scored {result['n']} items, not {FROZEN_N}")
            rows.append({"root": str(root), "seed": ok[a]["seed"], "a": a, "b": b,
                         "macro_f1_a": ok[a]["macro_f1"], "macro_f1_b": ok[b]["macro_f1"],
                         **result})

    if not rows:
        raise SystemExit("no complete ablation pair found")
    frame = pd.DataFrame(rows)
    frame["above"] = frame["ci95"].map(lambda ci: ci[0] > 0)
    frame["below"] = frame["ci95"].map(lambda ci: ci[1] < 0)
    order = {pair: i for i, pair in enumerate(comparisons)}
    table = (
        frame.groupby(["a", "b"])
        .agg(seeds=("seed", "count"), n=("n", "first"),
             mean_a=("macro_f1_a", "mean"), mean_b=("macro_f1_b", "mean"),
             mean=("delta_macro_f1", "mean"), sd=("delta_macro_f1", lambda s: s.std(ddof=1)),
             min=("delta_macro_f1", "min"), max=("delta_macro_f1", "max"),
             ci_above_zero=("above", "sum"), ci_below_zero=("below", "sum"))
        .reset_index()
    )
    table["order"] = [order[(a, b)] for a, b in zip(table.a, table.b, strict=True)]
    table = table.sort_values("order").drop(columns="order").reset_index(drop=True)

    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / f"{stem}.json").write_text(
        json.dumps({"comparisons": [list(c) for c in comparisons], "per_seed": rows,
                    "summary": table.to_dict(orient="records"), "missing": missing},
                   indent=2, default=str) + "\n", encoding="utf-8")

    lines = ["| A | B | Seeds | n | Mean F1 A | Mean F1 B | Mean Δ (A−B) | SD | Min | Max "
             "| CI>0 | CI<0 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    tex = ["% Generated by scripts/ablation_analysis.py; do not edit.",
           f"\\newcommand{{\\{macro}}}{{%"]
    for r in table.itertuples():
        lines.append(f"| {r.a} | {r.b} | {r.seeds} | {r.n} | {r.mean_a:.3f} | {r.mean_b:.3f} | "
                     f"{r.mean:+.3f} | {r.sd:.3f} | {r.min:+.3f} | {r.max:+.3f} | "
                     f"{r.ci_above_zero} | {r.ci_below_zero} |")
        tex.append(f"\\texttt{{{r.a.replace('_', chr(92) + '_')}}} & "
                   f"\\texttt{{{r.b.replace('_', chr(92) + '_')}}} & "
                   f"{r.mean_a:.3f} & {r.mean_b:.3f} & "
                   f"${r.mean:+.3f}$ & {r.sd:.3f} & ${r.min:+.3f}$ to ${r.max:+.3f}$ & "
                   f"{r.ci_above_zero}/{r.seeds} & {r.ci_below_zero}/{r.seeds} \\\\")
    tex.append("}")
    if missing:
        lines += ["", f"Missing pairs: {missing}"]
    (args.dest / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.dest / f"{stem}_tables.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
