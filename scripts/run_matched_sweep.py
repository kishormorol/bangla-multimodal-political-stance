"""Run every committed model config under the corrected matched protocol.

Primary population: the readable-image items, headline input, for every text and
vision-language config. Secondary population: all items, headline input, text
configs only. Folds, source groups and inner validation come from
`bmpb.crossval`; nothing here selects or tunes anything.

A model that fails is recorded as failed with its error, not dropped silently.

    python -u scripts/run_matched_sweep.py --out experiments/matched
    python -u scripts/run_matched_sweep.py --seed 43 --out experiments/seeds/seed-43

`--seed` overrides the config seed, which sets both the fold assignment and
model initialisation; within one seed every model still shares the same folds.
"""

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import pandas as pd
import yaml

from bmpb.crossval import cross_validate
from bmpb.data.dataset import load_image
from bmpb.paths import CONFIGS, CORPUS

TEXT = ["majority", "tfidf_logreg", "banglabert", "bangla_electra", "mbert", "xlmr", "mt5"]
MULTIMODAL = ["countvec_vit", "clip", "align", "blip", "vilt", "flava"]
# Added after the matched results, by the author's decision (HANDOFF.md,
# 2026-11-18): each VL model with blank and with shuffled photos, the
# TF-IDF + ViT pair for tfidf_logreg, and an image-only floor.
ABLATION = [f"{m}_{mode}" for m in MULTIMODAL for mode in ("blank", "shuffled")] + [
    "tfidf_vit", "tfidf_vit_shuffled", "vit_only"]
# Follow-up ablations (author's decision, HANDOFF.md, 2026-11-22).
FOLLOWUP = ["tfidf_vit_l2", "tfidf_vit_l2_shuffled", "clip_en", "clip_en_blank"]


def populations():
    corpus = pd.read_csv(CORPUS)
    corpus["text"] = corpus["title"]
    corpus["has_image"] = [load_image(path) is not None for path in corpus.image_path]
    paired = corpus[corpus.has_image].reset_index(drop=True)
    return {"paired_headlines": paired, "all_headlines": corpus}


# Frozen populations: (items, input_sha256) of the published matched run
# (reports/submission/matched.json). A run on anything else is refused, so a
# bundle with missing photos cannot silently shrink the paired population.
FROZEN = {
    "paired_headlines": (149, "6d2b0d7addc9d55208c2e5bcf8a59a12554f1b928d92b7695785003e372b9247"),
    "all_headlines": (198, "94765d304904656ecac37c975d20a9f661ce2e93fc92538a14e8c8b50e563d45"),
}


def check_frozen(frames, wanted):
    problems = []
    for population, (n, sha) in FROZEN.items():
        if wanted and population not in wanted:
            continue
        frame = frames[population]
        if len(frame) != n or digest(frame) != sha:
            problems.append(f"{population}: {len(frame)} items, sha {digest(frame)[:12]} "
                            f"(frozen: {n} items, sha {sha[:12]})")
    if problems:
        raise SystemExit("input does not match the frozen populations:\n  " + "\n  ".join(problems)
                         + "\nRe-run `make data ingest` until every image is present, "
                         "or pass --unpinned for an exploratory run that must not be reported.")


def digest(frame):
    columns = ["item_id", "text", "label", "source_url"]
    # Fixed "\n": pandas defaults to os.linesep, which would make the checksum
    # differ between Windows and the Linux GPU runs that produced FROZEN.
    text = frame[columns].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="experiments/matched")
    parser.add_argument("--only", nargs="*", help="restrict to these config names")
    parser.add_argument("--population", nargs="*", help="restrict to these populations")
    parser.add_argument("--seed", type=int, help="override the seed in every config")
    parser.add_argument("--unpinned", action="store_true",
                        help="skip the frozen-population check (exploratory only)")
    args = parser.parse_args()

    plan = {"paired_headlines": TEXT + MULTIMODAL + ABLATION + FOLLOWUP, "all_headlines": TEXT}
    frames = populations()
    if not args.unpinned:
        check_frozen(frames, args.population)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "sweep_summary.json"
    rows = json.loads(summary_path.read_text()) if summary_path.exists() else []
    done = {(r["population"], r["name"]) for r in rows if r.get("status") == "ok"}

    unknown = set(args.only or ()) - {n for names in plan.values() for n in names}
    if unknown:
        raise SystemExit(f"--only names not in the sweep: {sorted(unknown)}")
    for population, names in plan.items():
        if args.population and population not in args.population:
            continue
        frame = frames[population]
        for name in names:
            if args.only and name not in args.only:
                continue
            if (population, name) in done:
                print(f"skip {population}/{name}: already done", flush=True)
                continue
            kind = "multimodal" if name in MULTIMODAL + ABLATION + FOLLOWUP else "text"
            config = CONFIGS / kind / f"{name}.yaml"
            if args.seed is not None:
                settings = yaml.safe_load(config.read_text())
                settings["seed"] = args.seed
                config = out / "configs" / f"{name}.yaml"
                config.parent.mkdir(parents=True, exist_ok=True)
                config.write_text(yaml.safe_dump(settings, sort_keys=False))
            started = time.time()
            print(f"=== {population} / {name} (n={len(frame)})", flush=True)
            try:
                result = cross_validate(config, corpus=frame, out=out / population)
                row = {**result, "status": "ok"}
            except Exception as error:  # noqa: BLE001 - record and continue
                traceback.print_exc()
                row = {"name": name, "status": "failed", "error": repr(error)}
            row.update(
                population=population,
                input_items=len(frame),
                input_sha256=digest(frame),
                wall_seconds=round(time.time() - started, 1),
            )
            rows = [r for r in rows if (r["population"], r["name"]) != (population, name)]
            rows.append(row)
            summary_path.write_text(json.dumps(rows, indent=2) + "\n")
            status = row["status"]
            score = f" macro-F1={row['macro_f1']:.3f}" if status == "ok" else ""
            print(f"--- {population} / {name}: {status}{score}", flush=True)


if __name__ == "__main__":
    main()
