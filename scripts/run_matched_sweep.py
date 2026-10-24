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
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from bmpb.config import ExperimentConfig, pin_model_revisions
from bmpb.crossval import cross_validate, execution_identity
from bmpb.data.dataset import load_image
from bmpb.paths import CONFIGS, CORPUS

TEXT = ["majority", "tfidf_logreg", "banglabert", "bangla_electra", "mbert", "xlmr", "mt5"]
MULTIMODAL = ["countvec_vit", "clip", "align", "blip", "vilt", "flava"]


def populations():
    corpus = pd.read_csv(CORPUS)
    corpus["text"] = corpus["title"]
    corpus["has_image"] = [load_image(path) is not None for path in corpus.image_path]
    paired = corpus[corpus.has_image].reset_index(drop=True)
    return {"paired_headlines": paired, "all_headlines": corpus}


def digest(frame):
    columns = ["item_id", "text", "label", "source_url"]
    return hashlib.sha256(frame[columns].to_csv(index=False).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="experiments/matched")
    parser.add_argument("--only", nargs="*", help="restrict to these config names")
    parser.add_argument("--population", nargs="*", help="restrict to these populations")
    parser.add_argument("--seed", type=int, help="override the seed in every config")
    args = parser.parse_args()

    plan = {"paired_headlines": TEXT + MULTIMODAL, "all_headlines": TEXT}
    frames = populations()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "sweep_summary.json"
    rows = json.loads(summary_path.read_text()) if summary_path.exists() else []

    for population, names in plan.items():
        if args.population and population not in args.population:
            continue
        frame = frames[population]
        for name in names:
            if args.only and name not in args.only:
                continue
            old = next(
                (r for r in rows if r.get("population") == population and r.get("name") == name),
                None,
            )
            started = time.time()
            print(f"=== {population} / {name} (n={len(frame)})", flush=True)
            try:
                kind = "multimodal" if name in MULTIMODAL else "text"
                config = CONFIGS / kind / f"{name}.yaml"
                settings = yaml.safe_load(config.read_text())
                if args.seed is not None:
                    settings["seed"] = args.seed
                config = out / "configs" / f"{name}.yaml"
                config.parent.mkdir(parents=True, exist_ok=True)
                config.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
                resolved = pin_model_revisions(ExperimentConfig.load(config))
                config.write_text(
                    yaml.safe_dump(asdict(resolved), sort_keys=False), encoding="utf-8"
                )
                identity = execution_identity(resolved, frame)
                if old and old.get("status") == "ok" and old.get("identity") == identity:
                    run_dir = (out / old.get("run_dir", "")).resolve()
                    manifest = run_dir / "run.json"
                    complete = all(
                        (run_dir / name).is_file()
                        for name in ("config.yaml", "predictions.csv", "cv.json", "run.json")
                    )
                    if run_dir.is_relative_to(out.resolve()) and complete:
                        saved = json.loads(manifest.read_text(encoding="utf-8"))
                        saved_cv = json.loads((run_dir / "cv.json").read_text(encoding="utf-8"))
                        valid_artifacts = all(
                            saved.get("artifacts", {}).get(path.name)
                            == hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (
                                run_dir / "config.yaml",
                                run_dir / "predictions.csv",
                                run_dir / "cv.json",
                            )
                        )
                        if (
                            saved.get("identity") == identity
                            and saved_cv.get("identity") == identity
                            and valid_artifacts
                        ):
                            old.update(
                                {
                                    key: value
                                    for key, value in saved_cv.items()
                                    if key not in {"population", "run_dir"}
                                }
                            )
                            old.update(input_items=len(frame), input_sha256=digest(frame))
                            write_json(summary_path, rows)
                            print(
                                f"skip {population}/{name}: matching run and artifacts exist",
                                flush=True,
                            )
                            continue
                result = cross_validate(config, corpus=frame, out=out / population)
                run_dir = Path(result["run_dir"]).resolve().relative_to(out.resolve())
                row = {**result, "status": "ok", "identity": identity, "run_dir": str(run_dir)}
            except Exception as error:  # noqa: BLE001 - record and continue
                traceback.print_exc()
                row = {"name": name, "status": "failed", "error": repr(error)}
                if old and old.get("run_dir"):
                    row["previous_run_dir"] = old["run_dir"]
            row.update(
                population=population,
                input_items=len(frame),
                input_sha256=digest(frame),
                wall_seconds=round(time.time() - started, 1),
            )
            rows = [r for r in rows if (r.get("population"), r.get("name")) != (population, name)]
            rows.append(row)
            write_json(summary_path, rows)
            status = row["status"]
            score = f" macro-F1={row['macro_f1']:.3f}" if status == "ok" else ""
            print(f"--- {population} / {name}: {status}{score}", flush=True)


if __name__ == "__main__":
    main()
