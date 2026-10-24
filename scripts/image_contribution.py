"""Run the fixed image contribution comparison on a verified local input bundle.

Run from the repository root with the CUDA environment:
    .venv-gpu/bin/python -m scripts.image_contribution
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from bmpb.config import DataConfig, ExperimentConfig, load_yaml
from bmpb.crossval import make_folds
from bmpb.data.dataset import load_image
from bmpb.data.groups import source_groups
from bmpb.metrics import score
from bmpb.models.baselines import _char_word_vectorizer
from bmpb.paths import CONFIGS, EXPERIMENTS, REPORTS, ROOT
from scripts.matched_analysis import RESAMPLES, SEED, macro_f1, paired_difference
from scripts.run_matched_sweep import digest


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def verified_inputs(config: dict) -> tuple[pd.DataFrame, dict[int, list[np.ndarray]], dict]:
    bundle = (ROOT / config["bundle"]).resolve()
    manifest_path = bundle / "data/processed/input_manifest.json"
    population_path = bundle / "data/processed/population_manifest.json"
    report_path = REPORTS / "submission/matched.json"
    for path, key in (
        (manifest_path, "input_manifest_sha256"),
        (population_path, "population_manifest_sha256"),
        (report_path, "matched_report_sha256"),
    ):
        if sha256(path) != config[key]:
            raise ValueError(f"{path.name} differs from the fixed configuration")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle):
            raise ValueError("input manifest path escapes the bundle")
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            raise ValueError(f"input snapshot changed: {relative}")
    population = json.loads(population_path.read_text(encoding="utf-8"))
    ids = population["item_ids"]
    if len(ids) != len(set(ids)) or len(ids) != config["items"]:
        raise ValueError("population membership is incomplete or duplicated")
    corpus = pd.read_csv(bundle / "data/processed/corpus.csv", keep_default_na=False)
    if len(corpus) != manifest["items"] or corpus.item_id.duplicated().any():
        raise ValueError("corpus does not match its manifest")
    frame = corpus[corpus.item_id.isin(ids)].reset_index(drop=True)
    frame["text"] = frame["title"]
    if set(frame.item_id) != set(ids) or not frame.title.str.strip().ne("").all():
        raise ValueError("selected headlines are missing")
    if frame.label.value_counts().to_dict() != config["label_counts"]:
        raise ValueError("selected labels differ from the fixed population")
    if digest(frame) != config["headline_sha256"]:
        raise ValueError("headline input checksum differs")
    group_key = DataConfig.load(bundle / "configs/data.yaml").split["group_key"]
    frame["group"] = source_groups(frame, group_key)
    if dict(zip(frame.item_id, frame.group, strict=True)) != population["groups"]:
        raise ValueError("source groups differ from the frozen membership")
    if frame.group.nunique() != config["groups"]:
        raise ValueError("source group count differs")
    paths = []
    for value in frame.image_path:
        path = (bundle / "data/raw" / value).resolve()
        if not path.is_relative_to(bundle / "data/raw") or not path.is_file():
            raise ValueError("selected image is absent or outside the bundle")
        if str(path.relative_to(bundle)) not in manifest["files"]:
            raise ValueError("selected image was not verified by the input manifest")
        paths.append(str(path))
    frame["image_path"] = paths
    matched = json.loads(report_path.read_text(encoding="utf-8"))["runs"]
    paired = [row for row in matched if row["population"] == "paired_headlines"]
    if not paired or any(row["fold_item_ids"] != population["fold_item_ids"] for row in paired):
        raise ValueError("saved matched runs disagree on evaluation membership")
    folds = {seed: make_folds(frame, config["folds"], group_key, seed) for seed in config["seeds"]}
    observed = [[str(frame.iloc[i].item_id) for i in fold] for fold in folds[42]]
    if observed != population["fold_item_ids"]:
        raise ValueError("seed 42 folds differ from the saved evaluation")
    return frame, folds, {"files": manifest["files"], "groups": population["groups"]}


def image_features(frame: pd.DataFrame, config: dict, out: Path) -> np.ndarray:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModel

    vision = config["vision"]
    if vision["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this fixed run requires the local CUDA device")
    torch.set_num_threads(config["threads"])
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(0.60)
    directory = Path(
        snapshot_download(
            vision["repository"],
            revision=vision["revision"],
            allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
        )
    )
    names = ["config.json", "preprocessor_config.json", "model.safetensors"]
    model_hashes = {name: sha256(directory / name) for name in names}
    processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True)
    model, loading = AutoModel.from_pretrained(
        directory, use_safetensors=True, local_files_only=True, output_loading_info=True
    )
    if any(loading.values()):
        raise ValueError(f"pretrained image model did not load completely: {loading}")
    model.eval()
    model.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    model.to(vision["device"])
    torch.cuda.synchronize()
    started = time.perf_counter()
    features = []
    with torch.no_grad():
        for start in range(0, len(frame), vision["batch_size"]):
            images = [
                load_image(path, size=vision["image_size"])
                for path in frame.image_path.iloc[start : start + vision["batch_size"]]
            ]
            if any(image is None for image in images):
                raise ValueError("a selected image failed decoding; cancel the comparison")
            inputs = processor(images=images, return_tensors="pt").to(vision["device"])
            features.append(model(**inputs).last_hidden_state[:, 0].cpu().numpy())
            print(
                f"image features: {min(start + len(images), len(frame))}/{len(frame)}", flush=True
            )
    torch.cuda.synchronize()
    features = np.vstack(features)
    if not np.isfinite(features).all() or np.any(np.linalg.norm(features, axis=1) == 0):
        raise ValueError("image vectors are invalid")
    features = normalize(features, norm="l2")
    with (out / "image_features.npy").open("xb") as handle:
        np.save(handle, features, allow_pickle=False)
    write_json(
        out / "vision.json",
        {
            "source": vision,
            "model_files_sha256": model_hashes,
            "loading_info": loading,
            "processor": processor.to_dict(),
            "feature_sha256": sha256(out / "image_features.npy"),
            "shape": list(features.shape),
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    )
    return features


def fit_conditions(frame, features, folds, config, seed, out):
    if frame.item_id.duplicated().any() or not frame.label.isin([0, 1, 2]).all():
        raise ValueError("predictions require unique items and valid labels")
    if features.shape[0] != len(frame) or not np.isfinite(features).all():
        raise ValueError("image features do not match the evaluation population")
    membership = np.concatenate(folds)
    if sorted(membership.tolist()) != list(range(len(frame))):
        raise ValueError("folds must score every item exactly once")
    group_sets = [set(frame.iloc[test].group) for test in folds]
    if sum(map(len, group_sets)) != len(set.union(*group_sets)):
        raise ValueError("a source group crosses outer folds")
    text_config = ExperimentConfig.load(ROOT / config["text_config"])
    image_block = sparse.csr_matrix(features, dtype=np.float64)
    predicted = {name: np.full(len(frame), -1, dtype=int) for name in "ABC"}
    probability = {name: np.zeros((len(frame), 3)) for name in "ABC"}
    assignment = np.full(len(frame), -1, dtype=int)
    for fold, test in enumerate(folds):
        train = np.setdiff1d(np.arange(len(frame)), test)
        vectorizer = _char_word_vectorizer("tfidf", text_config.params)
        text_train = vectorizer.fit_transform(frame.text.iloc[train])
        text_test = vectorizer.transform(frame.text.iloc[test])
        text_hash = hashlib.sha256(
            json.dumps(
                {term: int(index) for term, index in vectorizer.vocabulary_.items()},
                sort_keys=True,
            ).encode()
            + vectorizer.idf_.tobytes()
        ).hexdigest()
        matrices = {
            "A": (text_train, text_test),
            "B": (
                sparse.hstack([text_train, image_block[train]], format="csr"),
                sparse.hstack([text_test, image_block[test]], format="csr"),
            ),
            "C": (image_block[train], image_block[test]),
        }
        assignment[test] = fold
        for name, (x_train, x_test) in matrices.items():
            width = text_train.shape[1]
            if name in "AB" and (
                (x_train[:, :width] != text_train).nnz or (x_test[:, :width] != text_test).nnz
            ):
                raise ValueError("the text block changed between conditions")
            offset = width if name == "B" else 0
            if name in "BC" and (
                (x_train[:, offset:] != image_block[train]).nnz
                or (x_test[:, offset:] != image_block[test]).nnz
            ):
                raise ValueError("the image block changed between conditions")
            started = time.perf_counter()
            classifier = LogisticRegression(**config["classifier"])
            with threadpool_limits(config["threads"]), warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                classifier.fit(x_train, frame.label.iloc[train])
            if not np.array_equal(classifier.classes_, [0, 1, 2]):
                raise ValueError("training fold lost a label")
            predicted[name][test] = classifier.predict(x_test)
            probability[name][test] = classifier.predict_proba(x_test)
            if not np.isfinite(classifier.coef_).all():
                raise ValueError("classifier coefficients are invalid")
            record = {
                "seed": seed,
                "fold": fold,
                "condition": name,
                "train_ids": frame.item_id.iloc[train].tolist(),
                "test_ids": frame.item_id.iloc[test].tolist(),
                "shape": list(x_train.shape),
                "text_vocabulary_idf_sha256": text_hash if name in "AB" else None,
                "text_block_unchanged": True if name in "AB" else None,
                "image_block_shared": True if name in "BC" else None,
                "iterations": classifier.n_iter_.tolist(),
                "seconds": time.perf_counter() - started,
            }
            with (out / "fits.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            print(f"seed {seed}, fold {fold}, condition {name}: fitted", flush=True)
    frames, scores = {}, {}
    for name in "ABC":
        if (
            not np.isin(predicted[name], [0, 1, 2]).all()
            or not np.isfinite(probability[name]).all()
        ):
            raise ValueError("predictions are incomplete or invalid")
        result = frame[["item_id", "label", "group"]].copy()
        result["fold"] = assignment
        result["predicted"] = predicted[name]
        for label in range(3):
            result[f"p{label}"] = probability[name][:, label]
        with (out / f"seed_{seed}_{name}.csv").open("x", encoding="utf-8", newline="") as handle:
            result.to_csv(handle, index=False)
        frames[name] = result
        scores[name] = score(frame.label, predicted[name], bootstrap=False).to_dict()
    return frames, scores


def group_score_samples(predictions):
    frame = predictions["A"]
    members = [group.index.to_numpy() for _, group in frame.groupby("group")]
    rng = np.random.default_rng(SEED)
    labels = frame.label.to_numpy()
    values = {name: predictions[name].predicted.to_numpy() for name in "ABC"}
    samples = np.empty((RESAMPLES, 3))
    for iteration in range(RESAMPLES):
        index = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        for column, name in enumerate("ABC"):
            samples[iteration, column] = macro_f1(labels[index], values[name][index])
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=CONFIGS / "ablations/image_contribution.yaml"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    config = load_yaml(args.config)
    if sklearn.__version__ != config["sklearn_version"]:
        raise ValueError("scikit learn version differs from the fixed protocol")
    if config["bootstrap"] != {"resamples": RESAMPLES, "seed": SEED} or config["augmentation"]:
        raise ValueError("bootstrap or augmentation differs from the fixed protocol")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("commit the reviewed implementation and configuration before running")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = (args.out or EXPERIMENTS / f"{config['name']}_{timestamp}").resolve()
    if not out.is_relative_to(EXPERIMENTS.resolve()):
        raise ValueError("new outputs must stay in experiments")
    out.mkdir(parents=True, exist_ok=False)
    try:
        frame, folds, inputs = verified_inputs(config)
        implementation = sorted((ROOT / "src/bmpb").rglob("*.py")) + [
            Path(__file__).resolve(),
            ROOT / "scripts/matched_analysis.py",
            ROOT / "scripts/run_matched_sweep.py",
            args.config.resolve(),
            ROOT / config["text_config"],
        ]
        write_json(
            out / "plan.json",
            {
                "config": config,
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "code_sha256": {
                    str(path.relative_to(ROOT)): sha256(path) for path in implementation
                },
                "software": {
                    name: importlib.metadata.version(name)
                    for name in [
                        "numpy",
                        "pandas",
                        "scikit-learn",
                        "scipy",
                        "torch",
                        "transformers",
                        "Pillow",
                        "huggingface-hub",
                    ]
                },
                "python": platform.python_version(),
                "input_files": inputs["files"],
                "item_ids": frame.item_id.tolist(),
                "groups": inputs["groups"],
                "fold_item_ids": {
                    seed: [[str(frame.iloc[i].item_id) for i in fold] for fold in parts]
                    for seed, parts in folds.items()
                },
                "classifier": LogisticRegression(**config["classifier"]).get_params(),
            },
        )
        features = image_features(frame, config, out)
        fitted = {
            seed: fit_conditions(frame, features, folds[seed], config, seed, out)
            for seed in config["seeds"]
        }
        results = []
        for seed, (predictions, scores) in fitted.items():
            samples = group_score_samples(predictions)
            for column, name in enumerate("ABC"):
                scores[name]["macro_f1_ci95"] = np.percentile(
                    samples[:, column], [2.5, 97.5]
                ).tolist()
                scores[name]["interval_unit"] = "source group, conditional on saved predictions"
            paired_options = {}
            if "groups" in inspect.signature(paired_difference).parameters:
                paired_options["groups"] = frame.set_index("item_id")["group"]
            difference = paired_difference(predictions["B"], predictions["A"], **paired_options)
            independent_interval = np.percentile(samples[:, 1] - samples[:, 0], [2.5, 97.5])
            if not np.array_equal(independent_interval, difference["ci95"]):
                raise ValueError("independent group bootstrap disagrees with the paired helper")
            lo, hi = difference["ci95"]
            results.append(
                {
                    "seed": seed,
                    "scores": scores,
                    "B_minus_A": difference,
                    "direction": "positive" if lo > 0 else "negative" if hi < 0 else "unresolved",
                    "C_minus_A": {
                        "delta_macro_f1": scores["C"]["macro_f1"] - scores["A"]["macro_f1"],
                        "ci95": np.percentile(samples[:, 2] - samples[:, 0], [2.5, 97.5]).tolist(),
                    },
                }
            )
        write_json(out / "results.json", {"status": "complete", "seeds": results})
        artifacts = {path.name: sha256(path) for path in sorted(out.iterdir()) if path.is_file()}
        write_json(out / "run.json", {"status": "complete", "artifacts": artifacts})
        print(f"Results saved to {out}", flush=True)
    except Exception as error:
        write_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
