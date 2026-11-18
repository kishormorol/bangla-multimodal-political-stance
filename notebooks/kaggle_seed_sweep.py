# Kaggle kernel: one seed of the matched sweep, with the image ablation, on the
# frozen 149 / 198 items.
#
# Input: a private Kaggle dataset holding bmpb-bundle.tar.gz (committed code at
# the commit in bundle-manifest.json, plus the corpus, augmentation table and
# image folders of the published matched run). Needs the GPU accelerator and
# internet access (HuggingFace checkpoints).
# The runner refuses to start unless both populations match the frozen
# checksums, so a bundle with missing photos fails here instead of shrinking n.
# Output: /kaggle/working/bmpb-seed-<SEED>.tar.gz, read locally by
# scripts/seed_analysis.py and scripts/ablation_analysis.py.

import glob
import json
import os
import subprocess
import sys
import tarfile

import torch

SEED = 42  # one kernel per seed; the launcher rewrites this line

assert torch.cuda.is_available(), "Enable the GPU accelerator for this kernel"
print(torch.cuda.get_device_name(0), "seed", SEED, flush=True)

WORK = "/kaggle/working/bmpb"
archives = glob.glob("/kaggle/input/**/bmpb-bundle.tar.gz", recursive=True)
if archives:
    os.makedirs(WORK, exist_ok=True)
    with tarfile.open(archives[0]) as archive:
        archive.extractall(WORK)
else:  # Kaggle unpacks some archives on upload
    manifests = glob.glob("/kaggle/input/**/bundle-manifest.json", recursive=True)
    assert manifests, "bundle not found under /kaggle/input"
    subprocess.run(["cp", "-r", os.path.dirname(manifests[0]), WORK], check=True)
os.chdir(WORK)
print(json.load(open("bundle-manifest.json")), flush=True)

pip = [sys.executable, "-m", "pip", "install", "-q"]
subprocess.run(pip + ["--no-deps", "-e", "."], check=True)
subprocess.run(pip + ["typer", "rich", "git+https://github.com/csebuetnlp/normalizer"], check=True)

out = f"experiments/seeds149/seed-{SEED}"
env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
result = subprocess.run(
    [sys.executable, "-u", "scripts/run_matched_sweep.py", "--seed", str(SEED), "--out", out],
    env=env,
)
print("sweep exit code", result.returncode, flush=True)

if os.path.exists(out):
    subprocess.run(["cp", "bundle-manifest.json", out], check=True)
    with tarfile.open(f"/kaggle/working/bmpb-seed-{SEED}.tar.gz", "w:gz") as archive:
        archive.add(out, arcname=f"seed-{SEED}")
    summary = json.load(open(f"{out}/sweep_summary.json"))
    print({r["status"] for r in summary}, len(summary), "runs", flush=True)
print("done", flush=True)
