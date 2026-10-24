# Experiment identity and verification

The matched sweep records a resolved configuration before training. Remote
model references are resolved to immutable repository commits, and both the
model and its processor load that revision. `params.revision` applies to the
configured pretrained model. `params.vision_revision` applies to the ViT
source in the hybrid model. Supplying a full commit avoids another reference
lookup. Local model directories are hashed by file names and bytes.
With `HF_HUB_OFFLINE=1`, an unpinned source uses the cached config's snapshot
commit. An absent cache fails clearly before training. Lookup and preparation
failures are recorded per model so that one failed source does not abort the
remaining sweep.

The run identity also records input IDs, text and label hashes, image byte
hashes, frozen source groups, folds, augmentation availability and membership,
implementation hashes, and software versions. Missing augmentation files are
recorded as unavailable rather than as applied augmentation. Per fold records
identify the augmented items actually attached to training.

Each CV run has a unique directory containing `config.yaml`, `cv.json`,
`predictions.csv`, and a final `run.json` manifest. The manifest identifies the
inputs and contains checksums of the other artifacts. The sweep summary
points to that exact directory. Its summary is replaced atomically.

Resume requires matching identity and valid artifacts. An incompatible or
incomplete run is not accepted as completed, and a new run keeps the older
directory. A malformed metadata document fails visibly rather than being
silently treated as a successful experiment. These checks prevent accidental
reuse; they are not a signature or proof against deliberate alteration.
Resume restores summary fields from the verified `cv.json` before skipping
training. A regression check reproduced the earlier failure with an altered
summary score, then confirmed that resume repairs it without adding a run.

The analysis scripts load the exact referenced artifacts and check their
checksums and summary fields. Pairing requires the same examples, labels,
outer folds, and frozen groups. Across seeds, aggregation requires comparable
configurations, inputs, augmentation, implementation, and software. The seed
and its fold assignment may differ. Duplicate seed entries are rejected.
An aggregate retains the identities and artifact paths. One seed has no
sample standard deviation, which is represented as null in JSON and NA in
tables.

Older bundles without this metadata remain historical evidence. Use
`scripts/verify_reports.py` to inspect committed summary arithmetic. Do not
invent manifests for missing historical inputs or mix those records with new
verified runs.

## Freeze inputs before transfer

After preparing and reviewing the intended corpus and image availability,
create an input manifest with `python scripts/input_manifest.py --write`.
The command records current file hashes and missing file states without
overwriting an existing manifest. Keep the manifest with the private input
bundle. Check it after transfer with `python scripts/input_manifest.py`.
A changed corpus, source table, configuration, or referenced image stops
verification. A manifest preserves the chosen snapshot; it does not certify
its annotation meaning, permissions, or suitability as an evaluation set.

`notebooks/run_seeds_colab_pro.ipynb` records its pinned source checkout and
verifies the supplied local data bundle before training. It saves new runs
and their aggregate analysis under a separate Drive directory. It does not
silently substitute a public dataset snapshot.
Each model still resumes through the same identity checks. Review and freeze
the notebook's source reference and the input bundle before running a study.
The notebook's JSON and Python cells were parsed locally. Its Colab flow and
GPU training have not been executed. The older CV and seed notebooks are
marked as historical workflows.
The pinned commit was fetched from the configured public repository. A reused
checkout must be clean before installation. A local fixture confirmed that
the notebook rejects an edited checkout and accepts a clean checkout.

## Local checks

The checks below need the data and analysis dependencies already used by the
project. They use synthetic records for the experiment flow and do not load
original news content or pretrained model weights.
The report check requires files from
[the evidence PR](https://github.com/kishormorol/bangla-multimodal-political-stance/pull/2).
The experiment check is included in the model PR.

```sh
mkdir -p .tmp
rustc --edition=2021 checks/reports.rs -o .tmp/check_reports
.tmp/check_reports .venv/bin/python
rustc --edition=2021 checks/experiments.rs -o .tmp/check_experiments
.tmp/check_experiments .venv/bin/python
```

The experiment check covers resume, a changed seed, corrupt prediction
rejection, preservation of the old output, aggregation without paired models,
retained identity, valid JSON, duplicate seed rejection, and changed input
manifest rejection.

The BLIP wrapper was first checked with a tiny synthetic retrieval checkpoint
under Transformers 4.40.0 and 4.44.2 on CPU. Changing pixels changed its logits.
Frozen and unfrozen encoder gradients followed the configured setting. The
synthetic vision weights used an initialization scale of 0.02 to make the
check sensitive to pixels; the tiny model's default near zero scale would
make that probe uninformative. These checks provide no new benchmark score.

The final CPU check used the full public retrieval checkpoint at revision
`bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c`. Its source file hash, software
versions, numerical differences, and gradient checks are recorded in
`reports/verification/pretrained_blip.json`. The checkpoint loaded with no
missing, unexpected, or mismatched keys. The wrapper features exactly matched
the native retrieval model's first image conditioned text state. Changing
the image changed both the features and the classifier logits. The frozen
encoder received no gradients. After enabling encoder gradients, both text
and vision parameters received finite, nonzero gradients. The classifier
received gradients in both conditions.

To repeat the check without dataset files:

1. Download the named revision's `pytorch_model.bin`, configuration, and
   processor files. Verify the weight file against the recorded SHA256.
   Load its state with `torch.load(weights_only=True, map_location="cpu")`.
   Pass that state to `BlipForImageTextRetrieval.from_pretrained` with
   `output_loading_info=True`, then save it locally with safe serialization.
   Save the processor alongside it. The format conversion preserves the
   loaded model weights and avoids unrestricted pickle loading in the wrapper.
2. Use the recorded Torch and Transformers versions, two CPU threads,
   `torch.manual_seed(42)`, and `configs/multimodal/blip.yaml`. Replace only
   `pretrained` with the local converted directory. Build through the normal
   registry and select evaluation mode.
3. Use a solid red RGB image and an RGB noise image, both 384 by 384 pixels.
   Generate noise with `np.random.default_rng(42).integers(0, 256,
   (384, 384, 3), dtype=np.uint8)`. Give both the fixed synthetic sentence
   `সরকারের সিদ্ধান্ত নিয়ে আলোচনা` and label zero. Use the repository
   `Collator` with the configured maximum text length.
4. Compare `_blip_multimodal_features` with the native backbone call using
   `use_itm_head=True` and `question_embeds[:, 0]`. Require exact equality,
   finite logits with shape `[2, 3]`, and a change exceeding `1e-5` in both
   features and logits when the image changes. Backpropagate cross entropy
   once with the frozen encoder, then once on the first example after
   enabling encoder gradients. Require finite, nonzero classifier gradients,
   no frozen encoder gradients, and finite, nonzero text and vision gradients
   when enabled.

The measured check took 18.12 seconds after download on this workstation.
It used synthetic inputs, performed no optimizer steps, and produced no
benchmark predictions. It does not verify GPU execution or training quality.

The same checkpoint also passed a local CUDA check on a GTX 1650 Ti. The
record is `reports/verification/pretrained_blip_gpu.json`. A batch of two
synthetic images produced image dependent outputs and exactly matched the
native retrieval features. CPU and GPU logits differed by at most
`1.27e-7`, within absolute and relative tolerances of `1e-4`.
One AdamW step updated the classifier with finite gradients, while the
frozen backbone received no gradients. Peak reserved GPU memory was
1,138,753,536 bytes. The GPU probe took 1.216 seconds after loading and moving
the model. Neither the timing nor the memory figure describes a full study.

To repeat the CUDA check, use the same converted checkpoint and synthetic
inputs described above with Torch 2.5.1 and CUDA 12.4. Load the configuration
with `ExperimentConfig.load`, seed Torch with 42, and build the wrapper once.
Compare evaluation logits on CPU and CUDA before training. Check the native
features and changed image response on CUDA, then switch to training mode.
Use AdamW on trainable parameters with the configured learning rate and
weight decay. Backpropagate cross entropy once with both synthetic labels
set to zero, then require finite gradients and changed classifier weights.
Do not interpret the synthetic loss as a stance result. Reset peak memory
statistics after moving the model, and synchronize CUDA around timing.

GitHub CI passed on all four review branches after Actions was activated on
the fork. Its existing workflow ran lint, formatting, the Python tests, and
the import check. No local Python test runner was used. The Rust checks and
the CPU and GPU probes provide separate coverage.
