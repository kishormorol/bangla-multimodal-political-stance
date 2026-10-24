# Bangla Multimodal Political Stance

Detecting political stance in Bangla news from the headline *and* the photo that
runs with it. The label is the stance a story's *framing* takes toward the
government — a property of the individual item, not an outlet-level bias score,
which 198 items across 39 outlets could not support. The repository was named
`bangla-multimodal-political-bias` until 2026-09-21; GitHub redirects the old
URL. This repository holds the dataset pipeline, the model comparison,
and the evaluation protocol behind the paper.

Three-way stance: `govt_critique` (0), `neutral` (1), `govt_leaning` (2), over
198 annotated items from 39 Bangladeshi and international outlets.

The collection includes reused items from
[BanglaBias (Lia et al., 2025)](https://aclanthology.org/2025.banglalp-1.5/).
See the [dataset provenance audit](docs/dataset-provenance.md) for row-level
overlap, unmatched items, and metadata exceptions. The current implementation
groups duplicate source URLs. The committed matched report and seed report
describe different item counts and require separate interpretation. See the
[evidence status](docs/evidence_status.md) for reproducible arithmetic checks
and the limits of the available artifacts.

The [recovered input record](docs/input_recovery.md)
and [proposed image comparison](docs/replication_protocol.md) distinguish
available replication inputs from experiments that have not run.

## Quick start

```bash
make install        # .venv + the package in editable mode
make data           # mirror the Drive folder into data/raw/
make ingest         # raw sheets -> data/processed/corpus.csv
make splits         # grouped, leakage-free train/val/test
bmpb audit          # data health report — read this before trusting a number

make train CONFIG=configs/text/tfidf_logreg.yaml
make leaderboard    # reports/tables/leaderboard.md
```

`make reproduce` runs the whole chain and trains every config.

## Layout

```
configs/            one YAML per model; data.yaml describes the corpus
  text/             banglabert, bangla_electra, mbert, xlmr, mt5, tfidf, majority
  multimodal/       clip, align, blip, flava, vilt, countvec_vit
src/bmpb/
  data/             download, ingest, preprocess, augment, splits, dataset
  models/           registry + text / multimodal / baseline families, heads
  audit.py          data health report
  train.py          one config -> one run directory
  evaluate.py       rescoring, published-prediction scoring, leaderboard
  metrics.py        macro-F1, bootstrap intervals, annotator agreement
  cli.py            the `bmpb` command
data/               not committed; see data/README.md
experiments/        one directory per run: config, metrics, predictions
reports/            figures and the results table
paper/              current manuscript mirror; see docs/paper.md
tests/
```

Adding a model means adding a YAML file, not editing training code. A `family`
in the config picks a builder from `src/bmpb/models/registry.py`; `bmpb models`
lists what is registered and which configs use it.

## What the pipeline produces

`experiments/<name>-<timestamp>/` holds `config.yaml`, `run.json` (git commit,
seed, split sizes, device, wall time), `predictions.csv` and `metrics.json`.
Every number in the results table traces back to one of these.

`make leaderboard` scores local runs and the prediction CSVs from the Drive
folder. The following table is a historical snapshot, with different
evaluation populations. Current manuscript tables instead use the separate
committed summaries in `reports/submission/`.

| Model | Source | n | Accuracy | Macro-F1 | 95% CI |
| --- | --- | ---: | ---: | ---: | --- |
| BanglaBERT | published | 47 | 0.723 | 0.719 | 0.578–0.836 |
| mBERT | published | 47 | 0.638 | 0.649 | 0.509–0.775 |
| tfidf_logreg | this repo | 29 | 0.586 | 0.554 | 0.332–0.731 |
| ViLT | published | 30 | 0.467 | 0.452 | 0.270–0.631 |
| ALIGN | published | 30 | 0.600 | 0.423 | 0.268–0.543 |
| CLIP | published | 30 | 0.433 | 0.386 | 0.209–0.573 |
| XLM-R | published | 47 | 0.362 | 0.215 | 0.136–0.300 |
| mT5 | published | 47 | 0.319 | 0.161 | 0.107–0.213 |
| majority | this repo | 29 | 0.276 | 0.144 | 0.081–0.206 |

Read that table with its caveats, not around them:

* Individual confidence interval overlap does not establish equivalence.
  Model comparisons require paired predictions and uncertainty for the
  difference on the same evaluation items.
* **`published` rows were produced on splits that leak** augmented variants
  across the train/test boundary, so they are optimistic. `data/README.md` has
  the counts.
* **`n` differs by row.** Text models are scored on 47 items, multimodal models
  on 30 — different populations, not a like-for-like comparison.
* **mT5 predicted `Neutral` for all 47 test rows.** That is a degenerate run,
  not a low score; scoring flags it automatically.

## Findings that shape the work

Three things surfaced while building the pipeline, each documented in full in
[`data/README.md`](data/README.md):

1. The matched experiments use headlines as model inputs. Reusing an article
   label does not establish independent headline stance ground truth.
2. **The published splits leak.** 25 source articles are shared between train
   and test. `make splits` fixes the ordering; the fix costs ~18 test items.
3. Stored article and image labels agree on 72 of 152 jointly labelled items.
   Disagreement does not establish that image features improve prediction.

The saved audit reports rounded pairwise agreement of κ = 0.73 for the two
annotation columns with 197 jointly labelled items. The third column has much
lower coverage. Original roles and adjudication still require study records.

## Evaluation protocol

* **Macro-F1** is the headline metric — the corpus is 52% one class, so accuracy
  rewards following the majority.
* **Every score carries a 2000-round bootstrap interval.** At this sample size a
  point estimate on its own is not interpretable.
* **Two floors in every table:** majority-class and TF-IDF + logistic regression.
  At 198 items a character n-gram baseline is a real competitor, and a
  transformer that does not clear it has not earned its place.
* **Evaluation sets keep the natural label distribution**; only training is
  balanced.

## Requirements

Python 3.10+. `make install` creates `.venv` and installs everything; Torch and
Transformers are only needed for the neural configs, and the data pipeline plus
the sklearn baselines run without them. On Apple silicon the trainer uses MPS
automatically — use the arm64 interpreter, not Anaconda's x86 build.

## Citing

See [`CITATION.cff`](CITATION.cff). The code is MIT; the dataset is not — the
headlines and photographs belong to their outlets. See `data/README.md`.
