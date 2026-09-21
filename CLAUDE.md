# Working in this repo

Bangla multimodal political stance detection: 198 annotated news items (headline +
photo), three-way stance, text and vision-language baselines under one protocol.

## Ground rules

**Label ids are frozen.** `0 = govt_critique, 1 = neutral, 2 = govt_leaning`.
Every published prediction CSV in `data/raw/` is keyed to them. Renumbering
silently invalidates results; `tests/test_ingest.py` pins the mapping.

**Augment after splitting, never before.** The originally published splits leak
augmented variants of the same article across the train/test boundary. If a
change makes `tests/test_splits.py` fail on the leakage assertion, the change is
wrong, not the test.

**Evaluation sets keep the natural label distribution.** Only training is
balanced. Do not "fix" the imbalance in val/test.

**Every reported score needs its `n` and its interval.** Test sets are 29–47
items; a 2-point macro-F1 difference is noise. `bmpb.metrics.score` attaches the
bootstrap interval and flags degenerate single-class runs — do not strip those
notes when summarizing.

**The text field is headlines** (median 8 words), not article bodies. Describe
results as headline stance classification.

## Adding a model

Add a YAML to `configs/text/` or `configs/multimodal/`. If the architecture fits
an existing `family` in `src/bmpb/models/registry.py`, no code changes are
needed. A new family is a function returning `(model, processor)` decorated with
`@register("name")`. `bmpb models` lists what exists.

## Commands

```bash
make install                                  # .venv + editable install
make data ingest splits                       # rebuild data/processed/
bmpb audit                                    # data health report
make train CONFIG=configs/text/banglabert.yaml
make leaderboard                              # reports/tables/leaderboard.md
make test lint
```

`make data` hits Google's rate limit on bulk folder downloads; it resumes, so
re-run it rather than working around it.

## Conventions

* Paths come from `src/bmpb/paths.py` — no hardcoded strings.
* Anything needed to rerun a number lives in a config, not in code.
* A run writes `config.yaml`, `run.json`, `predictions.csv` and `metrics.json`
  into `experiments/<name>-<timestamp>/`. Keep that contract: the results table
  is built from it.
* `data/` and `experiments/` are gitignored. Do not commit dataset files — the
  headlines and photos belong to the outlets they came from.
* Use the arm64 interpreter on Apple silicon, not Anaconda's x86 build.

## Training does not run on this Mac

The data pipeline, the sklearn baselines and the whole test suite run locally.
Fine-tuning the transformers does not:

* **MPS hangs.** The first forward+backward of BanglaBERT on `mps` never
  returns — not slow, hung, at 0% CPU.
* **CPU is pathologically slow.** One batch-8, 256-token step did not finish in
  100 seconds under torch 2.14 on Python 3.14, which is the likeliest cause.

So run the CV sweep on a GPU: `notebooks/run_cv_colab.ipynb` does the whole
thing on a Colab T4 and hands back `experiments/`, which `bmpb leaderboard`
reads. Don't rediscover this by waiting on a local run.

Two other environment traps that cost real time:

* **Redirecting python output buffers it.** `python ... > log` holds the log
  until the process exits, so a working run looks stalled. Use `python -u` or
  `PYTHONUNBUFFERED=1`. Piping through `tail` is worse — it emits nothing until
  stdin closes.
* **The sandbox blocks HuggingFace's CDN.** `from_pretrained` stalls with no
  error. Cache checkpoints in a network-enabled step, then train with
  `HF_HUB_OFFLINE=1`.
