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

<!-- ledger:brief -->

# Agent brief: From Words to Images: A Multimodal Benchmark for Bangla Political Stance Detection

This repository is one paper in a submission plan. Several Claude and Codex sessions, on different accounts, work on these papers in parallel. You start with no memory of earlier sessions: this file and `HANDOFF.md` are how the work carries over.

## The paper

What the paper is and how this repository is laid out are described above in this file and in `README.md`.

**This repository is public and the paper is under double-blind review.** Never add the venue, the review cycle, a deadline, an author name or affiliation, or a link to the submission plan to anything committed here — not the paper, not the README, not a commit message. The deadline is in the private plan; ask the author.

The lead author decides the science; agents support it.

## Where things stand

Read `HANDOFF.md` first: the newest entry is the current state. If it is missing or stale, ask the author which step is next before starting.

What each step means here, and what counts as done:

- `novelty` — searched the method's own vocabulary as well as the topic's, within the last week, and nothing does this already. **The author decides.**
- `design` — research question, data, baselines, metrics and analysis fixed in writing *before* results are seen. **The author decides.**
- `data` — data obtained, its licence and ethics status recorded, loading script committed, split frozen.
- `experiments` — every number the paper will report is produced by a committed script from committed configs, with seeds, `n` and intervals.
- `draft` — the full manuscript, every claim traceable to a result or a verified citation. **The author owns the argument.**
- `review` — read by someone other than the lead, and their comments resolved.
- `submit` — submitted; the manuscript id goes in the ledger.

## Rules that are not negotiable

A paper that breaks one of these is worse than no paper: it can be desk-rejected, retracted, or damage the author's record.

1. **No invented citations.** Every reference must be looked up — DOI, Crossref, arXiv, ACL Anthology, PubMed — and its BibTeX taken from that source. Open it and confirm it says what it is cited for. If you cannot verify one, mark it `% UNVERIFIED` in the source and list it in `HANDOFF.md`.
2. **No number without a source.** Every figure, table entry and percentage in the manuscript comes from a committed script or output file in this repository. Note which one beside it (a LaTeX comment is fine). Never type a result in by hand, round it favourably, or carry one over from a draft.
3. **No invented data.** Synthetic, simulated or placeholder data may exercise code. It must be labelled as such where it lives, and nothing derived from it may appear in the manuscript.
4. **Report what happened.** A negative or null result is reported, not tuned away. Do not change the metric, the split, the baselines or the hypothesis after seeing results without the author's explicit decision, recorded in `HANDOFF.md`.
5. **The author decides the science.** Novelty, design, framework, the claims the paper makes, and anything that changes its contribution are the author's calls. Propose; don't decide.
6. **Stay in this paper.** Don't pull code, text or data from the author's other paper repositories unless the author asks — several papers share themes, and duplicated text is self-plagiarism.

## How to work

- **Work on a branch** named `agent/<step>-<short-topic>`, and push it. The author reviews and merges; don't push to the default branch unless told to.
- **Small, honest commits.** The message says what changed and what was verified.
- **Never write a `Ledger:` line yourself.** Those lines record steps in the plan. Put the line you would suggest in `HANDOFF.md`; the author adds it when merging, once they agree the step is done.
- **When asked to review** another session's work, check claims against outputs and citations against sources, and report every discrepancy. Do not fix silently.

## Before you stop: update HANDOFF.md

Add an entry at the top of `HANDOFF.md` (create it if it is missing):

```markdown
## <YYYY-MM-DD> · <Claude or Codex> · <branch>
Done: what changed, with file paths.
Verified: what was checked, and how.
Not verified / open: anything uncertain, any `% UNVERIFIED` citation.
Needs the author: decisions only they can make.
Suggested: `Ledger: done <step>` (if a step is complete)
```

The tool and the session are recorded so the author can write the venue's AI-use disclosure accurately. AI tools cannot be authors; their use must be declared.

<!-- /ledger:brief -->
