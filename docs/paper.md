# Paper

The manuscript lives in Overleaf:
https://www.overleaf.com/project/69929902bff33e708abb74d9

`paper/` is a git-subtree mirror of that project, so the manuscript and the code
that produced its numbers share one history. Nothing hand-written belongs in
`paper/` — a `subtree push` would send it back to Overleaf — so this guide lives
here instead.

## Mirroring Overleaf into this directory

Overleaf projects expose a git remote (Overleaf menu → Git). One-time setup:

The remote is already configured and the first import is done. For a fresh
clone:

```bash
git remote add overleaf https://git.overleaf.com/69929902bff33e708abb74d9
git subtree add --prefix=paper overleaf main --squash
```

Overleaf's default branch here is `main`, not `master`.

Then, to pull the current manuscript and push edits back:

```bash
git subtree pull --prefix=paper overleaf main --squash   # Overleaf -> repo
git subtree push --prefix=paper overleaf main            # repo -> Overleaf
```

Pull before editing locally: Overleaf commits on every keystroke-ish save, so
the remote moves whenever the project is open in a browser.

LaTeX build artifacts (`.aux`, `.bbl`, `.log`, …) under `paper/` are gitignored.

## Keeping the numbers honest

Tables and figures should be generated, not retyped:

```bash
make leaderboard      # reports/tables/leaderboard.{md,csv}
bmpb audit            # the dataset statistics quoted in the data section
```

`reports/tables/leaderboard.csv` is the source for the results table, and every
row traces to a run directory under `experiments/` that records its git commit
and seed.

## Two claims in the manuscript that the artifacts contradict

These are not wording problems. Both are methodology claims that the files in
`data/raw/` do not support, and both are the kind a reviewer checks.

**"No fixed train-test split is used"** (§ Methodology). The manuscript says the
dataset is evaluated with stratified 5-fold cross-validation. But the Drive
folder ships a fixed split (`train_set.csv` 218 / `val_set.csv` 47 /
`test_set.csv` 47), every prediction file has exactly 47 rows, and the label
column of `BanglaBERT.csv` is identical, in order, to `test_set.csv`. Scoring
those files reproduces Table "Performance of Transformer-Based Text Models"
to three decimals:

| Model | Paper acc / F1 | Recomputed acc / F1 |
| --- | --- | --- |
| BanglaBERT | 0.723 / 0.719 | 0.723 / 0.719 |
| Bangla-ELECTRA | 0.468 / 0.462 | 0.468 / 0.462 |
| mBERT | 0.638 / 0.649 | 0.638 / 0.649 |
| XLM-RoBERTa | 0.362 / 0.215 | 0.362 / 0.215 |
| mT5 | 0.319 / 0.161 | 0.319 / 0.161 |

So the reported numbers come from one fixed split, not from cross-validation.

**"Augmentation is applied only on the training folds … ensuring no augmented
samples appear in validation data"** (§ Text Augmentation). In the shipped
splits, `val_set.csv` contains 14 augmented rows of 47 and `test_set.csv`
contains 20 of 47, and 25 source articles are shared between train and test
(17 between train and val). Augmented variants of training articles are being
scored as held-out test items.

Either the artifacts are stale relative to the code that produced the tables, or
the two claims need correcting. Deciding which needs the training code — the
Drive folder's "Code" entry is a Google Doc, which is also what breaks the
recursive download (see `data/README.md`). Until that is resolved, the
recomputed column above is the evidence that the tables came from the fixed,
leakage-affected split.

## Claims the manuscript needs to state

These come out of `bmpb audit` and are documented in `data/README.md`. Each one
changes how a result should be worded:

* The text field is **headlines** (median 8 words), so the task is headline
  stance classification.
* The originally published splits **leak augmented variants** across the
  train/test boundary (25 shared source articles between train and test), so
  scores computed on them are optimistic. Report either the regrouped splits or
  both, labelled.
* Text models are scored on 47 items and vision-language models on 30 —
  different populations. Report `n` per row.
* With that sample size the **bootstrap intervals overlap** across the
  vision-language models; they should be described as indistinguishable rather
  than ranked.
* **mT5 collapsed to a single class** on the test set. Report it as a degenerate
  run or rerun it as an encoder + classification head (`configs/text/mt5.yaml`).
* Inter-annotator agreement is **κ = 0.73** between the two annotators with full
  coverage; the third labelled 32 of 198 items and should be reported separately.
* Headline and image labels agree on only **47.4%** of items labelled both ways.
