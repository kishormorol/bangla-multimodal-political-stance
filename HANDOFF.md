## 2026 October 23, Codex, completed review and image contribution

Done: added `scripts/image_contribution.py`, its fixed configuration, one
synthetic Rust check, and `docs/image_contribution.md`. Ran the comparison
locally on the frozen 149 item input bundle. The aggregate evidence is in
`reports/verification/image_contribution.json`. Historical results and labels
remain unchanged.

Verified: both the original run and a repeat from upstream main finished
75 fits. Image feature bytes, all 15 prediction CSVs and every result matched
exactly. Both sets of 20 artifact checksums matched. All 156 frozen input
files were verified. A separate text run through the existing CV runner
matched all 149 predictions. Independent calculations reproduced all five
fold assignments' metrics and source group intervals. Adding images reduced
primary macro F1 from 0.5642 to 0.4588, with a paired interval for the decrease
of 0.0230 to 0.1812.

Review: repaired sweep summary recovery, documentation links, and the
manuscript's separate 86 item seed description. The combined checkout passed
all four Rust behavior checks, Ruff, Black and import checks. The manuscript
compiled to nine pages. The changed page was inspected. Spacing and local
font warnings remain, with no overflow or undefined reference reported.
The CPU CI workflow now includes the existing synthetic behavior checks.

Not verified: historical image byte identity, label visibility, semantic image
pairings, independent events or outlets, and performance on future news.
The intervals condition on fitted predictions and omit model refitting.
Upstream CI still requires maintainer approval. Colab execution and the
complete export from the unavailable unannotated source remain unchecked.

Needs the author: decide whether to adopt the result in the paper. The original
data, image features, and item predictions remain local. The focused study is
[upstream PR 6](https://github.com/kishormorol/bangla-multimodal-political-stance/pull/6).
It targets main without including the companion PR's model changes. Proposed
manuscript wording remains in `docs/image_contribution.md` for author review.

## 2026-10-12 · Claude (Opus 5.5, Claude Code) · agent/experiments-grouped-evaluation (seeds)

Done: the author asked for a multi-seed rerun on Colab. Added `--seed` to
`scripts/run_matched_sweep.py`, `notebooks/run_seeds_colab.ipynb` (seeds 43–46,
resumable to Google Drive) and `scripts/seed_analysis.py`. The aggregation was
fixed before any extra seed ran: per-model mean/SD/range of macro-F1, and
per-pair paired differences with counts of seeds whose interval is above or
below zero. The bundle is at `experiments/colab/bmpb-bundle.tar.gz` (local,
gitignored, built at bfe2dd3, article bodies stripped).

Verified: the seed override and the analysis were smoke-tested locally on the
sklearn baselines (seed 43 TF-IDF paired macro-F1 0.506 vs 0.543 at seed 42).

Not verified / open: no extra seed has run on a GPU yet. CountVec+ViT fails
locally only because torchvision is missing here; Colab has it.

Needs the author: run the notebook and return `bmpb-seeds.tar.gz`; then unpack
it into `experiments/seeds/` and run
`cd scripts && python seed_analysis.py ../experiments/matched ../experiments/seeds/seed-*`.

## 2026-10-12 · Claude (Opus 5.5, Claude Code) · agent/experiments-grouped-evaluation

Done: ran the corrected matched sweep on a Kaggle T4 (private kernel
`bmpb-matched-sweep`) at commit 1feab08: 13 configs on the 149 readable-image
headline items, 7 text configs on all 198 headlines, all from committed configs,
seed 42, no tuning. Added `scripts/run_matched_sweep.py`,
`scripts/matched_analysis.py` (paired comparisons fixed before results: each VL
model vs TF-IDF and vs BanglaBERT, source-group bootstrap, 2,000 resamples),
`scripts/summarize_training_log.py`, `notebooks/kaggle_matched_sweep.py`.
Outputs: `reports/submission/matched.{json,md}`, `matched_tables.tex` (copied
to `paper/latex/` and `\input`), `early_stopping.json`. Raw runs and the Kaggle
log are in `experiments/matched/` (gitignored). Rewrote the abstract, the
contributions, the protocol, Results, Discussion, Conclusion and Limitations
around the matched results; moved historical tables to Appendix A. Added
verified citations for BERT, XLM-R, mT5, ELECTRA, CLIP, ALIGN, ViLT, BLIP,
FLAVA, ViT; fixed the missing 7th author (Xuelong Li) on `li2025survey`.

Result (null for multimodality): TF-IDF 0.543 (0.457–0.619, n=149) is best;
best VL is CountVec+ViT 0.425; every VL model is below TF-IDF in paired tests
(Δ −0.117 to −0.218, CIs exclude 0) and indistinguishable from BanglaBERT
(Δ −0.042 to +0.058, CIs include 0). Corrected BanglaBERT is 0.366 on 198 items
vs 0.533 historically.

Verified: all 20 runs status ok, every item predicted once (runner asserts).
Kaggle bundle reproduced committed sklearn baselines exactly before upload.
34 tests pass. PDF compiles with no errors, overfull boxes or undefined
references; main text ends on page 6; all 8 pages inspected. The four
pre-existing citations were checked against Crossref; new ones were taken from
the ACL Anthology `.bib` export and arxiv.org/bibtex, with titles checked.

Not verified / open: single seed and single fold assignment; inner-validation
sets are 25–33 items and 65/75 neural fold runs early-stopped (29 at epoch 1–2),
so the encoders may be under-trained. The paired comparisons confound image use
with text representation. The arXiv-only citations may have published venue
versions (CLIP/ALIGN/ViLT ICML 2021, BLIP ICML 2022, FLAVA CVPR 2022, ViT and
ELECTRA ICLR); DBLP was unreachable, so those were not swapped in. The
upload bundle had article bodies stripped. Data was uploaded as a private Kaggle
dataset `kishormorol/bmpb-matched-bundle`, deleted after download (see below);
the private kernel's output still contains the working copy of the bundle.
`reports/submission/audit.json` was untracked before this session; left as is.

Needs the author: whether the paper's claims and title should now centre the
null multimodal result; whether to add multi-seed or larger-validation reruns
(a design change, your call); annotation procedures and dataset contribution as
before; whether to delete the Kaggle kernel too.

Suggested: `Ledger: done experiments` (if you accept single-seed matched runs as
the experiment of record).

## 2026-10-10 · Codex · agent/experiments-grouped-evaluation

Done: repaired shared source grouping and independent neural validation in
`src/bmpb/crossval.py` and `src/bmpb/data/`; recorded the verified Image_29 URL
correction without changing labels. Added provenance audit and headline baseline
scripts, aggregate outputs under `reports/submission/`, and a frozen-population
BanglaBERT notebook. Revised the manuscript to distinguish corrected headline
baselines from historical exploratory neural results.

Verified: 34 tests pass; source-URL regression tests failed before the repair.
Headline TF-IDF pooled macro-F1 is 0.496 on 198 items and 0.543 on 149 readable-image
items; intervals and fold variation are in the saved outputs. The six-page PDF
compiled without overflow or undefined citations and every page was inspected.

Not verified / open: corrected neural and multimodal training has not run.
Image_154/Image_155 remain conservatively grouped pending source-link verification.
Collection and annotation records for additional items remain incomplete.
Dataset and portable input bundles are local and are not committed.

Needs the author: run the frozen BanglaBERT experiment on a CUDA GPU; confirm
annotation procedures and dataset contribution before finalizing claims.
