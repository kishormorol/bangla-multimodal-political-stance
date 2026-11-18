## 2026-11-18 · Claude (Opus 5.5 + Fable 5.2 reviewer, Claude Code) · agent/review-presubmission

Done: pre-submission review. Fixed the seed-stability table (header had 5 cols
for 6-col rows; 13 LaTeX errors) and corrected abstract/discussion seed claims
to match seeds.md (commit 1c5ac96). Two independent agent reviews (number audit;
Fable 5.2 reviewer) — not a human review, so the `review` step is not done.

Done (later, same session): scripts/run_matched_sweep.py now refuses to run
unless both populations match the published checksums (149 items 6d2b0d7a..,
198 items 94765d30..); --unpinned bypasses it for exploratory runs only. Added
notebooks/run_seeds149_colab.ipynb (clones this branch, saves to
MyDrive/bmpb/seeds149, cell 4b gates training on the check).
Verified: the guard rejects the current local data (86 / f0a3a9.., 198 /
96446c..); 34 tests pass; ruff clean. The notebook itself has not been run.
Found: the local mirror has only 90 raw images and no processed images, and
gdown is rate-limited (no progress). Even the 198-headline corpus hashes
differently from the published run, so the corpus text has changed since then:
the author must find which column changed before the reseed can pass the gate.

Author decision (2026-11-18, in this session, after seeing results): add the
image ablation, TF-IDF + image features, and an image-only baseline.
Done: config key image_ablation (none/blank/shuffled; src/bmpb/config.py,
crossval.ablate_images, dataset.BLANK_IMAGE); countvec_vit family gains
params.text_features (count/tfidf/none). New configs in configs/multimodal/:
<vl>_blank and <vl>_shuffled for all six VL models, tfidf_vit (exact
tfidf_logreg features + frozen ViT), tfidf_vit_shuffled, vit_only. They run in
the paired sweep. scripts/ablation_analysis.py fixes the comparisons before any
run (real vs blank, real vs shuffled, tfidf_vit vs tfidf_logreg and vs
tfidf_vit_shuffled, vit_only vs majority), asserts n=149, and writes
reports/submission/ablation.{json,md} and ablation_tables.tex.
seed_analysis.py now prints n per model and leaves ablation rows out of the
paper tables (its LaTeX output is byte-identical on the old seeds).
notebooks/run_seeds149_colab.ipynb runs all of it: 35 runs per seed.
Verified: 59 tests pass (new tests/test_ablation.py: derangement, blank image,
configs differ from their parents only in the ablation, tfidf_vit features
equal tfidf_logreg's); ruff clean on changed files; analysis scripts smoke-run
on the old seeds. No ablation run has happened; the ViT path is untested
locally (HF CDN blocked here).

Verified: PDF compiles with 0 errors, 0 overfull boxes. Checked by hand that
every paired_headlines seed run has items_scored 86 and seeds.json paired
differences have n 86 (grep over experiments/seeds/ and reports/submission/).

Not verified / open — BLOCKING:
- The five-seed results (abstract 0.562, 0.402, -0.192..-0.277, 3-5/5 counts;
  Tables seed_stability/seed_diff) are on n=86, not 149. The Colab bundle had
  only 86 decodable images; run_matched_sweep.py defines the paired population
  as "images load_image can decode". The paper states 149 throughout.
- matched.json "seed 42" (Kaggle) and seeds.json seed 42 (Colab) differ in
  population and folds, even on 198 items (input sha 94765d.. vs 200ff2..).
- Unsourced numbers: median 8 words, 127 bodies/71 headlines, 20 augmented test
  rows, 63.2%, 114 variants, appendix tables (no % source).
- matched.json runs have git_commit null.

Needs the author:
1. (Correction, required) Rerun seeds 42-46 on the pinned 149 item IDs with an
   asserted n and checksum; regenerate Tables 1-2 from that same sweep.
2. (Design change, author decides) image ablation within VL models
   (real/blank/shuffled image); TF-IDF+image-features vs TF-IDF; image-only
   baseline. Otherwise narrow the "adding the photograph" claim.
3. (Optional) fixed-epoch BanglaBERT/mBERT check; matched zero-shot VLM row;
   expand related work.
4. A human co-author or colleague review.

Suggested: none yet — `experiments` is not done until item 1 is resolved.

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
