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
