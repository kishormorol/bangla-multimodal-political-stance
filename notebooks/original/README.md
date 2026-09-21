# Original Colab notebooks

The two notebooks that produced every number in the first submission, preserved
unmodified for provenance. They were linked from a Google Doc ("Code") in the
Drive folder:

* `01_text_models_fixed_split.ipynb` — Colab `10HNecNOmO4ZtwvaLritVfzUTvMXWcBbf`
* `02_image_multimodal.ipynb` — Colab `1jtGbly-DybCK5Qa6Wogh8JnfsHKV8BtO`

Do not develop in these. They are the record of what was run; `src/bmpb/` is
where the corrected pipeline lives.

## What protocol produced which table

Tracing every `train_test_split` and `StratifiedKFold` call against the result
tables in `paper/latex/acl_latex.tex`:

| Paper table | Notebook | Protocol | Items scored |
| --- | --- | --- | --- |
| Transformer-Based Text Models | nb1 cell 13 | `train_test_split(test_size=0.3, random_state=42)` on the **augmented** 312-row set | 47 |
| Classical ML Models | nb1 | same fixed split | 47 |
| Image-Based Models | nb2 cell 29 | `StratifiedKFold(n_splits=5)` on the 149-item image set | 149 |
| Baseline Multimodal Models | nb2 cells 42–49, 64 | `train_test_split(test_size=0.2, random_state=42)`, one split | 30 |
| **Proposed Multimodal Models (P1–P6)** | nb2 cell 55 | `StratifiedKFold(n_splits=5)`, no augmentation | 149 |

Three consequences for the manuscript:

1. **"No fixed train-test split is used" is false for three of the five tables.**
   Only the image-based and proposed-model results are cross-validated.
2. **The text-model table is computed on a test set containing augmented
   variants of training articles** (20 of its 47 rows are augmented, and 25
   source articles are shared with train). Those numbers are optimistic.
   Scoring the shipped prediction CSVs reproduces that table to three decimals,
   which is how the protocol was identified.
3. **The headline claim compares two protocols.** P1's 0.6315 is a 5-fold mean
   over 149 items; ALIGN's 0.600 is one split of 30 items, whose 95% bootstrap
   interval is roughly ±0.17. The claim that P1 "surpasses ALIGN" is not
   supported by a comparison of that shape.

The fix is to run every model under one protocol — grouped stratified 5-fold CV
over the same population, with per-fold standard deviations and intervals.
