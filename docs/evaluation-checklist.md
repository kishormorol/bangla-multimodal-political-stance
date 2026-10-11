# Evaluation readiness checklist

Prepared October 10, 2026. **The existing manuscript is not submission-ready.**

## Verified evidence

The current corpus contains 198 items (103 critique, 53 neutral, 42 leaning)
from 39 outlet keys. It mixes 127 recovered articles and 71 headlines. There
are 150 image paths, but only 149 decoded successfully during this audit.
Use readable images, rather than path existence, to define the paired population.

Two annotators overlap on 197 items, with Cohen's kappa 0.731; the third annotator
has 32 labels. Text and image stance labels agree on 72 of 152 jointly labelled
items (47.4%). This is label disagreement; it does not establish that a
multimodal classifier improves on a text classifier.

Fresh baselines are in `reports/submission/baselines.{md,json}`. Rebuild with:

```bash
.venv/bin/python scripts/submission_baselines.py
```

The primary input is explicitly the headline, independent of article backfill.
TF-IDF uses character 2–5-grams, logistic regression, balanced class weights,
and the published augmented headlines attached only to training parents.
The majority floor uses original training frequencies without augmentation.
Five source-grouped folds, seed 42, preserve natural test class distributions.
Every original item receives exactly one out-of-fold prediction. Report pooled
macro-F1, its 2,000-resample interval, and fold mean/sample SD separately.
The interval measures item-resampling uncertainty, not variability across seeds
or unseen outlets. Existing historical tables are not directly comparable.

## Scientific decisions needed before finalizing

1. **Dataset provenance: overlap verified.** The [provenance audit](dataset-provenance.md)
   finds 145 consistently matched items after one verified URL repair, and 53 unmatched
   items against [Lia et al. (2025)](https://aclanthology.org/2025.banglalp-1.5/).
   Credit the reused resource. Document additional items, image pairing, and
   annotations. Duplicate source grouping is repaired and the headline baselines have been rerun. Neural and multimodal reruns remain outstanding.
2. **Contribution.** The existing abstract promises cross-attention superiority.
   The saved predictions do not support that comparison under one protocol.
   Either complete the matched neural experiments or recast the submission as
   an evaluation/resource audit with a focused, defensible contribution.
3. **Annotation target.** Establish what evidence annotators saw, how the final
   item label was selected, the government/time context, and whether separate
   image annotations were newly collected. Avoid equating stance with factual
   truth, misinformation, or outlet-wide bias.
4. **Earlier submission.** If previously submitted for review, retrieve the reviews,
   submission link, and required point-by-point revision response.

## Minimum experiment set for the model paper

Freeze the readable paired item list and headline text before training. Run
majority, TF-IDF, BanglaBERT, an image-only model, and the proposed fusion model
on exactly the same outer folds. Include concatenation as the fusion ablation;
any retained claim about cross-attention needs its implemented model, config,
saved predictions, and comparison against concatenation. The registry currently
does not establish reproduction of all P1–P6 architectures in the old paper.

The corrected CV implementation now selects neural checkpoints on grouped
inner validation from outer training data. The old implementation selected
checkpoints on the outer test fold; neural runs from it require rerunning.
Augmentation follows the inner training split, so validation remains untouched.
Every new run records fold item IDs, checkpoint selection, augmentation setting,
and git commit. Save the exact modified code with the runs until committed.

Compare paired predictions using item-matched uncertainty for the difference
in macro-F1, rather than inferring significance from separate confidence
intervals. Inspect per-class F1, label disagreement cases, and confusion
matrices. Multi-seed and outlet-held-out checks would strengthen conclusions;
do not claim those checks were completed if they were not.

Neural training requires a GPU: this checkout documents MPS hangs and unusably
slow CPU training. `notebooks/run_cv_colab.ipynb` is the starting point, but its
default sweep mixes article inputs and populations. Use the pinned headline
and paired population explicitly for the primary comparison.

## Manuscript changes

- Use item-level political **stance** consistently; describe headline inputs.
- Replace the abstract's unsupported superiority claim and 63.2% headline
  result unless verified by the matched experiment.
- Replace the universal CV claim: the original notebooks used several protocols.
- Correct the claim of balanced labels; the original corpus is imbalanced.
- Distinguish collected image records, existing paths, and readable images.
- Separate historical reproduction from new, leakage-controlled results.
- Remove unverified preprocessing, deduplication, training, focal-loss, and
  attribution claims unless supported by code and records.
- Document uncertainty and avoid ranking models without paired evidence.
- Complete dataset lineage, limitations, ethical considerations, and release
  terms; dataset content is not covered by the code's MIT license.

The supported replacement prose is in `docs/evaluation-manuscript.md`. It is a
working draft, not a final submission, and does not resolve provenance or
replace missing neural results.

## Manuscript checks

Verify the intended venue's page limit, anonymization and formatting requirements,
limitations, embedded fonts, supplementary materials and disclosure checklist.
Render and inspect the PDF before submission. Do not invent consent,
compensation, permissions, or reviewer eligibility.
