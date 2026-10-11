Updated October 10: verified Image_29 URL correction applied; shared source grouping repaired and headline baselines rerun. Historical neural scores still need corrected runs. See the LaTeX manuscript for current provenance counts (145 consistent matches, 53 unmatched).

# Evaluating Bangla Political Stance in News Headlines and Images

Working replacement draft, October 10, 2026. Dataset overlap is now verified in
`docs/dataset-provenance.md`. The baseline scores below use corrected source grouping. Annotation procedure, prior-submission status,
and the final experimental contribution remain unresolved. These notes must
be resolved before conversion into the submission PDF.

## Abstract

Political stance classification in Bangla news requires a clear distinction
between item-level framing and outlet-level bias. We audit a collection of
198 news items with government-critique, neutral, and government-leaning labels
and construct a source-grouped evaluation protocol. The historical text splits
contain augmented variants of the same source articles in training and test
data, while historical text and multimodal results use different evaluation
populations. We therefore report fresh out-of-fold headline predictions with
augmentation restricted to training data. Character n-gram TF-IDF with logistic
regression achieves pooled macro-F1 of 0.496 (95% bootstrap interval
0.415–0.569) on all 198 items. On the 149-item subset with readable images,
the corresponding text baseline achieves 0.543 (0.457–0.619). Separately,
text and image stance labels agree on 72 of 152 jointly labelled items.
These analyses establish a reproducible baseline and expose the requirements
for a valid assessment of multimodal gains on this collection.

## 1. Introduction

A news headline and its accompanying photograph can frame a political event
in different ways. Classifying that framing requires specifying the stance
target, the evidence available to annotators, and the unit of prediction.
Here the unit is an individual news item and the target is its annotated
stance toward the government. The labels do not establish factual correctness
or a persistent political orientation of the publisher.

Small datasets make evaluation design consequential. Augmented variants can
transmit source-specific information across a random row split. In addition,
a text model tested on all available items cannot establish a multimodal
comparison against a model evaluated only where images are available.
Our analysis addresses both issues by keeping source groups intact and
reporting a separate matched population for future multimodal comparisons.

The contribution supported by the current artifacts is an audit of data and
evaluation lineage, with reproducible headline baselines. A claim of a new
multimodal architecture or improved fusion requires additional experiments.

## 2. Related Work and Provenance

[Lia et al. (2025)](https://aclanthology.org/2025.banglalp-1.5/) introduce a
200-article Bangla political stance benchmark with government-leaning,
government-critique, and neutral annotations. The present collection includes
reused items from that resource: 145 of 198 processed items consistently match
by normalized headline and/or source URL after one verified URL repair, and 53
are unmatched to the audited upstream snapshot. Image metadata and separate
image stance fields are present locally; their collection and annotation
procedures must be documented before claiming them as new contributions.

[Bhattacharjee et al. (2022)](https://aclanthology.org/2022.findings-naacl.98/)
introduce BanglaBERT and Bangla language-understanding benchmarks. BanglaBERT
is an appropriate monolingual comparator for a completed neural evaluation.
Its historical score in this project cannot substitute for a matched rerun.

## 3. Collection and Annotation Audit

The canonical table contains 198 items from 39 normalized outlet keys. Label
counts are 103 government-critique, 53 neutral, and 42 government-leaning.
The table originates from a 200-row raw sheet, but the precise exclusion
rationale requires confirmation from collection records.

Two annotators each cover 197 items. Their Cohen's kappa on their 197-item
overlap is 0.731. A third annotator covers 32 items, with kappas of 0.454 and
0.196 against the other two annotators. Unequal overlap sizes preclude treating
the three pairs as equivalent evidence of annotation reliability.

Among 152 items with both text and image stance labels, 72 agree (47.4%).
This is descriptive disagreement between modality labels, rather than a
measurement of classifier improvement. Annotation instructions, adjudication,
annotator expertise, and the source of final labels must be documented.

The processed text column mixes 127 recovered articles and 71 headlines.
The primary experiment uses the headline field for every item, avoiding an
uncontrolled mixture of input lengths and source-dependent recovery rates.
Of 150 image paths in the processed table, 149 successfully decode in the
audited snapshot. We use those 149 items to define the paired population.

## 4. Evaluation

We assign original items to five source-grouped, approximately label-stratified
folds with seed 42. Every item is predicted once by a model that did not train
on its source group. Evaluation folds retain the observed label distribution.
The full-corpus and paired-population experiments use separate fold assignments;
models compared within either population must share its assignments.

The historical augmented table contains 114 variants of 198 original items.
Only variants whose parents belong to the current training split are attached.
The TF-IDF classifier uses character-within-word n-grams of length 2–5,
minimum document frequency 2, at most 50,000 features, and logistic regression
with C=1 and balanced class weights. Vectorization is fitted within training.
The majority baseline predicts the most frequent original training label and
does not use augmentation, so it represents the natural class-frequency floor.

For neural experiments, the implementation now reserves grouped inner
validation from each outer training partition for checkpoint selection.
Augmentation is applied after that inner split. The outer test fold must never
select checkpoints or hyperparameters. No new neural results are reported here.

We report pooled macro-F1 with a 2,000-round item-level percentile bootstrap
interval, accuracy, and the mean and sample standard deviation of fold
macro-F1. Pooled F1 is not the arithmetic mean of fold F1. Bootstrap intervals
condition on the saved predictions and do not measure retraining variability.

## 5. Results

| Population | Model | n | Accuracy | Pooled macro-F1 | 95% interval | Fold F1 mean ± SD |
| --- | --- | ---: | ---: | ---: | --- | --- |
| All headlines | Majority | 198 | 0.520 | 0.228 | 0.207–0.248 | 0.228 ± 0.002 |
| All headlines | TF-IDF + logistic regression | 198 | 0.566 | 0.496 | 0.415–0.569 | 0.487 ± 0.085 |
| Paired headlines | Majority | 149 | 0.483 | 0.217 | 0.191–0.239 | 0.217 ± 0.004 |
| Paired headlines | TF-IDF + logistic regression | 149 | 0.570 | 0.543 | 0.457–0.619 | 0.535 ± 0.103 |

The linear classifier exceeds the class-frequency floor in both populations.
The paired subset's higher score does not establish a benefit from images:
both runs use text alone and the populations differ. Fold variability remains
substantial for TF-IDF, with standard deviations of 0.119 and 0.094.

Historical text predictions score 47 rows and include augmented test items;
historical baseline vision-language predictions score 30 rows. The original
proposed-model notebook uses five folds over a 149-item image table. These
protocols do not support a common ranked comparison. The historical
cross-attention accuracy of 63.2% cannot establish superiority over a baseline
tested on a different split or population.

## 6. Conclusion

Source grouping and explicit population definitions are prerequisites for
evaluating this small Bangla political stance collection. The verified headline
baselines provide reference points for a future matched multimodal evaluation.
The current evidence establishes neither a cross-attention advantage nor
generalization to unseen outlets, events, or time periods.

## Limitations

The collection is small, imbalanced, and dominated by a few outlets. Random
source-grouped folds do not remove shared event, outlet, or image information
unless those relations are explicitly grouped. Readable-image availability
restricts the population and may be selective. Modality disagreement may
reflect framing differences, annotation ambiguity, or both. The bootstrap does
not represent uncertainty in labels or model retraining. We have not completed
new neural comparisons or multi-seed robustness checks. The origins of the
unmatched items and image annotations require documentation before submission.
Duplicate source links and a headline/URL mismatch require repair before the
provisional baseline results can be used in the submission.

## Ethical Considerations

Political stance annotations concern news framing and should not be used as
evidence of a publisher's trustworthiness or as automated grounds for sanction.
Political context and annotator interpretation can influence labels. The code
license does not grant redistribution rights to article text or photographs.
Any release must distinguish annotations and source URLs from copyrighted
content and document the applicable permissions. Consent, compensation, and
institutional review information must come from the original study records.
