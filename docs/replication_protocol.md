# Recorded protocol for the image contribution comparison

## Material passport

| Field | Value |
| :--- | :--- |
| Origin | Academic research suite, experiment planning |
| Date | 23 October 2026 |
| Status | Original protocol recorded before execution, not registered |
| Scope | One comparison using fixed text and image representations |

The proposal was written after historical results were available. Its new
comparisons must be fixed before their predictions are inspected. The existing
items have already supported development, so they are not an untouched test set.
Author adoption into the paper remains a separate decision.

## Execution update

The comparison has now run locally and has been repeated from an independent
branch based directly on upstream main. The implementation, aggregate results,
verification records and proposed manuscript wording are in
[the focused study PR](https://github.com/kishormorol/bangla-multimodal-political-stance/pull/6).
The remaining sections preserve the proposal recorded at revision
`f99ab56f38956345fbbb92af90c94d2317c86e0f`, including its statements about
work that had not run at that point. The manuscript does not adopt the new
result. That decision remains with the author.

## Question and available evidence

The question is whether adding one fixed image representation improves
prediction of the stored final labels when the text representation and
training procedure are held fixed. The historical comparisons change text
representations along with image use, so they do not isolate that contribution.
The directional hypothesis is that B scores higher than A. The corresponding
null is a difference at or below zero. A practical effect threshold is a
separate choice and is not substituted for that null.

Use the 149 item membership recovered from the saved paired report. The new
local bundle has a currently decoded image for every selected item. Its text
checksum and seed 42 folds match the saved record. Historical identity of
image bytes remains unknown. Details are in `docs/input_recovery.md` and
`reports/verification/input_recovery.json`.
The exact bundle and its per item path mapping are required inputs. Do not
select replacement files from the broader raw folders. Use `title` as the
text field, and verify the saved headline checksum before feature extraction.

The prediction unit is one item, with labels 0, 1, and 2 as stored. The current
source components give 148 groups. Keep Image_154 and Image_155 together.
These components do not certify independence across events or outlets.
The result concerns prediction of existing labels. It cannot certify that the
labels describe stance visible in a headline and photograph.

## Fixed comparisons

| Condition | Features | Purpose |
| :--- | :--- | :--- |
| A | Character text features | Text reference |
| B | The exact text block from A, followed by fixed image features | Primary image addition comparison |
| C | The exact image block used in B | Image reference |

Use the text representation in `configs/text/tfidf_logreg.yaml`. The vectorizer
uses character word boundaries, ngrams of length 2 to 5, minimum document
frequency 2, at most 50,000 features, sublinear term frequency, and L2 row
normalization. Fit it on each training fold only. Reuse the fitted text block
between A and B within that fold. Do not refit a different text representation
for B.

The proposed image source is the frozen ViT named in
`configs/multimodal/countvec_vit.yaml`. Its public model metadata resolved to
revision `b4569560a39a0f1af58e3ddaf17facf20ab919b0` on 23 October 2026.
Use that revision for both model and processor. No weights or image features
were downloaded to prepare this proposal. Record local weight hashes,
software, device, and extraction batch size. Use the final CLS vector, as in
`src/bmpb/models/baselines.py`, with the encoder in evaluation mode and gradients
disabled. L2 normalize each image vector. Keep the text block numerically
unchanged when appending the image block. Do not tune block weights.
Do not normalize the concatenated vector. Keeping the original text block
also keeps its coefficient penalty unchanged when all image coefficients are
zero. Scaling that text block only in B would change its regularization.
Use `bmpb.data.dataset.load_image` at size 224, which converts to RGB and
resizes with bicubic interpolation, then pass that result to the pinned ViT
processor. Record the complete processor configuration. Do not mix direct
original image processing with this two step route.

All conditions use scikit learn 1.9.1 logistic regression with C 1.0, balanced
training class weights, 2,000 maximum iterations, solver `lbfgs`, tolerance
0.0001, intercept fitting enabled, and L2 regularization through `l1_ratio=0`.
Set `dual=False`, `warm_start=False`, `verbose=0`, `n_jobs=None`, and
`intercept_scaling=1`. Use random state 42. That solver is deterministic;
fold seeds below change membership rather than this setting. Record all
constructor parameters before running. The observed version has no
`multi_class` parameter, so do not copy an older API setting into the proposal.
No condition gets a parameter search or extra selection budget. A convergence
warning invalidates the affected seed before interpretation. Preserve it.
Any change of solver, tolerance, or iteration limit requires a new common
configuration and a new protocol version before rerunning all conditions and
seeds. It must not be chosen from their prediction scores.

Use original training rows only, with augmentation disabled in every condition.
The candidate deliberately avoids the published augmentation because a changed
headline would not supply a corresponding independent image. This proposal
requires a new study configuration and does not change the historical protocol.

The existing CountVec and ViT model does not implement this comparison. It uses
a different word count representation. A later implementation must reuse the
same character text features for A and B, keep sparse text features sparse,
and save predictions and feature identities. No such model study has run here.

## Folds, scores, and uncertainty

Seed 42 is the primary comparison. Use the saved five outer folds, which the
current grouping code reproduces. Every condition must score every selected
item exactly once, with identical training and held out membership. Use pooled
macro F1 over the fixed three labels. Report class support, per class F1,
accuracy, and confusion matrices as secondary descriptions.

The primary quantity is macro F1 of B minus macro F1 of A. Compute its paired
95 percent interval with 2,000 resamples of complete source groups. Reuse
the `paired_difference` function in `scripts/matched_analysis.py` and its fixed
bootstrap seed, passing B predictions first and A predictions second. Its CLI
hardcodes historical model names, so a reviewed adapter is needed for these
new conditions. The resampling keeps
paired predictions together. It conditions on fitted models and observed
predictions, so it does not include uncertainty from refitting training sets.
Overlapping CV training sets and unrecorded event relationships remain limits.

Seeds 43 to 46 are a predeclared secondary fold sensitivity analysis under the
same configuration. They are included before seed 42 is inspected. These are
the same articles assigned to different folds.
With deterministic frozen features and a linear solver, the intended variation
is principally fold assignment. It is not five independent datasets. Report
each difference and interval, then describe the spread of point estimates.
Do not construct an inferential interval by treating five seed scores or five
copies of each item as independent observations.

Compare C with A as a secondary description. The primary decision concerns
B versus A. Do not turn a favorable secondary result into the primary outcome.

## Decision rules and precision

An interval wholly above zero supports improved prediction under this fixed
setup. An interval wholly below zero supports reduced prediction. An interval
including zero leaves the direction unresolved. None of the outcomes decides
whether photographs contain political stance in general.

A useful effect threshold has not been accepted by the author. Before using
language about a useful gain, record a threshold and its reason before looking
at the new results. Report sensitivity at 0.02, 0.05, and 0.10 macro F1 as
explicit illustrative thresholds if no single threshold is chosen. An upper
bound below a threshold can rule out a gain of that size under the conditional
interval. It does not prove a zero effect. Apply interval rules to the primary
comparison. Secondary seed means have no sampling interval from this protocol.

The historical character text score on 149 items is 0.543, with a recorded
interval from 0.457 to 0.619. That width illustrates uncertainty in an existing
score; it is not a power calculation for the new paired difference. Pairing,
class support, group sizes, and correlated errors determine the new precision.
Do not promise detection of a two point or five point gain from 149 items alone.

The next sample size calculation, if needed, must use a stated precision target
or smallest meaningful effect. It must use paired pilot predictions and the
actual grouping structure. The current dataset supplies no untouched pilot
and confirmation split. A claim about future news would require a separately
collected evaluation cohort kept out of development. No such addition is
needed merely to report the present comparison as inconclusive.

## Competing explanations and the next decisive check

If B improves, correct image association is one possible explanation. Added
feature dimensions, source cues, or incidental visual correlations are others.
A later shuffled image control can test whether the correct pairing contributes
beyond adding an image feature block. Predefine that control before running it,
and compare it with B using the same folds and fixed model settings. A gain
that survives shuffled pairing would weaken a claim about paired visual evidence.
A negative control is not required to report an inconclusive or negative B
result. Adding it would be justified by the named uncertainty after a positive
primary result, rather than by a search for a favorable score.
That control is outside this proposed study. Any later use is an explicitly
exploratory follow up with its own protocol and results.

If C performs poorly, the fixed encoder or linear classifier may fail to use
available visual information. Poor C performance alone does not show that the
images lack stance information. If folds disagree, inspect training diagnostics
and source distributions. Preserve every result before proposing a new model.

## Execution conditions and cancellation

Verify the input manifest, selected identifiers, class counts, source groups,
fold membership, and model revision before extracting features. Keep original
news files and historical numerical reports unchanged. Store all new artifacts
under a separately identified experiment directory.

Cancel a paired comparison if a condition loses items, changes folds, changes
labels, fits preprocessing on held out records, or lacks complete predictions.
A failed condition cannot be replaced by a historical score. Record failures
and stop the affected comparison. The same rule applies if numerical warnings
leave one condition incomplete.

The user has authorized new local GPU work that preserves historical artifacts.
The BLIP implementation check passed on the local GPU, as recorded in
`docs/run_verification.md`. The proposed comparison still needs a reviewed
implementation and configuration before execution. Paid compute and external
data transfer remain outside the approved scope. Narrowing the manuscript claim
can make this study unnecessary.
The planned work is one image feature extraction over 149 files and 75 linear
classifier fits, from three conditions, five folds, and five fold assignments.
No tuning search is included. Runtime and cost have not been measured. Measure
extraction and fitting time before committing to any larger study.

Independent Bangla review is needed for a claim about the meaning of the labels
or the correctness of image pairings. Original annotation records are needed
to establish what annotators saw and how final labels were decided. Those are
separate evidence requirements. Model training cannot replace them.
