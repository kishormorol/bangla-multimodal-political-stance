# Local comparison of image contribution

The configuration `configs/ablations/image_contribution.yaml` fixes the
comparison before execution. Its protocol is the proposal at Git revision
`f99ab56f38956345fbbb92af90c94d2317c86e0f`, in
[`docs/replication_protocol.md` at that revision](https://github.com/WhiteHades/bangla-multimodal-political-stance/blob/f99ab56f38956345fbbb92af90c94d2317c86e0f/docs/replication_protocol.md). That proposal was written after historical
results were available. This experiment predicts the existing stored labels
on a recovered input snapshot.

The runner compares character text features, the same text features with a
fixed image feature block, and that image block alone. It checks the text
and image blocks for exact equality across the relevant conditions. It uses
the frozen 149 item population, the saved seed 42 folds, and four additional
fold assignments fixed in the configuration. All conditions use the same
classifier parameters and 2,000 iteration limit. Augmentation is disabled.

Before any model fitting, the implementation review added descriptive source
group intervals for each condition's pooled macro F1 and for the secondary
image versus text comparison. The primary comparison and its decision rules
remain those in the proposal. The primary interval from the existing paired
helper must exactly match the separately calculated group bootstrap interval.
These intervals condition on fitted predictions. They do not include model
refitting, annotation, event, or outlet uncertainty. Repeated fold seeds reuse
the same items.

Run from the repository root with the separate CUDA environment:

```sh
.venv-gpu/bin/python -m scripts.image_contribution
```

The command requires a clean, committed implementation. Its default output is
a new directory under `experiments/`. An existing output directory is rejected.
The runner verifies the input bundle, freezes its plan, and downloads only the
named public model revision. It writes image features and processor metadata,
training diagnostics, predictions, scores, and an artifact checksum manifest.
Failures remain in the new directory with a failure record. Historical reports
and input files are not output destinations.

The small local check uses explicitly synthetic text and image features:

```sh
mkdir -p .tmp
rustc --edition=2021 checks/image_contribution.rs -o .tmp/check_image_contribution
.tmp/check_image_contribution .venv-gpu/bin/python
```

It checks that the added image block can affect predictions, that all three
conditions score the same held out items, and that duplicate IDs or fold members
are rejected without changing completed outputs. It does not run a model or
produce a benchmark score.
The existing CI workflow runs the synthetic checks with its normal Python
environment, model downloads offline, and CUDA disabled.

After the experiment, repeat the text condition through the existing CV runner
with the same selected corpus, seed, folds, and augmentation disabled. Require
identical item predictions before interpreting the new comparison. Recalculate
the summaries from saved predictions and verify every artifact checksum.
Independent Bangla review and historical image byte identity remain separate
evidence requirements. Author adoption of the study remains a separate decision.

## Completed local run

The run finished on 23 October 2026 at implementation revision
`cb359f50b09358ede5ca67dbff6c5830ddc635e4`, recorded in
`reports/verification/image_contribution.json`. The aggregate record contains
the configuration, source and artifact checksums, software versions, model
loading details, every condition's scores, and verification results. It contains
no headlines, photos, image features, or item prediction records.

All conditions scored the same 149 items in 148 source groups. Class counts
were 72 government critique, 43 neutral, and 34 government leaning. Seed 42
used the saved five folds. The primary pooled results were:

| Condition | Macro F1 | Conditional 95 percent interval |
| :--- | ---: | :--- |
| A, text | 0.5642 | 0.4820 to 0.6420 |
| B, text and images | 0.4588 | 0.3752 to 0.5423 |
| C, images | 0.4074 | 0.3262 to 0.4854 |

Adding the fixed image features reduced the primary score by 0.1054. The
paired interval indicates a decrease of 0.0230 to 0.1812. The predeclared
decision rule therefore supports reduced prediction under the fixed setup.
The primary upper bound is below each illustrative gain threshold of 0.02,
0.05, and 0.10. No accepted threshold for a useful effect was chosen.

All four additional fold assignments also gave lower point estimates for B.
The intervals for seeds 43 and 44 excluded zero. The intervals for seeds 45
and 46 included zero. The five point estimates ranged from a decrease of
0.0469 to a decrease of 0.1224. Every seed's results remain in the aggregate
record. Reusing the items does not create five independent datasets.

The local NVIDIA GeForce GTX 1650 Ti extracted the 149 image vectors in
4.66 seconds after model loading. Peak GPU memory reserved was 482,344,960
bytes. The 75 classifier fits used the CPU and took 1.99 seconds in total
within the recorded fit timers. The fit limit was 2,000 iterations, and the
largest observed count was 16. Downloading, model loading, text preprocessing,
and statistical analysis are outside those timing claims.

Verification checked all 20 run artifact checksums and all 156 frozen input
files. A separate run through `bmpb.crossval.cross_validate`, with augmentation
disabled, produced identical text predictions for all 149 items. Its labels,
groups, and folds also matched. Independent calculations from saved predictions
reproduced the confusion counts, macro F1, accuracy, condition intervals, and
paired intervals for all five seeds. The numerical tolerance was 0.000000000000001.
The historical numerical reports and input files remain unchanged.

The experiment supports a limited conclusion about stored label prediction
with one frozen image representation and one linear classifier. It does not
establish that photographs lack useful stance information. The images' identity
with historical training inputs is unknown, and the labels and semantic image
pairings still need independent Bangla review. The items have already supported
development, and source grouping does not establish event or outlet independence.
The intervals omit model refitting and annotation uncertainty.

The unchanged protocol and the complete aggregate results allow another
researcher to rerun the comparison against the exact frozen bundle. Access
to that local bundle is necessary to verify the prediction records. No new
model search or shuffled image experiment is justified by the negative primary
result alone. Adopting the result in the manuscript requires the author's
decision.

## Independent branch verification

The study branch starts directly from upstream revision `a935d11`. Its helper
adapter supplies the frozen groups to the original upstream analysis interface
and also accepts the interface in the companion model PR. A complete repeat
at revision `4050e40f9fa35ba47c98e1288591cdac07a3c077` produced identical image feature bytes,
all 15 prediction CSVs, and every score and interval. The aggregate record
contains the repeat implementation and artifact checksums. No new conditions,
parameters, or evaluation items were introduced. The original run remains
unchanged. The branch can therefore be reviewed without including the model PR.
Both runs used the same local GPU, frozen files and software versions. The
repeat verifies branch separation and local execution. An external researcher
has not replicated the experiment.

## Proposed manuscript wording

The following paragraph is a proposal for the author. It is not inserted into
the manuscript. Its numbers come from the aggregate record above.

We evaluated the addition of frozen ViT image features to identical character TF IDF features on 149 recovered items in 148 source groups. The primary saved fold assignment gave macro F1 0.5642 for text and 0.4588 for text and images. Their conditional 95 percent source group intervals were 0.4820 to 0.6420 and 0.3752 to 0.5423, respectively. Adding images reduced macro F1 by 0.1054, with a paired conditional 95 percent interval for the decrease of 0.0230 to 0.1812. All four additional fold assignments gave lower point estimates, and two of their intervals included zero. The comparison uses a fixed encoder and linear classifier to predict stored labels. It does not establish that images lack stance information or that the recovered image bytes match the historical inputs. The intervals condition on saved predictions and omit uncertainty from refitting, annotations, events and outlets. The repeated fold assignments reuse the same items, which have already supported development.
