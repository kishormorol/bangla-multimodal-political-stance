# Evidence available in this checkout

Checked on 23 October 2026 against revision `a935d11`.

The committed summaries support arithmetic checks. They do not contain the
original prediction files or training logs. A successful check of a confusion
matrix does not reproduce model training or establish annotation validity.

Run the check from the repository root with:

```sh
python scripts/verify_reports.py
```

The check uses the Python standard library and does not change the reports.
It recalculates accuracy and macro F1 from the 20 matched confusion matrices.
It checks the 100 saved seed scores against the model summaries and the 60
paired differences against their source scores and aggregate summaries.

The matched report records 149 items for the paired population and 198 for
the headline population. All 60 seed comparison records instead report 86
items and 85 source groups. The seed run entries omit item membership and
input hashes. The seed results must therefore be described separately from
the 149 item matched report until the original artifacts establish their
relationship. Matching a population size alone would not establish identity.

All 20 matched report entries have a null Git revision. Fold membership and an
input digest survive, but those fields do not identify image bytes, software
versions, or the exact training implementation.

The early stopping summary records 75 neural fold runs, with 65 early stopped
runs. Among those 65 runs, 29 selected a checkpoint from epoch 1 or 2. The
summary does not say that those runs stopped at epoch 1 or 2.

The original reports remain unchanged. Their confidence intervals require
the original predictions and grouping records for independent recalculation.
Original annotation procedures, unresolved collection origins, and image
pairing semantics require further evidence.

## Independent baseline reproduction

The public source table was subsequently recovered through the download URL
behind the repository's Drive link. Its SHA256 is
`49bddb145fb380019fc2fb6f9690a8f1494355128c074b0421cfd1f02060e94e`,
which exactly matches `raw_sha256` in the committed provenance report.
The processed source, annotation source, image metadata table, and published
augmentation table were also recovered and kept under ignored `data/raw/`.

Ingest reproduces annotation coverage of 32, 197, and 197 records. Selecting
the 149 item membership from the matched report reproduces its text input
checksum. The complete 198 item text population also reproduces its checksum.

The original implementation at `a935d11` was then run in an isolated checkout.
Both the majority classifier and character ngram classifier reproduce their
saved macro F1, confusion matrices, fold memberships, and bootstrap intervals
on both populations. See
[`original_baselines.json`](../reports/verification/original_baselines.json)
for exact scores, source checksums, and the software versions used.

The reproduction used the saved item membership rather than the incomplete
local image download to select text examples. It verifies text experiments,
not image availability, historical neural training, or original predictions
for every individual item. The original experiment software versions remain
unknown even though the four baseline results reproduce in the recorded
current environment.

To repeat the check, use a separate checkout at `a935d11` and ingest the
recovered sources. Set `text` to `title`. Keep all 198 rows for the complete
population. For the paired population, select the union of `fold_item_ids`
from its matched report entry while preserving corpus order. Confirm the
input checksums before calling `bmpb.crossval.cross_validate` with each of
`configs/text/majority.yaml` and `configs/text/tfidf_logreg.yaml`. Pass the
selected corpus explicitly and use a new output directory. Compare the saved
fold membership, confusion matrices, point scores, and intervals. The partial
image download must not determine membership for this text reproduction.

A small Rust check exercises the report command and verifies that it rejects
a deliberately corrupted copy while preserving the committed inputs:

```sh
mkdir -p .tmp
rustc --edition=2021 checks/reports.rs -o .tmp/check_reports
.tmp/check_reports .venv/bin/python
```
