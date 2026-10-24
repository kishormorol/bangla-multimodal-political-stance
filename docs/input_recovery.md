# Recovered inputs for independent replication

Checked on 23 October 2026. The aggregate record is
`reports/verification/input_recovery.json`. News content and image bytes remain
local under ignored data directories.

## What was recovered

All 13 paired model entries in `reports/submission/matched.json` contain the
same 149 item identifiers and outer folds. The processed image table also has
149 identifiers, but only 122 are in that saved evaluation set. The aggregate
record lists the 27 identifiers on each side of the difference.

The public Drive folders supplied a current image file for every saved item.
The recovery preferred the processed copy when listed and used the original
image folder for the remaining items. The selected files comprise 122 processed
copies and 27 originals. Every file decoded fully. No two selected files share
an exact byte hash or an exact decoded RGB pixel hash, including dimensions.
Those checks do not exclude near duplicates, shared events, or outlet effects.

The selected headlines, labels, and source URLs reproduce the saved text input
checksum. The current grouping code gives 148 source groups. It reproduces the
saved five outer folds for seed 42 exactly. Image_154 and Image_155 remain in
one source group because their source URLs match.

The recovery checks establish current availability and mechanical record
associations. Historical image hashes and original neural prediction artifacts
remain unavailable. The recovered bytes must not be described as proven copies
of the historical training inputs. Filename agreement also does not establish
that a photograph belongs to its article. Independent review is still needed.

## Private artifacts

The new files are under `data/raw/recovered_20261023/`. The inventory at
`data/interim/replication_recovery_20261023.json` records each identifier,
Drive file identifier, retrieval method, decoded dimensions, byte count,
and SHA256. Existing source files were copied, and missing files were retrieved
through the public Drive download endpoint. No source artifact was overwritten.
Firecrawl verified the public download route. It returned an image description
rather than original bytes, so binary retrieval used the original endpoint.

The frozen bundle is `data/interim/replication_inputs_20261023/`. It contains
all 198 text records and the 149 selected image files. The other 49 records
have no image in this bundle. That restriction preserves the chosen comparison
population and does not assert that those articles never had photographs.
The bundle contains its data configuration, the five recovered source CSVs,
a population record, and an input manifest with 156 file states.

The earlier `data/processed/input_manifest.json` still verifies its original
198 records and 106 file states. Its bytes were preserved. The aggregate report
contains hashes of both snapshots and the private recovery inventory.

## How another researcher can check the recovery

First, recover the five source CSVs and verify the checksums recorded in
`reports/verification/original_baselines.json`. Obtain the image folder
listings from the identifiers in `configs/data.yaml` without modifying the
existing raw mirror.

Second, recover the saved paired population from the union of `fold_item_ids`
in a paired model entry. Confirm agreement across all 13 entries. Normalize
image identifiers with `bmpb.data.ingest.normalize_item_id`. Prefer the listed
processed image and otherwise select the original image with the same
identifier. Preserve download responses and the current source listing.

Third, decode each selected image with Pillow `load`. Hash its original bytes,
then hash decoded RGB dimensions and pixels to check exact duplicates. Compare
the population with the image table by identifiers. Reproduce the headline
input checksum and the seed 42 folds before training anything.

Finally, restore the private bundle into a clean verified implementation
checkout. The command `python scripts/input_manifest.py` requires the script
from [the experiment integrity PR](https://github.com/kishormorol/bangla-multimodal-political-stance/pull/3).
That script is not included in the evidence PR. Compare the verified manifest
hash with the aggregate record. The public `scripts/verify_reports.py` checks the aggregate's source
identity and count arithmetic. It does not access the private images or prove
that those files were downloaded correctly.

The current snapshot supports preparing an independent reevaluation of the
saved item population. It supplies no new performance result or evidence that
images improve classification.
