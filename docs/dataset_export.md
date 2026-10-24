# Local dataset export

`python scripts/upload_hf.py` builds and validates both dataset splits from the
local files named at the top of `scripts/upload_hf.py`. By default it is a dry
run and does not upload. Review the printed row, missing field, and image
inventory before using the result. `--push` publishes to the Hugging Face Hub;
`--repo ORG/NAME` selects another repository.

The annotated split reads `data/processed/corpus.csv`. Exported `headline`
comes from `title` when that CSV column exists, even if a particular cell is
blank. It falls back to `text` only when the `title` column is absent. The
numeric `label` is copied without changing the IDs: 0 is `govt_critique`, 1 is
`neutral`, and 2 is `govt_leaning`. IDs must be present and unique within a
source table, and annotated labels must be valid IDs.

The unannotated split reads
`data/raw/scraped/scraped_corpus_clean.csv`. Its `headline` field is copied as
provided, and its exported `label` is `-1`. The processed annotated table and
scraped table have different source fields and provenance. The export does
not merge them into one annotation history or establish that the annotated
labels were produced from headlines alone.

Image paths are resolved under their configured local image roots. Paths
outside those roots, absent files, and files that fail image decoding are
excluded as null image values. The per-split inventory reports images included,
missing, or rejected after a decode failure. It counts rows from the actual
input files rather than a fixed dataset-card total.

The local provenance findings and limitations are recorded in
[`dataset-provenance.md`](dataset-provenance.md). Neither that audit nor this
export establishes permission to redistribute outlet content or photographs.
