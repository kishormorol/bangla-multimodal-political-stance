# Annotation workflows

This repository has distinct annotation inputs. Keep their files and claims
separate; they do not share an automatic merge format.

## Local single annotator

Run `python scripts/annotate_local.py` to label rows from the configured
Hugging Face `unannotated` split. The script writes one CSV row per labelled
item, with `item_id`, `headline`, `outlet`, `date`, `source_url`, numeric
`label`, `label_name`, and `annotator`. It records one headline stance only.
It does not store image judgements, flags, notes, or a vote from another
annotator. The output is not accepted by `bmpb merge-annotations`.

Before resuming, the script checks the saved header, row width, item IDs,
label IDs, and agreement between each label ID and label name. A malformed
file or conflicting labels for the same ID stop the command before dataset
loading or appending. A row with an
empty label and label name remains pending. The command does not repair or
relabel existing records.

The local check uses synthetic CSV files and needs no dataset download:

```sh
mkdir -p .tmp
rustc --edition=2021 checks/annotations.rs -o .tmp/check_annotations
.tmp/check_annotations .venv/bin/python
```

## Artifact desk export and merge

`bmpb merge-annotations --export DIR` expects JSON under `DIR/items/` and
`DIR/annotations/`. Item records provide fields such as `id`, `title`, `body`,
`summary`, `source_url`, `outlet`, `published`, and `image_url`. Annotation
records use a `labels` mapping keyed by item ID; each record can contain
`story`, `image`, `flag`, and `note`. The merge writes rows with collected
annotations to `data/interim/annotated.csv` and items requiring adjudication
to `data/interim/adjudicate.csv`. Its current code uses a majority of at least
two story votes and sends flagged or unresolved items to adjudication. It
retains each annotator's image vote; those votes do not determine the story
label.

## Corpus ingest

The configured Drive corpus is a separate spreadsheet path. Ingest reads
`Final_Dataset.csv` fields including `Image_id`, `Title`, `Preprocessed_Text`,
`FINAL LABEL`, and `source link`, then joins annotator columns from
`Preprocesing.csv` by item ID. The resulting corpus preserves `title` and
`text` separately. The current config does not establish who saw which fields
or how the final labels were adjudicated. Do not infer those procedures from
the column names or from the separate Artifact desk workflow.

The label IDs remain `0 = govt_critique`, `1 = neutral`, and
`2 = govt_leaning` in each path. This file describes observed formats and
current code behavior; it does not set a new annotation target or voting rule.
