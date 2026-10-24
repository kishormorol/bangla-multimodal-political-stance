---
language:
- bn
license: cc-by-nc-4.0
task_categories:
- text-classification
- image-classification
- zero-shot-classification
task_ids:
- multi-class-classification
- sentiment-analysis
tags:
- political-stance-detection
- bangla
- bengali
- multimodal
- news
- south-asia
- bangladesh
- nlp
- vision-language
pretty_name: "BanglaPoliticalStance: Bangla Multimodal Political Stance Detection Dataset"
dataset_info:
  features:
  - name: item_id
    dtype: string
  - name: headline
    dtype: string
  - name: source_url
    dtype: string
  - name: outlet
    dtype: string
  - name: image
    dtype: image
  - name: date
    dtype: string
  - name: section
    dtype: string
  - name: label
    dtype: int32
  splits:
  - name: annotated
    num_examples: 198
  - name: unannotated
    num_examples: 14521
configs:
- config_name: default
  data_files:
  - split: annotated
    path: data/annotated-*
  - split: unannotated
    path: data/unannotated-*
citation: |
  @dataset{morol2026banglapoliticalstance,
    title={BanglaPoliticalStance: A Large-Scale Bangla Multimodal Political Stance Detection Dataset},
    author={Kishor Morol},
    year={2026},
    url={https://huggingface.co/datasets/kishormorol/BanglaPoliticalStance},
    note={14,719 Bangla news headlines with photographs from 392 outlets}
  }
---

# BanglaPoliticalStance

This dataset repository contains an annotated corpus split and a scraped
unannotated split. The published card snapshot lists 198 annotated rows and
14,521 unannotated rows. The local build reports counts and image coverage
from its current source files; these can differ from the published snapshot.

## Dataset Summary

The annotated source is a processed table with separate `title` and `text`
fields and a numeric final label. The unannotated source is a scraped table
with a `headline` field and no stance label. During export, `headline` is
populated from `title` when that column exists in the annotated table; `text`
is used only when the `title` column is absent. The unannotated `headline` is
copied as provided. This does not establish that the annotated labels were
assigned from headlines alone. The label IDs are:

| Label | ID | Description |
|---|---:|---|
| `govt_critique` | 0 | Critical of the government |
| `neutral` | 1 | Neutral reporting |
| `govt_leaning` | 2 | Favorable toward the government |

## Dataset Structure

The dataset has two splits:

### `annotated`
The published card snapshot has 198 rows. These come from the processed
annotated table. Each row must have a unique item ID and a valid numeric stance
label (0, 1, or 2). The export preserves the stored label ID. The local
provenance audit found consistent upstream matches for 145 processed rows and
no exact normalized match for 53 rows. The unmatched rows are not thereby
established as independently collected. See
[the repository provenance audit](https://github.com/kishormorol/bangla-multimodal-political-stance/blob/a935d118dd843cb40ed14bfeaa2a958c1e0ce111/docs/dataset-provenance.md) for the audit and
its limits. The stored annotation columns cover 32, 197, and 197 of 198 items.
The saved audit reports rounded pairwise Cohen's kappa of 0.73 for the
two columns with 197 jointly labelled items. Original annotator roles,
instructions, and final label selection require further study records.

### `unannotated`
The published card snapshot has 14,521 rows. These come from the scraped table
and are provided as unlabeled input.

**We welcome community contributions to annotate this data.**

## Features

| Feature | Type | Description |
|---|---|---|
| `item_id` | string | `Image_N` identifier in the original annotated corpus; scraped IDs use the collection script's 12-character MD5 URL hash |
| `headline` | string | `title` in the annotated source when present, otherwise `text`; scraped `headline` in the unannotated source |
| `source_url` | string | Original article URL |
| `outlet` | string | News outlet name |
| `image` | image | Accompanying news photograph |
| `date` | string | Publication date (when available) |
| `section` | string | News section (politics, national, etc.) |
| `label` | int32 | Numeric stance ID (annotated split); `-1` marks unannotated rows |

## Important Notes

1. **The text is headlines, not articles.** Median headline length is ~8 words. Results should be described as headline stance classification.

2. **Label IDs are frozen.** `0 = govt_critique, 1 = neutral, 2 = govt_leaning`. Do not renumber.

3. **Images may be absent.** Export checks local image files and decodability. Missing or unreadable files are represented as null; the build output reports included, missing, and decode-failed image counts for each split.

4. **Provenance and rights are not fully established.** The provenance audit identifies reused upstream items. This repository does not assert redistribution rights for outlet headlines or photographs. Check source terms and permissions before reuse.

## Usage

```python
from datasets import load_dataset

# Load annotated split (with labels)
ds = load_dataset("kishormorol/BanglaPoliticalStance", split="annotated")

# Load unannotated split (for annotation or self-supervised pretraining)
ds_new = load_dataset("kishormorol/BanglaPoliticalStance", split="unannotated")

# Example
print(ds[0]["headline"])  # Bangla headline
print(ds[0]["label"])     # 0, 1, or 2
ds[0]["image"].show()     # PIL Image
```

## License and reuse

The annotations and metadata are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
The headlines and photographs belong to their respective news outlets;
`source_url` records provenance, not permission. Check outlet terms before
redistributing article text or images.

## Contributing

We welcome contributions to:
- **Annotate** items in the unannotated split
- **Validate** existing annotations
- **Report** issues with data quality

Please open a discussion on this dataset's page if you'd like to contribute.
