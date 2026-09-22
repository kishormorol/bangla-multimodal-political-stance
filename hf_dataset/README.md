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
size_categories:
- 10K<n<100K
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
    dtype:
      class_label:
        names:
          '0': govt_critique
          '1': neutral
          '2': govt_leaning
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

A large-scale Bangla multimodal dataset for political stance detection, containing **14,719 news items** (headlines + photos) from **392 Bangladeshi news outlets**.

## Dataset Summary

BanglaPoliticalStance is the first large-scale multimodal Bangla political stance detection dataset. Each item consists of a news headline in Bangla and its accompanying photograph, collected from major Bangladeshi news portals. The dataset supports three-way stance classification:

| Label | ID | Description |
|---|---:|---|
| `govt_critique` | 0 | Critical of the government |
| `neutral` | 1 | Neutral reporting |
| `govt_leaning` | 2 | Favorable toward the government |

## Dataset Structure

The dataset has two splits:

### `annotated` (198 items)
Expert-annotated by 3 human annotators with inter-annotator agreement of κ=0.73 (Cohen's kappa between the two primary annotators). These items include gold-standard labels for benchmarking.

### `unannotated` (14,521 items)
Recently collected headlines and images from 392 Bangla news outlets, ready for annotation. These items do not have stance labels yet.

**We welcome community contributions to annotate this data.**

## Features

| Feature | Type | Description |
|---|---|---|
| `item_id` | string | Unique identifier (MD5 hash of URL) |
| `headline` | string | News headline in Bangla (median ~8 words) |
| `source_url` | string | Original article URL |
| `outlet` | string | News outlet name |
| `image` | image | Accompanying news photograph |
| `date` | string | Publication date (when available) |
| `section` | string | News section (politics, national, etc.) |
| `label` | ClassLabel | Stance label (annotated split only) |

## Source Outlets (Top 20)

| Outlet | Articles |
|---|---:|
| The Daily Star | 996 |
| Prothom Alo | 604 |
| Bangladesh Sangbad Sangstha (BSS) | 464 |
| BBC Bangla | 230 |
| Jugantor | 182 |
| Bangladesh Pratidin | 166 |
| Daily Ittefaq | 142 |
| The Business Standard | 118 |
| Jago News | 112 |
| bdnews24 | 102 |
| Samakal | 82 |
| Kaler Kantho | 79 |
| Naya Diganta | 74 |
| Ajker Patrika | 71 |
| Bangla Tribune | 67 |
| NTV | 65 |
| Risingbd | 64 |
| Dhaka Post | 96 |
| Dhaka Mail | 99 |
| Daily Inqilab | 98 |

...and 370+ more outlets.

## Important Notes

1. **The text is headlines, not articles.** Median headline length is ~8 words. Results should be described as headline stance classification.

2. **Label IDs are frozen.** `0 = govt_critique, 1 = neutral, 2 = govt_leaning`. Do not renumber.

3. **Image-text stance can differ.** In the annotated subset, article-level and image-level labels agree on only 47.4% of items — the photo often carries a different stance from the headline. This gap is the core argument for multimodal approaches.

4. **The annotated split has class imbalance.** `govt_critique` (103) > `neutral` (53) > `govt_leaning` (42). This reflects the real distribution and should not be artificially balanced for evaluation.

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

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{bangla_political_stance_2026,
  title={BanglaPoliticalStance: A Large-Scale Bangla Multimodal Political Stance Detection Dataset},
  author={Kishor Morol},
  year={2026},
  url={https://huggingface.co/datasets/kishormorol/BanglaPoliticalStance},
  note={14,719 Bangla news headlines with photographs from 392 outlets}
}
```

## License

The annotations and metadata are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The headlines and photographs belong to their respective news outlets — `source_url` records the provenance of each item. Please check each outlet's terms before redistributing article text or images.

## Contributing

We welcome contributions to:
- **Annotate** items in the unannotated split
- **Validate** existing annotations
- **Report** issues with data quality

Please open a discussion on this dataset's page if you'd like to contribute.
