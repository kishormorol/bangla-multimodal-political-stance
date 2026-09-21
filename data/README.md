# Dataset

Nothing in `data/` is committed. Run `make data` (or `bash scripts/fetch_data.sh`)
to mirror the shared Drive folder into `data/raw/`, then `make ingest && make splits`.

```
data/
  raw/          exact Drive mirror — 25 CSVs, images/, processed_images/
  interim/      scratch space for one-off transforms
  processed/    corpus.csv, train.csv, val.csv, test.csv, image_index.csv
```

Source folder: `https://drive.google.com/drive/folders/1KbjKIuktu97dtVtahyZXjw7dCc3EBzqc`

Google rate-limits bulk downloads from shared folders. A first run often stops
partway with *"Cannot retrieve the public link … have had many accesses"*; the
download resumes, so re-run it after a few minutes.

## What the corpus is

198 Bangla news items, each with a headline, a source URL, an accompanying
photo, and a three-way stance label.

| Label | id | Items |
| --- | ---: | ---: |
| `govt_critique` | 0 | 103 |
| `neutral` | 1 | 53 |
| `govt_leaning` | 2 | 42 |

**The label ids are fixed.** Every published prediction CSV in the Drive folder
is keyed to them, so renumbering would silently invalidate the existing results.
`tests/test_ingest.py` pins the mapping.

39 distinct outlets, led by BBC Bangla (39), Prothom Alo (37+4 under a second
spelling), Jugantor (13) and Samakal (10).

## Three things to know before quoting a number

### 1. The text is headlines, not articles

`Preprocessed_Text` holds the headline. Median length is **8 words** (54
characters); the longest is 176 characters. Every "text model" result in this
project is therefore **headline stance classification**, not document
classification. `bmpb audit` prints this warning on every run.

### 2. The published splits leak

`train_set.csv` / `val_set.csv` / `test_set.csv` (218/47/47) were cut from
`Balanced_Augmented_Dataset.csv` *after* augmentation. An augmented row keeps its
parent's `source_index`, so the same article appears on both sides:

```
source_index overlap  train/val: 17   train/test: 25   val/test: 3
```

A model can see a swap-augmented variant of a test headline during training, so
any score on that test set is optimistic. `make splits` replaces them: articles
are assigned to splits first, the 114 augmented rows follow their parent, and
only the training split gets augmented. `tests/test_splits.py` asserts there is
no group overlap, and `bmpb audit` reports the leakage in the published splits
so the comparison stays visible.

The originals stay in `data/raw/` and load via
`bmpb.data.splits.load_original_splits()` — use them only to reproduce numbers
that are already in the manuscript.

The grouped splits are smaller (219/30/29 with augmentation on train) and their
val/test halves are *deliberately unbalanced*: they carry the corpus's real
label distribution, because balancing an evaluation set makes accuracy look
better than the task is.

### 3. The image subset is a different population

Only 106 of 198 items resolve to an image on disk, and that subset is not
label-balanced the same way (50 / 33 / 23). The vision-language models are
evaluated on 30 items where the text models get 47. **Those columns are not
directly comparable**, and the results table prints `n` for every row.

## Annotation

Three annotators, with uneven coverage — Annotator 1 labelled 32 of 198 items:

| Pair | Cohen's κ | Overlap |
| --- | ---: | ---: |
| Annotator 2 ↔ 3 | 0.73 | 197 |
| Annotator 1 ↔ 2 | 0.45 | 32 |
| Annotator 1 ↔ 3 | 0.20 | 32 |

κ = 0.73 between the two annotators who labelled everything is substantial
agreement and is the number worth reporting; the Annotator 1 pairs rest on 32
items and should be described as such rather than averaged in.

Article-level and image-level labels agree on only **47.4%** of the 152 items
labelled both ways — the photo carries a different stance from the headline
about half the time. That gap is the argument for the multimodal framing, and it
is a finding rather than a data problem.

## Raw file inventory

**Corpus lineage** — each row is derived from the one above it:

| File | Rows | What changed |
| --- | ---: | --- |
| `Dataset.csv` | 200 | original annotation sheet, three annotator columns, trailing unnamed columns |
| `Political_Bias_Cleaned_Dataset.csv` | 200 | cleaning pass |
| `Preprocesing.csv` | 200 | adds `Preprocessed_Text` |
| `Preprocesing_Cleaned.csv` | 200 | annotator columns dropped |
| `Preprocesing_Numerical.csv` | 200 | adds `*_NUM` integer label encodings |
| `Final_Dataset.csv` | **198** | two unusable rows removed — **this is the corpus** |
| `Balanced_Augmented_Dataset.csv` | 312 | 198 original + 114 augmented (59 `swap`, 52 `all`, 3 `synonym`) |
| `train/val/test_set.csv` | 218/47/47 | split of the augmented set — **leaks, see above** |

**Image tables**

| File | Rows | Notes |
| --- | ---: | --- |
| `Final_Image_Dataset_Processed.csv` | 149 | `Image_id`, `Image_Path`, `Final_Label` |
| `Final_Image_Dataset_WithText.csv` | 149 | the same plus `Preprocessed_Text` |

**Published predictions** — scored automatically by `make leaderboard`:

| File | Test rows | Prediction column |
| --- | ---: | --- |
| `BanglaBERT.csv` | 47 | `banglabert` |
| `BanglaELECTRA_Predictions.csv` | 47 | `bangla_electra` |
| `mBERT_Predictions.csv` | 47 | `mbert` |
| `XLMRoBERTa_Predictions.csv` | 47 | `xlmroberta` |
| `mt5_predictions.csv` | 47 | `mt5_pred` (label strings, not ids) |
| `CLIP_Predictions.csv` | 30 | `Predicted_Label` |
| `BLIP_Predictions.csv` | 30 | `Predicted_Label` |
| `ALIGN_Predictions.csv` | 30 | `Predicted_Label` |
| `FLAVA_Predictions.csv` | 30 | `Predicted_Label` |
| `ViLT_Predictions.csv` | 30 | `Predicted_Label` |
| `CountVec_ViT_Predictions.csv` | 30 | `Predicted_Label` |
| `1_Predictions.csv` | 30 | unidentified model — same schema as the multimodal files |

`Final_Dataset_BanglaBERT.csv` (30 rows) is a BanglaBERT run over the image
subset rather than the text test set.

## Quirks that ingest handles

* `Image_id` is zero-padded in the first rows (`Image_01`) and not afterwards
  (`Image_2`) → normalized to `Image_<n>`.
* Labels appear as strings and as ids, with drifting case (`Govt critique` in
  the annotation sheet, `Govt Critique` in the numeric one).
* `Image_Path` points at the original Colab mount
  (`/content/drive/MyDrive/Political_Bias/processed_images/Image_2.jpg`) →
  rewritten relative to `data/raw/` and dropped when the file is absent.
* `images/` carries whatever extension the news CDN served — `.jpg`, `.webp`,
  `.avif`, `.svg`, and doubled suffixes like `Image_01.jpg.webp`.
  `processed_images/` is the uniform set the vision models consumed, and ingest
  prefers it.
* Outlet names vary by whitespace and case (`BBC Bangla`, `\t\nBBC Bangla`,
  ` BBC bangla`) → collapsed into `outlet_key`.
* `Dataset.csv` carries three trailing `Unnamed:` spreadsheet columns → dropped.

## Licensing

The code is MIT (see `LICENSE`). The dataset is **not** covered by it: the
headlines and photographs belong to the outlets listed in `newspaper name`, and
`source link` records where each item came from. Redistribute the annotations
and the URLs; do not redistribute the article text or images without checking
each outlet's terms.
