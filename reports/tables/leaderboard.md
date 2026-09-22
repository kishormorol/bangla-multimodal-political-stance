# Results

Macro-F1 is the headline metric; the 95% interval is a 2000-round item bootstrap
over pooled out-of-fold predictions. **Only rows marked `5-fold CV` are mutually
comparable** — they share one grouped, stratified protocol with augmentation applied
inside the training folds.

Rows marked `published` are scored from the prediction files in the Drive folder.
They come from three different protocols (see `notebooks/original/README.md`) and the
text-model rows are computed on a test set containing augmented variants of training
articles, so they are optimistic. They are kept here for traceability to the first
submission, not for comparison.

`per-fold` is the mean ± standard deviation across folds; a wide spread means the
pooled figure rests on folds that disagree.

| Model | Protocol | Population | n | Accuracy | Macro-F1 | 95% CI | Per-fold | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| BanglaBERT | published | — | 47 | 0.723 | 0.719 | 0.578–0.836 | — | test set is 47 items; differences smaller than the CI width are not meaningful |
| mBERT | published | — | 47 | 0.638 | 0.649 | 0.509–0.775 | — | test set is 47 items; differences smaller than the CI width are not meaningful |
| banglabert | 5-fold CV | all items | 198 | 0.576 | 0.533 | 0.458–0.606 | 0.522 ± 0.062 |  |
| tfidf_logreg | 5-fold CV | all items | 198 | 0.551 | 0.501 | 0.429–0.572 | 0.490 ± 0.091 |  |
| mbert | 5-fold CV | all items | 198 | 0.520 | 0.498 | 0.421–0.568 | 0.496 ± 0.087 |  |
| BanglaELECTRA | published | — | 47 | 0.468 | 0.462 | 0.308–0.596 | — | test set is 47 items; differences smaller than the CI width are not meaningful |
| ViLT | published | — | 30 | 0.467 | 0.452 | 0.270–0.631 | — | test set is 30 items; differences smaller than the CI width are not meaningful |
| ALIGN | published | — | 30 | 0.600 | 0.423 | 0.268–0.543 | — | test set is 30 items; differences smaller than the CI width are not meaningful |
| xlmr | 5-fold CV | all items | 198 | 0.424 | 0.421 | 0.349–0.488 | 0.350 ± 0.144 |  |
| BLIP | published | — | 30 | 0.467 | 0.415 | 0.240–0.583 | — | test set is 30 items; differences smaller than the CI width are not meaningful |
| FLAVA | published | — | 30 | 0.433 | 0.411 | 0.224–0.583 | — | test set is 30 items; differences smaller than the CI width are not meaningful |
| CLIP | published | — | 30 | 0.433 | 0.386 | 0.209–0.573 | — | test set is 30 items; differences smaller than the CI width are not meaningful |
| bangla_electra | 5-fold CV | all items | 198 | 0.374 | 0.372 | 0.308–0.436 | 0.333 ± 0.047 |  |
| mt5 | 5-fold CV | all items | 198 | 0.485 | 0.367 | 0.305–0.429 | 0.341 ± 0.072 |  |
| CountVec_ViT | published | — | 30 | 0.300 | 0.281 | 0.129–0.438 | — | test set is 30 items; differences smaller than the CI width are not meaningful |
| majority | 5-fold CV | all items | 198 | 0.520 | 0.228 | 0.207–0.248 | 0.228 ± 0.002 | model predicted a single class for every item (degenerate) |
| XLMRoBERTa | published | — | 47 | 0.362 | 0.215 | 0.136–0.300 | — | test set is 47 items; differences smaller than the CI width are not meaningful |
| mt5_predictions | published | — | 47 | 0.319 | 0.161 | 0.107–0.213 | — | test set is 47 items; differences smaller than the CI width are not meaningful; model predicted a single class for every item (degenerate) |
