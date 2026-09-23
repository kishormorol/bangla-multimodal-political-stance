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
| banglabert | 5-fold CV | all items | 198 | 0.576 | 0.533 | 0.458–0.606 | 0.522 ± 0.062 |  |
| clip | 5-fold CV | items with an image | 86 | 0.616 | 0.516 | 0.396–0.627 | 0.499 ± 0.163 | test set is 86 items; differences smaller than the CI width are not meaningful |
| tfidf_logreg | 5-fold CV | all items | 198 | 0.551 | 0.501 | 0.429–0.572 | 0.490 ± 0.091 |  |
| mbert | 5-fold CV | all items | 198 | 0.520 | 0.498 | 0.421–0.568 | 0.496 ± 0.087 |  |
| mt5 | 5-fold CV | all items | 198 | 0.520 | 0.497 | 0.427–0.569 | 0.481 ± 0.078 |  |
| countvec_vit | 5-fold CV | items with an image | 86 | 0.465 | 0.431 | 0.312–0.543 | 0.406 ± 0.093 | test set is 86 items; differences smaller than the CI width are not meaningful |
| xlmr | 5-fold CV | all items | 198 | 0.424 | 0.421 | 0.349–0.488 | 0.350 ± 0.144 |  |
| align | 5-fold CV | items with an image | 86 | 0.419 | 0.407 | 0.294–0.510 | 0.376 ± 0.049 | test set is 86 items; differences smaller than the CI width are not meaningful |
| bangla_electra | 5-fold CV | all items | 198 | 0.374 | 0.372 | 0.308–0.436 | 0.333 ± 0.047 |  |
| blip | 5-fold CV | items with an image | 86 | 0.430 | 0.366 | 0.268–0.464 | 0.326 ± 0.043 | test set is 86 items; differences smaller than the CI width are not meaningful |
| vilt | 5-fold CV | items with an image | 86 | 0.372 | 0.336 | 0.238–0.433 | 0.242 ± 0.149 | test set is 86 items; differences smaller than the CI width are not meaningful |
| flava | 5-fold CV | items with an image | 86 | 0.395 | 0.297 | 0.222–0.368 | 0.232 ± 0.051 | test set is 86 items; differences smaller than the CI width are not meaningful |
| majority | 5-fold CV | all items | 198 | 0.520 | 0.228 | 0.207–0.248 | 0.228 ± 0.002 | model predicted a single class for every item (degenerate) |
