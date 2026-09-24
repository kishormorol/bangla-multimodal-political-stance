# Text models scored on the 87-item image subset

Apples-to-apples comparison surface: only the 87 items that have an image.
Predictions come from each text model's latest 5-fold CV run, filtered to this subset.
Bootstrap CI is 2000 rounds on macro-F1.

| Model | n | Accuracy | Macro-F1 | 95% CI | F1_gc | F1_n | F1_gl |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| bangla_electra | 87 | 0.425 | 0.410 | 0.310–0.506 | 0.235 | 0.559 | 0.436 |
| banglabert | 87 | 0.621 | 0.590 | 0.474–0.696 | 0.697 | 0.560 | 0.514 |
| majority | 87 | 0.460 | 0.210 | 0.175–0.240 | 0.630 | 0.000 | 0.000 |
| mbert | 87 | 0.494 | 0.486 | 0.372–0.589 | 0.543 | 0.500 | 0.415 |
| mt5 | 87 | 0.460 | 0.359 | 0.268–0.448 | 0.653 | 0.125 | 0.298 |
| tfidf_logreg | 87 | 0.621 | 0.549 | 0.440–0.646 | 0.713 | 0.667 | 0.267 |
| xlmr | 87 | 0.437 | 0.439 | 0.331–0.538 | 0.385 | 0.545 | 0.388 |
