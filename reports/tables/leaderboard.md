# Results

Macro-F1 is the headline metric; the 95% interval is a 2000-round item bootstrap.
Rows marked `published` are scored from the prediction files in the Drive folder
and were produced on the original, leakage-affected splits — see data/README.md.

| Model | Source | n | Accuracy | Macro-F1 | 95% CI | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| BanglaBERT | published | 47 | 0.723 | 0.719 | 0.578–0.836 | test set is 47 items; differences smaller than the CI width are not meaningful |
| mBERT | published | 47 | 0.638 | 0.649 | 0.509–0.775 | test set is 47 items; differences smaller than the CI width are not meaningful |
| tfidf_logreg | this repo | 29 | 0.586 | 0.554 | 0.332–0.731 | test set is 29 items; differences smaller than the CI width are not meaningful |
| BanglaELECTRA | published | 47 | 0.468 | 0.462 | 0.308–0.596 | test set is 47 items; differences smaller than the CI width are not meaningful |
| ViLT | published | 30 | 0.467 | 0.452 | 0.270–0.631 | test set is 30 items; differences smaller than the CI width are not meaningful |
| ALIGN | published | 30 | 0.600 | 0.423 | 0.268–0.543 | test set is 30 items; differences smaller than the CI width are not meaningful |
| BLIP | published | 30 | 0.467 | 0.415 | 0.240–0.583 | test set is 30 items; differences smaller than the CI width are not meaningful |
| FLAVA | published | 30 | 0.433 | 0.411 | 0.224–0.583 | test set is 30 items; differences smaller than the CI width are not meaningful |
| CLIP | published | 30 | 0.433 | 0.386 | 0.209–0.573 | test set is 30 items; differences smaller than the CI width are not meaningful |
| CountVec_ViT | published | 30 | 0.300 | 0.281 | 0.129–0.438 | test set is 30 items; differences smaller than the CI width are not meaningful |
| XLMRoBERTa | published | 47 | 0.362 | 0.215 | 0.136–0.300 | test set is 47 items; differences smaller than the CI width are not meaningful |
| mt5_predictions | published | 47 | 0.319 | 0.161 | 0.107–0.213 | test set is 47 items; differences smaller than the CI width are not meaningful; model predicted a single class for every item (degenerate) |
| majority | this repo | 29 | 0.276 | 0.144 | 0.081–0.206 | test set is 29 items; differences smaller than the CI width are not meaningful; model predicted a single class for every item (degenerate) |
