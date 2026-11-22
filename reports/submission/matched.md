| Population | Model | Status | n | Accuracy | Pooled macro-F1 | 95% CI | Fold mean ± SD | F1 gc | F1 neu | F1 gl | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| paired_headlines | majority | ok | 149 | 0.483 | 0.217 | 0.191–0.239 | 0.217 ± 0.004 | 0.652 | 0.000 | 0.000 | model predicted a single class for every item (degenerate) |
| paired_headlines | tfidf_logreg | ok | 149 | 0.570 | 0.543 | 0.457–0.619 | 0.535 ± 0.103 | 0.627 | 0.588 | 0.413 |  |
| paired_headlines | banglabert | ok | 149 | 0.450 | 0.367 | 0.285–0.448 | 0.339 ± 0.125 | 0.568 | 0.194 | 0.340 |  |
| paired_headlines | bangla_electra | ok | 149 | 0.409 | 0.380 | 0.300–0.457 | 0.312 ± 0.108 | 0.483 | 0.286 | 0.372 |  |
| paired_headlines | mbert | ok | 149 | 0.463 | 0.446 | 0.359–0.529 | 0.408 ± 0.112 | 0.510 | 0.450 | 0.377 |  |
| paired_headlines | xlmr | ok | 149 | 0.423 | 0.344 | 0.265–0.421 | 0.277 ± 0.049 | 0.555 | 0.203 | 0.273 |  |
| paired_headlines | mt5 | ok | 149 | 0.463 | 0.361 | 0.283–0.436 | 0.333 ± 0.077 | 0.611 | 0.138 | 0.333 |  |
| paired_headlines | countvec_vit | ok | 149 | 0.470 | 0.425 | 0.345–0.508 | 0.410 ± 0.135 | 0.584 | 0.278 | 0.414 |  |
| paired_headlines | clip | ok | 149 | 0.436 | 0.387 | 0.308–0.463 | 0.356 ± 0.101 | 0.558 | 0.239 | 0.364 |  |
| paired_headlines | align | ok | 149 | 0.362 | 0.341 | 0.267–0.414 | 0.314 ± 0.053 | 0.459 | 0.321 | 0.244 |  |
| paired_headlines | blip | ok | 149 | 0.342 | 0.275 | 0.214–0.332 | 0.196 ± 0.074 | 0.471 | 0.041 | 0.312 |  |
| paired_headlines | vilt | ok | 149 | 0.436 | 0.340 | 0.260–0.415 | 0.275 ± 0.104 | 0.584 | 0.226 | 0.209 |  |
| paired_headlines | flava | ok | 149 | 0.349 | 0.325 | 0.252–0.399 | 0.283 ± 0.105 | 0.460 | 0.194 | 0.320 |  |
| all_headlines | majority | ok | 198 | 0.520 | 0.228 | 0.207–0.248 | 0.228 ± 0.002 | 0.684 | 0.000 | 0.000 | model predicted a single class for every item (degenerate) |
| all_headlines | tfidf_logreg | ok | 198 | 0.566 | 0.496 | 0.415–0.569 | 0.487 ± 0.085 | 0.675 | 0.457 | 0.356 |  |
| all_headlines | banglabert | ok | 198 | 0.465 | 0.366 | 0.298–0.437 | 0.357 ± 0.090 | 0.598 | 0.244 | 0.257 |  |
| all_headlines | bangla_electra | ok | 198 | 0.364 | 0.346 | 0.279–0.411 | 0.311 ± 0.131 | 0.307 | 0.478 | 0.253 |  |
| all_headlines | mbert | ok | 198 | 0.556 | 0.491 | 0.416–0.564 | 0.483 ± 0.090 | 0.647 | 0.463 | 0.364 |  |
| all_headlines | xlmr | ok | 198 | 0.460 | 0.383 | 0.316–0.454 | 0.327 ± 0.065 | 0.602 | 0.250 | 0.297 |  |
| all_headlines | mt5 | ok | 198 | 0.434 | 0.342 | 0.274–0.409 | 0.320 ± 0.034 | 0.584 | 0.195 | 0.247 |  |

Paired macro-F1 difference (vision-language minus text), readable-image items, source-group bootstrap with 2000 resamples:

| Model | Text reference | n | Groups | Δ macro-F1 | 95% CI | Share Δ>0 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| countvec_vit | tfidf_logreg | 149 | 148 | -0.117 | -0.211 to -0.023 | 0.006 |
| countvec_vit | banglabert | 149 | 148 | +0.058 | -0.059 to +0.175 | 0.835 |
| clip | tfidf_logreg | 149 | 148 | -0.156 | -0.258 to -0.044 | 0.002 |
| clip | banglabert | 149 | 148 | +0.020 | -0.095 to +0.133 | 0.637 |
| align | tfidf_logreg | 149 | 148 | -0.201 | -0.317 to -0.083 | 0.001 |
| align | banglabert | 149 | 148 | -0.026 | -0.131 to +0.081 | 0.328 |
| blip | tfidf_logreg | 149 | 148 | -0.268 | -0.366 to -0.166 | 0.000 |
| blip | banglabert | 149 | 148 | -0.092 | -0.200 to +0.014 | 0.042 |
| vilt | tfidf_logreg | 149 | 148 | -0.203 | -0.315 to -0.088 | 0.001 |
| vilt | banglabert | 149 | 148 | -0.027 | -0.133 to +0.084 | 0.316 |
| flava | tfidf_logreg | 149 | 148 | -0.218 | -0.327 to -0.105 | 0.000 |
| flava | banglabert | 149 | 148 | -0.042 | -0.148 to +0.065 | 0.230 |
