| Population | Model | Seeds | n | Mean macro-F1 | SD | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all_headlines | bangla_electra | 5 | 198 | 0.350 | 0.042 | 0.279 | 0.379 |
| all_headlines | banglabert | 5 | 198 | 0.412 | 0.037 | 0.364 | 0.453 |
| all_headlines | majority | 5 | 198 | 0.228 | 0.000 | 0.228 | 0.228 |
| all_headlines | mbert | 5 | 198 | 0.452 | 0.050 | 0.406 | 0.521 |
| all_headlines | mt5 | 5 | 198 | 0.370 | 0.048 | 0.311 | 0.432 |
| all_headlines | tfidf_logreg | 5 | 198 | 0.505 | 0.018 | 0.480 | 0.529 |
| all_headlines | xlmr | 5 | 198 | 0.361 | 0.028 | 0.327 | 0.393 |
| paired_headlines | align | 5 | 149 | 0.337 | 0.043 | 0.286 | 0.386 |
| paired_headlines | align_blank | 5 | 149 | 0.342 | 0.014 | 0.324 | 0.355 |
| paired_headlines | align_shuffled | 5 | 149 | 0.326 | 0.028 | 0.281 | 0.348 |
| paired_headlines | bangla_electra | 5 | 149 | 0.349 | 0.046 | 0.299 | 0.404 |
| paired_headlines | banglabert | 5 | 149 | 0.358 | 0.036 | 0.309 | 0.398 |
| paired_headlines | blip | 5 | 149 | 0.320 | 0.029 | 0.275 | 0.350 |
| paired_headlines | blip_blank | 5 | 149 | 0.325 | 0.025 | 0.288 | 0.353 |
| paired_headlines | blip_shuffled | 5 | 149 | 0.319 | 0.015 | 0.297 | 0.337 |
| paired_headlines | clip | 5 | 149 | 0.422 | 0.058 | 0.364 | 0.492 |
| paired_headlines | clip_blank | 5 | 149 | 0.255 | 0.031 | 0.215 | 0.296 |
| paired_headlines | clip_shuffled | 5 | 149 | 0.315 | 0.030 | 0.274 | 0.351 |
| paired_headlines | countvec_vit | 5 | 149 | 0.405 | 0.024 | 0.381 | 0.434 |
| paired_headlines | countvec_vit_blank | 5 | 149 | 0.451 | 0.028 | 0.426 | 0.497 |
| paired_headlines | countvec_vit_shuffled | 5 | 149 | 0.395 | 0.047 | 0.325 | 0.439 |
| paired_headlines | flava | 5 | 149 | 0.301 | 0.023 | 0.276 | 0.325 |
| paired_headlines | flava_blank | 5 | 149 | 0.275 | 0.035 | 0.231 | 0.317 |
| paired_headlines | flava_shuffled | 5 | 149 | 0.289 | 0.029 | 0.268 | 0.336 |
| paired_headlines | majority | 5 | 149 | 0.217 | 0.000 | 0.217 | 0.217 |
| paired_headlines | mbert | 5 | 149 | 0.465 | 0.016 | 0.444 | 0.489 |
| paired_headlines | mt5 | 5 | 149 | 0.326 | 0.060 | 0.222 | 0.376 |
| paired_headlines | tfidf_logreg | 5 | 149 | 0.524 | 0.021 | 0.497 | 0.543 |
| paired_headlines | tfidf_vit | 5 | 149 | 0.401 | 0.007 | 0.395 | 0.411 |
| paired_headlines | tfidf_vit_shuffled | 5 | 149 | 0.331 | 0.040 | 0.281 | 0.374 |
| paired_headlines | vilt | 5 | 149 | 0.306 | 0.029 | 0.272 | 0.340 |
| paired_headlines | vilt_blank | 5 | 149 | 0.307 | 0.018 | 0.288 | 0.336 |
| paired_headlines | vilt_shuffled | 5 | 149 | 0.317 | 0.019 | 0.294 | 0.339 |
| paired_headlines | vit_only | 5 | 149 | 0.396 | 0.016 | 0.373 | 0.417 |
| paired_headlines | xlmr | 5 | 149 | 0.385 | 0.026 | 0.367 | 0.429 |

| VL model | Reference | Seeds | Mean Δ | SD | Min | Max | CI>0 | CI<0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| align | banglabert | 5 | -0.021 | 0.075 | -0.111 | +0.077 | 0 | 1 |
| align | tfidf_logreg | 5 | -0.187 | 0.050 | -0.234 | -0.112 | 0 | 4 |
| blip | banglabert | 5 | -0.039 | 0.049 | -0.112 | +0.013 | 0 | 1 |
| blip | tfidf_logreg | 5 | -0.205 | 0.043 | -0.268 | -0.166 | 0 | 5 |
| clip | banglabert | 5 | +0.064 | 0.079 | -0.034 | +0.151 | 2 | 0 |
| clip | tfidf_logreg | 5 | -0.102 | 0.049 | -0.156 | -0.047 | 0 | 3 |
| countvec_vit | banglabert | 5 | +0.047 | 0.030 | +0.005 | +0.078 | 0 | 0 |
| countvec_vit | tfidf_logreg | 5 | -0.119 | 0.023 | -0.158 | -0.102 | 0 | 4 |
| flava | banglabert | 5 | -0.058 | 0.032 | -0.109 | -0.030 | 0 | 0 |
| flava | tfidf_logreg | 5 | -0.224 | 0.016 | -0.251 | -0.211 | 0 | 5 |
| vilt | banglabert | 5 | -0.052 | 0.032 | -0.097 | -0.022 | 0 | 0 |
| vilt | tfidf_logreg | 5 | -0.218 | 0.028 | -0.267 | -0.203 | 0 | 5 |
