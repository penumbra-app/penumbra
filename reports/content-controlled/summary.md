# Controlled content-model experiment

Same enriched movie snapshot, 603 users and 20,156 test ratings. Choices were frozen on validation before this run's test evaluation. The historical holdout had already been inspected, so results are exploratory.

| Family | Validation pairwise | Test pairwise | NDCG@10 | RMSE | MAE | Text scoring seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 59.972% | 57.469% | 0.773654 | 1.019924 | 0.776546 | 6.984 |
| shared_catalog | 59.343% | 57.580% | 0.777279 | 1.021399 | 0.777759 | 0.545 |
| separate_fields | 59.579% | 57.579% | 0.776357 | 1.021181 | 0.777577 | 0.556 |
| semantic | 59.214% | 57.604% | 0.775363 | 1.021052 | 0.777885 | 0.474 |
| ridge | 59.427% | 57.435% | 0.774776 | 1.021397 | 0.777985 | 5.876 |
| top_k | 59.959% | 57.452% | 0.775004 | 1.019677 | 0.776337 | 5.755 |

## Validation choices

Overall choice: `personal_tfidf`.

Best representation: `personal_tfidf`.

- baseline: `personal_tfidf`
- shared_catalog: `shared_tfidf`
- separate_fields: `fields-plot0.25`
- semantic: `qwen`
- ridge: `personal_tfidf+ridge-a10`
- top_k: `personal_tfidf+top50`

## Paired differences against current selected baseline

Percentage-point changes in pairwise accuracy; 10,000 user bootstrap resamples, seed 42. The 99% intervals apply a Bonferroni correction for the five primary family comparisons.

| Family | Change (pp) | 95% interval (pp) | 99% interval (pp) |
| --- | ---: | --- | --- |
| shared_catalog | +0.111 | [-0.451, +0.677] | [-0.643, +0.848] |
| separate_fields | +0.109 | [-0.477, +0.718] | [-0.684, +0.897] |
| semantic | +0.135 | [-0.583, +0.867] | [-0.823, +1.078] |
| ridge | -0.034 | [-0.461, +0.397] | [-0.595, +0.537] |
| top_k | -0.017 | [-0.061, +0.025] | [-0.075, +0.036] |

## Timing and interpretation

Text scoring timings include per-user fitting, matrix selection and scoring for all held-out test candidates; they exclude the common structured core, metric computation, shared-index construction and one-off embedding inference. They are single-run batch timings, not end-to-end request latency or full-catalog serving benchmarks. `results.json` records index build and encoder times separately.

Core settings, text weight/prior/cap and recency are fixed. Ridge replaces only the text adjustment and uses the same rating residual target; it is not a complete replacement of the structured model. Semantic cosine is clipped at zero as in the existing LSA scorer.

Shared IDF uses the full static catalog and is transductive. Descriptions are retrospective TMDB metadata; timestamp-filtered other-user tags are identical across methods. No held-out ratings enter representation fitting, user fitting or selection. Candidate texts pass through frozen transforms only. Evaluation ranks held-out rated movies, not all movies.

NDCG, RMSE and MAE intervals are secondary exploratory analyses; the primary selection objective remains pairwise accuracy. All intervals and history-size slices are available in `results.json`. The serving preset has not been changed.
