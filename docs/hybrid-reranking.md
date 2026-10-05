# Hybrid and Reranking

## Purpose

Penumbra's hybrid layer decides which movies should appear first after combining evidence from the project's personalization systems.

The hybrid layer currently works with five primary ranking signals:

- `collaborative`: matrix-factorization prediction
- `personal`: the older genre-based personal preference signal
- `content`: standalone content-model prediction
- `quality`: Bayesian-adjusted movie-quality signal
- `popularity`: `ln(1 + rating_count)`

The content system also exposes `content_confidence`, which is useful for experiments involving confidence-aware ranking.

The main ranking model remains a learned pairwise logistic reranker. Week 6 also tested pointwise gradient-boosted trees as a nonlinear alternative, but the GBT family did not beat the logistic baseline on validation and was rejected.

---

# Current Ranking Architecture

Conceptually:

`Content + Collaborative + Personal + Quality + Popularity -> Learned Reranker -> Relevance Ranking -> Diversity Reranking -> Top-K`

For a single user:

1. Build the user's historical profile from the allowed chronological training portion.
2. Produce collaborative, personal, content, quality, and popularity signals for candidate movies.
3. Convert candidate signals into the feature representation expected by the learned reranker.
4. Produce a relevance score for each unseen movie.
5. Retain a larger high-relevance candidate pool.
6. Apply genre-based diversity reranking.
7. Return the final Top-K recommendations.

For group recommendations, Week 7 will consume the final individual-user scores produced by this pipeline.

---

# Inputs

Penumbra currently evaluates on MovieLens latest-small.

Base inputs include:

- User ID
- User ratings
- Rating timestamps
- Movie IDs
- Movie titles
- Movie genres
- Population rating history

Week 6 additionally uses IMDb-derived metadata to enrich the movie table.

The enriched content dataset contains coverage for:

- Director: 9,717 / 9,742 movies
- Runtime: 9,718 / 9,742
- Release era: 9,742 / 9,742
- Cast: 9,366 / 9,742
- Language: unavailable from the current IMDb import
- Unavailable IMDb matches: 23 movies

The generated enriched dataset is:

`data/movies-enriched.csv`

The selected standalone content configuration is:

`without_language`

The reproducible content report is:

`docs/week6-content-repro.json`

---

# Outputs

The learned ranking path produces a ranking score for each candidate movie.

Relevant output information can include:

- `movie_id`
- `title`
- `genres`
- learned ranking score
- `personal_score`
- `content_score`
- `content_confidence`
- `collaborative_score`
- `quality_score`
- `popularity`

The learned reranker score is a relative ranking score. It is not a calibrated 1-5 star prediction.

---

# Ranking Signals

## Personal Preference

The personal score captures whether the user historically rates the candidate's genres above or below the user's own normal rating level.

It uses information such as:

- genre overlap
- rating recency
- release-year similarity
- amount of historical evidence

This signal is weaker than MF as a standalone predictor but has repeatedly shown complementary ranking information.

---

## Content Score

The standalone content model builds a user taste profile from movie metadata.

Week 6 content scoring uses the validated `without_language` configuration with IMDb-enriched metadata.

The content model can use:

- genre
- director
- runtime
- release era
- cast
- language when available

The content prediction is separate from the older personal genre score.

The content model also returns a confidence value describing how well-supported the prediction is.

---

## Content Confidence

Content confidence measures how much evidence supports a content-model prediction.

Week 5 analysis showed that content accuracy increased with confidence, motivating Week 6 experiments that attempted to use confidence inside the final reranker.

A simple linear confidence feature did not provide a clean improvement.

Week 6 therefore also tested GBTs because nonlinear trees can theoretically learn interactions such as:

`high content score + high content confidence -> stronger positive effect`

However, the tested pointwise GBT models still underperformed the pairwise logistic baseline on validation.

Content confidence is therefore retained as a useful diagnostic and future experimental signal, but it is not part of the selected logistic feature set.

---

## Collaborative Score

The collaborative signal is produced by biased matrix factorization.

The selected MF configuration is:

- Latent factors: `20`
- Learning rate: `0.005`
- Regularization: `0.02`
- Epochs: `40`
- Prior strength: `5.0`
- Random seed: `42`
- `shrink_latent=True`

MF learns:

- global rating behavior
- user bias
- movie bias
- latent user factors
- latent movie factors

The predicted collaborative score is rating-like and becomes one feature in the hybrid reranker.

---

## Quality Score

Quality is based on a Bayesian-adjusted movie rating.

Conceptually:

`quality_rating = (rating_count * movie_average + prior_strength * global_average) / (rating_count + prior_strength)`

and:

`quality_score = quality_rating - global_average`

MF and Quality are highly redundant, but Week 6 ablations showed that removing Quality from the strongest five-signal logistic hybrid still reduced macro and pair-weighted ranking performance.

Quality therefore remains in the selected hybrid feature set.

---

## Popularity

Popularity is:

`popularity = ln(1 + rating_count)`

The logarithm prevents extremely popular movies from dominating simply because they have much larger raw rating counts.

---

# Pairwise Logistic Reranking

## Training Objective

The logistic reranker learns from pairwise preference differences.

For a preferred movie `P` and less-preferred movie `L`:

`X = features(P) - features(L), label = 1`

The reverse example is also included:

`X = features(L) - features(P), label = 0`

Pairs with equal ratings are skipped.

This creates a balanced pairwise classification dataset whose objective is directly aligned with relative ranking.

---

## Feature Scaling

Pairwise feature differences are standardized using `StandardScaler`.

The fitted scaler is stored with the trained logistic-regression model and reused at inference time.

---

## Chronological Information Boundary

For the final linear hybrid experiments:

- First 60% of each eligible user's ratings: user profile and upstream models
- Next 20%: pairwise reranker training
- Final 20%: held-out evaluation

The split is deterministic.

Ratings are ordered by timestamp, with movie ID used as a deterministic tie-breaker.

Upstream personalization models are built inside the permitted information boundary:

- Content sees the user's first-60% profile only.
- Hybrid MF is trained from eligible users' first-60% profiles.
- Population features exclude the target user's own ratings where required.
- Final held-out ratings do not become upstream evidence.

---

# Week 6 Advanced Improvements

## 1. IMDb-Enriched Content Integration

Week 6 first replaced the limited base movie metadata with the IMDb-enriched movie table.

The standalone content evaluator selected:

`without_language`

This configuration is recorded in:

`docs/week6-content-repro.json`

The enriched content data improved the hybrid enough to justify retaining the richer content pipeline for later experiments.

---

## 2. Linear Content-Confidence Experiment

A five-feature linear experiment tested:

`Content + Content Confidence + MF + Quality + Popularity`

The learned confidence coefficient was very small.

Final held-out results were:

- Macro pairwise: `63.547%`
- Pair-weighted: `68.056%`
- NDCG@10: `0.9069`

The raw confidence feature did not produce a clean improvement and was not promoted.

The main interpretation is that an independent linear confidence term is not the same as learning an interaction between confidence and the content prediction.

---

## 3. Personal Signal Reintroduction

Week 5 diagnostics showed that Personal often rescued errors made by stronger standalone signals.

Week 6 therefore tested Personal inside the enriched hybrid.

The five-signal logistic model uses:

`MF + Personal + Content + Quality + Popularity`

Its standardized coefficients in the Week 6 training run were:

| Feature | Coefficient |
| --- | ---: |
| Collaborative / MF | +0.690199 |
| Personal | +0.147118 |
| Content | +0.472316 |
| Quality | +0.076812 |
| Popularity | +0.261745 |

These are standardized logistic coefficients and should not be interpreted as raw feature weights.

---

# Week 6 Linear Ablation Study

All Week 6 linear ablations use the IMDb-enriched movie table and the validated content configuration.

Users evaluated: `600`

Held-out preference pairs: `633,032`

| Configuration | Macro | Pair-Weighted | NDCG@10 |
| --- | ---: | ---: | ---: |
| **MF + Personal + Content + Quality + Popularity** | **63.645%** | **68.648%** | **0.9080** |
| MF + Personal + Content + Popularity | 63.445% | 68.556% | **0.9080** |
| MF + Content + Quality + Popularity | 63.419% | 68.503% | 0.9066 |
| MF + Content + Popularity | 63.307% | 68.423% | 0.9061 |
| Personal + Content + Quality + Popularity | 63.293% | 68.529% | 0.9070 |
| Content + Quality + Popularity | 63.253% | 68.401% | 0.9064 |

Main findings:

- Reintroducing Personal improved the enriched hybrid.
- Removing Quality hurt the five-signal model slightly, despite strong MF/Quality redundancy.
- Removing MF also hurt performance.
- The five-signal logistic model is the strongest Week 6 hybrid across the combined metric profile.
- No otherwise-identical `no_content` linear ablation was run, so the marginal contribution of Content inside the five-signal logistic model was not isolated directly.

The original three-feature ML reranker remains an important reference:

| Model | Macro | Pair-Weighted | NDCG@10 |
| --- | ---: | ---: | ---: |
| Original ML: Personal + Quality + Popularity | **63.876%** | 68.516% | 0.9051 |
| Week 6 five-signal Hybrid | 63.645% | **68.648%** | **0.9080** |

There is therefore no strict winner across all three metrics:

- Original ML retains a `0.231 pp` macro advantage.
- Week 6 Hybrid gains `0.132 pp` pair-weighted accuracy.
- Week 6 Hybrid gains `0.0029` NDCG@10.

For Week 6 hybrid development, the five-signal logistic model is the selected advanced hybrid candidate because it is the strongest tested hybrid and improves two of the three ranking metrics relative to the original ML benchmark.

---

# Week 6 Gradient-Boosted Tree Experiment

## Motivation

The Week 6 roadmap called for testing a stronger nonlinear reranker.

GBT was attractive because nonlinear trees can theoretically learn interactions that pairwise linear logistic regression cannot represent directly.

The initial GBT feature set was:

- collaborative / MF
- personal
- content
- content confidence
- quality
- popularity

Unlike the logistic reranker, the first GBT implementation was pointwise.

Each development movie produced one feature row and the target was the user's actual rating.

The model used `GradientBoostingRegressor`.

---

## GBT Development Split

GBT model selection did not use the final held-out 20%.

For GBT validation, the timeline was:

- First 60%: user profile and upstream models
- Next 10%: GBT / validation-logistic training
- Next 10%: GBT validation
- Final 20%: untouched

The logistic comparator was retrained on the same development-training portion so the comparison was apples-to-apples.

Validation evaluation contained:

- Users evaluated: `571`
- Preference pairs: `167,009`

Because this validation slice is smaller and earlier than the final test slice, its absolute NDCG values should not be compared directly with the final-20% NDCG values.

---

## Full GBT Training Diagnostics

The first six-feature GBT produced these feature importances:

| Feature | Importance |
| --- | ---: |
| Collaborative / MF | 0.794176 |
| Personal | 0.094723 |
| Content | 0.073066 |
| Quality | 0.015759 |
| Popularity | 0.012368 |
| Content Confidence | 0.009908 |

Feature importance describes how the tree used features for splitting. It is not equivalent to a feature's marginal contribution to ranking metrics.

---

## GBT Validation Ablations

Five GBT configurations were evaluated on the same validation slice:

| Model | Macro | Pair-Weighted | NDCG@10 |
| --- | ---: | ---: | ---: |
| **Validation Logistic** | **64.460%** | **68.547%** | **0.9327** |
| Full GBT | 63.506% | 67.818% | 0.9314 |
| GBT - Confidence | 63.654% | 67.831% | 0.9312 |
| GBT - Quality | 63.238% | 67.773% | 0.9307 |
| GBT - Popularity | 64.155% | 67.603% | 0.9320 |
| GBT - Content | 63.572% | 67.610% | 0.9305 |

Every tested GBT configuration lost to the pairwise logistic comparator on all three validation metrics.

The closest macro/NDCG variant was `GBT - Popularity`:

- Macro delta vs Logistic: `-0.306 pp`
- Pair-weighted delta: `-0.944 pp`
- NDCG@10 delta: `-0.0007`

The strongest GBT by pair-weighted accuracy was `GBT - Confidence`, but it still lost:

- Macro delta: `-0.806 pp`
- Pair-weighted delta: `-0.716 pp`
- NDCG@10 delta: `-0.0015`

---

## GBT Decision

GBT is rejected for Week 6.

The nonlinear model did not provide evidence that it should replace the pairwise logistic reranker.

The most likely structural explanation is that the logistic model is trained directly on pairwise preference ordering, while the tested GBT is a pointwise rating regressor whose training objective is only indirectly related to the final ranking metrics.

Because GBT failed on validation, Penumbra did not train or evaluate a GBT on the final held-out 20%.

The final test set therefore remained unused for GBT model selection.

---

# Diversity Reranking

After learned relevance ranking, Penumbra retains a larger candidate pool and greedily selects a diverse final slate.

Current candidate-pool rule:

`candidate_pool_size = 5 * requested_recommendation_count`

Genre similarity uses Jaccard similarity:

`similarity(A, B) = |A intersection B| / |A union B|`

For each candidate:

`adjusted_score = original_ml_score - diversity_weight * maximum_genre_similarity`

Current diversity weight:

`0.15`

The returned recommendation score remains the original learned relevance score. The diversity-adjusted score is temporary and depends on the slate already selected.

---

# Current Week 6 Decision

The Week 6 reranking conclusion is:

**Keep pairwise logistic regression. Reject the tested pointwise GBT family.**

Selected Week 6 hybrid feature set:

`MF + Personal + Content + Quality + Popularity`

Final held-out metrics for that hybrid:

- Macro pairwise: `63.645%`
- Pair-weighted accuracy: `68.648%`
- NDCG@10: `0.9080`

GBT was evaluated only on the dedicated validation slice and was not promoted to final testing.

---

# Reproducibility

The hybrid pipeline uses:

- deterministic chronological splitting
- deterministic movie-ID tie-breaking
- explicit random seeds
- leakage-safe profile / training / evaluation boundaries
- separately saved experimental artifacts
- shared evaluation candidates for ablation comparisons
- a dedicated validation slice for Week 6 GBT model selection

Earlier one-off Week 6 hybrid evaluations used an evaluation-time MF configuration that did not exactly match the selected 40-epoch `shrink_latent=True` MF used during training.

Those one-off results are superseded by the corrected Week 6 ablation benchmark documented above, which uses a matching MF configuration.

---

# Known Limitations

- No otherwise-identical linear `no_content` ablation was run for the final five-signal logistic Hybrid.
- Content confidence showed diagnostic value but did not improve the tested linear or pointwise-GBT rerankers.
- The tested GBT used a pointwise regression objective rather than a ranking-specific boosted-tree objective.
- MovieLens latest-small is much smaller and less metadata-rich than a production movie catalog.
- IMDb enrichment still leaves some metadata unavailable, especially language.
- Offline ratings do not capture short-term viewing intent.
- Popularity can bias the system toward widely rated movies.
- Diversity still uses explicit genre overlap rather than semantic similarity.
- Final model selection involves several ranking metrics, and no single model currently dominates every metric.
- Evaluation on MovieLens does not directly represent real-world user satisfaction.

---

# Relevant Files

## Core Hybrid

- `src/hybrid/genre_recommender.py`
- `src/hybrid/ml_reranker.py`
- `src/hybrid/content_adapter.py`
- `src/hybrid/diversity.py`
- `src/hybrid/recommender.py`
- `src/hybrid/gbt_reranker.py`

## Training

- `src/scripts/train_ml_reranker.py`
- `src/scripts/train_hybrid_reranker.py`
- `src/scripts/week3_ablations.py`
- `src/scripts/week6_ablations.py`
- `src/scripts/train_week6_gbt.py`
- `src/scripts/week6_gbt_ablations.py`

## Evaluation

- `src/evaluation/ranking.py`
- `src/evaluation/splits.py`
- `src/evaluation/metrics.py`
- `src/evaluation/reports.py`
- `src/scripts/evaluate_ranking.py`
- `src/scripts/evaluate_week3_ablations.py`
- `src/scripts/evaluate_week6_ablations.py`
- `src/scripts/evaluate_week6_gbt.py`
- `src/scripts/evaluate_week6_gbt_ablations.py`

## Week 6 Reports

- `docs/week6-content-repro.json`
- `docs/week6-ablation-results.json`
- `docs/week6-gbt-ablation-results.json`

## Saved Models

Historical original reranker:

- `src/models/ml_reranker.joblib`

Historical Hybrid V1:

- `src/models/hybrid_v1_reranker.joblib`

Week 3 ablations:

- `src/models/ablations/`

Week 6 linear ablations:

- `src/models/week6_ablations/`

Week 6 GBT validation ablations:

- `src/models/week6_gbt_ablations/`

The GBT artifacts are experimental validation models and are not selected production rerankers.

---

# Benchmark History

## Week 6 Linear Hybrid

Users evaluated: `600`

Test pairs: `633,032`

| Model | Macro | Pair-Weighted | NDCG@10 |
| --- | ---: | ---: | ---: |
| Original ML | **63.876%** | 68.516% | 0.9051 |
| **Week 6 five-signal Hybrid** | 63.645% | **68.648%** | **0.9080** |

## Week 6 GBT Validation

Users evaluated: `571`

Validation pairs: `167,009`

| Model | Macro | Pair-Weighted | NDCG@10 |
| --- | ---: | ---: | ---: |
| **Validation Logistic** | **64.460%** | **68.547%** | **0.9327** |
| Best GBT macro/NDCG: GBT - Popularity | 64.155% | 67.603% | 0.9320 |

GBT was rejected on validation and was not evaluated on the final 20%.

## Week 5 Primary Evaluation

Users evaluated: `600`

Test pairs: `633,032`

| Model | Macro | Pair-Weighted | NDCG@10 |
| --- | ---: | ---: | ---: |
| Movie-average baseline | 62.608% | 67.333% | 0.9002 |
| Heuristic | 60.633% | 65.130% | 0.8986 |
| Original ML | **63.876%** | **68.516%** | **0.9051** |
| Matrix Factorization | 62.634% | 67.265% | 0.8988 |
| Content | 57.024% | 61.965% | 0.8929 |
| Hybrid V1 | 63.287% | 67.872% | 0.9050 |

Week 5 established that complementary information matters more than simply adding individually strong models.

## Week 3

Week 3 integrated standalone Content and MF into the reranker and established the first complete hybrid architecture.

Those historical models and ablations remain preserved for reproducibility, but Week 6 supersedes them as the current advanced-hybrid benchmark.
