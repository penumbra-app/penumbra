# Evaluation

## Purpose

Penumbra uses offline evaluation to determine whether changes to the recommendation system improve ranking quality on unseen user preferences.

Evaluation is treated as a model-selection tool rather than simply a reporting step. New recommendation signals and hybrid configurations are compared against existing baselines under the same held-out evaluation procedure. A more complex model is not adopted unless the evaluation provides evidence that it improves recommendation quality.

Week 5 expanded the evaluation framework beyond pairwise ranking accuracy to include Top-K ranking quality, rating-prediction error, signal diagnostics, ablation analysis, and targeted error analysis.

---

## Datasets

Current evaluation uses the MovieLens latest-small dataset.

Dataset size:

- Ratings: 100,836
- Users: 610
- Rated movies: 9,724
- Movie rows: 9,742

MovieLens provides explicit user ratings, timestamps, movie IDs, titles, and genres.

The content model supports additional metadata including director, runtime, release era, and language. However, the base MovieLens movies dataset does not provide all of these fields, which limits the strength of the currently evaluated content model.

---

## Data Splitting

Evaluation uses a deterministic chronological split for each eligible user.

Ratings are ordered by timestamp and divided into:

- First 60%: profile / upstream model training
- Next 20%: pairwise reranker training
- Final 20%: held-out evaluation

Movie ID is used as a deterministic tie-breaker when ratings share the same timestamp.

The final 20% is not used to construct the user's profile, train the reranker, or provide upstream personalization evidence.

For hybrid evaluation, content profiles and collaborative models are constructed inside the permitted training boundary before predictions are generated for held-out movies.

This structure attempts to simulate the realistic task of predicting future preferences from past behavior while preventing held-out ratings from leaking into model inputs.

---

## Metrics

### Pairwise Ranking Accuracy

For every pair of held-out movies with different user ratings, the model is evaluated on whether it ranks the more highly rated movie above the less highly rated movie.

Credit is:

- 1.0 for the correct ordering
- 0.0 for the incorrect ordering
- 0.5 for a predicted tie

Two aggregates are reported.

**Macro pairwise accuracy** calculates accuracy independently for each user and then averages across users. Every evaluated user therefore contributes equally.

**Pair-weighted accuracy** aggregates decisions across all held-out pairs. Users contributing more valid pairs consequently have more influence.

### NDCG@10

Normalized Discounted Cumulative Gain at 10 measures the quality of the model's top-ranked held-out movies.

Unlike pairwise accuracy, NDCG@10 gives greater importance to placing highly relevant movies near the top of the recommendation ranking.

Higher values are better, with 1.0 representing the ideal ordering for the evaluated candidates.

### MAE

Mean Absolute Error measures the average absolute difference between predicted ratings and actual ratings.

Lower values are better.

MAE is reported only for models that produce rating-like predictions.

### RMSE

Root Mean Squared Error measures rating-prediction error while penalizing large errors more heavily than MAE.

Lower values are better.

RMSE is reported only for models that produce rating-like predictions.

Ranking-only logistic rerankers do not produce calibrated rating predictions, so MAE and RMSE are not reported for those models.

---

## Baselines

The evaluation suite currently compares:

**Movie-average baseline**

Uses population-level movie rating information and provides a simple non-personalized reference point.

**Heuristic recommender**

Combines the user's historical personal genre preference with Bayesian-adjusted movie quality.

**Original ML reranker**

Pairwise logistic regression using:

- personal preference
- quality
- popularity

This remains the strongest currently evaluated ranking configuration.

**Matrix Factorization**

Biased matrix factorization using user biases, movie biases, and latent user/movie factors.

**Content Model**

Predicts preference from the user's learned movie-metadata preferences.

**Hybrid V1**

Pairwise logistic regression using:

- content prediction
- collaborative matrix-factorization prediction
- quality
- popularity

Additional feature combinations are evaluated separately through ablation experiments.

---

## Evaluation Procedure

For each eligible user:

1. Ratings are chronologically split into profile, pairwise-training, and held-out evaluation portions.
2. Required user profiles and upstream models are constructed using only permitted training information.
3. Predictions are generated for held-out movies.
4. Differently rated held-out movies are converted into preference pairs.
5. Pairwise ordering accuracy is calculated.
6. Held-out movies are ranked to calculate NDCG@10.
7. Rating-like models are additionally evaluated using MAE and RMSE.
8. Per-user results are aggregated into macro and pair-weighted metrics.

The primary Week 5 evaluation contains:

- Users evaluated: 600
- Held-out preference pairs: 633,032

All evaluated configurations use the same final held-out portion so results can be compared directly.

---

## Current Results

### Week 5 Primary Evaluation

| Model | Pairwise | Pair-Weighted | NDCG@10 | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movie-average baseline | 62.608% | 67.333% | 0.9002 | 0.7891 | 0.9395 |
| Heuristic | 60.633% | 65.127% | 0.8986 | 0.7587 | 0.9220 |
| **Original ML** | **63.876%** | **68.516%** | **0.9051** | — | — |
| Matrix Factorization | 62.634% | 67.265% | 0.8988 | **0.7143** | **0.8685** |
| Content | 55.622% | 59.679% | 0.8850 | 0.7806 | 0.9441 |
| Hybrid V1 | 63.287% | 67.872% | 0.9050 | — | — |

Hybrid V1 compared with the original ML reranker:

- Macro pairwise: -0.589 percentage points
- Pair-weighted: -0.644 percentage points
- NDCG@10: -0.0001

The original ML reranker therefore remains the strongest currently evaluated ranking configuration.

Matrix factorization produces the lowest MAE and RMSE among models that output rating-like predictions.

---

## Interpretation

Week 5 shows that adding individually useful recommendation components does not automatically create a stronger hybrid.

The most important finding is that Hybrid V1 performs slightly below the simpler original ML reranker despite incorporating standalone content and collaborative models.

Diagnostic analysis provides evidence for why.

### MF and Quality Are Highly Redundant

Matrix factorization and movie quality agree on the direction of approximately 92.1% of evaluated preference pairs.

Their conditional rescue rates are also low:

- When Quality is wrong, MF is correct on approximately 11.9% of those pairs.
- When MF is wrong, Quality is correct on approximately 12.1% of those pairs.

This indicates that the two signals frequently encode similar ranking information. Adding MF alongside Quality therefore provides less new information than its standalone performance might suggest.

### Personal Preference Provides Complementary Information

Personal preference is weaker as an isolated ranking signal, but it frequently makes correct decisions when stronger signals fail.

For example:

- When MF is wrong, Personal rescues approximately 54.3% of those pairs.
- When Quality is wrong, Personal rescues approximately 54.0%.

This helps explain why removing Personal from Hybrid V1 can hurt performance even though MF is a stronger standalone signal.

Standalone strength and usefulness inside a hybrid are therefore different properties.

### Content Contains Useful but Currently Weak Information

The current standalone Content model reaches:

- 55.622% macro pairwise accuracy
- 59.679% pair-weighted accuracy
- 0.8850 NDCG@10

Content performance varies substantially with its confidence.

Pair-weighted content accuracy rises from approximately:

- 56.3% in the low-confidence group
- 60.5% in the medium-confidence group
- 62.2% in the high-confidence group

This suggests that content predictions are not equally reliable across candidates.

Hybrid V1 currently uses the content prediction itself but does not provide content confidence to the reranker.

The currently evaluated dataset also lacks much of the richer metadata supported by the content implementation, limiting the content model's ability to exploit director, runtime, language, and other signals.

### Hybrid and Original ML Make Different Errors

Across 633,032 held-out pairs:

- Both correct: 396,256
- Original ML only correct: 37,470
- Hybrid V1 only correct: 33,395
- Both wrong: 165,911

The original ML reranker therefore wins 4,075 more unique pairwise decisions than Hybrid V1.

This relatively small difference is consistent with their nearly identical NDCG@10 scores, but it is sufficient for the original reranker to remain the stronger validated model.

These findings motivate further hybrid-design experiments rather than replacing the existing production ranking configuration.

---

## Known Limitations

- Evaluation currently uses MovieLens latest-small, which is much smaller and less metadata-rich than a production movie catalog.
- The content model supports richer metadata than is available in the current evaluation dataset.
- Offline ratings do not capture short-term viewing intent.
- Explicit MovieLens ratings do not perfectly represent real-world viewing behavior or user satisfaction.
- Pairwise accuracy measures relative ordering but does not capture every property of recommendation quality.
- NDCG@10 evaluates ranking quality within the available held-out candidates rather than a full production-scale candidate catalog.
- MAE and RMSE apply only to models that produce rating-like predictions.
- Current confidence signals are analyzed diagnostically but are not yet fully incorporated into hybrid ranking.
- Current hybrid training uses linear pairwise logistic regression, limiting the interactions it can learn between signals.
- Diversity reranking requires separate evaluation because the primary Week 5 model comparison focuses on relevance ranking.
- Results from MovieLens should not be interpreted as directly comparable to internal recommendation metrics reported by production streaming platforms.

---

## Relevant Files

### Core Evaluation

- `src/evaluation/metrics.py`
- `src/evaluation/ranking.py`
- `src/evaluation/reports.py`
- `src/evaluation/splits.py`

### Primary Evaluation

- `src/scripts/evaluate_ranking.py`

### Ablation Evaluation

- `src/scripts/week3_ablations.py`
- `src/scripts/evaluate_week3_ablations.py`

### Week 5 Diagnostics

- `src/scripts/evaluate_signal_usefulness.py`
- `src/scripts/retest_old_ml.py`
- `src/scripts/analyze_week5_errors.py`

### Tests

- `tests/evaluation/`

### Saved Models

- `src/models/ml_reranker.joblib`
- `src/models/hybrid_v1_reranker.joblib`
- `src/models/ablations/`