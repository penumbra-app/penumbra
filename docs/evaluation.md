# Evaluation

## Purpose

Penumbra uses offline evaluation to decide whether recommendation-system changes actually improve ranking quality on unseen user preferences.

Evaluation is treated as a model-selection tool rather than only a reporting step.

A more complicated model is not kept merely because it is newer or more sophisticated. It must be supported by controlled evaluation.

Week 5 expanded the evaluation system beyond pairwise accuracy.

Week 6 used that framework to:

- reproduce enriched Content
- test content-confidence integration
- reintroduce Personal into the Hybrid
- run targeted linear feature ablations
- test a nonlinear GBT reranker
- reject GBT when it failed validation

---

# Datasets

Current evaluation uses MovieLens latest-small.

Dataset size:

- Ratings: `100,836`
- Users: `610`
- Rated movies: `9,724`
- Movie rows: `9,742`

MovieLens provides:

- user IDs
- movie IDs
- ratings
- timestamps
- titles
- genres

Week 6 also uses IMDb-derived enrichment for the Content model.

Generated enriched movie file:

`data/movies-enriched.csv`

Week 6 IMDb coverage:

- Director: `9,717`
- Runtime: `9,718`
- Release era: `9,742`
- Cast: `9,366`
- Language: `0`
- Unavailable movie IDs: `23`

The selected standalone Content configuration is:

`without_language`

Its reproducibility report is:

`docs/week6-content-repro.json`

---

# Data Splitting

## Standard Final Evaluation Split

The main evaluation uses a deterministic chronological split for each eligible user:

- First 60%: profile / upstream models
- Next 20%: reranker training
- Final 20%: held-out evaluation

Ratings are sorted by:

1. timestamp
2. movie ID as a deterministic tie-breaker

The final 20% is excluded from:

- user-profile construction
- reranker training
- upstream content evidence
- upstream collaborative evidence

For the Hybrid:

- Content profiles use the user's first 60%.
- Hybrid MF is trained from eligible users' first-60% profiles.
- Population reference features exclude the target user's own ratings where required.
- Final evaluation labels remain outside the permitted feature boundary.

---

## Week 6 GBT Validation Split

GBT model selection uses a separate validation structure:

- First 60%: profile / upstream models
- Next 10%: GBT and validation-logistic training
- Next 10%: validation
- Final 20%: untouched final test

This was introduced so GBT architecture and feature decisions would not be selected using the final test set.

The final 20% was not used for GBT because every tested GBT variant failed to beat the validation logistic baseline.

---

# Metrics

## Macro Pairwise Accuracy

For every user, held-out movies with different ratings are converted into preference pairs.

A model receives:

- `1.0` if it orders the pair correctly
- `0.0` if it orders the pair incorrectly
- `0.5` if it predicts a tie

Each user's accuracy is calculated independently and then averaged.

Every user therefore receives equal weight.

Higher is better.

---

## Pair-Weighted Accuracy

Correct pairwise decisions are aggregated across all evaluated pairs.

Users who contribute more valid preference pairs therefore have more influence.

Higher is better.

Macro and pair-weighted accuracy are both retained because they answer different questions.

---

## NDCG@10

NDCG@10 measures whether highly relevant movies appear near the top of the ranked held-out set.

Higher is better.

`1.0` represents the ideal ordering of the evaluated candidates.

Absolute NDCG values should only be compared when models are evaluated on the same candidate split.

For example, Week 6 GBT validation NDCG values around `0.93` should not be directly compared with final-20% NDCG values around `0.90`, because the candidate sets differ in size and chronology.

---

## MAE

Mean Absolute Error measures average absolute rating-prediction error.

Lower is better.

MAE is reported only for rating-like models.

---

## RMSE

Root Mean Squared Error measures rating-prediction error with greater penalty for large mistakes.

Lower is better.

Ranking-only logistic rerankers do not produce calibrated rating predictions, so MAE and RMSE are not reported for them.

---

# Baselines and Compared Models

## Movie-Average Baseline

A simple population movie-rating baseline.

---

## Heuristic Recommender

Combines personal genre preference and Bayesian-adjusted movie quality.

---

## Original ML Reranker

Pairwise logistic regression using:

- Personal
- Quality
- Popularity

This remains an important historical benchmark.

---

## Matrix Factorization

Biased MF using:

- global rating behavior
- user bias
- movie bias
- latent user factors
- latent movie factors

Selected MF configuration:

- factors: `20`
- learning rate: `0.005`
- regularization: `0.02`
- epochs: `40`
- prior strength: `5.0`
- random seed: `42`
- `shrink_latent=True`

---

## Content

Standalone metadata-based user preference model.

Week 6 uses IMDb-enriched movie metadata and the selected `without_language` configuration.

---

## Hybrid V1

Historical four-feature logistic hybrid:

- Content
- MF
- Quality
- Popularity

---

## Week 6 Five-Signal Hybrid

Pairwise logistic regression using:

- MF
- Personal
- Content
- Quality
- Popularity

This is the strongest tested Week 6 Hybrid across the combined metric profile.

---

## Week 6 GBT

Pointwise `GradientBoostingRegressor` using a subset of:

- MF
- Personal
- Content
- Content Confidence
- Quality
- Popularity

GBT was evaluated only on the dedicated validation slice because it failed validation.

---

# Evaluation Procedure

For standard final evaluation:

1. Chronologically split each eligible user's ratings.
2. Build user profiles and upstream models only from permitted information.
3. Generate predictions for held-out movies.
4. Convert differently rated held-out movies into preference pairs.
5. Calculate macro pairwise accuracy.
6. Calculate pair-weighted accuracy.
7. Rank held-out candidates and calculate NDCG@10.
8. Calculate MAE/RMSE for rating-like models.
9. Aggregate per-user results.

For Week 6 GBT validation:

1. Keep the first 60% as the profile.
2. Split the middle 20% into development-train and validation halves.
3. Train GBT and its logistic comparator only on the first 10%.
4. Evaluate both on the same second 10%.
5. Keep the final 20% untouched unless a GBT configuration earns promotion.

---

# Current Results

## Week 6 Linear Hybrid Ablations

These are the main final-20% Week 6 Hybrid results.

Users evaluated: `600`

Test pairs: `633,032`

| Configuration | Macro | Pair-Weighted | NDCG@10 |
| --- | ---: | ---: | ---: |
| **MF + Personal + Content + Quality + Popularity** | **63.645%** | **68.648%** | **0.9080** |
| MF + Personal + Content + Popularity | 63.445% | 68.556% | **0.9080** |
| MF + Content + Quality + Popularity | 63.419% | 68.503% | 0.9066 |
| MF + Content + Popularity | 63.307% | 68.423% | 0.9061 |
| Personal + Content + Quality + Popularity | 63.293% | 68.529% | 0.9070 |
| Content + Quality + Popularity | 63.253% | 68.401% | 0.9064 |

The best Week 6 Hybrid is therefore:

`MF + Personal + Content + Quality + Popularity`

---

## Week 6 Hybrid vs Original ML

| Model | Macro | Pair-Weighted | NDCG@10 |
| --- | ---: | ---: | ---: |
| Original ML | **63.876%** | 68.516% | 0.9051 |
| Week 6 five-signal Hybrid | 63.645% | **68.648%** | **0.9080** |

Differences:

- Macro: `-0.231 pp`
- Pair-weighted: `+0.132 pp`
- NDCG@10: `+0.0029`

There is no strict winner across all three metrics.

The original ML model retains the best macro accuracy.

The Week 6 Hybrid achieves better pair-weighted accuracy and NDCG@10.

The project goal is not specifically to beat the original ML benchmark; the goal is to maximize the overall ranking metric profile while keeping changes supported by evaluation.

---

## Week 6 Linear Content-Confidence Experiment

A linear Hybrid with raw Content Confidence was tested before the final ablation study.

Features:

- Content
- Content Confidence
- MF
- Quality
- Popularity

Results:

- Macro: `63.547%`
- Pair-weighted: `68.056%`
- NDCG@10: `0.9069`

The confidence coefficient was very small.

This experiment was rejected because raw confidence did not provide a clean overall improvement.

The result does not prove that confidence itself is useless. It shows that an independent linear confidence feature is not equivalent to learning a confidence-content interaction.

---

# Week 6 GBT Validation

## Motivation

GBT was tested as a nonlinear alternative to logistic regression.

The main hypothesis was that trees could learn interactions such as:

`content matters more when content confidence is high`

The tested GBT was pointwise rather than pairwise.

It predicted actual user rating from candidate features and then used the predicted score for ranking.

---

## Validation Logistic Baseline

For a fair comparison, the logistic comparator was retrained on exactly the same development-training slice used by the GBT.

Validation set:

- Users evaluated: `571`
- Preference pairs: `167,009`

Validation Logistic:

- Macro: `64.460%`
- Pair-weighted: `68.547%`
- NDCG@10: `0.9327`

---

## GBT Ablation Results

| Model | Macro | Pair-Weighted | NDCG@10 |
| --- | ---: | ---: | ---: |
| **Validation Logistic** | **64.460%** | **68.547%** | **0.9327** |
| Full GBT | 63.506% | 67.818% | 0.9314 |
| GBT - Confidence | 63.654% | 67.831% | 0.9312 |
| GBT - Quality | 63.238% | 67.773% | 0.9307 |
| GBT - Popularity | 64.155% | 67.603% | 0.9320 |
| GBT - Content | 63.572% | 67.610% | 0.9305 |

Every GBT variant lost to Logistic on all three validation metrics.

---

## GBT Deltas vs Logistic

| Model | Macro Delta | Pair-Weighted Delta | NDCG@10 Delta |
| --- | ---: | ---: | ---: |
| Full GBT | -0.954 pp | -0.729 pp | -0.0014 |
| GBT - Confidence | -0.806 pp | -0.716 pp | -0.0015 |
| GBT - Quality | -1.222 pp | -0.774 pp | -0.0021 |
| GBT - Popularity | -0.306 pp | -0.944 pp | -0.0007 |
| GBT - Content | -0.888 pp | -0.937 pp | -0.0022 |

The best GBT by macro and NDCG@10 was `GBT - Popularity`.

The best GBT by pair-weighted accuracy was `GBT - Confidence`.

Neither was competitive enough to replace Logistic.

---

# Interpretation

## Personal Is Complementary

Week 5 diagnostics showed that Personal was weaker standalone but often corrected mistakes made by MF and Quality.

Week 6 confirmed that bringing Personal back into the enriched Hybrid improves the metric profile.

The five-signal model outperformed the otherwise comparable four-feature IMDb baseline.

---

## MF and Quality Are Redundant but Both Still Help

Week 5 showed that MF and Quality agree on roughly `92.1%` of pairwise decisions.

Their learned coefficients also compete with one another.

However, Week 6 ablations showed that removing either one from the strongest five-signal model reduced performance.

The correct conclusion is therefore not that one should automatically be removed.

Instead:

- they are highly redundant
- but both still contribute enough to justify retention in the tested logistic Hybrid

---

## Content Is Useful but Its Marginal Linear Contribution Was Not Fully Isolated

Week 6 retained enriched Content in the strongest Hybrid.

However, no otherwise-identical linear `no_content` ablation was run.

Therefore the marginal contribution of Content inside the final five-signal logistic model was not isolated directly.

GBT did include a `no_content` ablation, but GBT itself uses a different learning objective and cannot substitute for that missing linear ablation.

---

## Content Confidence Did Not Earn Promotion

Content confidence showed useful diagnostic behavior in Week 5.

However:

- raw linear confidence did not improve the Hybrid cleanly
- removing confidence from GBT slightly improved macro and pair-weighted performance relative to Full GBT
- all GBT variants still lost to Logistic

Confidence remains available for future research but is not part of the selected logistic Hybrid.

---

## Pairwise Logistic Beat Pointwise GBT

The main Week 6 nonlinear experiment failed.

A likely structural reason is objective alignment:

- Logistic is trained directly on pairwise preference ordering.
- GBT was trained to predict numeric ratings pointwise.
- Penumbra's primary ranking metrics evaluate relative ordering and Top-K ranking.

The simpler linear model therefore had a training objective more directly aligned with the evaluation task.

This result supports keeping pairwise logistic regression rather than adopting a more complex model without evidence.

---

# Week 5 Primary Evaluation

Week 5 remains the main baseline diagnostic benchmark that motivated Week 6.

Users evaluated: `600`

Held-out preference pairs: `633,032`

| Model | Pairwise | Pair-Weighted | NDCG@10 | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movie-average baseline | 62.608% | 67.333% | 0.9002 | 0.7891 | 0.9395 |
| Heuristic | 60.633% | 65.130% | 0.8986 | 0.7587 | 0.9220 |
| **Original ML** | **63.876%** | **68.516%** | **0.9051** | — | — |
| Matrix Factorization | 62.634% | 67.265% | 0.8988 | **0.7143** | **0.8685** |
| Content | 55.622% | 59.679% | 0.8850 | 0.7806 | 0.9441 |
| Hybrid V1 | 63.287% | 67.872% | 0.9050 | — | — |

Week 5 diagnostics established:

- MF and Quality are highly redundant.
- Personal provides strong complementary information.
- Content is confidence-sensitive.
- More signals do not automatically produce a better Hybrid.

Those findings directly motivated the Week 6 experiments.

---

# Superseded Week 6 One-Off Evaluations

Some early Week 6 one-off Hybrid runs used an evaluation-time MF configuration that did not exactly match the selected MF configuration used during training.

The mismatch involved the selected 40-epoch `shrink_latent=True` configuration.

Those one-off Hybrid numbers are superseded by the corrected Week 6 linear-ablation benchmark documented above.

The corrected benchmark uses a matching MF configuration during training and evaluation.

---

# Current Model-Selection Decision

For Hybrid / Reranking:

**Keep pairwise logistic regression. Reject the tested pointwise GBT family.**

Selected Week 6 Hybrid candidate:

`MF + Personal + Content + Quality + Popularity`

Final held-out Hybrid metrics:

- Macro: `63.645%`
- Pair-weighted: `68.648%`
- NDCG@10: `0.9080`

GBT was rejected on validation and was not evaluated on the final 20%.

---

# Known Limitations

- MovieLens latest-small is much smaller than a production recommendation dataset.
- Offline explicit ratings do not perfectly represent real-world watch behavior or satisfaction.
- NDCG@10 is calculated over available held-out candidates rather than a production-scale catalog.
- The final five-signal logistic Hybrid did not receive a matched linear `no_content` ablation.
- Confidence signals remain incompletely exploited by the selected linear model.
- The tested GBT used pointwise rating regression rather than a ranking-specific tree objective.
- Final model quality is multi-objective: macro, pair-weighted accuracy, and NDCG@10 can disagree.
- Diversity requires its own evaluation because the primary model comparisons focus on relevance ranking.
- IMDb enrichment remains incomplete for some fields, especially language.
- Offline MovieLens results should not be treated as directly comparable to private production recommender metrics.

---

# Relevant Files

## Core Evaluation

- `src/evaluation/metrics.py`
- `src/evaluation/ranking.py`
- `src/evaluation/reports.py`
- `src/evaluation/splits.py`

## Primary Evaluation

- `src/scripts/evaluate_ranking.py`

## Historical Ablations

- `src/scripts/week3_ablations.py`
- `src/scripts/evaluate_week3_ablations.py`

## Week 5 Diagnostics

- `src/scripts/evaluate_signal_usefulness.py`
- `src/scripts/retest_old_ml.py`
- `src/scripts/analyze_week5_errors.py`

## Week 6 Linear Evaluation

- `src/scripts/week6_ablations.py`
- `src/scripts/evaluate_week6_ablations.py`

## Week 6 GBT

- `src/hybrid/gbt_reranker.py`
- `src/scripts/train_week6_gbt.py`
- `src/scripts/evaluate_week6_gbt.py`
- `src/scripts/week6_gbt_ablations.py`
- `src/scripts/evaluate_week6_gbt_ablations.py`

## Week 6 Reports

- `docs/week6-content-repro.json`
- `docs/week6-ablation-results.json`
- `docs/week6-gbt-ablation-results.json`

## Models

- `src/models/ml_reranker.joblib`
- `src/models/hybrid_v1_reranker.joblib`
- `src/models/ablations/`
- `src/models/week6_ablations/`
- `src/models/week6_gbt_ablations/`

The Week 6 GBT models are validation artifacts, not selected final rerankers.

---

# Week 6 Evaluation Conclusion

Week 6 successfully tested both richer inputs and a more expressive nonlinear reranker.

The final evidence supports three conclusions:

1. The enriched five-signal pairwise logistic Hybrid is the strongest tested Hybrid across the combined final-test metric profile.
2. The tested pointwise GBT family does not improve ranking quality and should not replace Logistic.
3. Model complexity is not itself an improvement; the chosen model must remain tied to measured ranking performance.

Week 7 can therefore proceed using the pairwise logistic Hybrid as the advanced single-user reranking foundation while work shifts toward group recommendation and production hardening.
