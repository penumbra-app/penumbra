# Hybrid and Reranking

## Purpose

The hybrid recommendation system combines personalized user-taste signals, collaborative predictions, and global movie signals to determine which movies should rank highest for a user.

Development currently includes three related ranking systems:

1. A heuristic genre-based recommender that produces personal preference and movie-quality signals.
2. A learned pairwise logistic-regression reranker using personal preference, quality, and popularity.
3. Experimental hybrid rerankers that combine content-based predictions, matrix-factorization predictions, personal preference, quality, and popularity.

The original learned reranker remains the strongest model under the current held-out pairwise evaluation.

Week 3 integrated the standalone content and collaborative models into the reranking pipeline and tested whether those additional signals improved ranking performance. They did not outperform the existing reranker. The Week 3 experiments and ablations are preserved as evidence rather than replacing the stronger existing model.

---

## Inputs

The system currently uses the MovieLens latest-small dataset.

Required inputs include:

- User ID
- User ratings
- Rating timestamps
- Movie IDs
- Movie titles
- Movie genres
- Ratings from the overall user population

Depending on the ranking configuration, the hybrid system can also consume:

- Standalone content-model predictions
- Matrix-factorization predictions
- Personal preference scores
- Movie quality scores
- Movie popularity

The original trained reranker is stored at:

`src/models/ml_reranker.joblib`

The Week 3 Hybrid V1 model is stored at:

`src/models/hybrid_v1_reranker.joblib`

Week 3 ablation models are stored under:

`src/models/ablations/`

---

## Outputs

The current ML recommendation path returns ranked movies containing information such as:

- `movie_id`
- `title`
- `genres`
- `ml_score`
- `personal_score`
- `quality_score`
- `popularity`

Experimental hybrid configurations can additionally use:

- `content_score`
- `collaborative_score`

A higher ML or hybrid score means the trained reranker prefers that movie more strongly.

These scores are ranking scores and should not be interpreted directly as predicted 1–5 star ratings.

---

## Core Idea

The hybrid system represents each candidate movie using signals that capture different kinds of evidence about whether a user will prefer it.

Available ranking signals currently include:

1. Personal preference score
2. Content-model prediction
3. Collaborative matrix-factorization prediction
4. Movie quality score
5. Movie popularity

The strongest current reranker uses:

- personal preference
- quality
- popularity

Week 3 introduced Hybrid V1 using:

- content prediction
- collaborative matrix-factorization prediction
- quality
- popularity

Additional ablations tested different combinations of all five available signals.

Rather than manually assigning final weights, the system learns their relative importance using pairwise logistic regression.

For two movies A and B:

- If the user rated A higher than B, the feature difference `A - B` receives label `1`.
- The reverse feature difference `B - A` receives label `0`.

A logistic-regression model learns which feature differences tend to correspond to the movie a user preferred.

At inference time, candidate movies receive the same feature representation used during training. The trained scaler standardizes those features and the logistic-regression decision function produces the final ranking score.

---

# Algorithm Flow

## 1. Build the User's Historical Profile

The user's ratings are collected and ordered using timestamps.

More recent ratings receive greater weight through exponential decay.

The current recency half-life is 4 years, meaning a rating that is four years older than the user's newest rating receives approximately half as much recency weight.

---

## 2. Calculate the User's Baseline Rating

The user's average rating is computed using the recency weights.

This represents the user's normal rating level.

For example, a user who generally gives high ratings should not automatically appear to strongly prefer every genre.

---

## 3. Measure Personal Genre Preference

For every candidate movie genre, the system looks at movies of that genre previously rated by the user.

Historical evidence is weighted by:

- rating recency
- similarity between the historical movie's release year and the candidate movie's release year

The weighted genre average is compared with the user's overall weighted average.

Conceptually:

`genre preference = weighted genre rating - weighted user average`

The result is reduced when there is very little evidence for that genre.

The confidence calculation is:

`confidence = evidence / (evidence + confidence_prior)`

The current confidence prior is `5.0`.

This prevents one unusually high rating from creating an extremely strong genre preference.

---

## 4. Calculate Movie Quality

Movie quality is based on a Bayesian-adjusted average rating.

Movies with very few ratings are pulled toward the overall dataset average instead of trusting their raw average immediately.

Conceptually:

`quality_rating = (rating_count × movie_average + prior_strength × global_average) / (rating_count + prior_strength)`

The quality feature used by the recommender is:

`quality_score = quality_rating - global_average`

Therefore:

- positive quality score = above-average movie quality
- negative quality score = below-average movie quality

---

## 5. Produce the Heuristic Score

Before the learned reranker was introduced, the system estimated a movie rating using:

`predicted_rating = user_average + personal_score + quality_weight × quality_score`

The result is clipped to the MovieLens rating range of `0.5` to `5.0`.

The current heuristic quality weight is `0.20`.

---

## 6. Diversity Reranking for the Heuristic Recommender

The heuristic recommender initially sorts movies by predicted rating and considers its top 50 candidates.

It then greedily selects recommendations while applying a penalty when a candidate repeats genres already present in selected recommendations.

The current repetition penalty is `0.08` per overlapping genre.

This prevents the recommendation list from becoming unnecessarily repetitive.

This diversity step is currently part of `recommend_by_genre`.

This diversity step is specific to the older heuristic recommender.

The final learned recommendation pipeline now uses a separate genre-based diversity reranker after ML relevance ranking, described below.

---

## 7. Calculate Popularity

Popularity is defined as:

`popularity = ln(1 + number of ratings)`

The logarithm prevents extremely popular movies from dominating the feature scale simply because they have many more ratings.

---

## 8. Produce the Content Score

The content score is produced by the standalone content model in `src/content/`.

The model builds a user taste profile from that user's historical ratings and predicts how strongly a candidate movie matches the learned metadata preferences.

The standalone content model can incorporate signals including:

- genre
- director
- runtime
- release era
- language

For Week 3 hybrid experiments, `PredictionResult.predicted_score` is used as the content feature.

The content model also produces confidence information, but content confidence is not currently included as a Hybrid V1 reranking feature.

---

## 9. Produce the Collaborative Score

The collaborative score is produced by biased matrix factorization.

Matrix factorization learns:

- user biases
- movie biases
- latent user factors
- latent movie factors

The model predicts a rating-like score representing how strongly the learned collaborative system expects a user to prefer a movie.

For Week 3 hybrid experiments, this predicted score becomes the `collaborative_score`.

The collaborative model also produces confidence information, but collaborative confidence is not currently included as a Hybrid V1 reranking feature.

---

## 10. Create Chronological Training and Evaluation Partitions

For each eligible user, historical ratings are ordered chronologically and divided into:

- First 60%: user profile
- Next 20%: pairwise training movies
- Final 20%: held-out evaluation movies

The split is deterministic.

If multiple ratings share the same timestamp, movie ID is used as a deterministic tie-breaker.

Split fractions are validated so invalid configurations cannot silently produce incorrect partitions.

The profile, training, and test partitions are required to:

- contain no overlapping rows
- collectively cover the user's complete rating history

---

## 11. Maintain a Leakage-Safe Information Boundary

The system prevents held-out evaluation ratings from influencing features or models used to predict those same preferences.

For the original ML pipeline, population-level reference data is constructed inside the chronological training boundary and the target user's own ratings are excluded when appropriate.

For Week 3 hybrid training, the information boundary is stricter for upstream personalization models:

- First 60% → content profiles, collaborative training, and population signals
- Next 20% → pairwise labels for training the hybrid reranker
- Final 20% → untouched evaluation

The matrix-factorization model used by the Week 3 hybrid pipeline is trained once on eligible users' first-60% profile ratings.

The content model for a target user is constructed from that user's first-60% profile.

Population-level signals used by the hybrid pipeline exclude the target user's ratings.

This prevents pairwise-training labels and final evaluation labels from becoming upstream personalization evidence.

---

## 12. Create Pairwise Training Examples

The profile portion is used to construct user preference information.

Movies in the pairwise-training portion are represented using the features available to the ranking configuration being trained.

Pairs of differently rated movies are then created.

For a preferred movie P and less-preferred movie L:

`X = features(P) - features(L), label = 1`

The reverse example is also included:

`X = features(L) - features(P), label = 0`

This creates a balanced pairwise classification dataset.

Pairs with equal ratings are skipped because there is no preference direction to learn.

---

## 13. Standardize the Features

Before training, the feature matrix is standardized using `StandardScaler`.

This prevents features with naturally larger numerical ranges from receiving extra influence simply because of their scale.

The same fitted scaler is saved with the trained model and reused during inference.

---

## 14. Train Logistic Regression

A logistic-regression classifier is trained on the pairwise feature differences.

The exact features depend on the ranking configuration being tested.

The original ML reranker learns from:

- personal preference
- quality
- popularity

Hybrid V1 learns from:

- content
- collaborative
- quality
- popularity

Week 3 ablation models test additional combinations.

Training uses an explicit `random_state=42` to support reproducibility.

The trained scaler and logistic-regression model are stored together as a `TrainedRanker`.

---

## 15. Rank Unseen Movies

During inference:

1. Movies already rated by the user are removed.
2. Every unseen candidate receives the features required by the selected ranking model.
3. Features are transformed using the saved scaler.
4. Logistic regression's decision function produces a ranking score.
5. Movies are sorted by ranking score from highest to lowest.
6. A larger high-relevance candidate pool is retained.
7. The candidate pool is reranked using genre-based diversity.
8. The requested number of final recommendations is returned.


---

## 16. Diversity Reranking After Learned Ranking

The final recommendation pipeline does not simply return the highest-scoring movies from the learned reranker.

Instead, the learned model first produces a larger pool of high-relevance candidates. The current final pipeline requests:

`candidate_pool_size = 5 × requested_recommendation_count`

For example, requesting 10 final recommendations first produces the top 50 candidates according to the learned reranker.

The diversity stage then greedily constructs the final recommendation slate.

The highest-ranked candidate is always selected first.

For each remaining position, every remaining candidate receives an adjusted selection score:

`adjusted_score = original_ml_score - diversity_weight × maximum_genre_similarity`

Genre similarity is measured using Jaccard similarity:

`similarity(A, B) = |A ∩ B| / |A ∪ B|`

where A and B are the genre sets of two movies.

For a candidate movie, the system compares it against every movie already selected and uses the maximum similarity:

`maximum_genre_similarity = max(similarity(candidate, selected_movie))`

The current diversity weight is:

`0.15`

The candidate with the highest adjusted score is selected next, and the process repeats until the requested number of recommendations has been chosen.

Using maximum similarity instead of summing similarity across the entire slate prevents common genres such as Drama from accumulating an excessively large penalty simply because they appear in several selected movies.

This stage creates a relevance-diversity tradeoff:

- `diversity_weight = 0` preserves the original ML ranking.
- A small positive weight can promote a slightly lower-scoring movie when it is substantially different from movies already selected.
- An excessively large weight could sacrifice too much predicted relevance for variety.

The returned `score` remains the original learned ranking score. The diversity-adjusted score is temporary and depends on the recommendations already selected, so it is not treated as a permanent movie score.

---

# Features / Signals

## Personal Score

Measures whether the user historically rates the candidate's genres above or below their own normal rating level.

Affected by:

- genre overlap
- rating recency
- release-year similarity
- amount of historical evidence

---

## Content Score

Produced by the standalone content model.

It represents the content model's predicted preference for the candidate movie based on the target user's learned metadata preferences.

Current content modeling includes signals such as:

- genre
- director
- runtime
- release era
- language

The content score is a separate signal from the older personal genre score.

---

## Collaborative Score

Produced by biased matrix factorization.

It represents predicted preference based on rating behavior learned across the user population.

The model incorporates:

- global rating behavior
- user bias
- movie bias
- latent user factors
- latent movie factors

---

## Quality Score

Measures how much the movie's Bayesian-adjusted rating differs from the global MovieLens average.

This provides a general movie-quality signal independent of a particular user's tastes.

---

## Popularity

Defined as:

`ln(1 + rating count)`

This gives the model information about how widely rated a movie is without allowing raw rating counts to grow excessively large.

---

# Training

The reranking system uses pairwise logistic regression.

Users with fewer than 10 ratings are skipped when the chronological split cannot provide useful profile, training, and evaluation portions.

The pairwise dataset is balanced because every positive feature-difference example is paired with its reversed negative example.

## Original Reranker

The original learned reranker uses:

- personal
- quality
- popularity

This remains the strongest currently measured ranking configuration.

## Hybrid V1

Week 3 introduced a four-feature hybrid:

- content
- collaborative
- quality
- popularity

The Week 3 Hybrid V1 training run produced:

- Users used: 603
- Training examples: 1,418,086
- Positive examples: 709,043
- Negative examples: 709,043

The learned standardized coefficients were:

| Feature | Coefficient |
| --- | ---: |
| Content | +0.4051 |
| Collaborative | +0.7898 |
| Quality | -0.0215 |
| Popularity | +0.2513 |

The collaborative signal received the largest positive coefficient.

Content received a smaller positive coefficient, popularity provided a modest positive signal, and quality was close to neutral in this configuration.

The Hybrid V1 model is saved separately from the original reranker so previous models and benchmarks remain reproducible.

---

# Week 3 Hybrid Integration

Week 3 connected the standalone content and collaborative models to the learned reranking pipeline.

The goal was to determine whether the specialized content and collaborative personalization systems provide complementary information that improves final ranking performance.

## Hybrid Candidate

The Week 3 pipeline introduced a shared hybrid candidate representation containing:

- `content_score`
- `collaborative_score`
- `quality_score`
- `popularity`

The collaborative score comes from biased matrix factorization.

The content score comes from the standalone content model.

Quality and popularity provide global movie-level evidence.

## Hybrid V1

Hybrid V1 uses:

`Content + MF + Quality + Popularity`

Like the original ML reranker, Hybrid V1 uses standardized pairwise feature differences and logistic regression.

The model was trained under the 60/20/20 chronological information boundary and evaluated on the untouched final 20%.

---

# Evaluation

The system uses chronological holdout evaluation rather than randomly mixing old and new ratings.

For each eligible user:

- 60% of historical ratings build the profile.
- 20% are used for pairwise training.
- 20% remain held out for evaluation.

For the held-out portion, pairs of movies with different actual ratings are compared.

A ranking receives:

- `1.0` credit if it orders the pair correctly
- `0.0` credit if it orders the pair incorrectly
- `0.5` credit if the model predicts a tie

Two forms of aggregate accuracy are reported.

### Macro Accuracy

Accuracy is calculated separately for each user and then averaged.

Every user therefore receives equal weight regardless of how many valid test pairs they contribute.

### Pair-Weighted Accuracy

Correct pairwise decisions are aggregated across all evaluated pairs.

Users who contribute more valid movie pairs therefore have more influence on this metric.

Both metrics are retained because they answer different questions about ranking performance.

---

# Week 3 Hybrid V1 Benchmark

Users evaluated: 600

Test pairs: 633,032

## Macro Accuracy

| Model | Accuracy |
| --- | ---: |
| Movie-average baseline | 62.608% |
| Heuristic | 60.633% |
| **Original ML reranker** | **63.876%** |
| Matrix factorization | 62.634% |
| Content | 55.026% |
| Hybrid V1 | 63.253% |

Hybrid V1 improvement:

- vs baseline: +0.646 percentage points
- vs matrix factorization: +0.620 percentage points
- vs content: +8.228 percentage points
- vs original ML: -0.623 percentage points

## Pair-Weighted Accuracy

| Model | Accuracy |
| --- | ---: |
| Movie-average baseline | 67.333% |
| Heuristic | 65.127% |
| **Original ML reranker** | **68.516%** |
| Matrix factorization | 67.265% |
| Content | 59.095% |
| Hybrid V1 | 67.798% |

Hybrid V1 improvement:

- vs baseline: +0.466 percentage points
- vs matrix factorization: +0.534 percentage points
- vs content: +8.703 percentage points
- vs original ML: -0.717 percentage points

Hybrid V1 therefore improved substantially over standalone content and modestly over standalone matrix factorization, but it did not outperform the existing ML reranker.

Because the hybrid did not win, additional ablation experiments were performed rather than replacing the stronger baseline.

---

# Week 3 Ablation Study

Six additional feature combinations were trained and evaluated to determine which signals were helping or hurting ranking performance.

The experiments were:

1. MF + Personal + Quality + Popularity
2. MF + Quality + Popularity
3. MF + Content
4. MF + Personal
5. MF + Personal + Content
6. MF + Personal + Content + Quality + Popularity

All six experiments used the same pairwise logistic-regression approach and chronological information boundary.

## Results

| Configuration | Macro | Pair-Weighted |
| --- | ---: | ---: |
| **Original ML: Personal + Quality + Popularity** | **63.876%** | **68.516%** |
| Hybrid V1: MF + Content + Quality + Popularity | 63.253% | 67.798% |
| All 5 signals | 63.024% | 67.883% |
| MF + Personal + Quality + Popularity | 62.962% | 67.899% |
| MF + Personal | 62.556% | 67.714% |
| MF + Personal + Content | 62.534% | 67.682% |
| MF + Content | 62.441% | 67.626% |
| MF + Quality + Popularity | 62.374% | 66.364% |

The strongest ablation by macro accuracy was the five-signal model.

The strongest ablation by pair-weighted accuracy was MF + Personal + Quality + Popularity.

Neither surpassed the original Personal + Quality + Popularity reranker.

---

## Ablation Coefficients

The six experimental models learned the following standardized coefficients:

| Configuration | MF | Personal | Content | Quality | Popularity |
| --- | ---: | ---: | ---: | ---: | ---: |
| MF + Personal + Quality + Popularity | +0.7504 | +0.4441 | — | +0.0140 | +0.2603 |
| MF + Quality + Popularity | +0.7293 | — | — | +0.0947 | +0.2104 |
| MF + Content | +0.8550 | — | +0.3759 | — | — |
| MF + Personal | +0.8533 | +0.4136 | — | — | — |
| MF + Personal + Content | +0.8540 | +0.4991 | -0.0903 | — | — |
| All 5 | +0.7422 | +0.5220 | -0.0834 | +0.0234 | +0.2604 |

These coefficients correspond to standardized features and should not be interpreted as raw feature weights.

When Personal and Content were included together, the learned Content coefficient became slightly negative.

This suggests that the current content prediction provides little additional ranking information once the existing personal preference signal is available.

Adding matrix factorization to the original Personal + Quality + Popularity feature set also reduced held-out performance rather than improving it.

---

# Week 3 Conclusion

Week 3 successfully integrated real content and collaborative predictions into the hybrid reranker, but the resulting models did not outperform the existing learned reranker.

The strongest currently measured ranking model therefore remains:

`Personal + Quality + Popularity → Pairwise Logistic Regression`

The Week 3 experiments show that:

- Matrix factorization is a useful standalone personalization signal but did not provide enough complementary information to improve the existing reranker.
- The standalone content model currently performs substantially below the other ranking approaches on pairwise evaluation.
- Adding Content to MF + Personal produced essentially no improvement and slightly reduced held-out accuracy.
- Adding MF to the existing Personal + Quality + Popularity configuration reduced held-out performance.
- Combining more signals does not automatically produce a stronger ranking model.
- Experimental hybrid models should not replace a simpler baseline unless held-out evaluation supports the change.

The Week 3 hybrid implementations and saved models are retained for reproducibility and future experimentation even though they are not the current winning ranking configuration.

---

# Reproducibility

The training and evaluation pipeline includes explicit reproducibility safeguards.

These include:

- deterministic chronological splitting
- deterministic movie-ID tie-breaking for equal timestamps
- validated split fractions
- explicit `random_state=42` for logistic regression
- deterministic matrix-factorization initialization
- regression tests for information-boundary invariants
- separately saved model artifacts for different experimental configurations

Previous complete retraining and evaluation runs have produced identical training counts, test-pair counts, model metrics, and diagnostics under the tested environment.

Experimental models are stored separately rather than overwriting stronger or historically important models.

---

# Inference

The original learned-ranking inference function is:

`recommend_with_ml(...)`

It:

1. Loads the saved ranker.
2. Finds the user's watched movies.
3. Removes watched movies from the candidate set.
4. Computes personal, quality, and popularity features.
5. Standardizes the features using the saved scaler.
6. Calculates the logistic-regression decision score.
7. Sorts unseen movies by ML score.

The final Week 4 recommendation pipeline builds on this ranking through:

`recommend_for_user(...)`

Conceptually:

`unseen movies → learned ML ranking → high-relevance candidate pool → diversity reranking → final Top-K`

The final recommendation layer uses the strongest currently validated learned reranker rather than forcing the Week 3 Hybrid V1 model into production, because the original Personal + Quality + Popularity reranker still performs better on held-out evaluation.

Detailed public API usage is documented in the repository's main README.

---

# Evaluation Sanity Checks

The evaluation pipeline verifies properties including:

- the number of stored user results matches the reported number of evaluated users
- per-user pair counts sum to the total pair count
- every evaluated user contributes at least one valid pair
- every evaluated user has enough ratings for the chronological split
- reported accuracies remain within `[0, 1]`
- chronological partitions do not overlap
- hidden evaluation data remains outside the permitted training boundary

The full automated test suite currently passes.

---

# Reliability Tests

Automated tests protect the chronological and information-boundary logic.

Tests verify:

- chronological ordering
- deterministic behavior
- deterministic equal-timestamp tie-breaking
- no overlap between profile, training, and test partitions
- complete coverage of every rating
- rejection of invalid split fractions
- exclusion of held-out test ratings from inappropriate model inputs
- exclusion of target-user ratings from target-user population reference features where required
- model and evaluation data-contract behavior

Current full automated test suite:

`131 passed`

---

# Important Hyperparameters

Current genre and heuristic parameters:

- Recency half-life: `4.0 years`
- Release-year penalty: `0.005 per year`
- Minimum release-year weight: `0.50`
- Genre confidence prior: `5.0`
- Heuristic quality weight: `0.20`

Heuristic recommender diversity:

- Repetition penalty: `0.08 per overlapping genre`
- Candidate pool: `50 movies`

Final learned recommender diversity:

- Genre similarity: Jaccard similarity
- Diversity weight: `0.15`
- Candidate pool multiplier: `5× requested recommendation count`
- Similarity aggregation: maximum similarity to any already-selected movie

Movie-quality Bayesian prior strength:

- `10.0`

Chronological split:

- Profile: `60%`
- Pairwise training: `20%`
- Evaluation: `20%`

Popularity transformation:

- `ln(1 + rating_count)`

Logistic-regression random seed:

- `42`

Matrix-factorization configuration used by the hybrid experiments:

- Latent factors: `20`
- Learning rate: `0.005`
- Regularization: `0.02`
- Epochs: `20`
- Prior strength: `5.0`
- Random seed: `42`

---

# Known Limitations

- The strongest current reranker still relies primarily on personal genre preference, movie quality, and popularity.
- The current standalone content model performs substantially below the strongest reranker under pairwise evaluation.
- Matrix-factorization predictions have been integrated experimentally but have not improved the strongest reranker.
- Content and collaborative confidence signals are not yet included in the learned hybrid feature set.
- Personalization remains heavily influenced by genres rather than deeper semantic relationships between individual movies.
- Popularity can introduce bias toward widely rated movies.
- Diversity currently uses only explicit genre overlap and does not yet model semantic similarity between movies.
- The current diversity weight and candidate-pool multiplier are initial engineering choices and have not yet been validated through the Week 5 Top-K evaluation.
- The model is trained globally rather than training a separate ranker for every individual user.
- MovieLens metadata is limited compared with production movie data.
- Release-year similarity includes manually designed behavior.
- Genre relationships are primarily explicit rather than learned semantically.
- Pairwise accuracy measures relative preference ordering and does not capture every aspect of recommendation quality.
- Offline MovieLens evaluation cannot fully represent short-term viewing intent or real-world user satisfaction.
- More features do not necessarily provide complementary information, as demonstrated by the Week 3 ablations.

---

# Future Improvements

Possible improvements include:

- investigate why the standalone content model underperforms the existing personal preference signal
- test content and collaborative confidence as reranking features
- add richer semantic movie information such as keywords and plot text
- test TF-IDF and embedding-based movie representations
- test stronger ranking models such as gradient-boosted trees after the core pipeline is stable
- evaluate the relevance-diversity tradeoff using Top-K metrics and controlled ablations
- test richer movie-similarity signals for diversity beyond explicit genre overlap
- evaluate ranking quality using NDCG@10 and additional Top-K metrics
- optimize hyperparameters using validation data
- improve cold-start behavior for users with little rating history
- add calibrated predicted-enjoyment scores in addition to ranking scores
- evaluate on larger or richer datasets
- investigate group recommendation and multi-user preference aggregation
- investigate short-term viewing intent in addition to long-term user taste
- evaluate the system with real users after the offline recommender is stable

---

# Relevant Files

## Primary Hybrid Implementation

- `src/hybrid/genre_recommender.py`
- `src/hybrid/ml_reranker.py`
- `src/hybrid/content_adapter.py`
- `src/hybrid/diversity.py`
- `src/hybrid/recommender.py`

## Training

- `src/scripts/train_ml_reranker.py`
- `src/scripts/train_hybrid_reranker.py`
- `src/scripts/week3_ablations.py`

## Saved Models

Original learned reranker:

- `src/models/ml_reranker.joblib`

Hybrid V1:

- `src/models/hybrid_v1_reranker.joblib`

Week 3 ablations:

- `src/models/ablations/`

## Evaluation

- `src/evaluation/ranking.py`
- `src/evaluation/splits.py`
- `src/evaluation/metrics.py`
- `src/evaluation/reports.py`
- `src/scripts/evaluate_ranking.py`
- `src/scripts/evaluate_week3_ablations.py`

## Tests

Hybrid tests:

- `tests/hybrid/`

Evaluation tests:

- `tests/evaluation/`

## Demonstration

- `main.py`

---

# Benchmark History

Dataset: MovieLens latest-small

Evaluation split:

- 60% profile
- 20% pairwise training
- 20% held-out testing

---

## Week 3 Hybrid Integration and Ablation Benchmark

Users evaluated: 600

Test pairs: 633,032

### Primary Model Comparison

| Model | Macro | Pair-Weighted |
| --- | ---: | ---: |
| Movie-average baseline | 62.608% | 67.333% |
| Heuristic | 60.633% | 65.127% |
| **Original ML reranker** | **63.876%** | **68.516%** |
| Matrix factorization | 62.634% | 67.265% |
| Content | 55.026% | 59.095% |
| Hybrid V1 | 63.253% | 67.798% |

Hybrid V1 features:

- content prediction
- matrix-factorization prediction
- movie quality
- popularity

Hybrid V1 training:

- Users used: 603
- Pairwise examples: 1,418,086
- Positive examples: 709,043
- Negative examples: 709,043

Hybrid V1 standardized coefficients:

- Content: +0.4051
- Collaborative: +0.7898
- Quality: -0.0215
- Popularity: +0.2513

### Week 3 Ablation Results

| Configuration | Macro | Pair-Weighted |
| --- | ---: | ---: |
| **Original ML** | **63.876%** | **68.516%** |
| Hybrid V1 | 63.253% | 67.798% |
| All 5 | 63.024% | 67.883% |
| MF + Personal + Quality + Popularity | 62.962% | 67.899% |
| MF + Personal | 62.556% | 67.714% |
| MF + Personal + Content | 62.534% | 67.682% |
| MF + Content | 62.441% | 67.626% |
| MF + Quality + Popularity | 62.374% | 66.364% |

Week 3 conclusion:

The existing Personal + Quality + Popularity pairwise logistic reranker remains the strongest measured ranking model.

Integrating content and collaborative predictions successfully produced a complete hybrid architecture, but neither Hybrid V1 nor six additional feature ablations surpassed the existing reranker on the untouched final-20% evaluation.

The hybrid experiments are retained as reproducible experimental results rather than replacing the stronger baseline.

---

## Week 2 Leakage-Safe and Reproducible Benchmark

Users evaluated: 601

Test pairs: 695,795

Macro results:

- Movie-average baseline accuracy: 62.730%
- Heuristic accuracy: 60.342%
- Matrix-factorization accuracy: 62.773%
- ML reranker accuracy: 63.686%
- ML improvement over heuristic: +3.344 percentage points
- ML improvement over movie-average baseline: +0.956 percentage points
- ML improvement over matrix factorization: +0.914 percentage points

Pair-weighted results:

- Movie-average baseline: 67.084%
- Heuristic: 64.965%
- Matrix factorization: 66.964%
- ML reranker: 68.282%

Training run:

- Users used: 605
- Pairwise training examples: 1,530,814
- Positive examples: 765,407
- Negative examples: 765,407

Reproducibility:

Two complete train-and-evaluate runs produced identical training counts, test-pair counts, accuracies, confidence intervals, and diagnostics.

Reliability changes introduced in Week 2:

- deterministic chronological splitting
- explicit validation of split fractions
- deterministic tie-breaking for equal timestamps
- tests proving complete split coverage
- tests proving no overlap between profile, training, and test partitions
- global held-out boundary excluding eligible users' evaluation ratings from population-level features
- target-user exclusion from population reference features
- automated leakage-regression tests
- explicit logistic-regression random seed
- retraining under the stronger information boundary
- reproducibility verification through repeated full training/evaluation runs
- evaluation sanity checks
- pair-weighted reporting
- per-user analysis
- bootstrap confidence intervals
- correlation diagnostics

---

## Earlier Week 2 Intermediate Benchmark

An intermediate Week 2 run produced:

- Users evaluated: 601
- Test pairs: 747,790
- Baseline accuracy: 62.630%
- Heuristic accuracy: 60.625%
- Matrix-factorization accuracy: 62.743%
- ML accuracy: 64.024%

This is not the final Week 2 benchmark.

Subsequent reliability changes altered the valid evaluation-pair set. After the pipeline was finalized, two complete retraining and evaluation runs produced the identical 695,795-pair benchmark documented above.

---

## Frozen Week 1 Benchmark

Users evaluated: 601

Test pairs: 747,939

Results:

- Movie-average baseline accuracy: 62.792%
- Heuristic accuracy: 60.597%
- ML reranker accuracy: 63.926%
- ML improvement over heuristic: +3.329 percentage points
- ML improvement over movie-average baseline: +1.134 percentage points

Training run:

- Users used: 604
- Pairwise training examples: 1,609,532
- Positive examples: 804,766
- Negative examples: 804,766

The Week 1 benchmark removed direct target-user leakage.

Population-level movie quality and popularity statistics excluded the target user's own ratings when generating that user's features.

During the Week 2 reliability audit, an additional information-boundary issue was discovered: population-level features could still use held-out ratings belonging to other users.

Week 2 introduced a stronger global chronological boundary in which held-out evaluation ratings from eligible users are excluded from population-level feature construction.

The Week 1 benchmark remains frozen as a historical benchmark.

---

## Superseded Initial Benchmark

An earlier evaluation produced:

- Heuristic accuracy: 61.995%
- ML reranker accuracy: 66.950%
- Improvement: +4.955 percentage points
- Test pairs: 941,248

This result is not an official benchmark.

The initial implementation calculated population-level movie quality and popularity using the complete ratings dataset.

As a result, hidden ratings belonging to the user being evaluated could influence features used to evaluate that same user.

The evaluation pipeline was subsequently corrected, strengthened during the Week 2 reliability audit, and the model was retrained.