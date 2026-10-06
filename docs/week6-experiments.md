# Week 6: measured text improvements

The selected content preset uses **TF-IDF unigrams and bigrams**, text weight
**0.25**, text evidence prior **5**, maximum text adjustment **0.5 rating points**,
core regularization **1**, genre weight **1**, and release-era weight **0.25**.
Other core settings retain their previous values. This is the best configuration
on the validation set among the 162 configurations tested, not a claim of a
globally optimal model. The content demo now loads this frozen preset.

## Results

| Configuration | Validation pairwise | Test pairwise | Test NDCG@10 | Test RMSE | Test MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing content baseline | 57.1908% | 55.1788% | 0.75447 | 1.03022 | 0.78637 |
| Best core-only configuration | 57.4095% | 54.9846% | 0.75419 | 1.03226 | 0.78875 |
| **Selected TF-IDF blend** | **57.4345%** | **55.3614%** | **0.75627** | **1.02651** | **0.78338** |
| Best LSA embedding blend | 57.3152% | 55.4902% | 0.75904 | 1.02591 | 0.78279 |

Test evaluation covers **603 users**, **20,156 ratings**, and **938,381 unequal-rating
pairs**. Nine users have only tied test ratings, so pairwise accuracy averages
over 594 users. The selected model gains **0.1826 percentage points** in pairwise
accuracy over the original content baseline. Its paired user-bootstrap 95%
confidence interval is **[-0.3149, +0.6826] percentage points** (2,000 resamples,
seed 42). The interval includes zero: the improvement is small and statistically
inconclusive on this dataset.

LSA performs better on the final test point estimates but was worse on validation.
It remains an experimental option. Switching to it after seeing test results
would tune on the holdout. The selected TF-IDF configuration stays frozen.

This evaluation concerns the standalone content model. The older 66.9% reranker
result in the project README uses a different model, split, and global rating
aggregates; it is not directly comparable to these results. `main.py` retains
that separate legacy pipeline.

## Protocol and combinations

1. Order each user's ratings by timestamp and movie ID. Use approximately 60%
   for the profile, 20% for validation, and 20% for test. Keep entire timestamp
   groups together. Seven users lack sufficient holdouts after grouping and are
   excluded. Require at least five profile ratings and two in each holdout.
2. Freeze the initial profile for both holdouts. No validation or test ratings
   enter the user mean, residuals, recency weights, vocabulary, IDF, or SVD fit.
   No global rating-quality or popularity aggregates are used.
3. Include only tags available at or before that user's training cutoff; exclude
   the evaluation user's own tags. Fit text on movies in that user's history;
   transform candidate text without refitting.
4. Measure 18 core configurations: regularization `{1, 5, 20}`, genre weights
   `{0.5, 1, 2}`, release-era weights `{0, 0.25}`. This includes the unchanged
   baseline. Recency remains fixed at a 365-day half-life with floor 0.5.
5. Cross the top three core configurations with 96 TF-IDF configurations:
   unigrams / unigrams+bigrams, minimum document frequency `{1, 2}`, text prior
   `{1, 5}`, and text weight `{0.25, 0.5, 1, 2}`. Vocabulary is capped at 10,000
   features, with English stopwords, Unicode accent normalization, sublinear TF,
   and L2 normalization. Text adjustment cap remains fixed at 0.5.
6. TF-IDF improved validation, so test 48 LSA combinations using its best
   tokenization: dimensions `{8, 32}` crossed with the same three core models,
   two priors, and four weights. LSA uses training-only truncated SVD and positive
   cosine similarity; insufficient corpora fall back to sparse TF-IDF. This is
   a small latent semantic experiment, not a pretrained sentence model.
7. Choose the highest user-macro validation pairwise accuracy, breaking ties by
   lower RMSE then configuration name. Write `selected_config.json` before
   scoring final test labels. Report each preselected family leader on test.

Pairwise prediction ties within 1e-12 receive half credit; equal-rating pairs
are omitted. NDCG@10 uses `2**rating - 1` gains and averages tied score orderings.
RMSE is the square root of mean per-user MSE; MAE and NDCG are also user-macro.
The candidates are each user's held-out rated movies. These metrics measure
ordering among observed movies, not retrieval quality over the entire catalog.

## Data coverage and limitations

The official [MovieLens small archive](https://grouplens.org/datasets/movielens/latest/)
was checked against the bundled movies and ratings, record for record, before
adding its 3,683 tag applications across 1,572 movies. MovieLens describes these
as user-created words or short phrases in its
[data README](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html).
The original README and license are preserved in
[`data/MOVIELENS-README.txt`](../data/MOVIELENS-README.txt).

After temporal and author filtering, text covers 26.42% of training ratings,
20.72% of validation ratings, and **17.29% of test ratings**. Untagged movies stay
neutral for the text component. Plot fields are implemented and tested with
fixtures; this real-data benchmark contains **no plot summaries**. No plots were
fabricated or substituted with titles. Enriched plots/static keywords supplied
later must be available at the evaluation cutoff; rerun selection for that data.

The selected model's test pairwise accuracy changes by history size:

| Training ratings | Users | Baseline | Selected |
| --- | ---: | ---: | ---: |
| 5–19 | 128 | 49.3891% | 49.0062% |
| 20–99 | 316 | 55.3872% | 55.7447% |
| 100+ | 159 | 59.2109% | 59.4829% |

The short-history slice regresses slightly. No slice-specific settings were
retuned after observing test results. More keyword/plot coverage and a new
holdout would be needed before making stronger claims.

## Serving performance and checks

On this Apple Silicon machine with one numerical thread, recommending from
9,742 catalog movies took the following median of three warm runs. The fitted
user profiles are cached; candidate text is transformed in bounded batches.

| User / history ratings | Baseline warm | Selected warm | Selected cold |
| --- | ---: | ---: | ---: |
| 49 / 12 | 131 ms | 158 ms | 161 ms |
| 528 / 43 | 139 ms | 164 ms | 164 ms |
| 414 / 1,618 | 117 ms | 123 ms | 128 ms |

Timing is machine-dependent; these are latency samples, not throughput claims.
The original search and holdout report took 46.1 seconds. Serving predictions
match the vectorized evaluator across all test users to within **8.9e-16**.

All **106 tests pass**, including positive/negative preferences, plot input,
empty/stopword/OOV text, duplicate keywords, cold start, sparse histories,
recency, caps, explanation reconciliation, cached and batched serving, LSA,
training-only vocabulary, future/own-tag exclusion, timestamp ties, metrics,
holdout-label perturbation, and the saved serving preset. The original 81
tests remain green. Both selected and baseline demo paths run successfully.

## Reproduce and use

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-experiments.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.evaluate_content --output-dir reports/week6
.venv/bin/python -m src.content.demo --user-id 414 --limit 3
python3 -m src.content.demo --baseline --limit 3
```

The experiment dependencies are pinned separately from the legacy application's
requirements. Experiments do not need further downloads once the included tags
are present. A repeat run reproduced the selected preset and every validation
and test metric exactly; timings are recorded separately.

```python
from src.content import load_selected_model

# MovieMetadata accepts keywords=("space exploration",) and plot="...".
# Supply only metadata available when making the recommendation.
model = load_selected_model(ratings, movies)
recommendations = model.recommend(user_id=1, limit=10)
```

`ContentModel(...)` still provides the previous defaults for existing callers;
`load_selected_model(...)` and the demo use
[`src/content/selected_config.json`](../src/content/selected_config.json).
Experiment reruns write reports without silently changing that serving preset.

Artifacts: [all metrics and protocol](../reports/week6/results.json),
[validation leaderboard](../reports/week6/validation.csv),
[per-user test metrics](../reports/week6/test_users.csv), and
[locked selection](../reports/week6/selected_config.json). The results record
input SHA-256 hashes, environment versions, cohort counts, and runtime samples.

Implementation references:
[scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
and [truncated SVD / LSA](https://scikit-learn.org/stable/modules/decomposition.html#lsa).
