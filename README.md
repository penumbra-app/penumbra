# Flick

Flick is an in-progress personalized movie recommendation system built on the MovieLens dataset.

The project combines a heuristic recommendation pipeline with a supervised machine-learning reranker that learns how to order movies from user preference data.

## Data and team handoff

Dataset CSVs, enriched metadata, API credentials, and per-user reports are local-only.
Restore the private data bundle before running data-dependent commands. See
[the hybrid team handoff](docs/hybrid-handoff.md) for setup and integration.
The completed enrichment selected a content model with **57.47% pairwise accuracy**
and **0.7737 NDCG@10**; see [the full report](reports/tmdb-full/results.json).

## Week 6: measured content improvements

For current evaluation, use the [full-catalog and future-confirmation workflow](docs/evaluation-environment.md).
For the new dataset, follow the [MovieLens 32M preparation and reserved evaluation guide](docs/movielens-32m.md).
The historical metrics below remain development/regression evidence.

The standalone content model now supports keyword and plot TF-IDF, optional LSA
embeddings, and a reproducible validation/test experiment. Comparing **162
configurations** selected a small TF-IDF unigram/bigram blend. On **603 held-out
users**, pairwise accuracy changed from **55.18% to 55.36%**, NDCG@10 from
**0.7545 to 0.7563**, and RMSE from **1.0302 to 1.0265**. The accuracy improvement
is small and its 95% confidence interval includes zero.

The content demo uses the saved validation-selected preset. Keyword experiments
use time-filtered MovieLens tags; plot support is fixture-tested because real
plots are not bundled. See [the experiment report](docs/week6-experiments.md)
for every configuration, limitations, latency, and reproduction instructions.
These content-model results use a different protocol from the legacy reranker
below and are not directly comparable.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-experiments.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.content.demo --user-id 414
.venv/bin/python -m src.evaluate_content
```

## TMDB enrichment

Add your TMDB API Read Access Token to the ignored `.env` file using
[`.env.example`](.env.example), then fetch movie metadata:

```bash
.venv/bin/python -m src.enrich_movies --limit 100
.venv/bin/python -m src.explore_data --movies-file data/movies_enriched.csv
.venv/bin/python -m src.content.demo --user-id 414 --movies-file data/movies_enriched.csv
```

This adds plot overviews, keywords, cast, directors, runtime, and language using
MovieLens's official TMDB ID mappings. Successful responses are cached; omit
`--limit` to fetch the remaining catalog. See [setup and evaluation instructions](docs/tmdb-enrichment.md).

This product uses the TMDB API but is not endorsed or certified by TMDB.

## Legacy reranker results

The learned reranker was evaluated against the original heuristic ranking system using held-out user ratings.

| Model              |   Pairwise Ranking Accuracy |
| ------------------ | --------------------------: |
| Heuristic baseline |                       62.0% |
| ML reranker        |                       66.9% |
| Improvement        | **+4.95 percentage points** |

Evaluation covered:

* **601 users**
* **941,183 held-out movie pairs**
* Chronological **60% / 20% / 20%** profile, training, and testing splits

Pairwise accuracy measures whether the system correctly ranks the movie that a user actually rated higher above a movie they rated lower.

## How It Works

Flick currently uses two ranking stages.

### 1. Personalized Heuristic Scoring

The baseline recommender builds a user taste profile from previous ratings and scores unseen movies using:

* **Genre preferences** derived from the user's rating history
* **Recency weighting**, so newer ratings contribute more strongly
* **Release-year compatibility** between previously rated movies and candidates
* **Confidence weighting**, so preferences supported by more evidence receive greater influence
* **Bayesian-adjusted movie quality**, which prevents movies with very few ratings from being overvalued
* **Diversity-aware reranking** to reduce excessive genre repetition in recommendation lists

### 2. Machine-Learned Reranking

Flick also includes a pairwise learning-to-rank model trained using MovieLens user preferences.

For each movie, the current model uses:

* Personalized preference score
* Bayesian movie-quality score
* Log-scaled movie popularity

Training examples are created from pairs of movies rated differently by the same user.

For example, if a user rated:

```text
Movie A: 5 stars
Movie B: 3 stars
```

the model learns that Movie A should rank above Movie B based on the difference between their feature vectors.

The current implementation uses:

* `StandardScaler` for feature normalization
* `LogisticRegression` from scikit-learn as the pairwise ranking model

A linear model was chosen as an interpretable baseline before introducing more complex ranking methods.

## Data Split

Each sufficiently active user's rating history is ordered chronologically and divided into:

```text
First 60%  -> Build user preference profile
Next 20%   -> Generate ML training comparisons
Final 20%  -> Evaluate ranking performance
```

This allows the system to learn from earlier behavior and evaluate against later ratings that were not used as pairwise training labels.

## Architecture

```text
MovieLens Ratings
        |
        v
User Taste Profile
        |
        v
Candidate Movie Generation
        |
        v
Personalization / Quality / Popularity Features
        |
        v
Pairwise ML Reranker
        |
        v
Ranked Recommendations
```

The longer-term architecture is designed to support additional recommendation signals:

```text
Content / Genre Model ----\
Collaborative Filtering ---\
Additional User Signals ----> Hybrid Feature Layer
Movie Quality ------------/           |
Popularity --------------/            v
                                Learned Reranker
                                      |
                                      v
                          Group / Diversity Reranking
                                      |
                                      v
                           Final Recommendations
```

## Dataset

Flick currently uses the MovieLens small dataset:

* **100,836 ratings**
* **9,742 movies**
* **610 users**

The included data files are located in:

```text
data/
├── movies.csv
└── ratings.csv
```

## Project Structure

```text
flick/
├── data/
│   ├── movies.csv
│   └── ratings.csv
│
├── src/
│   ├── content/
│   │   ├── baselines.py
│   │   ├── demo.py
│   │   ├── errors.py
│   │   ├── genres.py
│   │   ├── model.py
│   │   ├── profiles.py
│   │   ├── README.md
│   │   ├── scoring.py
│   │   └── schemas.py
│   ├── data_processing/
│   │   └── movielens.py
│   ├── models/
│   │   └── ml_reranker.joblib
│   ├── evaluate_ranking.py
│   ├── explore_data.py
│   ├── genre_recommender.py
│   ├── load_data.py
│   └── ml_reranker.py
│
├── tests/
│   ├── fixtures.py
│   ├── test_baselines.py
│   ├── test_content_model.py
│   ├── test_genres.py
│   ├── test_profiles.py
│   ├── test_scoring.py
│   └── test_schemas.py
│
├── main.py
├── requirements.txt
└── README.md
```

## Standalone Content Model

The standalone content model lives in `src/content/`. It builds a cached user taste profile from personal rating residuals, learns genre, director, runtime, release-decade, and language preferences, and predicts requested movies with capped feature adjustments added to the user's baseline. Missing details are neutral; director, runtime, and language require enriched movie metadata. Final scores are clamped to `0` through `5`, and optional debug output exposes every intermediate value.

The batch API preserves user order and then movie order. `predict_unseen` directly removes movies each user has already rated and returns deterministic unseen-movie predictions with an optional per-user limit. Unknown users and movie IDs raise explicit errors, while known movies with missing or unknown genres safely fall back to the user's baseline. More advanced candidate retrieval remains outside this package.

Week 2 adds evidence-based regularization, configurable recency weighting that preserves older ratings, and confidence based on matching genre support. Predictions include structured genre reasons. The standalone `predict_batch(profile, movies)` API also accepts unseen movies outside the training catalog. See [the content-model specification](docs/content-model.md) for formulas, configuration, and the Week 1/2 completion checklist. Confidence measures evidence support, not calibrated prediction accuracy.

The standalone model is not yet connected to `main.py`, the existing heuristic recommender, or the ML reranker.

Run the full suite, including text experiments, after installing the experiment
dependencies above:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Inspect a real MovieLens user's profile and three unseen-movie calculations with:

```bash
.venv/bin/python -m src.content.demo
```

### Key Files

`src/content/model.py`
Provides cached profile construction, single prediction, and deterministic batch prediction.

`src/content/profiles.py`
Builds the versioned taste profile containing the user baseline, genre preferences, evidence counts, and metadata coverage.

`src/content/scoring.py`
Calculates the bounded genre component, weighted adjustment, score clamping, and optional debug values.

`src/content/schemas.py`
Defines validated model inputs, prediction outputs, reason signals, and prediction debug data.

`genre_recommender.py`
Builds personalized genre preferences, calculates movie-quality scores, applies recency/year weighting, and contains the heuristic recommendation logic.

`ml_reranker.py`
Builds pairwise training examples, trains the logistic-regression ranker, loads/saves the trained model, and performs learned reranking.

`evaluate_ranking.py`
Evaluates the heuristic and ML ranking systems against held-out user preferences.

`main.py`
Runs the current ML-powered recommendation pipeline for a selected MovieLens user.

## Running Flick

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate recommendations

```bash
python main.py
```

The current demo selects a MovieLens user and outputs their highest-ranked unseen movies along with the ranking signals used by the model.

Example output:

```text
1. Example Movie
   Genres: Drama|Thriller
   ML score: 1.742
   Personal score: +0.126
   Quality score: +0.481
   Popularity: 6.532
```

## Evaluation

The heuristic and learned models are evaluated on movie pairs from each user's held-out test ratings.

For two test movies with different ratings:

```text
Actual ratings:
A = 4.5
B = 2.5
```

the ranking is considered correct when:

```text
score(A) > score(B)
```

Tied predicted scores receive half credit.

The final metric is the mean pairwise accuracy across evaluated users so that highly active users do not completely dominate the result.

Current benchmark:

```text
Users evaluated:     601
Test pairs:          941,183

Heuristic accuracy:  61.997%
ML accuracy:         66.948%
Difference:          +4.952 percentage points
```

## Current Limitations

Flick is still under active development.

Current limitations include:

* The learned ranker currently uses only three ranking features
* Movie popularity and quality are currently calculated from global dataset aggregates
* Pairwise example generation is quadratic in the number of training movies for each user
* Collaborative filtering has not yet been incorporated
* The ML recommendation path has not yet been combined with the full diversity/group reranking pipeline

These provide several directions for future experimentation and evaluation.

## Roadmap

Planned work includes:

* Collaborative filtering
* Additional personalized ranking features
* Hybrid recommendation scoring
* Group recommendations
* Fairness-aware group aggregation
* Diversity-aware post-ML reranking
* More efficient pair sampling
* Additional ranking metrics such as NDCG@K and Recall@K
* Stronger temporal evaluation with training-only aggregate statistics
* User-facing application interface

## Tech Stack

* Python
* pandas
* NumPy
* scikit-learn
* joblib

## Status

Flick is currently under active development.

## Week 4 — Flick V1 content recommendations

The standalone content model now includes cast with shared credit and a small
score cap, neutral handling of missing metadata, cold-start profiles, and reasons
that report evidence and score contributions. `ContentModel.recommend(user_id,
limit=10)` ranks unseen movies; `.venv/bin/python -m src.content.demo` runs it on MovieLens.
This path is separate from the existing ML reranker in `main.py`.
See [the content model guide](src/content/README.md#week-4-complete-single-user-recommendations)
for usage and simple explanations. Cast requires enriched metadata; the bundled
MovieLens CSV does not include it.
