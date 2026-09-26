# Penumbra

[Penumbra](https://penumbra.mov) is a personalized movie recommendation system designed to learn what individual users enjoy and rank movies accordingly.

The project is currently focused on building and evaluating the recommendation engine that will power the future Penumbra application.

## Recommendation System

Penumbra's recommendation architecture is being developed around three components:

### Content-Based Recommendation

Models a user's preferences from properties of movies they have previously rated, such as genres and other movie metadata.

### Collaborative Filtering

Learns from rating patterns across users to identify movies a user may enjoy based on the behavior of users with similar preferences.

### Hybrid / Reranking

Combines recommendation signals and reranks candidate movies to produce the final personalized ranking.

The current hybrid system uses a pairwise logistic regression reranker with three features:

- Personal preference score
- Movie quality score
- Movie popularity

Future versions will incorporate outputs from both the content-based and collaborative models into a unified hybrid ranking system.

## Current Benchmark

**Dataset:** MovieLens 1M

### Evaluation Protocol

For each eligible user, ratings are ordered chronologically and divided into:

- First 60% → preference profile
- Next 20% → pairwise training
- Final 20% → held-out testing

Population-level quality and popularity statistics exclude the target user's ratings during training and evaluation to prevent the user's hidden ratings from influencing features used to predict that same user's preferences.

### Week 1 Frozen Benchmark

| Model | Pairwise Accuracy |
| --- | ---: |
| Movie-average baseline | 62.792% |
| Handcrafted heuristic | 60.597% |
| **ML reranker** | **63.926%** |

The ML reranker improves by:

- **+3.329 percentage points** over the handcrafted heuristic
- **+1.134 percentage points** over the movie-average baseline

Evaluation included:

- **601 users**
- **747,939 held-out rating pairs**
- Identical results across repeated evaluation runs
- Full automated test suite passing before the benchmark was frozen

The benchmark is frozen under the Git tag:

`week1-reranker-benchmark`

More detailed methodology and architecture are documented in `docs/hybrid-reranking.md`.

## Current Project Status

Penumbra is currently in the recommender-system development stage.

Current work includes:

- Content-based recommendation
- Collaborative filtering
- Learned hybrid reranking
- Leakage-safe offline evaluation
- Recommendation-system testing
- Architecture and integration planning

Planned development includes:

- Integration of content and collaborative model outputs
- Stronger hybrid ranking
- Recommendation quality analysis and evaluation
- Group recommendations
- Diversity and fairness-aware reranking
- Application and backend integration

## Repository Structure

```text
src/
├── content/          # Content-based recommendation
├── collaborative/    # Collaborative filtering
├── hybrid/           # Hybrid models and reranking
├── evaluation/       # Offline recommender evaluation
├── data_processing/  # Dataset processing
└── models/           # Saved model artifacts

tests/
├── content/
├── collaborative/
└── hybrid/

docs/                 # Architecture and model documentation
```

## Dataset

Current development and evaluation use the MovieLens 1M dataset.

MovieLens provides approximately one million movie ratings from thousands of users and is used as an offline benchmark while Penumbra's recommendation system is developed.

## Development

Install dependencies:

```bash
pip install -r requirements.txt
```

data/
├── movies.csv
└── ratings.csv


## Standalone Content Model

The standalone content model lives in `src/content/`. It builds a cached user taste profile from personal rating residuals and learns preferences across genre, director, runtime, release decade, and language. Predictions combine capped feature adjustments with the user's baseline, while missing metadata remains neutral. Final scores are clamped to `0` through `5`, and optional debug output exposes intermediate scoring values.

The batch API preserves user order and then movie order. `predict_unseen` directly removes movies each user has already rated and returns deterministic unseen-movie predictions with an optional per-user limit. Unknown users and movie IDs raise explicit errors, while known movies with missing or unknown genres safely fall back to the user's baseline. More advanced candidate retrieval remains outside this package.

Week 2 adds evidence-based regularization, configurable recency weighting that preserves older ratings, and confidence based on matching genre support. Predictions include structured genre reasons. The standalone `predict_batch(profile, movies)` API also accepts unseen movies outside the training catalog. See [the content-model specification](docs/content-model.md) for formulas, configuration, and the Week 1/2 completion checklist. Confidence measures evidence support, not calibrated prediction accuracy.

The standalone model is not yet connected to `main.py`, the existing heuristic recommender, or the ML reranker.

Run its tests with:

```bash
python3 -m unittest discover -s tests -v
```

Inspect a real MovieLens user's profile and three unseen-movie calculations with:

```bash
python3 -m src.content.demo
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


## Project Goal

Penumbra's goal is to move beyond generic movie popularity and produce recommendations that reflect each user's individual taste.

The current system establishes the first reproducible baseline for measuring that goal. Future model changes will be evaluated against frozen benchmarks rather than judged only by whether individual recommendations appear reasonable.
