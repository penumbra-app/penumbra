# Content Model — Weeks 1–3

The standalone content recommender implements:

```text
ratings → UserTasteProfile → unseen movie → PredictionResult
                          → list of movies → batch predictions
```

Week 2 adds effective evidence counts, regularization toward the user's baseline,
gentle recency weighting, and confidence based on candidate genre support.
The existing `PredictionResult` contract is preserved: user ID, movie ID, score
from 0 to 5, confidence from 0 to 1, structured reason signals, and optional debug.

Week 3 adds director, runtime, release-decade, and language preferences. Each
feature has a cap before and after weighting; the final rating stays within 0–5.
Missing metadata stays neutral. Supply `directors=("Name",)`,
`runtime_minutes=105`, `release_year=1998`, and `language="en"` on `MovieMetadata`
to use these signals. The bundled CSV only supplies release years, so the other
three require enriched records. See the specification for buckets and weights.

## Quick start

```python
from src.content import (
    MovieMetadata, UserRating, build_profile, predict_one, predict_batch,
)

movies = (
    MovieMetadata(1, "Space adventure", ("Sci-Fi",)),
    MovieMetadata(2, "Period drama", ("Drama",)),
)
ratings = (UserRating(1, 1, 5.0), UserRating(1, 2, 1.0))
profile = build_profile(1, ratings, movies)
unseen = MovieMetadata(3, "New space adventure", ("Sci-Fi",))
result = predict_one(profile, unseen, include_debug=True)
print(f"User {result.user_id} + unseen movie {result.movie_id} "
      f"→ predicted {result.predicted_score:.3f} / 5")
print(f"Confidence: {result.confidence:.3f}")
batch = predict_batch(profile, [unseen])
```

This prints a prediction of `3.333 / 5` and confidence `0.048`. The small
adjustment and low confidence reflect only one supporting Sci-Fi rating.
Standalone prediction accepts metadata for movies outside the training catalog.

For cached profiles and catalog IDs:

```python
from src.content import ContentModel, ProfileConfig, ScoringConfig

model = ContentModel(
    ratings, (*movies, unseen),
    config=ScoringConfig(confidence_prior_count=5.0),
    profile_config=ProfileConfig(
        regularization_strength=5.0,
        recency_half_life_days=365.0,
        minimum_recency_weight=0.5,
    ),
)
one = model.predict_one(1, 3)
batch = model.predict(user_ids=(1,), movie_ids=(3,))
unseen_results = model.predict_unseen(user_ids=(1,), limit=10)
```

`predict_batch` preserves candidate order and duplicates. `ContentModel.predict`
returns the Cartesian product in user order, then movie order. `predict_unseen`
excludes rated movies and selects ascending movie IDs, with an optional limit
per user; it is not a top-score ranking. Unknown catalog IDs raise explicit
`UnknownUserError` or `UnknownMovieError` errors.

## Implementation

| Module | Responsibility |
| --- | --- |
| `schemas.py` | Validated inputs, predictions, reasons, and debug fields |
| `baselines.py` | Personal mean rating and residuals |
| `genres.py` | Normalize multi-genre residuals and aggregate weighted evidence |
| `features.py` | Learn director, runtime, decade, and language preferences |
| `reliability.py` | Profile configuration and reproducible recency weights |
| `profiles.py` | Versioned `content-v3` profiles and metadata coverage |
| `scoring.py` | Single and standalone batch predictions with confidence |
| `model.py` | Catalog lookup, cached profiles, and unseen filtering |
| `demo.py` | Real MovieLens profile and unseen movie predictions |

See [the full model specification](../../docs/content-model.md) for formulas,
defaults, missing-data policies, the Week 1–3 checklist, and limitations.
Confidence expresses evidence support; it is not a calibrated accuracy probability.

## Verification

From the repository root, with no third-party packages required for these paths:

```bash
python3 -m unittest discover -s tests -v
python3 -m src.content.demo
```

The demo reads `data/ratings.csv` and `data/movies.csv`, and prints actual unseen
movie scores, confidence, raw counts, effective evidence, and shrunk preferences.
