# Content Model — Weeks 1–6

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
per user; it is not a top-score ranking. New users receive an empty profile with a neutral score (2.5 by default), zero
confidence, and no reasons. Unknown movie IDs raise `UnknownMovieError`.

## Implementation

| Module | Responsibility |
| --- | --- |
| `schemas.py` | Validated inputs, predictions, reasons, and debug fields |
| `baselines.py` | Personal mean rating and residuals |
| `genres.py` | Normalize multi-genre residuals and aggregate weighted evidence |
| `features.py` | Learn director, runtime, decade, and language preferences |
| `reliability.py` | Profile configuration and reproducible recency weights |
| `profiles.py` | Versioned `content-v4` profiles and metadata coverage |
| `scoring.py` | Single and standalone batch predictions with confidence |
| `model.py` | Catalog lookup, cached profiles, and unseen filtering |
| `demo.py` | Real MovieLens profile and unseen movie predictions |

See [the full model specification](../../docs/content-model.md) for formulas,
defaults, missing-data policies, the Week 1–4 checklist, and limitations.
Confidence expresses evidence support; it is not a calibrated accuracy probability.

## Verification

From the repository root, install numerical dependencies for the text tests and
selected demo. The core model and baseline demo still require only Python:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-experiments.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.content.demo
python3 -m src.content.demo --baseline
```

The demo reads `data/ratings.csv` and `data/movies.csv`, and prints actual unseen
movie scores, confidence, raw counts, effective evidence, and shrunk preferences.

## Week 4: complete single-user recommendations

```python
# Highest scores first; already-rated movies are excluded.
recommendations = model.recommend(user_id=1, limit=10)
new_user_recommendations = model.recommend(user_id=999, limit=10)
```

- **Cast without oversized influence:** pass `cast=("Actor A", "Actor B")` or an
  enriched record with `cast="Actor A|Actor B"`. Names are trimmed and deduplicated
  ignoring case. Each rated movie shares its credit across its actors; candidate
  preferences are averaged, not added. Cast weight and component cap both default
  to 0.25. The cap applies before and after weighting.
- **Missing data and new users:** missing details make no score adjustment.
  Ratings with missing movie records still contribute to the user's mean.
  With no ratings, `ProfileConfig(cold_start_score=2.5)` supplies a neutral starting
  point with zero confidence. All movies initially tie, so movie ID breaks ties;
  these are not personalized picks. Sparse histories retain the existing shrinkage
  and low evidence confidence.
- **Supported explanations:** each reason names a matched feature, its effective
  evidence, and `score_contribution` in rating points. Contributions allocate each
  feature's net adjustment to matching preferences in its direction, after caps
  and weighting; opposing preferences already reduce that net adjustment.
  `strength = score_contribution / 5`. Reasons sum to the score adjustment before
  the final 0–5 clamp. Missing, unknown, disabled, neutral, and fully cancelled
  features emit no reasons. These are model explanations, not causal claims.

`recommend` scores every unseen movie before limiting and orders by score,
confidence, then movie ID. `predict_unseen` retains its original ID-order behavior.
Run `.venv/bin/python -m src.content.demo` for ranked results and reason contributions.
The bundled MovieLens CSV has no cast: cast behavior is verified with enriched
fixtures in `tests/test_week4.py`; real cast preferences require enriched data.

## Week 6: text and measured configuration selection

`MovieMetadata` accepts optional `keywords: tuple[str, ...]` and `plot: str`.
The record loader accepts pipe-separated keywords. Missing text and terms absent
from the training vocabulary supply no score adjustment. TF-IDF and optional
LSA are fitted only on the user's rated movies, then cached. Candidate text never
changes IDF or vocabulary. See [`text.py`](text.py).

```python
from src.content import TextConfig, ContentModel, load_selected_model

selected = load_selected_model(ratings, movies)  # Frozen Week 6 preset
experimental = ContentModel(ratings, movies, text_config=TextConfig(
    source="combined", ngram_max=2, weight=0.25,
    regularization_strength=5, max_adjustment=0.5,
))
results = selected.recommend(user_id=1, limit=10)
```

Text scores are similarity-weighted, recency-weighted rating residuals divided
by similarity evidence plus a prior. The weighted text component is capped and
added to the core score before the final 0–5 clamp. Text reasons report the
actual adjustment and matching similarity mass. Confidence takes the larger
of core support and text support, avoiding addition of overlapping evidence;
it remains uncalibrated. English stopwords are used; multilingual text needs
separate validation.

For the standalone batch API, pass `text_profile=model.build_text_profile(user_id)`
to `predict_batch(...)`. For time-sensitive tag data, use
`with_keywords_as_of` in `src/data_processing/text_metadata.py` before constructing
a model. Static enrichment is assumed available as of prediction time.

The [Week 6 report](../../docs/week6-experiments.md) records 162 combinations,
separate validation selection and test evaluation, uncertainty, sparse coverage,
and serving latency. The demo uses the selected preset; `--baseline` uses the
original defaults. The legacy `main.py` reranker is a separate application path.

## Optional TMDB enrichment

Use `python3 -m src.enrich_movies --limit 100` after configuring a local TMDB
credential. The command creates a cached metadata snapshot with plot, keyword,
cast, director, runtime, and language fields. Load it through
`movie_metadata_from_records`, or pass `--movies-file data/movies_enriched.csv`
to the content demo. See [TMDB setup and usage](../../docs/tmdb-enrichment.md).
