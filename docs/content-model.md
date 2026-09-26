# Content model specification

## Scope and completion

The standalone model predicts a user's 0–5 rating from personal rating history
and movie genres, directors, runtime, release decade, and language. It returns the existing `PredictionResult`, including a 0–1
confidence score. It does not use other users' preferences.

| Milestone | Deliverable | Implementation / verification |
| --- | --- | --- |
| Week 1 | Reusable `UserTasteProfile` | `build_profile`, cached `ContentModel.build_profile`; profile tests |
| Week 1 | Standalone unseen-movie prediction | `predict_one(profile, movie)`; scoring and reliability tests |
| Week 1 | Profile plus movie-list batch prediction | `predict_batch(profile, movies)`; order, generators, duplicates, empty inputs, single/batch equality |
| Week 1 | Catalog batch and unseen filtering | `ContentModel.predict` and `predict_unseen`; model tests |
| Week 1 | Model documentation and runnable demo | This specification, package README, `python3 -m src.content.demo` |
| Week 2 | Evidence counts and regularization | Raw movie counts, effective evidence, shrunk genre residuals; exact arithmetic and sparse/dense tests |
| Week 2 | Recency weighting preserving older taste | Configurable decay with positive floor; old, missing, disabled, and as-of tests |
| Week 3 | Four metadata preferences and per-feature caps | `features.py`, enriched record adapter, feature tests |
| Week 2 | Evidence-based confidence | Candidate support and history factor; coverage, overlap, sparse-history, and batch tests |

## Inputs and outputs

`UserRating(user_id, movie_id, rating, timestamp=None)` accepts finite ratings
from 0 to 5 and optional non-negative integer Unix timestamps in seconds.
`MovieMetadata` supplies a positive movie ID, title, and genres. Other metadata
fields supply director, runtime, release-era, and language signals; cast remains unused.

`build_profile(user_id, ratings, movies, config=None)` accepts rating iterables
and either a movie iterable or a mapping keyed by movie ID. It selects only the
requested user's ratings. `UserTasteProfile` stores the personal baseline,
rating count, per-genre evidence, configuration, coverage metadata, and version
`content-v3`. Profiles are immutable; rebuild them when training data or settings
change. `ContentModel` caches them for its input snapshot.

The public prediction contract remains:

```text
user_id, movie_id, predicted_score, confidence, reason_signals, debug (optional)
```

Use `result.to_dict()` for a nested serializable representation. Single and batch
APIs return the same values, including confidence, reasons, and debug when enabled.

## Learning genre preferences

The personal baseline remains the arithmetic mean of all supplied user ratings:

```text
b = sum(rating_i) / rating_count
residual_i = rating_i - b
```

Preserving this baseline keeps the user's overall rating scale stable. Recency
weights apply to genre learning, not to the baseline. A single rating therefore
has zero residual and supplies no directional genre preference.

Trim genre whitespace and deduplicate case-insensitively within and across
movies. A rating for a movie with `k_i` unique genres contributes
`residual_i / k_i` to each genre. Thus extra genre labels cannot increase the
total residual contributed by one movie.

### Recency

With age in days and half-life `h`, each observation has weight:

```text
w_i = floor + (1 - floor) * 2 ** (-age_days_i / h)
```

Defaults are `h = 365` and `floor = 0.5`: a new rating has weight 1, a rating one
year old has weight 0.75, and very old ratings retain at least half their weight.
This changes relative genre influence gently without deleting older preferences.

The default reference timestamp is the user's latest supplied dated rating.
This makes historical datasets and repeated runs reproducible without consulting
the wall clock. It measures recency within the supplied history; pass an explicit
`reference_timestamp` when absolute staleness matters. Ratings later than that
explicit as-of time raise `ValueError`; callers must supply an appropriate
training cutoff instead of silently learning from future ratings.

Missing timestamps receive the floor weight when a reference exists. If all
timestamps and the explicit reference are absent, every rating gets weight 1.
Set `recency_half_life_days=None` to disable recency weighting. Even in that mode,
an explicit as-of boundary is validated.

### Evidence and shrinkage

For genre `g`, accumulate:

```text
movie_count_g = number of distinct rated movies containing g
E_g = sum(w_i for rated movies containing g)
T_g = sum(w_i * residual_i / k_i for rated movies containing g)
mean_contribution_g = T_g / E_g
regularized_contribution_g = T_g / (E_g + lambda)
```

The default `lambda = 5` acts as five units of neutral prior evidence. A genre
with one effective observation retains `1/6` of its raw preference; ten retain
`10/15`. Neutral means zero residual, so weak preferences move predictions
toward the user's baseline. Both positive and negative preferences shrink.
Set `regularization_strength=0` to disable shrinkage.

`GenrePreference` exposes `movie_count`, `effective_evidence_count`, weighted
`total_contribution`, `mean_contribution`, and `regularized_contribution`.
`profile.preference_for(genre)` returns the regularized value;
`evidence_for` returns the raw count and `effective_evidence_for` the weighted
count. Unknown genres return zero. Effective counts are sums of bounded weights,
not a statistical effective-sample-size estimate. They are not summed across
genres to claim independent observations.

## Week 3 metadata preferences

`profile.feature_preferences` stores director, runtime, release-era, and language
preferences with raw movie counts and effective evidence. These use the same
baseline residuals, recency weights, and regularization as genres. Multiple
unique directors split a movie's residual equally. Names and languages are
trimmed and case-folded. Language values must use a consistent vocabulary
(e.g. `en` throughout); aliases such as `English` are not automatically translated.

Runtime buckets are `<90`, `90–<120`, `120–<150`, and `150+` minutes. Release eras
are calendar decades (1990–1999 maps to `1990s`). Missing values add no evidence;
unseen values have neutral preferences. Candidate values are averaged within
each feature, including unknown values, so adding directors cannot multiply it.

| Feature | Default weight | Component and final adjustment cap |
| --- | --- | --- |
| Genre | 1.0 | ±1.0 |
| Director | 0.5 | ±0.5 |
| Runtime | 0.25 | ±0.25 |
| Release era | 0.25 | ±0.25 |
| Language | 0.25 | ±0.25 |

For each feature, clamp the mean preference, multiply by its weight, and clamp
again to the same cap. Thus weights below one reduce influence, while arbitrarily
large finite weights cannot bypass the cap. Configurable caps must be in `[0, 5]`.
Use `<feature>_weight` and `max_abs_<feature>_component` in `ScoringConfig`, with
`release_era` as the era feature name. The default total adjustment is at most
2.25 points in either direction, and the final score stays in `[0, 5]`.
Debug's `feature_components` records candidate and matched values, raw and bounded
components, weights, final adjustments, and support for each new feature.

The MovieLens adapter accepts optional enriched columns `directors` (pipe-separated
names or a sequence), `runtime_minutes`, `release_year`, and `language`. Explicit
years override title years. Blank and numeric NaN values count as missing.
The bundled MovieLens data has title years but no director, runtime, or language
metadata; those preferences need enriched input. No metadata is fabricated.

## Scoring and confidence

Each unique candidate genre contributes its regularized preference, or zero
when unknown. Average over all candidate genres, including unknown ones:

```text
raw_genre_component = average(regularized candidate preferences)
bounded_genre_component = clamp(raw_genre_component, -bound, bound)
genre_adjustment = clamp(genre_weight * bounded_genre_component, -bound, bound)
score = clamp(b + genre_adjustment + sum(other_feature_adjustments), 0, 5)
```

The default bound and genre weight are both 1. Missing or entirely unknown
genres supply zero genre adjustment and support; other metadata can still contribute. Debug's `raw_genre_component`
means before the component bound, but after profile regularization.

Confidence measures available evidence supporting the candidate's genre estimate:

```text
history_support = rating_count / (rating_count + prior)
genre_support = average(E_g / (E_g + prior) over ALL candidate genres)
confidence = history_support * average(support of enabled, present feature types)
```

The default positive `prior` is 5 (`ScoringConfig.confidence_prior_count`). Unknown
genres have zero support. A candidate without genres has zero support. Averaging
instead of summing genre support prevents the same multi-genre observations from
multiplying confidence. More matching evidence increases confidence; partial
coverage and discounted older evidence reduce it. Neutral preferences can still
have support. Disabling a feature weight or component bound removes its adjustment, reasons,
and confidence contribution. Missing feature types are excluded from the support
average; present but unknown values supply zero support.

Confidence is a bounded evidence heuristic, not a calibrated probability that
the rating will be correct. These defaults have not been optimized against a
held-out evaluation set. Conflicting ratings and prediction error variance are
not currently included in confidence.

Nonzero matched preferences produce `ReasonSignal` entries for each feature type. `strength`
is the regularized preference divided by 5 and bounded to `[-1, 1]`;
`evidence_count` is effective evidence. Signs indicate affinity or aversion;
strengths describe individual learned preferences, not additive final-score
contributions or probabilities. Unknown and neutral preferences produce no reason.

Debug includes the baseline, candidate/matched/unknown genres, components,
weight, unclamped score, clamping status, genre support, and effective evidence
for each matched genre.

## API behavior and edge cases

- `predict_one(profile, movie, config=None, include_debug=False)` scores any
  supplied movie metadata, including movies outside the training catalog.
- `predict_batch(profile, movies, config=None, include_debug=False)` accepts an
  iterable, preserves order and duplicates, and returns a tuple; empty input
  returns `()`.
- `ContentModel(ratings, movies, config=None, profile_config=None)` accepts
  separate scoring and profile settings. `predict(user_ids, movie_ids)` returns
  a Cartesian product in user order, then movie order, not zipped pairs.
- `predict_unseen(user_ids, limit=None)` excludes rated movies, sorts remaining
  IDs, and limits separately per user. The limit selects IDs, not top scores.
- Standalone and explicit catalog prediction can score a previously rated movie;
  callers select unseen candidates, or use `predict_unseen` for filtering.
- No ratings for a user raises `UnknownUserError`; unknown catalog movie IDs
  raise `UnknownMovieError`. No global cold-start baseline is invented.
- Missing movie metadata and empty/blank genre metadata contribute to the
  baseline but supply no genre evidence. Coverage counts report these gaps.
- Duplicate user/movie ratings are rejected when building a profile so repeated
  events cannot inflate evidence. Callers must resolve reratings before input.
  Duplicate movie IDs and mismatched movie mapping keys are also rejected.
- Profile and scoring configuration reject invalid ranges, nonfinite numbers,
  and boolean values.

For Week 1 arithmetic comparisons, build profiles with
`ProfileConfig(regularization_strength=0, recency_half_life_days=None)`.
Version 3 defaults intentionally produce different scores; rebuild saved
profiles rather than treating old raw means as regularized estimates.

## Validation and integration boundary

Run `python3 -m unittest discover -s tests -v` for the deterministic suite.
It checks exact profile and prediction arithmetic, baseline preservation,
regularization, recency, confidence, error paths, serialization, single/batch
consistency, and unseen filtering. `python3 -m src.content.demo` exercises the
included MovieLens CSV data and prints user/movie scores and confidence.

The model remains standalone and is not wired into `main.py`, the existing
heuristic recommender, or the ML reranker. Collaborative filtering, hybrid
ranking, diversity, group aggregation, and cast signals remain
outside this content-only milestone. Any later accuracy claim should use
training-only profiles and temporal held-out evaluation.
