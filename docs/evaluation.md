# Evaluation

## Content: Week 5 procedure

Run from the repository root:

```powershell
python -m src.scripts.evaluate_content --splits-output data/content-splits.csv
```

The report is [content-evaluation-results.json](content-evaluation-results.json).
It records input SHA-256 hashes, split hash/counts, feature coverage, all candidate
configurations, validation/test metrics, and the selected configuration. The
optional CSV exports each `(userId, movieId, rating, timestamp, split)` assignment
for teammates to reuse. This runner evaluates content only; it does not claim
that collaborative or hybrid models have passed this protocol.

### Shared data split

Use `src.evaluation.splits.build_user_evaluation_split`, which delegates to the
existing hybrid `chronological_split`: each user's earliest 60% builds the
profile, the next 20% is validation, and the final 20% is test (integer cutoffs).
The shared helper calls its middle partition `train`, because hybrid uses it to
train a reranker; content uses it for feature selection. Every content variant
uses identical assignments. Profiles, baselines, recency reference times, and
feature evidence are fitted on the first partition only. No test refit occurs.
Test labels cannot affect validation selection.

The bundled snapshot contains 100,836 ratings, 610 users, and 9,742 catalog
movies. It produces 60,255 training, 20,164 validation, and 20,417 test ratings.
The runner rejects duplicate user/movie observations and missing timestamps
before splitting. Missing catalog metadata is retained as an empty-feature movie,
so difficult candidates are not silently dropped.

Timestamp ties follow the shared helper's existing pandas ordering, not a new
content-specific split. Reuse the exported assignments for exact cross-model
comparisons, especially across pandas versions. This is per-user chronology,
not a single global cutoff; cross-user models need an explicitly agreed temporal
protocol before making strict real-time claims.

### Metrics

| Metric | Definition |
| --- | --- |
| MAE | Mean absolute 0–5 rating error over held-out observations |
| RMSE | Square root of mean squared rating error; primary validation selection criterion |
| Pairwise accuracy (macro) | Per-user fraction of correctly ordered unequal-rating pairs, averaged over eligible users |
| Pairwise accuracy (micro) | Correct pair credit divided by all eligible pairs |
| NDCG@10 | Binary relevance: rating >= 4 is liked; discount `1/log2(rank+1)`; macro average over users with a liked candidate |

Equal actual ratings do not create preference pairs. Predicted score ties receive
0.5 pair credit. NDCG uses expected gain across each entire tied-score block,
including ties crossing rank 10, so a constant baseline cannot benefit from lucky
movie-ID ordering. Users with no unequal-rating pairs or no liked movies are
excluded only from the corresponding ranking metric; counts are reported.
They remain in rating-error metrics. The test report has 601 pairwise-eligible
users, 941,183 pairs, and 591 NDCG-eligible users.

Candidates are the movies rated in that held-out partition, not the full unseen
catalog. NDCG therefore measures ordering within observed choices; it is not a
measurement of full-catalog recommendation quality, exposure, or online lift.

### Baselines, ablations, and retention

Compare a personal-mean baseline, genres only, all implemented features (including
cast weight 0.25), and six variants that each disable exactly one feature's weight.
Profile shrinkage/recency settings stay fixed. Select the lowest validation RMSE;
exact ties prefer fewer enabled features and then a stable variant name. Selection
happens before scoring test data. Store the resulting `selected_config` with
features lacking training evidence disabled. Test ablations are descriptive and
must not be used for additional tuning.

The original bundled CSV has genre metadata for 9,708 movies and release years for 9,729.
Cast, director, runtime, and language coverage are all zero. Equal ablation scores
for those features do not show they are useful or harmless. That run cannot
validate those features; the subsequent IMDb run below evaluates them. Cast
remains disabled in the library default, and is enabled by the enriched report.
Future feature additions
must improve the prespecified validation criterion before promotion. No new
model family was introduced for Week 5.

## Recorded content results

These values come from the committed JSON report, using the bundled CSV snapshot.
The original run's chosen effective configuration enables genre weight 1.0 and release-era
weight 0.25. Its report label is `without_cast`; other unavailable features are
also disabled in `selected_config`.

| Configuration | Validation RMSE | Test MAE | Test RMSE | Test pairwise macro | Test NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| User mean | 0.959111 | 0.766893 | 0.991444 | 0.500000 | 0.671557 |
| Genres only / remove era | 0.947235 | 0.754816 | 0.979537 | 0.544824 | 0.713490 |
| Genres + era (selected) | 0.940812 | 0.750897 | 0.975247 | 0.550698 | 0.728973 |
| Remove genre (era only on this data) | 0.952581 | 0.762782 | 0.987046 | 0.510226 | 0.697419 |

The selected configuration's test pairwise micro accuracy is 0.599850. It improves
RMSE by about 1.63% relative to the personal mean on this snapshot. These are
point estimates, not significance tests or evidence that more metadata will help.
The data contains no cold users in this split; cold-start behavior is verified
with deterministic tests, not an empirical cold-start accuracy claim.

## IMDb-enriched results (September 28, 2026)

[content-enriched-evaluation-results.json](content-enriched-evaluation-results.json)
records the completed run on `data/movies-enriched.csv`. It uses exactly the same
rating split as the original run, verified by matching split SHA-256 hashes.
[content-imdb-import.json](content-imdb-import.json) records source/archive hashes
and the 23 unmatched IMDb title IDs. See [metadata coverage](metadata-sources.md#completed-local-imdb-import).

Validation selected `without_language`: genre weight 1.0, director 0.5, runtime
0.25, release era 0.25, cast 0.25, and language 0. The full configuration ties
because this dataset supplies no original-language values. All other feature
removals worsen validation RMSE.

| Configuration | Validation RMSE | Test MAE | Test RMSE | Test pairwise macro | Test NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| User mean | 0.959111 | 0.766893 | 0.991444 | 0.500000 | 0.671557 |
| Genres only | 0.947238 | 0.754811 | 0.979551 | 0.544748 | 0.713665 |
| Selected / full | 0.931784 | 0.742606 | 0.966774 | 0.567768 | 0.751358 |
| Remove genre | 0.942967 | 0.754046 | 0.978021 | 0.544557 | 0.735940 |
| Remove director | 0.937148 | 0.747603 | 0.971614 | 0.564328 | 0.746843 |
| Remove runtime | 0.935122 | 0.745697 | 0.970218 | 0.556000 | 0.740320 |
| Remove release era | 0.938063 | 0.746400 | 0.970892 | 0.563346 | 0.742888 |
| Remove cast | 0.931982 | 0.742759 | 0.966932 | 0.568177 | 0.751831 |

The selected model's pairwise micro accuracy is 0.621958. Its test RMSE improves
from the original model's 0.975247 to 0.966774, and NDCG@10 from 0.728973 to
0.751358. This comparison includes IMDb year/genre updates as well as added
director, runtime, and cast features; it does not isolate metadata enrichment's
individual effects. The table's feature removals do hold the enriched snapshot
fixed.

Cast improves validation RMSE by only 0.000198 and test RMSE by 0.000158. Its
test pairwise accuracy and NDCG are slightly worse than the no-cast variant.
It is retained under the prespecified validation-RMSE criterion, not claimed as
a significant or universal ranking improvement. Language remains unvalidated.

## Reproduce the enriched run

After following [metadata access](metadata-sources.md):

```powershell
python -m src.scripts.enrich_content --imdb-directory data
python -m src.scripts.evaluate_content --movies data/movies-enriched.csv --output docs/content-enriched-evaluation-results.json --splits-output data/content-splits.csv
python -m src.content.demo --movies data/movies-enriched.csv --evaluation-report docs/content-enriched-evaluation-results.json --user-id 1 --count 10
```

Keep the same ratings and split assignments. Record metadata snapshot hashes and
coverage. Current IMDb/TMDB aggregates, popularity, revenue, awards, and reviews
can postdate historical ratings; do not insert them into historical profiles as
though available at the prediction time. The current model excludes these fields.
Present-day metadata corrections still limit strict historical reconstruction.

MovieLens latest-small is a development dataset; GroupLens discourages treating
its changing latest releases as a permanent research benchmark. The committed
input hashes fix this project's snapshot. For publication, agree on a fixed
benchmark and protocol before comparing models. See the
[official dataset description](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html).

## Verification and relevant files

`python -m pytest -q` covers exact arithmetic, cast evidence budgets/caps, sparse
metadata, cold users, reason attribution, ingestion joins, API retry behavior,
ranking tie treatment, shared split identity, and test-label isolation.

- `src/evaluation/content.py`: content metrics, ablations, and evaluation.
- `src/scripts/evaluate_content.py`: CLI and reproducible report output.
- `src/evaluation/splits.py`: shared split entry point.
- `tests/evaluation/test_content.py`: metric and leakage regressions.
- `docs/content-evaluation-results.json`: recorded run and selected configuration.
- `docs/content-enriched-evaluation-results.json`: IMDb-enriched metrics and configuration.
- `docs/content-imdb-import.json`: source and output hashes for the local import.
