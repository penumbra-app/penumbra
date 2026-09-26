# Collaborative filtering implementation

Role 2 learns from user/movie rating behavior. It supplies predicted ratings and
confidence to other components. All week 3/4 code, tests, model artifacts and
experiment records live under `src/collaborative/`.

## Week 3 tuning

Run from the repository root:

```powershell
.venv/Scripts/python.exe -m src.collaborative.tune
```

Users with at least 10 ratings are ordered by timestamp, then movie ID:

1. First 60%: fit candidate models.
2. Next 20%: choose settings by minimum RMSE, using MAE to break ties.
3. Final 20%: evaluate only after selection is locked.

Users with fewer than 10 ratings contribute fitting data only. The selected
configuration and the previous default are refitted on profile plus validation
(the first 80%) for their final comparison with the movie-average baseline.
The final 20% has been examined in prior project work, so this is a development
benchmark rather than a newly untouched test set. The random seed is 42.

The bounded search starts with the current defaults (20 factors, 0.005 learning
rate, 0.02 regularization, 20 epochs), then changes one setting at a time:
10 factors, 40 factors, learning rate 0.01, regularization 0.1, or 40 epochs.
Each trained candidate is scored with and without confidence shrinkage, giving
12 configurations from six training runs. This is not an exhaustive search.
The existing constructor defaults remain unchanged; the saved selected model
contains the validation-selected configuration.

`artifacts/tuning_results.json` records every configuration, validation metrics,
confidence buckets, training-error histories, dataset hash, environment versions,
selected settings and final holdout metrics. Nonexistent confidence buckets have
null metrics. `artifacts/selected_mf.joblib` is the selected model fitted on 80%;
it has not been refitted on the final test ratings.

## How learning works

The unshrunk training prediction is:

```
global_mean + user_bias + movie_bias + dot(user_factors, movie_factors)
```

Global mean is the average training rating. User bias learns rating generosity
or harshness. Movie bias learns general movie reception. The dot product
captures the individual user/movie match using learned vectors.

For each observed rating, error is `actual - predicted`. Stochastic gradient
descent adjusts biases and factors in the direction that reduces squared error.
A regularization penalty pulls weights toward zero to discourage overfitting.
Learning rate controls the update size, factor count controls model capacity,
and epochs control how many times the training ratings are revisited.
Training history records unshrunk, unclipped training RMSE; validation scoring
uses the actual configured inference behavior and clips ratings to 0.5-5.

## Confidence and unknown IDs

For known IDs, confidence is:

```
(user_count / (user_count + prior_strength))
* (movie_count / (movie_count + prior_strength))
```

Counts come only from fitted data. With `shrink_latent=True`, confidence multiplies
the latent interaction at inference. With False, confidence is still returned but
does not change the score. Validation selects between these choices. Confidence
is an evidence heuristic, not a calibrated probability or a similarity score.

| User | Movie | Prediction before clipping | Confidence |
|---|---|---|---|
| Known | Unknown | Global mean + user bias | 0.1 |
| Unknown | Known | Global mean + movie bias | Half of movie evidence |
| Unknown | Unknown | Global mean | 0 |

New ratings do not automatically update an existing model. Refit to learn a new
user or movie representation. The component does not generate a final ranked
movie list or filter watched movies; those tasks belong to the hybrid layer.

## Week 4 inference and persistence

```python
from src.collaborative import BiasedMatrixFactorization

model = BiasedMatrixFactorization.load(
    "src/collaborative/artifacts/selected_mf.joblib"
)
predictions = model.predict([23, 23], [1, 50])
```

Output columns stay `user_id`, `movie_id`, `predicted_score`, `confidence`.
Pairs retain their input order, including duplicates. Empty lists return an empty
DataFrame with the same columns. Scores stay in 0.5-5 and confidence in 0-1.
IDs must be finite, nonnegative integers (integral numeric values accepted;
strings, booleans, fractional IDs and values beyond signed int64 are rejected).
Training ratings must be finite numeric values from 0.5 to 5. Missing or invalid
training rows raise errors instead of silently disappearing.

The model validates training settings and reports numerical divergence. An
unfitted model cannot predict or save. Save writes a version-1 envelope containing
the model, settings and training metadata, then replaces the destination after
serialization succeeds. Load validates artifact version/type, settings, array
shapes, finite weights, index mappings and evidence counts. Legacy raw model
objects remain supported; objects without shrinkage settings retain unshrunk
inference. Only load trusted joblib files, because they use pickle serialization.
The default save/load path is now `src/collaborative/artifacts/collaborative_mf.joblib`;
explicit old paths remain usable. The tuning artifact is named `selected_mf.joblib`.

Run the saved-model example (also supports direct file execution):

```powershell
.venv/Scripts/python.exe -m src.collaborative.demo --user 23 --movies 1 50 999999999
```

## Validation

```powershell
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests src/tests src/collaborative
```

Tests cover input validation, reproducibility, pair order and duplicates, sparse
and unknown IDs, empty batches, unfitted operations, persistence roundtrips,
legacy artifacts, incompatible/corrupt weights, chronological boundaries and
selection independence from held-out test labels. The tuning run additionally
requires exact save/load agreement on every final held-out prediction.

The historical evaluator remains available under `evaluation/`. Its command-line
default is chronological; its legacy Python API default remains random. Its
results are standalone rating diagnostics, not the hybrid team's pairwise ranking
benchmark. No hybrid, shared evaluation or content implementation is changed.


## Completed week 3 and 4 results

The six training runs produced 12 validation configurations. Selection used
validation RMSE only (MAE for ties); the final holdout did not affect selection.

Selected configuration: 20 factors, learning rate 0.005, regularization 0.02,
40 epochs, prior strength 5, seed 42, confidence shrinkage enabled.

Rows: 60,255 profile, 20,164 validation, 20,417 held-out test; 80,419 rows for final refitting.

Selected validation RMSE: 0.886420; MAE: 0.682463.

| Model | Final RMSE | Final MAE | Within 0.5 |
|---|---:|---:|---:|
| Baseline | 1.004890 | 0.777937 | 42.00% |
| Current MF | 0.898750 | 0.693342 | 45.61% |
| Selected MF | 0.889162 | 0.684513 | 46.07% |

Selected-model held-out confidence diagnostics:

| Confidence | Count | RMSE | MAE |
|---|---:|---:|---:|
| [0.00, 0.20) | 2,760 | 0.9721 | 0.7701 |
| [0.20, 0.40) | 1,670 | 0.9083 | 0.6993 |
| [0.40, 0.60) | 2,339 | 0.8863 | 0.6832 |
| [0.60, 0.80) | 5,209 | 0.8772 | 0.6757 |
| [0.80, 1.00] | 8,439 | 0.8647 | 0.6594 |

88 tests pass. Full held-out predictions and confidence match exactly after
save/load. The saved-model demo also ran successfully. Joblib emits a NumPy
array-shape deprecation warning during serialization tests; no tests fail.

These results are a development benchmark, not a statistically validated claim
of generalization. The search is small, and the final holdout was seen earlier
in project work. Use the selected artifact explicitly; the constructor defaults
and the other roles' models remain unchanged.
