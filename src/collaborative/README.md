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

## Week 5 shared evaluation and ablations

Run from the repository root:

```bash
python -m src.collaborative.evaluation.benchmark
```

This writes `artifacts/benchmark/results.json`, `per_user_metrics.csv`, and
`splits.csv`. The split export is generated locally and excluded from Git.
The JSON records settings, environment versions, split counts/hash, validation
selection and every variant's validation/test metrics. The CSV retains per-user
ranking metrics and eligibility counts for paired follow-up analysis.

Week 5 uses the exact shared split helper used by the content evaluator,
including timestamp ties and small histories. Fit on the first 60%, compare
variants on the next 20%, and evaluate on the final 20% without refitting.
This differs from Week 3/4's 80%-refitted benchmark, so those numerical results
must not be compared as if only the model changed. The 80%-trained saved artifact
is intentionally not loaded here. A fresh 40-epoch model is fitted on 60%.

MAE/RMSE measure rating error. Pairwise accuracy excludes equal actual ratings;
predicted ties earn half credit. Macro accuracy averages users equally, while
micro accuracy weights by eligible pairs. NDCG@10 uses binary relevance
(`rating >= 4`), logarithmic rank discounts and expected gain across entire tied
score groups, including ties crossing position 10. Users with no liked candidate
are excluded from NDCG and counted explicitly; users with no unequal-rating pairs
are excluded from pairwise accuracy. Candidate sets contain held-out rated movies
only, not the entire unseen catalog. Content and collaborative now share the
metric implementation in `src/evaluation/metrics.py`; the old content import
remains available.

Ablations remove one inference contribution using identical fitted weights:
confidence shrinkage, latent interaction, user bias, or movie bias. These are
**frozen-weight inference ablations**, not separately retrained architectures.
The movie-average baseline is fitted to the same 60%. Validation RMSE selects
a diagnostic variant (alphabetical exact ties); the benchmark does not promote
or overwrite any model artifact.

Measured on the bundled 100,836 ratings (60,255 train / 20,164 validation /
20,417 test):

| Variant | Test RMSE | Test MAE | Pairwise macro | NDCG@10 |
|---|---:|---:|---:|---:|
| Movie-average baseline | 1.012584 | 0.783652 | 0.617786 | 0.767512 |
| Full MF | 0.921151 | 0.709007 | 0.623813 | 0.766685 |
| Without shrinkage | 0.922237 | 0.709517 | 0.621326 | 0.765844 |
| Without latent interaction | 0.922700 | 0.710547 | 0.625853 | 0.767165 |
| Without user bias | 1.012283 | 0.774715 | 0.623993 | 0.766579 |
| Without movie bias | 0.985164 | 0.778612 | 0.513597 | 0.707796 |

Validation selects full MF by RMSE (0.888279). Full MF improves held-out rating
error and pairwise accuracy over the baseline, but **does not beat the baseline
on held-out NDCG@10**. Removing latent interaction slightly improves held-out
ranking while worsening rating error; removing shrinkage raises validation NDCG
but lowers held-out NDCG. These small differences are not evidence of statistical
significance. No ranking-based promotion is justified from this run.
User bias is largely a per-user offset, so its ranking effect should be small
except where score clipping changes ties. Movie bias contributes strongly to
both rating error and ranking in this experiment.

This remains a development benchmark: the dataset and 40-epoch configuration
were examined in earlier work. Per-user chronological splits are not a global
time cutoff, so they do not establish strictly future-only population behavior.
Shared metrics/splits are available for other roles; this report evaluates the
collaborative component, not an integrated hybrid pipeline.

Reproduction note: this run uses Python 3.14.7, NumPy 2.5.3 and pandas 3.0.6
(complete versions are in the report). Its split SHA-256 is
`e4f31a01ce732608b5e520109ccb4d9415bb387703f71832e19a609bc983f172`.
Reconstructing the content evaluator's assignments in this environment produces
the exact same hash. The older committed content report has a different hash;
do not compare that archived report as an identical-split experiment. Re-run
content in the same environment or reuse exported assignments for team comparisons.
The shared helper does not specify a secondary sort key for timestamp ties.

## Separate ranking-oriented collaborative experiment

The Week 6 Role 2 experiment is `PairwiseMatrixFactorization` in
`pairwise_factorization.py`. It supplies predictions and confidence; it does not
select candidates, filter watched movies, or produce final individual/group
rankings. The existing `BiasedMatrixFactorization`, package exports, default demo,
and `artifacts/selected_mf.joblib` remain the working baseline.

### Objective and training

The hypothesis is that teaching MF to order a user's higher-rated movies above
lower-rated movies can improve NDCG@10 beyond rating-error training alone.
The new model first fits a fresh 40-epoch baseline on training ratings. It copies
those weights and performs five additional epochs with two updates per row:

1. A squared rating-error update, including L2 regularization, keeps training
   anchored to star ratings.
2. Sample another observed movie from a different rating bucket for the same
   user. Orient the pair from higher to lower actual rating. Minimize
   `rank_weight * log(1 + exp(-(score_high - score_low)))`.

A rating bucket is chosen uniformly among the user's other rating levels; a row
is then chosen uniformly within that bucket. This is not uniform sampling over
all possible preference pairs. Equal-rating pairs are excluded; users without
unequal ratings receive rating updates only. Unrated movies are never used as
negative examples. Storage is proportional to observed ratings rather than all
possible pairs. Separate random streams keep rating-row order identical across
ranking weights, including the zero-weight control.

Prediction is `global_mean + user_bias + movie_bias + user_factors @ movie_factors`,
clipped to 0.5–5. User/global offsets cancel in pairwise differences but remain
learned by the rating objective. The experiment does not shrink latent scores at
inference. Confidence still uses training evidence counts and the baseline's
unknown-ID fallbacks; it is not a calibrated probability. Scores remain rating
estimates because of the rating objective, but their held-out calibration/error
must still be measured.

### Run and selection protocol

```bash
python -m src.collaborative.evaluation.pairwise_experiment
```

All models use the shared chronological 60/20/20 split. Only the first 60% trains
weights. The fixed search tests ranking weights 0.1 and 0.5, learning rate 0.002,
regularization 0.02, 20 factors, and five additional epochs, on seeds 42, 43, 44.
The zero-weight control uses the same extra rating training and unshrunk inference;
it helps distinguish the pairwise loss from extra training/inference changes.
The unchanged 40-epoch, shrinkage-enabled MF is also scored on every seed.

Select the positive weight with highest mean validation NDCG@10 across seeds;
exact ties favor the smaller weight. Lock selection before computing any test
metrics. No refitting uses validation or test labels. Test metrics use the same
held-out rated candidate sets, binary relevance (`rating >= 4`), tie handling,
and eligible-user policy as the shared benchmark.

The prespecified improvement gate requires all of:

- Mean validation NDCG beats both baseline and rating-only control.
- Test NDCG beats both comparators on every seed (at least three seeds).
- The lower bound of each paired-user bootstrap 95% interval is positive.
- Test RMSE stays within 1% of baseline RMSE on every seed.

Bootstrap differences are paired by user and seed; differences are averaged
across seeds per user before resampling users 2,000 times. This avoids counting
repeated users as independent observations. It measures user variation on this
development holdout, not generalization to fresh data. The holdout has previously
been examined; further experimentation must not be described as a fresh test.
No run automatically replaces the existing model, even if the gate passes.

### Artifacts and independent inference

The runner saves `artifacts/pairwise/results.json`, `per_user_metrics.csv`, and
`selected_pairwise.joblib`. The model artifact is the validation-selected positive
weight on the first prespecified seed (42), not the best test seed. Results retain
all validation candidates, training histories, per-seed test metrics, input/split
hashes, source hashes, settings, versions, confidence intervals and the gate result.
Save/load predictions are required to match exactly on the full test partition.

```python
from src.collaborative.pairwise_factorization import PairwiseMatrixFactorization

model = PairwiseMatrixFactorization.load(
    "src/collaborative/artifacts/pairwise/selected_pairwise.joblib"
)
predictions = model.predict([23, 23], [1, 50])
```

Output remains `user_id`, `movie_id`, `predicted_score`, `confidence`, preserving
pair order and duplicates. Save uses a separate versioned artifact type and
atomic replacement. It rejects the baseline's current default artifact paths;
load rejects wrong types/versions and invalid weights. Only load trusted joblib
files. New model code and experiment results are confined to `src/collaborative/`.

### Recorded experiment outcome

Validation selected rank weight 0.5. Mean test metrics across seeds 42, 43, 44:

| Model | RMSE | MAE | NDCG@10 | Pairwise macro |
|---|---:|---:|---:|---:|
| Existing MF baseline | 0.920669 | 0.708834 | 0.766747 | 0.623058 |
| Matched rating-only control | 0.922401 | 0.709895 | 0.766265 | 0.621461 |
| Separate pairwise MF | 0.923794 | 0.711028 | 0.767370 | 0.621704 |

The pairwise model's mean NDCG improvement over baseline is 0.000623, with a
paired-user bootstrap 95% interval of [-0.001716, 0.002874]. Against the matched
rating-only control, the difference is 0.001104 with interval
[-0.000269, 0.002594]. Both intervals include zero. Seed 42 loses to the baseline;
seeds 43 and 44 improve. Rating error remains within the 1% guardrail, but worsens,
and pairwise accuracy is lower. **The improvement gate fails; retain existing MF.**
The experimental artifact is saved for inspection, not promoted.

Validation: 207 tests and 72 subtests pass. Coverage includes deterministic
training, observed-pair update direction, no fabricated pairs for equal ratings,
baseline weight immutability, bounded/unknown/empty/duplicate predictions,
invalid inputs, independent save/load, baseline overwrite prevention, corrupt
weights, paired bootstrap behavior, and test-label independence of selection
and fitted weights. All held-out scores and confidence match exactly after
save/load. File hashes confirm this work changed no files outside collaborative
and preserved the existing MF implementation and selected artifact (earlier
Week 5 changes elsewhere in the working tree predate this experiment).

## EASE collaborative model

`ease.py` implements a separate EASE model based on
[Harald Steck's paper](https://arxiv.org/abs/1905.03375).
The current MF and pairwise models, exports, and their saved artifacts are unchanged.
All EASE implementation, tests and experiment artifacts live under collaborative.

EASE learns linear item-to-item relationships from a binary user/movie matrix X:
training ratings >= 4 become 1; lower ratings and unobserved entries both become
0. This discards explicit rating magnitude and does not distinguish dislike from
lack of observation in its reconstruction objective. It minimizes
`||X - XB||² + lambda ||B||²` with `diag(B) = 0`, preventing a movie from copying
its own input directly to its output. The closed-form solution is
`P = inverse(X.T X + lambda I)` and `B[i,j] = -P[i,j]/P[j,j]` off the diagonal.

The implementation computes the same fitted-user predictions without allocating
a movie-by-movie matrix. With `Z = solve(lambda I + X X.T, X)` and
`d = (1 - column_sum(X * Z))/lambda`, the predictions are `X - Z/d`.
Cholesky solves the user-space system; fitted scores are cached. This is useful
for this dataset's relatively small user count. The dense cache scales with
users times movies and the solve with users squared; a configurable working-memory
estimate rejects oversized inputs before allocation. This is not a general
large-scale sparse EASE implementation. New users or changed histories require
refitting; this API does not ingest a new user's history at prediction time.

### Evaluation and interfaces

```bash
python -m src.collaborative.evaluation.ease_experiment
```

The runner tries regularization 1, 10, 100 and 1000 on the shared 60/20/20 split.
Fit uses 60%; select by raw validation NDCG@10, breaking exact ties toward larger
regularization. After selection, fit a nonnegative affine rating conversion on
validation observations only. Test labels affect neither weights, selection nor
calibration. Calibration explicitly rejects overlap with training user/movie
pairs. The runner compares with freshly trained, unchanged 40-epoch MF on seeds
42, 43 and 44. EASE itself is deterministic and needs no seed repeats.

```python
from src.collaborative.ease import EASE

model = EASE.load("src/collaborative/artifacts/ease/selected_ease.joblib")
raw = model.predict_ranking([23, 23], [1, 50])
ratings = model.predict([23, 23], [1, 50])
```

`predict_ranking` returns `user_id`, `movie_id`, `ranking_score`, `confidence`.
Raw scores are unconstrained and are not stars. `predict` requires calibration
and returns the common `predicted_score` rating contract, clipped to 0.5–5.
The affine conversion is monotonic before clipping; clipping can create ties.
No raw-score MAE/RMSE is reported. Raw and calibrated ranking metrics are both
reported so the rating conversion cannot silently change the ranking claim.

Confidence uses positive-interaction counts:
`user_likes/(user_likes+5) * movie_likes/(movie_likes+5)`. This is an evidence
heuristic, not a correctness probability. Unknown IDs return raw zero and zero
confidence; calibrated predictions then use the fitted intercept. Known users
or movies with no likes also have zero confidence. Empty batches and duplicate
requests preserve the common behavior. Save/load has an independent versioned
envelope, validates state, and refuses other models' reserved artifact names.
Only load trusted joblib artifacts.

### Recorded result

Validation selected lambda 1000 (the upper edge of the fixed search; this is not
an exhaustive optimum). On the held-out rated candidate sets:

| Model | NDCG@10 | Pairwise macro | RMSE | MAE |
|---|---:|---:|---:|---:|
| Existing MF, mean of 3 seeds | 0.766747 | 0.623058 | 0.920669 | 0.708834 |
| EASE raw ranking | 0.766811 | 0.592312 | not applicable | not applicable |
| EASE calibrated ratings | 0.766713 | 0.592313 | 1.034412 | 0.827635 |

EASE does not establish a ranking improvement and is substantially worse as a
rating predictor with this simple conversion. Retain MF. EASE remains an
independent experiment, not the default. Results are in
`artifacts/ease/results.json`; per-user metrics and the separately saved model
are in the same directory. The runner requires exact raw and calibrated
prediction agreement after save/load across the full test partition.
The report includes a paired-user bootstrap interval, settings, environment,
input/split/source hashes, timing and memory estimates. This is a previously
examined development holdout, not a fresh generalization test.
