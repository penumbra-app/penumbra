# Evaluation environment

For a downloadable offline cohort, use the [MovieLens 32M workflow](movielens-32m.md).
It reuses the enriched catalog, separates development and reserved users, and guards
a frozen final comparison. It does not need access to anonymized MovieLens people.

Use the full-catalog benchmark for development and a separately registered future
window for confirmation. The old rated-candidate benchmark is retained for
regression checks; it is no longer described as an untouched test.

## Development benchmark

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m src.evaluate_catalog develop
.venv/bin/python -m src.benchmark_content
```

Development outputs: `reports/evaluation/development/results.json` and ignored
`test_users.csv`. Latency outputs: `reports/evaluation/latency/results.json` and
ignored `request_samples.csv`. `--limit 3` runs a smoke check, explicitly labeled
as partial. Dataset files and API credentials remain local and ignored.

Each user trains on their first 60% of ratings and is evaluated on the next 20%,
without splitting identical timestamps. The last 20% is not used by this command.
The dataset has already been inspected, so this is development evidence rather
than a new holdout. Users need at least five history ratings and one development
outcome; this yields a different cohort from the earlier two-holdout protocol.

The candidate set contains every unseen movie with an observed interaction at or
before that user's cutoff. Earliest observed interaction is a conservative
availability proxy, not the actual release date. History movies and not-yet-known
movies are excluded. Held-out labels never determine candidate eligibility.
Genres/plots/keywords still come from today's retrospective snapshot; candidate
availability filtering does not turn these descriptions into historical metadata.

The primary metric is **macro NDCG@10**, with binary relevance for observed ratings
>=4. Secondary metrics are Recall@10/20, NDCG@20, judged recommendation fraction,
recommended-catalog coverage and history-size slices. Ties use score descending,
then movie ID ascending for all models. This common offline tie rule differs
from serving's score/confidence/movie-ID rule, which the latency benchmark uses.

Unobserved movies are unjudged, not known dislikes. Their gain is zero for the
known-positive metric, which can undercount good recommendations. Users with no
eligible positives have undefined NDCG/recall, not zero; their counts are explicit.
The report includes positive availability, users with no outcomes, and recall
against *all* observed positives as well as eligible positives. Very low judged
fractions mean the benchmark cannot establish real-world preference accuracy.

The fixed enriched preset is compared with its identical structured core without
text. No hyperparameters are chosen on this run. Old NDCG values are not directly
comparable: the candidate pool, relevance gains and eligible users have changed.
User-paired bootstrap intervals use 10,000 resamples and seed 42. The primary
NDCG@10 comparison has a 95% interval; other metrics/slices are exploratory.

For each metric the report estimates the detectable effect with 80% power and
the independent users needed for a specified worthwhile effect (default 0.01
absolute NDCG/recall). This normal approximation assumes the same paired-user
variance in a future sample; it is planning guidance, not a promise. Additional
ratings from the same people do not count as additional independent users.

## Team comparison contract

The benchmark exports `.cache/evaluation/development-contexts.jsonl.gz` with
each user's history, cutoff and exact candidate IDs. It contains no development
outcome labels. Share it privately alongside the frozen movie/tag snapshot and
the report's input hashes. All models must score every listed candidate once,
exclude listed history movies, and use the common tie/relevance rules above.
The content scorer uses only that user's history. Collaborative/hybrid training
must additionally exclude every other user's interaction later than the target
cutoff; using the entire ratings file would leak future interactions.
Timestamp-filter other-user tags and exclude the target user's own tags as the
content evaluator does. Share only aggregate metrics publicly.

## Future confirmation

The existing 2018 MovieLens data cannot supply a genuinely new test by reshuffling
it. Register a future collection window before obtaining outcome labels:

```bash
.venv/bin/python -m src.evaluate_catalog register \
  --plan reports/evaluation/future-plan.json --days 30 --minimum-effect 0.01
```

The generated default plan freezes the existing MovieLens snapshot as a workflow
demonstration. MovieLens users are anonymized; do not pretend new app users are
those same people by reusing their IDs. For a real prospective study, first gather
your own cohort's history, then register a new plan with `--ratings` pointing to
that history and `--plan` pointing to a new filename. Keep identity/movie mappings
consistent. This default plan does not create access to fresh MovieLens outcomes.

The registration freezes source hashes, input files, preset, candidate catalog,
global prediction time, user eligibility, primary metric, comparison and window
end. Existing history users with >=5 ratings form the cohort. All catalog metadata
is frozen before future outcomes; refreshing it invalidates the plan. This
addresses metadata timing prospectively without pretending historical metadata
is available. Keep the checkout and data snapshots frozen or retain copies for
the final run. If model changes are needed, register a new future window *before*
collecting or inspecting its labels; do not edit an existing plan.

After the window closes, supply newly collected interactions in a private CSV:

```text
userId,movieId,rating,timestamp
```

IDs must match the frozen catalog/cohort; timestamps are UTC Unix seconds, ratings
are 0–5. Each user/movie pair must be unique and absent from history. Preserve
real event timestamps; do not relabel old ratings as fresh data. Export the complete
window rather than filtering users to those with positive outcomes. The evaluator
retains cohort members without observations and reports out-of-cohort users.

```bash
.venv/bin/python -m src.evaluate_catalog confirm \
  --plan reports/evaluation/future-plan.json --outcomes data/future_ratings.csv
```

The tool refuses early access, changed source/data, repeated evaluation, and
old/repeated/out-of-window events. A local open marker is written before parsing
outcomes; failed attempts remain recorded for audit. This guards against accidental
peeking, not a determined user changing code/state. It cannot authenticate data
provenance or prove the team has not inspected labels elsewhere. No confirmation
result exists until real future data arrives. Observational evaluation still
does not establish causal online lift; controlled online exposure is separate.

## End-to-end performance

The benchmark measures the public `recommend` API over the full current catalog,
top-10 sorting and response serialization for short, median and longest histories.
Fresh-subprocess cold trials include startup, CSV/config loading, model/profile
construction and IPC. Warm trials start from loaded profiles. The concurrent
burst uses two workers and reports queue-inclusive p50/p95 and throughput.
Cold and warm process high-water RSS are reported separately.

Defaults: three cold trials and ten warm trials per history, plus nine requests
in the concurrent burst. Small-sample p95 is descriptive, not an SLO estimate.
This local microbenchmark excludes HTTP/network overhead and sustained production
load. Run it separately from accuracy/model-encoding jobs to avoid contention.
Response parity is verified across cold, warm and concurrent execution.

## Automated checks

GitHub Actions runs the synthetic-fixture suite on pushes and pull requests using
Python 3.13. It needs no private data, TMDB key, or model download. Full data
experiments remain local; CI validates correctness and leakage boundaries.
Workflow setup follows the official
[checkout](https://github.com/actions/checkout) and
[setup-python](https://github.com/actions/setup-python) documentation.
