# MovieLens 32M: private development and reserved offline cohorts

The first complete run is documented in [the 2026-10-08 experiment report](../reports/ml32m/summary.md).
Both 1,000-user cohorts have been evaluated. The existing reserved cohort is now
inspected; its single-use guard remains in place.

Download the official [MovieLens 32M ZIP](https://grouplens.org/datasets/movielens/32m/).
Leave it zipped and place it at `data/imports/ml-32m.zip`. This directory and the
prepared data are ignored by Git. The importer verifies all four CSV files against
the checksums in the [official README](https://files.grouplens.org/datasets/movielens/ml-32m-README.html).
It streams ratings directly from the archive and does not extract the full dataset.
Keep the original ZIP for reproducibility; no API token or new TMDB calls are needed.

Run these commands from the repository root:

```bash
.venv/bin/python -m src.prepare_ml32m --zip data/imports/ml-32m.zip
.venv/bin/python -m src.evaluate_ml32m develop --limit 10
```

Alternatively, pass `--zip "$HOME/Downloads/ml-32m.zip"` to read the download in
place. The importer needs the existing `data/movies_enriched.csv`, `data/links.csv`
and `data/ratings.csv`. It writes a new dataset under `data/ml32m/`, leaving the
existing inputs unchanged. Expect checksum verification followed by progress every
two million scanned ratings. The full source is scanned even though the prepared
cohort is small; memory scales with selected histories, not all 32 million rows.

After the smoke run succeeds, run the development benchmark:

```bash
.venv/bin/python -m src.evaluate_ml32m develop
```

Smoke results go to `reports/ml32m/development-smoke/results.json`; full development
results go to `reports/ml32m/development/results.json`. Read the elapsed time from
the smoke report to estimate the full run; history sizes vary. No reserved metrics
are computed by either command. The smoke run and development run can be repeated
while fixing or choosing the content model. Use `--preset PATH` for another
compatible content preset. This wrapper supports the existing content preset
scorer, not the experimental ridge/embedding scorer.

## Fixed selection rules

The default is 1,000 development users and 1,000 reserved users. This is a laptop
starting budget, not a guarantee of statistical power. Use development paired-user
variance and reported eligible-user counts to assess sensitivity. Decide sample
sizes before opening reserved metrics; do not keep expanding a test until it passes.
`--development-users` and `--reserved-users` set sizes during preparation only.

The catalog is the intersection of 32M movies with your existing enriched catalog,
joined through unique, consistent TMDB/IMDb identities. Conflicting or ambiguous
joins are omitted. New-release movie IDs, titles and genres are retained. Existing
plots, keywords and structured metadata are reused, including records with missing
fields. This controls metadata coverage and avoids enriching the entire 32M movie
catalog. **It is a restricted-catalog benchmark, not an 87,585-movie benchmark.**

Users must have at least 20 ratings within this catalog, at least five history
ratings and two outcomes after the chronological split. Their first recorded
rating anywhere in 32M must be later than the latest timestamp in the old local
ratings file. An exact match to an old movie/rating/timestamp event excludes that
user, even if user IDs differ. IDs are anonymized across releases; matching IDs
alone cannot establish person identity. These conservative rules reduce overlap,
but cannot prove the people are disjoint if their recorded history changed.

A fixed SHA-256 rule assigns each eligible user to development or reserved, then
selects the lowest hashes within each partition. Selection never uses model scores,
positive-rating counts or measured performance. Failure to find the requested
number aborts without publishing partial output. Existing output cannot be
overwritten, preventing accidental regeneration during tuning.

Both cohorts use the first approximately 80% of each user's catalog ratings as
history and the final 20% as outcomes. Equal timestamps stay together. Real event
timestamps are preserved. All unseen catalog movies first observed anywhere in
the full 32M source by that user's cutoff are candidates. The full-source timestamp
scan is only an availability proxy; other users' preferences never train this
content model. No community tags are loaded: this first experiment uses existing
TMDB plot/keyword content and avoids mixing tag-user identities across releases.

The manifest records input hashes, selection rules, coverage and exclusion counts.
It and both rating cohorts stay private under `data/ml32m/`. Preparation necessarily
parses raw ratings for sampling and overlap screening; it never computes model
metrics. Avoid opening or otherwise analyzing the reserved labels during tuning.

## Freeze and evaluate once, after development

Choose the final preset and code using development results. Then:

```bash
.venv/bin/python -m src.evaluate_ml32m freeze
.venv/bin/python -m src.evaluate_ml32m evaluate-reserved
```

If development used a different `--preset`, pass that same path to `freeze`.
The plan at `reports/ml32m/offline-plan.json` freezes code, data, preset and the
primary comparison: binary observed-positive NDCG@10 for the selected content model
versus the same structured core without text. Recall@10/20 and NDCG@20 are secondary.
Reports use paired-user bootstrap intervals with 10,000 resamples. A positive 95%
primary interval is evidence of improvement for this benchmark; effect size and
the predeclared worthwhile difference (default 0.01 absolute NDCG) also matter.

The reserved evaluator verifies frozen inputs before parsing ratings and creates
a single-use marker keyed to the reserved ratings file. Another plan cannot reopen
the same cohort. Interrupted or failed attempts remain recorded; investigate before
any rerun, and disclose any outcome inspection. These are local workflow guards,
not access control against editing files or code. Do not delete markers to retune.

Reserved results are written to `reports/ml32m/reserved/results.json`. Per-user CSVs
are ignored by Git; share only aggregate reports publicly. API keys stay in the
ignored local `.env`, and these commands do not read them.

This is **new offline evidence**, not a future or causal online experiment. Metadata
is a retrospective snapshot, and unobserved movies are unjudged. Users without
eligible positive outcomes are counted and excluded from NDCG/recall macros. The
new population, history split, tags and candidate catalog differ from earlier
experiments; compare models within this run, not absolute scores across protocols.
The existing `evaluate_catalog register/confirm` commands remain separate tools for
a real prospective cohort you can collect from.
