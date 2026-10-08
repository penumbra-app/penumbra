# Content model handoff to the hybrid team

Use the [evaluation environment](evaluation-environment.md) for the current
full-catalog protocol, private candidate/history export, confirmation guards,
and complete-request performance measurements.

## Public code and private inputs

Commit source, tests, aggregate evaluation reports, configuration, and this guide.
Dataset CSVs, raw TMDB responses, per-user reports, `.env` files, and private
transfer archives stay outside Git. GitHub visibility is repository-wide:
a public repository cannot contain collaborator-only branches or folders.

The repository owner can invite the teammate through Settings > Collaborators
(or organization repository access). Public code is already readable; an invite
is needed for direct write access. For private inputs, share the local
`private_exports/content-handoff.tar.gz` through a restricted team drive or
separate private repository accessible only to the team. This archive is not
encrypted; use an access-controlled transfer, not a public URL or public release.
It contains exact datasets and evaluation outputs, not credentials or raw API cache.
Keep MovieLens's included README and TMDB attribution with the data.

The API key is not needed for scoring or evaluation. Each developer should obtain
their own token only if refreshing metadata, and put it in their own ignored `.env`.
Never commit real credentials, including to a private repository. Actions secrets
are for workflow execution, not a way to distribute local development credentials.

`data/movies.csv` and `data/ratings.csv` were already tracked before this handoff.
Removing them from the current tree does not erase earlier Git history or public
copies. If that historical exposure matters, the owner must separately coordinate
history cleanup; this commit does not rewrite history or change repository visibility.

## Restore and verify

From the repository root, obtain the archive privately, then run:

```bash
tar -xzf /path/to/content-handoff.tar.gz
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-experiments.txt
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m src.evaluate_content --movies-file data/movies_enriched.csv
```

The archive contains `data/`, original and enriched per-user results, and the
full enriched experiment directory. `data/private-handoff-manifest.json` lists
SHA-256 values for transferred files. The evaluated enriched CSV hash is also
recorded in `reports/tmdb-full/results.json`. No API calls are needed above.
The raw API cache remains local to the original machine; regenerate it with
`src.enrich_movies` and your own token only if needed.

## Model integration

Use the public adapters and pass only the user's allowed training/history ratings.
Do not pass validation or test ratings when constructing profiles. The model
uses MovieLens movie IDs, not TMDB IDs. Example (caller supplies `history_rows`
already filtered to the intended training cutoff):

```python
import csv
from src.content.selection import load_selected_model
from src.data_processing.movielens import (
    movie_metadata_from_records, user_ratings_from_records,
)
from src.data_processing.text_metadata import with_keywords_as_of

with open("data/movies_enriched.csv", newline="", encoding="utf-8") as f:
    movies = movie_metadata_from_records(tuple(csv.DictReader(f)))
with open("data/tags.csv", newline="", encoding="utf-8") as f:
    tags = tuple(csv.DictReader(f))
ratings = user_ratings_from_records(history_rows)
# Use one user's training history here; exclude their own tags as in evaluation.
cutoff = max(r.timestamp for r in ratings)
movies = with_keywords_as_of(movies, tags, cutoff, user_id)
model = load_selected_model(
    ratings, movies, config_path="reports/tmdb-full/selected_config.json",
)
features = [p.to_dict() for p in model.predict((user_id,), candidate_movie_ids)]
```

Pass `predicted_score` (0–5), `confidence` (0–1 evidence support, not calibrated
probability), and optional reason signals to the hybrid model, joined by
`user_id` and `movie_id`. For learning the hybrid model, generate these features
from histories that exclude the target ratings. Match temporal cutoffs across
all components. Candidate scores do not incorporate collaborative signals.

The default `src/content/selected_config.json` remains the original sparse-tag
preset. Explicitly use `reports/tmdb-full/selected_config.json` for the enriched
winner; merely supplying richer metadata to the demo does not select this preset.

## Results and limits

| Metric | Previous selected model | Enriched selected model |
| --- | ---: | ---: |
| Pairwise accuracy | 0.553614 | 0.574692 |
| NDCG@10 | 0.756267 | 0.773654 |
| RMSE | 1.026510 | 1.019924 |
| MAE | 0.783378 | 0.776546 |

603 evaluable users and 20,156 test ratings; 594 users have unequal-rating pairs.
The new winner uses TF-IDF unigrams, min_df=1, text prior=5, weight=1, cap=0.5.
It was selected on validation among 162 candidates. LSA did not win ranking.
The old-to-new paired user bootstrap (10,000 resamples, seed 42) gives a pairwise
improvement of 2.11 percentage points, 95% interval [0.95, 3.28]. This comparison
includes enrichment and configuration selection. The enriched selected-versus-
enriched-baseline interval in the report includes zero; text's incremental benefit
alone is not established. Current TMDB metadata is retrospective, the existing
test set has been inspected, and candidates are held-out rated movies rather than
the full catalog. Use a fresh holdout for confirmatory hybrid evaluation.
Shared-catalog TF-IDF and pretrained semantic embeddings are available in the
separate [controlled experiment](controlled-content-experiments.md); the serving
preset still uses per-user TF-IDF.
