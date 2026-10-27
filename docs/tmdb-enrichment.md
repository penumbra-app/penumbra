# TMDB movie enrichment

Flick can fetch plot overviews, keywords, directors, the first ten billed cast
members, runtime, and original language from TMDB. The import uses the official
MovieLens `links.csv` mapping. MovieLens IDs, titles, genres, and existing release
years are preserved; TMDB vote averages and popularity are not model features.

The client uses the Python standard library. It combines movie details,
`credits`, and `keywords` into one request, throttles requests, retries temporary
failures, and saves successful responses for reuse. See TMDB's
[authentication](https://developer.themoviedb.org/docs/authentication-application),
[combined requests](https://developer.themoviedb.org/docs/append-to-response), and
[rate-limit documentation](https://developer.themoviedb.org/docs/rate-limiting).

## Configure your credential

Obtain an **API Read Access Token** from your
[TMDB API settings](https://www.themoviedb.org/settings/api). Create `.env` in the
repository root using [`.env.example`](../.env.example) as a template:

```dotenv
TMDB_API_READ_ACCESS_TOKEN=your_token_here
```

Alternatively use `TMDB_API_KEY` for a v3 API key. Set one credential type.
Environment variables take precedence over `.env`. `TMDB_ACCESS_TOKEN` is also
accepted as an alias. Tokens are sent in the Authorization header; v3 keys use
TMDB's query parameter. Errors do not print either credential. `.env` and its
variants are ignored by Git; the example file contains no secrets. There is no
command-line token flag, so credentials do not enter shell history or process
arguments.

## Fetch metadata and resume

From the repository root:

```bash
# Pilot: fetch up to 100 uncached movies, keeping every movie in the output CSV.
.venv/bin/python -m src.enrich_movies --limit 100

# Fetch the remaining catalog, reusing the successful pilot responses.
.venv/bin/python -m src.enrich_movies

# Rebuild the CSV entirely from local cache without network or a credential.
.venv/bin/python -m src.enrich_movies --offline
```

`--limit` bounds movie fetches, not the number of output rows. Repeating
`--limit 100` advances to the next uncached movies. A fetch may retry a temporary
error up to three times. Cache entries are isolated by TMDB ID and language.
Use `--refresh` to replace cached snapshots, or `--language fr-FR` to request
another language. The current text model uses English stopwords, so its saved
configuration has not been validated for other languages.

Successful responses are retained immediately. If authentication or repeated
network failures stop a run, rerun the same command to reuse that progress.
The output CSV is replaced atomically only after a completed pass. Missing
mappings and HTTP 404 responses keep the original movie with neutral enrichment.
Rate-limit responses respect `Retry-After`; a requested delay above 30 seconds
stops the command with a resumable cooldown message. Invalid or incomplete
responses are not cached.

Generated files:

| Path | Contents |
| --- | --- |
| `data/tmdb_cache/<id>-<language>.json` | Successful raw response and fetch timestamp |
| `data/movies_enriched.csv` | Complete MovieLens catalog with optional extra fields |
| `data/movies_enriched.csv.metadata.json` | Coverage, status counts, timestamps, and input/output hashes |

These generated files are ignored by Git. The official `data/links.csv` is
provided in the private data bundle alongside movies and ratings from the same
MovieLens archive. `--data-dir`, `--output`, `--cache-dir`, and `--env-file` support
other local paths. Cache entries do not expire automatically, allowing repeatable
experiments; refresh them explicitly when you need a newer snapshot.

## Explore and recommend

```bash
.venv/bin/python -m src.explore_data --movies-file data/movies_enriched.csv
.venv/bin/python -m src.content.demo --user-id 414 --movies-file data/movies_enriched.csv
```

The explorer prints coverage for every added feature. The demo loads the
enrichment through the existing validated metadata adapter and uses it in the
content model. Neither command makes API calls. Without `--movies-file`, both
continue to use the original MovieLens CSV. `main.py` remains the separate
legacy reranker; TMDB enrichment is connected to the standalone content model.

The saved Week 6 preset was selected using sparse MovieLens tags. Its original
accuracy numbers do not establish how it performs with richer metadata.

## Evaluate an enriched snapshot

```bash
.venv/bin/python -m src.evaluate_content --movies-file data/movies_enriched.csv
```

With `--movies-file`, the default output directory is `reports/tmdb-experiment`.
The original Week 6 reports and serving preset are preserved. The report records
the exact input file, SHA-256, feature coverage, and fetch-date range. All models
in that run, including the baseline, use the same supplied metadata.

TMDB snapshots are marked **retrospective**: a fetch timestamp records when we
downloaded metadata, not when each field first existed. The evaluator labels
this explicitly and continues to exclude future/user-authored MovieLens tags
and held-out rating labels. It cannot claim a strictly historical metadata
backtest using today's plots or keywords. The existing test split has also
already been inspected; use a fresh holdout before making a new generalization
claim. No enriched model is promoted automatically.

## Verification

```bash
.venv/bin/python -m unittest tests.test_tmdb -v
.venv/bin/python -m unittest discover -s tests -v
```

The API tests use synthetic response fixtures, without credentials or network.
They cover both authentication methods, credential redaction, cache/refresh,
language separation, rate limits, timeouts and server failures, missing movies,
malformed responses, identity preservation, partial catalogs, resumed imports,
CSV round trips, and the exploration/recommendation entry points. Live enrichment
requires a configured credential and network access.

The integration was live-checked on 100 movies: all 100 supplied plot, cast,
director, runtime, and language data; 96 supplied keywords. Cache-only rebuilding
produced an identical CSV. The explorer, recommendation demo, and full evaluation
pipeline completed with that snapshot, and all 133 tests passed. The pilot
covers only 100 of 9,742 catalog movies; it is not a full-catalog accuracy study.

## Credits

Metadata is supplied by [The Movie Database (TMDB)](https://www.themoviedb.org/).
This product uses the TMDB API but is not endorsed or certified by TMDB.
TMDB's [FAQ](https://developer.themoviedb.org/docs/faq) explains attribution and
API usage terms. For a future graphical application, include an
[approved TMDB logo](https://www.themoviedb.org/about/logos-attribution) in its
About/Credits view alongside the attribution notice.

## Completed catalog evaluation

The full run enriched 9,621 of 9,742 movies; 113 were not found and eight lacked
mappings. Test text coverage reached 99.39%. The validation-selected TF-IDF
configuration achieved 57.47% pairwise accuracy, 0.7737 NDCG@10, 1.0199 RMSE,
and 0.7765 MAE. Aggregate artifacts are in `reports/tmdb-full/`; the earlier
100-movie check above is retained as historical verification. See
[the team handoff](hybrid-handoff.md) for use of the new preset.
