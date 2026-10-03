# Movie metadata sources and access

Source documentation checked September 28, 2026.

## Recommendation

The project uses **IMDb's non-commercial bulk datasets** for a broad, local
catalog of movie metadata: directors, principal cast, genres, years, and runtime.
The downloaded archives are in `data/`. Their records are joined to MovieLens by
IMDb ID, preserving the user-rating data used to train and evaluate the model.
TMDB remains an optional source for additional fields such as original language.

## Options and coverage

| Source | Useful coverage | Access and limits |
| --- | --- | --- |
| IMDb non-commercial | IDs, titles/types, years, runtime, up to three genres; directors/writers, principal credits, people, alternate titles; aggregate ratings/votes | Daily gzip TSV downloads; personal/non-commercial terms; a subset rather than the full IMDb database |
| TMDB | Movie descriptions, credits, languages, countries, studios, collections, budget/revenue, popularity and votes, keywords, regional releases/certifications | Authenticated API; broad catalog with uneven field completeness; cached per-movie responses work well for this project |
| IMDb licensed bulk | Broader title/people metadata, detailed credits, plots, awards, certifications, and separately packaged box-office/other products | Daily delivery through AWS Data Exchange; licensing/product selection required; pricing must be obtained from the provider |
| MovieLens | Individual user ratings, timestamps, genres, user tags, and external ID joins | Training/evaluation target data; not a replacement for broad film metadata |

Sources: [IMDb free schema](https://data.imdb.com/non-commercial-datasets/),
[TMDB official API schema](https://developer.themoviedb.org/openapi/tmdb-api.json),
[IMDb bulk overview](https://data.imdb.com/documentation/bulk-data-documentation/),
[IMDb title fields](https://data.imdb.com/documentation/bulk-data-documentation/data-dictionary/titles),
and [MovieLens file definitions](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html).

### Factor-to-source map

| Candidate factor | Practical source | Current content model |
| --- | --- | --- |
| Genre, release decade | MovieLens; IMDb basics; TMDB details | Implemented; supported by current validation |
| Director and cast | IMDb crew/principals; TMDB credits | Implemented and evaluated on local IMDb data |
| Runtime | IMDb basics or TMDB details | Implemented and evaluated on local IMDb data |
| Original language | TMDB `original_language` | Implemented; validation pending |
| Writers, composers, other crew | TMDB credits; IMDb licensed credits | Retained in TMDB cache, not scored |
| Plot/overview, themes/keywords | TMDB details and keywords; IMDb licensed titles | Raw research data, not scored |
| Production country, studio, franchise | TMDB details | Raw research data, not scored |
| Certification by region/release type | TMDB release dates | Raw research data, not scored |
| Budget, revenue, popularity, aggregate votes | TMDB details; IMDb ratings/box-office products as applicable | Excluded from current content scoring |
| Awards and detailed content advisories | IMDb licensed title and Parents Guide products | Requires appropriate data product; not imported |
| Personal rating history and recency | MovieLens ratings or application ratings | Implemented |

[TMDB credits](https://developer.themoviedb.org/reference/movie-credits),
[keywords](https://developer.themoviedb.org/reference/movie-keywords), and
[release dates](https://developer.themoviedb.org/reference/movie-release-dates)
provide separate resources. IMDb documents
[Parents Guide fields](https://data.imdb.com/documentation/bulk-data-documentation/data-dictionary/parentsGuide)
separately from title metadata.

IMDb `title.akas.language` describes an alternative title's language; it must not
be treated as the movie's original language. Free principal credits are not a
complete cast list, and `ordering` is not a guaranteed billing ranking. These
constraints follow the [free dataset schema](https://data.imdb.com/non-commercial-datasets/).

## Ready-to-use project integration

`data/links.csv` was recovered from the official
[MovieLens latest-small archive](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)
after verifying equality of all 9,742 movie records with `data/movies.csv`.
It supplies `movieId,imdbId,tmdbId`; do not join on title text. IMDb numeric IDs are
normalized with the `tt` prefix and at least seven digits. Provider person IDs
are stored as `imdb:nm...` or `tmdb:<id>` to keep namesakes distinct.

### TMDB: optional enrichment route

Create an API credential in your TMDB account and set the local environment
variable `TMDB_READ_ACCESS_TOKEN` to the API Read Access Token. The importer sends
it in an Authorization header, never a URL or output file. TMDB documents
[application authentication](https://developer.themoviedb.org/docs/authentication-application).

```powershell
python -m src.scripts.enrich_content --tmdb-cache data/tmdb-cache --fetch
```

Each uncached linked title requests:

```text
GET /3/movie/{tmdbId}?append_to_response=credits,keywords,release_dates,external_ids
```

The importer keeps complete JSON responses under `data/tmdb-cache/<tmdbId>.json`,
adds a fetch timestamp, and writes `data/movies-enriched.csv`. The cache retains
the wider metadata for later research while the CSV contains only model-supported
fields. Rerunning reuses cached responses; omit `--fetch` for a fully offline run
against an existing cache. There is no automatic refresh; replace a chosen cache
snapshot deliberately and rerun evaluation when updating metadata.

Requests are sequential, spaced, and bounded by timeouts. HTTP 429 and selected
server errors receive bounded retries; other errors stop the run, except 404
which leaves the movie unenriched. Authentication failures remain explicit.
See [TMDB rate limiting](https://developer.themoviedb.org/docs/rate-limiting).

IDs are checked against cached payloads, including IMDb IDs when both are present.
Mismatches are reported and excluded from enrichment. Original MovieLens titles
and available genres are preserved. Runtime <= 0 becomes missing; original
language, release year, directors, and source-ordered cast are normalized. Raw
TMDB names remain in the cache for displaying person-ID reason signals.

TMDB's daily exports contain IDs and a few attributes, **not full metadata**;
they do not replace the detail requests. See
[daily ID exports](https://developer.themoviedb.org/docs/daily-id-exports).

### IMDb: offline bulk route, no API credential

The importer reads these official files from the supplied directory; the current
local downloads are in `data/`:

- [title.basics.tsv.gz](https://datasets.imdbws.com/title.basics.tsv.gz)
- [title.crew.tsv.gz](https://datasets.imdbws.com/title.crew.tsv.gz)
- [title.principals.tsv.gz](https://datasets.imdbws.com/title.principals.tsv.gz)

Then run:

```powershell
python -m src.scripts.enrich_content --imdb-directory data
```

The importer streams compressed files and keeps only linked catalog records in
memory. It preserves person IDs. When
[name.basics.tsv.gz](https://datasets.imdbws.com/name.basics.tsv.gz) is present,
it also writes `movies-enriched.people.json`, mapping the catalog's person IDs to
display names. The demo loads this file automatically alongside the enriched CSV;
names do not replace the model's stable identities. Language remains missing
unless already present in the input.

All seven archives can stay compressed. Their uses in this project are:

| File | Use |
| --- | --- |
| `title.basics.tsv.gz` | Runtime/year enrichment and genre fallback |
| `title.crew.tsv.gz` | Director IDs |
| `title.principals.tsv.gz` | Principal actor/actress IDs in source order |
| `name.basics.tsv.gz` | Human-readable names for directors and actors |
| `title.akas.tsv.gz` | Available for alternative/localized title lookup; not imported |
| `title.episode.tsv.gz` | Available for episode-to-series relationships; not imported |
| `title.ratings.tsv.gz` | Available aggregate IMDb ratings/votes; not used as personal ratings |

The full downloads cover titles beyond MovieLens. `movies-enriched.csv` retains
the existing 9,742-movie project catalog; this import does not add all IMDb titles
as recommendation candidates. The raw archives remain available for a later
catalog expansion with an explicit application-ID mapping.

### Completed local IMDb import

The September 28, 2026 import scanned the downloaded basics, crew, principals,
and names archives successfully. It matched 9,719 catalog titles to IMDb basics;
23 titles retained their existing metadata. Coverage in the resulting CSV is:

| Field | Movies with metadata (of 9,742) |
| --- | ---: |
| Genres | 9,742 |
| Release year | 9,742 |
| Directors | 9,717 |
| Runtime | 9,718 |
| Principal cast | 9,366 |
| Original language | 0 |

`data/movies-enriched.csv` is about 2.1 MB. The accompanying person-name mapping
contains 44,564 names. [Import provenance](content-imdb-import.json) records the
four input archive hashes, CSV/name-map hashes, and unmatched IDs. This was a
production-size local import; no API call was needed. The seven original archives
are preserved in `data/` and excluded from Git.

### Outputs and reproducibility

Both import paths write a companion `movies-enriched.manifest.json` containing
source, generation time, input/output SHA-256 hashes, coverage, unavailable movie
IDs, rejected identity matches, and a person-name output hash when generated.
A generated timestamp does not assert that
all source files were retrieved that day; retain the original snapshot dates.
Partial coverage retains the original movie row and available fields. Output
must differ from the input movies/links files. Raw caches/dumps and default
enriched outputs are excluded from Git.

```powershell
python -m src.scripts.evaluate_content --movies data/movies-enriched.csv --output docs/content-enriched-evaluation-results.json
python -m src.content.demo --movies data/movies-enriched.csv --evaluation-report docs/content-enriched-evaluation-results.json --count 10
```

Use one consistent person-ID vocabulary for each training/candidate snapshot.
Do not concatenate IMDb and TMDB people as though their IDs were interchangeable.
The model's ten-person cutoff applies after ingestion; raw source metadata stays
available for future experiments. See [content model](content-model.md) for cast
normalization and [evaluation](evaluation.md) for promotion criteria.

## Access terms and research limits

IMDb's free datasets are offered for personal/non-commercial use, subject to
their linked terms; commercial deployments need appropriate licensing.
[IMDb access terms](https://data.imdb.com/non-commercial-datasets/).

TMDB's developer API is free for non-commercial use with attribution; commercial
use requires contacting TMDB. Its required attribution includes an approved logo
and the notice: “This product uses the TMDB API but is not endorsed or certified
by TMDB.” See the [TMDB FAQ](https://developer.themoviedb.org/docs/faq).

The added MovieLens links retain the dataset's research-use conditions, required
acknowledgment, and same-condition redistribution terms; commercial/revenue use
requires GroupLens permission. Cite F. Maxwell Harper and Joseph A. Konstan,
*The MovieLens Datasets: History and Context* (2015),
[DOI](https://doi.org/10.1145/2827872), and retain the
[official license and dataset description](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html)
with redistributed data. No IMDb/TMDB bulk data is committed here.

The bundled-data run validates only genres and release decade; see
[evaluation](evaluation.md) for the separate IMDb-enriched run. External API
fetching has automated mocked coverage but has not been run with a real TMDB
credential. The local IMDb route does not require an API token.
Current aggregate ratings, revenue, popularity, reviews, and
awards must not leak future information into historical evaluation. Missing
values and coverage differences must be reported rather than interpreted as
negative preferences.
