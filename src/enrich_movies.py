"""Fetch TMDB metadata by MovieLens links.csv IDs, with resumable caching.

Run: python -m src.enrich_movies --limit 100
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from src.data_processing.movielens import movie_metadata_from_record
from src.data_processing.tmdb import (
    ATTRIBUTION, MovieResponse, TMDBClient, TMDBError, TMDBNotFound, atomic_write,
)


ENRICHMENT_FIELDS = (
    "directors", "cast", "runtime_minutes", "release_year", "language", "keywords", "plot",
    "tmdbId", "tmdb_status", "tmdb_fetched_at", "tmdb_language", "metadata_source",
    "metadata_temporality",
)


def read_records(path: Path) -> tuple[dict, ...]:
    with path.open(encoding="utf-8", newline="") as source:
        return tuple(csv.DictReader(source))


def read_links(path: Path) -> dict[int, int | None]:
    links = {}
    for row in read_records(path):
        movie_id = int(row["movieId"])
        if movie_id <= 0 or movie_id in links:
            raise ValueError("links.csv must contain unique positive MovieLens IDs")
        value = row["tmdbId"].strip()
        if value and (not value.isascii() or not value.isdecimal() or int(value) <= 0):
            raise ValueError(f"Invalid TMDB mapping for MovieLens movie {movie_id}")
        links[movie_id] = int(value) if value else None
    return links


def _names(entries, limit=None):
    values, seen = [], set()
    for entry in entries:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name.strip():
            continue
        name = " ".join(name.replace("|", " ").split())
        if name.casefold() not in seen:
            values.append(name)
            seen.add(name.casefold())
        if limit is not None and len(values) >= limit:
            break
    return "|".join(values)


def enrich_record(base: dict, response: MovieResponse, language: str) -> dict:
    """Preserve MovieLens identity/genres; add only content, never vote scores."""
    payload = response.payload
    row = dict(base)
    runtime = payload.get("runtime")
    if isinstance(runtime, bool) or not isinstance(runtime, (int, float)) or not math.isfinite(runtime) or runtime <= 0:
        runtime = ""
    original = movie_metadata_from_record(base)
    year = original.release_year
    if year is None:
        date = payload.get("release_date")
        try:
            year = datetime.strptime(date, "%Y-%m-%d").year if isinstance(date, str) and date else None
        except ValueError:
            year = None
    overview = payload.get("overview")
    original_language = payload.get("original_language")
    credits = payload["credits"]
    cast = sorted((p for p in credits["cast"] if isinstance(p, dict)),
                  key=lambda p: p["order"] if type(p.get("order")) is int else 10**9)
    row.update(
        directors=_names(p for p in credits["crew"] if isinstance(p, dict) and p.get("job") == "Director"),
        cast=_names(cast, limit=10), runtime_minutes=runtime,
        release_year=year or "", language=original_language.strip() if isinstance(original_language, str) else "",
        keywords=_names(payload["keywords"]["keywords"]),
        plot=overview.strip() if isinstance(overview, str) else "",
        tmdbId=payload["id"], tmdb_status="ok", tmdb_fetched_at=response.fetched_at,
        tmdb_language=language, metadata_source="tmdb", metadata_temporality="retrospective",
    )
    movie_metadata_from_record(row)  # Fail before publishing incompatible metadata.
    return row


def metadata_summary(records) -> dict:
    records = tuple(records)
    fields = ("plot", "keywords", "directors", "cast", "runtime_minutes", "language")
    return {
        "movies": len(records),
        "coverage": {field: sum(bool(row.get(field)) for row in records) for field in fields},
        "enriched_movies": sum(row.get("metadata_source") == "tmdb" for row in records),
        "temporality": "retrospective" if any(row.get("metadata_source") == "tmdb" for row in records) else "original",
        "fetched_at_min": min((row["tmdb_fetched_at"] for row in records if row.get("tmdb_fetched_at")), default=None),
        "fetched_at_max": max((row["tmdb_fetched_at"] for row in records if row.get("tmdb_fetched_at")), default=None),
    }


def enrich_catalog(records, links, client, *, limit=None, refresh=False, offline=False, progress=None):
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("limit must be a non-negative integer or None")
    if offline and refresh:
        raise ValueError("offline and refresh cannot be combined")
    records = tuple(records)
    seen, rows = set(), []
    counts = dict(fetched=0, cache_hits=0, not_found=0, unmapped=0, deferred=0, requests=0)
    for base in records:
        movie_id = int(base["movieId"])
        if movie_id in seen:
            raise ValueError("movies.csv must contain unique MovieLens IDs")
        seen.add(movie_id)
        movie_metadata_from_record(base)
        tmdb_id = links.get(movie_id)
        row = {**base, "tmdbId": tmdb_id or "", "tmdb_status": "unmapped"}
        if tmdb_id is None:
            counts["unmapped"] += 1
        else:
            response = client.cached_movie(tmdb_id)
            budget_available = limit is None or counts["requests"] < limit
            if (response is None or refresh) and budget_available and not offline:
                counts["requests"] += 1
                try:
                    response = client.get_movie(tmdb_id, refresh=refresh)
                except TMDBNotFound:
                    row["tmdb_status"] = "not_found"
                    counts["not_found"] += 1
                    response = None
                if response is not None:
                    counts["fetched"] += int(not response.from_cache)
            if response is not None:
                row = enrich_record(base, response, client.language)
                counts["cache_hits"] += int(response.from_cache)
            elif row["tmdb_status"] != "not_found":
                row["tmdb_status"] = "deferred"
                counts["deferred"] += 1
        rows.append(row)
        if progress is not None and (len(rows) % 100 == 0 or len(rows) == len(records)):
            progress(len(rows), len(records), dict(counts))
    return tuple(rows), {**counts, **metadata_summary(rows)}


def write_catalog(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    fields.extend(field for field in ENRICHMENT_FIELDS if field not in fields)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(Path(path), output.getvalue())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, help="Default: DATA_DIR/movies_enriched.csv")
    parser.add_argument("--cache-dir", type=Path, help="Default: DATA_DIR/tmdb_cache")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--limit", type=int, help="Maximum uncached movie fetches this run (default: all)")
    parser.add_argument("--refresh", action="store_true", help="Replace cached responses from TMDB")
    parser.add_argument("--offline", action="store_true", help="Rebuild CSV from cache without network or credentials")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.refresh and args.offline:
        parser.error("--refresh and --offline cannot be combined")
    movie_path, links_path = args.data_dir / "movies.csv", args.data_dir / "links.csv"
    output = args.output or args.data_dir / "movies_enriched.csv"
    if output.resolve() in {p.resolve() for p in (movie_path, links_path, args.data_dir / "ratings.csv", args.env_file)}:
        parser.error("--output must be separate from the source data and credentials")
    cache_dir = args.cache_dir or args.data_dir / "tmdb_cache"

    last_requests = -1

    def progress(done, total, counts):
        nonlocal last_requests
        if counts["requests"] == last_requests and done != total:
            return
        last_requests = counts["requests"]
        print(f"{done}/{total} movies: {counts['fetched']} fetched, {counts['cache_hits']} cached, "
              f"{counts['deferred']} deferred", flush=True)

    try:
        factory = TMDBClient if args.offline else TMDBClient.from_environment
        kwargs = {} if args.offline else {"env_file": args.env_file}
        client = factory(cache_dir=cache_dir, language=args.language, **kwargs)
        rows, report = enrich_catalog(read_records(movie_path), read_links(links_path), client,
                                      limit=args.limit, refresh=args.refresh, offline=args.offline, progress=progress)
        report.update(
            schema_version=1, source="https://www.themoviedb.org/", attribution=ATTRIBUTION,
            language=args.language, generated_at=datetime.now(timezone.utc).isoformat(),
            input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (movie_path, links_path)},
        )
        write_catalog(output, rows)
        report["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        atomic_write(Path(str(output) + ".metadata.json"), json.dumps(report, indent=2) + "\n")
    except (TMDBError, OSError, ValueError, KeyError) as error:
        # Network errors are already sanitized by TMDBClient. Avoid dumping a
        # traceback, request headers, or JSON response bodies in the CLI.
        if isinstance(error, TMDBError):
            parser.exit(1, f"Enrichment stopped: {error}\n")
        parser.exit(1, "Enrichment stopped: check input files, CSV fields, and output permissions.\n")
    print(f"Saved {output} ({report['enriched_movies']}/{report['movies']} enriched movies)")
    print(ATTRIBUTION)


if __name__ == "__main__":
    main()
