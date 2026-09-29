"""Enrich MovieLens from official IMDb TSVs or cached TMDB movie details."""
import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.content.features import FEATURE_TYPES, feature_values
from src.data_processing.metadata import (
    enrich_from_imdb, enrich_from_tmdb, imdb_person_names, imdb_title_id, read_links, write_movies,
)
from src.data_processing.movielens import movie_metadata_from_records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_tmdb(movie_id: int, token: str) -> dict:
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?append_to_response=credits,keywords,release_dates,external_ids"
    request = Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            try:
                delay = float(error.headers.get("Retry-After", 2 ** attempt))
            except ValueError:
                delay = 2 ** attempt
            time.sleep(max(0, min(delay, 30)))
    raise RuntimeError("TMDB request failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movies", type=Path, default=Path("data/movies.csv"))
    parser.add_argument("--links", type=Path, default=Path("data/links.csv"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--imdb-directory", type=Path, help="Contains title.basics/crew/principals.tsv.gz")
    source.add_argument("--tmdb-cache", type=Path, help="Directory of <tmdbId>.json responses")
    parser.add_argument("--fetch", action="store_true", help="Fetch uncached TMDB details with TMDB_READ_ACCESS_TOKEN")
    parser.add_argument("--output", type=Path, default=Path("data/movies-enriched.csv"))
    args = parser.parse_args()
    if args.output.resolve() in (args.movies.resolve(), args.links.resolve()):
        parser.error("output must differ from input movies and links")
    if args.fetch and not args.tmdb_cache:
        parser.error("--fetch requires --tmdb-cache")
    links = read_links(args.links)
    with args.movies.open(encoding="utf-8-sig", newline="") as file:
        movies = movie_metadata_from_records(csv.DictReader(file))
    inputs = [args.movies, args.links]
    unavailable, rejected = [], []
    people_path = None
    if args.imdb_directory:
        paths = [args.imdb_directory / f"title.{name}.tsv.gz" for name in ("basics", "crew", "principals")]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            parser.error("missing required IMDb files: " + ", ".join(missing))
        audit = {}
        enriched = enrich_from_imdb(movies, links, *paths, audit=audit,
                                   progress=lambda message: print(message, flush=True))
        unavailable = audit["unavailable_movie_ids"]
        inputs.extend(paths)
        names_path = args.imdb_directory / "name.basics.tsv.gz"
        if names_path.exists():
            print("Resolving IMDb person names...", flush=True)
            names = imdb_person_names(enriched, names_path)
            people_path = args.output.with_suffix(".people.json")
            people_path.parent.mkdir(parents=True, exist_ok=True)
            people_path.write_text(json.dumps(names, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                                   encoding="utf-8")
            inputs.append(names_path)
    else:
        args.tmdb_cache.mkdir(parents=True, exist_ok=True)
        token = os.environ.get("TMDB_READ_ACCESS_TOKEN", "")
        enriched = []
        for index, movie in enumerate(movies):
            linked = links.get(movie.movie_id, {})
            tmdb_id = linked.get("tmdbId", "").strip()
            if not tmdb_id:
                unavailable.append(movie.movie_id)
                enriched.append(movie)
                continue
            if not tmdb_id.isdigit() or int(tmdb_id) <= 0:
                raise ValueError(f"Invalid TMDB ID for movie {movie.movie_id}")
            cache = args.tmdb_cache / f"{int(tmdb_id)}.json"
            if not cache.exists() and args.fetch:
                if not token:
                    parser.error("set TMDB_READ_ACCESS_TOKEN in your environment to fetch uncached movies")
                try:
                    payload = _fetch_tmdb(int(tmdb_id), token)
                except HTTPError as error:
                    if error.code != 404:
                        raise
                    unavailable.append(movie.movie_id)
                    enriched.append(movie)
                    continue
                payload["_fetched_at"] = datetime.now(timezone.utc).isoformat()
                pending = cache.with_suffix(".json.tmp")
                pending.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                pending.replace(cache)
                time.sleep(0.1)
            if not cache.exists():
                unavailable.append(movie.movie_id)
                enriched.append(movie)
                continue
            inputs.append(cache)
            payload = json.loads(cache.read_text(encoding="utf-8"))
            expected_imdb = imdb_title_id(linked.get("imdbId"))
            actual_imdb = imdb_title_id(payload.get("imdb_id"))
            if payload.get("id") != int(tmdb_id) or (expected_imdb and actual_imdb and expected_imdb != actual_imdb):
                rejected.append(movie.movie_id)
                enriched.append(movie)
            else:
                enriched.append(enrich_from_tmdb(movie, payload))
            if (index + 1) % 100 == 0:
                print(f"Processed {index + 1}/{len(movies)} movies", flush=True)
    write_movies(args.output, enriched)
    manifest = {
        "source": "imdb_noncommercial" if args.imdb_directory else "tmdb",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {str(path): _sha256(path) for path in inputs},
        "output_sha256": _sha256(args.output),
        "people_sha256": _sha256(people_path) if people_path else None,
        "movies": len(enriched), "unavailable_movie_ids": unavailable, "rejected_id_matches": rejected,
        "coverage": {kind: sum(bool(feature_values(movie, kind)) for movie in enriched) for kind in FEATURE_TYPES},
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("source", "movies", "coverage")}, indent=2))
    print(f"Saved {args.output} and {manifest_path}")


if __name__ == "__main__":
    main()
