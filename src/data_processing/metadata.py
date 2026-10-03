"""Join official metadata by external IDs; never guess matches from titles."""
from __future__ import annotations

import csv
import gzip
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from pathlib import Path

from src.content.schemas import MovieMetadata


def imdb_title_id(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    if text.casefold() in ("", "nan", "none", "\\n"):
        return None
    match = re.fullmatch(r"(?:tt)?(\d+)", text)
    if not match or int(match[1]) <= 0:
        raise ValueError(f"Invalid IMDb title ID: {text}")
    return f"tt{int(match[1]):07d}"


def read_links(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"movieId", "imdbId", "tmdbId"} <= set(reader.fieldnames or ()):
            raise ValueError("links.csv must contain movieId, imdbId, tmdbId")
        links = {}
        for row in reader:
            movie_id = int(row["movieId"])
            if movie_id <= 0 or movie_id in links:
                raise ValueError(f"invalid or duplicate linked movie ID: {movie_id}")
            imdb_title_id(row["imdbId"])
            links[movie_id] = row
        return links


def _tsv_rows(path: Path, columns: set[str]):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as source:
        # IMDb's TSV quotes are literal title characters, not CSV quoting.
        reader = csv.DictReader(source, delimiter="\t", quoting=csv.QUOTE_NONE)
        if not columns <= set(reader.fieldnames or ()):
            raise ValueError(f"{path.name} is missing columns: {sorted(columns)}")
        yield from reader


def _positive_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _year(value: object) -> int | None:
    number = _positive_number(value)
    return int(number) if number is not None and number.is_integer() else None


def enrich_from_imdb(movies: Iterable[MovieMetadata], links: Mapping[int, Mapping],
                     basics_path: Path, crew_path: Path, principals_path: Path,
                     *, audit: dict | None = None,
                     progress: Callable[[str], None] | None = None) -> tuple[MovieMetadata, ...]:
    """Stream full TSVs but retain only linked catalog titles and their credits.

    Principal actor/actress credits are ordered by the source's ordering field.
    That field is a row order, not a guaranteed billing or importance ranking.
    IMDb person IDs are retained as stable identities rather than merging namesakes.
    """
    movies = tuple(movies)
    title_ids = {m.movie_id: imdb_title_id(links.get(m.movie_id, {}).get("imdbId")) for m in movies}
    wanted = {title_id for title_id in title_ids.values() if title_id}
    basics, directors, cast = {}, {}, {}
    if progress:
        progress("Scanning IMDb title basics...")
    for row in _tsv_rows(basics_path, {"tconst", "startYear", "runtimeMinutes", "genres"}):
        if row["tconst"] in wanted:
            basics[row["tconst"]] = row
    if progress:
        progress(f"Matched {len(basics)} IMDb titles; scanning directors...")
    for row in _tsv_rows(crew_path, {"tconst", "directors"}):
        if row["tconst"] in wanted:
            directors[row["tconst"]] = tuple("imdb:" + person for person in
                row["directors"].split(",") if re.fullmatch(r"nm\d+", person))
    if progress:
        progress("Scanning IMDb principal cast...")
    for row in _tsv_rows(principals_path, {"tconst", "ordering", "nconst", "category"}):
        if (row["tconst"] in wanted and row["category"] in ("actor", "actress")
                and re.fullmatch(r"nm\d+", row["nconst"])):
            cast.setdefault(row["tconst"], []).append((int(row["ordering"]), "imdb:" + row["nconst"]))
    enriched = []
    for movie in movies:
        title_id = title_ids[movie.movie_id]
        row = basics.get(title_id, {})
        genres = tuple(g for g in row.get("genres", "").split(",") if g and g != "\\N")
        enriched.append(replace(movie,
            genres=movie.genres or genres,
            release_year=_year(row.get("startYear")) or movie.release_year,
            runtime_minutes=_positive_number(row.get("runtimeMinutes")) or movie.runtime_minutes,
            directors=directors.get(title_id) or movie.directors,
            cast=tuple(dict.fromkeys(person for _, person in sorted(cast.get(title_id, ())))) or movie.cast,
        ))
    if audit is not None:
        audit["unavailable_movie_ids"] = [movie.movie_id for movie in movies
                                          if title_ids[movie.movie_id] not in basics]
    return tuple(enriched)


def imdb_person_names(movies: Iterable[MovieMetadata], names_path: Path) -> dict[str, str]:
    """Resolve only catalog people, keeping stable identities separate from labels."""
    wanted = {person for movie in movies for person in movie.directors + movie.cast
              if person.startswith("imdb:")}
    names = {}
    for row in _tsv_rows(names_path, {"nconst", "primaryName"}):
        identity = "imdb:" + row["nconst"]
        if identity in wanted and row["primaryName"] not in ("", "\\N"):
            names[identity] = row["primaryName"]
    return names


def enrich_from_tmdb(movie: MovieMetadata, details: Mapping) -> MovieMetadata:
    """Normalize movie details with appended credits; keep raw JSON for research.

    Stable TMDB person IDs avoid collisions. Original language is used; the API
    response's translation language is not a property of the movie.
    """
    credits = details.get("credits") or {}
    cast_rows = sorted(credits.get("cast") or (), key=lambda row: (
        row.get("order") if isinstance(row.get("order"), int) else 10**9,
        row.get("id", 0)))

    def person_id(row):
        value = row.get("id")
        return f"tmdb:{value}" if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None

    cast = tuple(dict.fromkeys(identity for row in cast_rows if (identity := person_id(row))))
    directors = tuple(dict.fromkeys(identity for row in credits.get("crew") or ()
        if row.get("job") == "Director" and (identity := person_id(row))))
    date = str(details.get("release_date") or "")
    language = details.get("original_language")
    return replace(movie, genres=movie.genres or tuple(g["name"] for g in details.get("genres") or ()
                   if g.get("name")),
        directors=directors or movie.directors, cast=cast or movie.cast,
        runtime_minutes=_positive_number(details.get("runtime")) or movie.runtime_minutes,
        release_year=_year(date[:4]) or movie.release_year,
        language=language.strip().casefold() if isinstance(language, str) and language.strip() else movie.language)


def write_movies(path: Path, movies: Iterable[MovieMetadata]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=("movieId", "title", "genres", "directors", "cast",
                                                   "runtime_minutes", "release_year", "language"))
        writer.writeheader()
        for movie in movies:
            writer.writerow(dict(movieId=movie.movie_id, title=movie.title,
                genres="|".join(movie.genres), directors="|".join(movie.directors), cast="|".join(movie.cast),
                runtime_minutes=movie.runtime_minutes, release_year=movie.release_year, language=movie.language))
