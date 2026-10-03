import csv
import gzip
import json
import sys
from urllib.error import HTTPError
from unittest.mock import patch

import pytest

from src.content import MovieMetadata
from src.data_processing.metadata import enrich_from_imdb, enrich_from_tmdb, imdb_person_names, imdb_title_id, read_links, write_movies
from src.data_processing.movielens import movie_metadata_from_records
from src.scripts.enrich_content import _fetch_tmdb, main


def test_imdb_normalization_preserves_title_identity():
    assert imdb_title_id("0114709") == "tt0114709"
    assert imdb_title_id("tt0114709") == "tt0114709"
    assert imdb_title_id("12345678") == "tt12345678"
    assert imdb_title_id("") is None
    with pytest.raises(ValueError):
        imdb_title_id("114709.0")


def test_streamed_imdb_join_orders_actor_credits_and_handles_missing_values(tmp_path):
    sources = {
        "basics": "tconst\tstartYear\truntimeMinutes\tgenres\n"
                  "tt0000001\t2001\t100\tComedy\n"
                  "tt0000002\t\\N\t\\N\t\\N\n",
        "crew": "tconst\tdirectors\n"
                "tt0000001\tnm0000005,nm0000006\n"
                "tt0000002\t\\N\n",
        "principals": "tconst\tordering\tnconst\tcategory\n"
                      "tt0000001\t4\tnm0000002\tactress\n"
                      "tt0000001\t2\tnm0000001\tactor\n"
                      "tt0000001\t1\tnm0000009\tdirector\n"
                      "tt9999999\t1\tnm0000010\tactor\n"
                      "tt0000001\t5\tnm0000001\tactor\n",
    }
    paths = []
    for name, text in sources.items():
        path = tmp_path / f"{name}.tsv.gz"
        with gzip.open(path, "wt", encoding="utf-8") as target:
            target.write(text)
        paths.append(path)
    movies = (MovieMetadata(10, "M (1999)", ("Action",), release_year=1999), MovieMetadata(20, "Missing"),
              MovieMetadata(30, "Unlinked"))
    audit = {}
    result = enrich_from_imdb(movies, {10: {"imdbId": "1"}, 20: {"imdbId": "2"}}, *paths, audit=audit)
    assert audit["unavailable_movie_ids"] == [30]
    assert result[0].genres == ("Action",)
    assert result[0].release_year == 2001
    assert result[0].runtime_minutes == 100
    assert result[0].cast == ("imdb:nm0000001", "imdb:nm0000002")
    assert result[0].directors == ("imdb:nm0000005", "imdb:nm0000006")
    assert result[0].language is None
    assert result[1:] == movies[1:]
    output = tmp_path / "movies.csv"
    write_movies(output, result)
    with output.open(encoding="utf-8", newline="") as source:
        assert movie_metadata_from_records(csv.DictReader(source)) == result


def test_imdb_names_resolve_catalog_people_without_merging_namesakes(tmp_path):
    path = tmp_path / "name.basics.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as target:
        target.write("nconst\tprimaryName\nnm1\tSame Name\nnm2\tSame Name\nnm3\tUnused\nnm4\t\\N\n")
    movies = (MovieMetadata(1, "M", directors=("imdb:nm1",), cast=("imdb:nm2", "imdb:nm4", "tmdb:3")),)
    assert imdb_person_names(movies, path) == {"imdb:nm1": "Same Name", "imdb:nm2": "Same Name"}


def test_tmdb_keeps_stable_person_ids_order_and_original_language():
    result = enrich_from_tmdb(MovieMetadata(1, "M"), dict(
        original_language=" FR ", runtime=0, release_date="2001-01-01", genres=[dict(name="Drama")],
        credits=dict(cast=[dict(id=2, name="Same Name", order=2), dict(id=1, name="Same Name", order=0)],
                     crew=[dict(id=3, job="Director"), dict(id=4, job="Writer")]),
    ))
    assert result.cast == ("tmdb:1", "tmdb:2")
    assert result.directors == ("tmdb:3",)
    assert result.language == "fr"
    assert result.runtime_minutes is None
    assert result.release_year == 2001
    assert enrich_from_tmdb(MovieMetadata(1, "M"), dict(credits=None, genres=None)) == MovieMetadata(1, "M")


def test_cli_rejects_mismatched_tmdb_identity_and_reports_missing_cache(tmp_path):
    movies_path, links_path = tmp_path / "movies.csv", tmp_path / "links.csv"
    write_movies(movies_path, (MovieMetadata(1, "A"), MovieMetadata(2, "B"), MovieMetadata(3, "C")))
    links_path.write_text("movieId,imdbId,tmdbId\n1,1,101\n2,2,102\n3,3,103\n", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "101.json").write_text(json.dumps(dict(id=101, imdb_id="tt0000001", runtime=90)), encoding="utf-8")
    (cache / "102.json").write_text(json.dumps(dict(id=102, imdb_id="tt9999999", runtime=120)), encoding="utf-8")
    output = tmp_path / "enriched.csv"
    with patch.object(sys, "argv", ["enrich", "--movies", str(movies_path), "--links", str(links_path),
                                    "--tmdb-cache", str(cache), "--output", str(output)]):
        main()
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["rejected_id_matches"] == [2]
    assert manifest["unavailable_movie_ids"] == [3]
    assert manifest["coverage"]["runtime"] == 1


def test_tmdb_rate_limit_retry_and_auth_failure():
    error = HTTPError("https://api.themoviedb.org", 429, "Rate limited", {"Retry-After": "1"}, None)
    with patch("src.scripts.enrich_content.urlopen", side_effect=error) as request, \
         patch("src.scripts.enrich_content.time.sleep") as sleep:
        with pytest.raises(HTTPError):
            _fetch_tmdb(1, "test-token")
        assert request.call_count == 4
        assert sleep.call_count == 3
        assert request.call_args.args[0].get_header("Authorization") == "Bearer test-token"
        assert "test-token" not in request.call_args.args[0].full_url
    with patch("src.scripts.enrich_content.urlopen", side_effect=HTTPError("url", 401, "Unauthorized", {}, None)) as request:
        with pytest.raises(HTTPError):
            _fetch_tmdb(1, "bad-token")
        assert request.call_count == 1


def test_link_mapping_rejects_duplicates(tmp_path):
    path = tmp_path / "links.csv"
    path.write_text("movieId,imdbId,tmdbId\n1,1,1\n1,2,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_links(path)
