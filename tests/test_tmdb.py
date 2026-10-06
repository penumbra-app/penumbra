import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

from src.content import load_selected_model, UserRating
from src.data_processing.movielens import movie_metadata_from_records, movie_metadata_from_record
from src.data_processing.tmdb import (
    MovieResponse, TMDBAuthError, TMDBClient, TMDBError, TMDBNotFound,
    _NoRedirects, read_credentials,
)
from src.enrich_movies import enrich_catalog, enrich_record, main, read_links, read_records, write_catalog


def payload(tmdb_id=101):
    # Synthetic API fixture, never published as real movie enrichment.
    return {
        "id": tmdb_id, "title": "Different translated title", "overview": "A pilot, a robot.\nAn adventure.",
        "runtime": 105, "release_date": "2005-02-03", "original_language": "en",
        "vote_average": 9.9, "vote_count": 100000, "popularity": 1234,
        "credits": {"cast": [{"name": "Second Actor", "order": 1}, {"name": "First Actor", "order": 0}],
                    "crew": [{"name": "Director One", "job": "Director"}, {"name": "Writer", "job": "Writer"}]},
        "keywords": {"keywords": [{"name": "Space"}, {"name": "space"}, {"name": "Robots"}]},
    }


def response(data=None):
    return io.BytesIO(json.dumps(data if data is not None else payload()).encode())


def http_error(status, headers=None):
    return HTTPError("https://api.themoviedb.org/?api_key=secret-key", status,
                     "sensitive server error", headers or {}, io.BytesIO(b"sensitive response"))


class TMDBClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def client(self, opener, **kwargs):
        return TMDBClient(token="secret-token", cache_dir=self.path, opener=opener,
                          min_interval=0, sleep=Mock(), **kwargs)

    def test_bearer_auth_combined_request_timeout_and_secret_free_cache(self):
        opener = Mock(return_value=response())
        client = self.client(opener)
        result = client.get_movie(101)
        request = opener.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")
        self.assertNotIn("secret-token", request.full_url)
        self.assertEqual(urlparse(request.full_url).path, "/3/movie/101")
        self.assertEqual(parse_qs(urlparse(request.full_url).query)["append_to_response"], ["credits,keywords"])
        self.assertEqual(opener.call_args.kwargs["timeout"], 15)
        self.assertFalse(result.from_cache)
        self.assertNotIn("secret-token", next(self.path.glob("*.json")).read_text())
        cached = TMDBClient(cache_dir=self.path, opener=Mock(side_effect=AssertionError("network"))).get_movie(101)
        self.assertEqual(cached.payload, result.payload)
        self.assertEqual(cached.fetched_at, result.fetched_at)
        self.assertTrue(cached.from_cache)

    def test_api_key_auth_fallback(self):
        opener = Mock(return_value=response())
        client = TMDBClient(api_key="secret-key", cache_dir=self.path, opener=opener)
        client.get_movie(101)
        request = opener.call_args.args[0]
        self.assertEqual(parse_qs(urlparse(request.full_url).query)["api_key"], ["secret-key"])
        self.assertIsNone(request.get_header("Authorization"))
        self.assertNotIn("secret-key", next(self.path.glob("*.json")).read_text())

    def test_missing_credentials_fails_before_network(self):
        opener = Mock()
        client = TMDBClient(cache_dir=self.path, opener=opener)
        with self.assertRaisesRegex(TMDBAuthError, "TMDB_API_READ_ACCESS_TOKEN"):
            client.get_movie(101)
        opener.assert_not_called()

    def test_429_honors_retry_after_and_success_is_cached(self):
        opener = Mock(side_effect=[http_error(429, {"Retry-After": "3"}), response()])
        client = self.client(opener)
        client.get_movie(101)
        self.assertEqual(opener.call_count, 2)
        client._sleep.assert_called_once_with(3)
        self.assertIsNotNone(client.cached_movie(101))

    def test_long_retry_after_stops_without_retrying_too_soon(self):
        opener = Mock(side_effect=http_error(429, {"Retry-After": "120"}))
        client = self.client(opener)
        with self.assertRaisesRegex(TMDBError, "cooldown"):
            client.get_movie(101)
        self.assertEqual(opener.call_count, 1)
        client._sleep.assert_not_called()

    def test_server_and_transport_failures_are_bounded_and_redacted(self):
        for errors in ([http_error(503) for _ in range(3)],
                       [URLError("secret-token secret-key") for _ in range(3)]):
            opener = Mock(side_effect=errors)
            client = self.client(opener, max_retries=2)
            with self.assertRaises(TMDBError) as caught:
                client.get_movie(101)
            self.assertEqual(opener.call_count, 3)
            self.assertNotIn("secret", str(caught.exception))
            self.assertEqual(len(list(self.path.glob("*.json"))), 0)

    def test_auth_failures_and_404_do_not_retry_or_cache(self):
        for status, error_type in ((401, TMDBAuthError), (403, TMDBAuthError), (404, TMDBNotFound)):
            opener = Mock(side_effect=http_error(status))
            with self.assertRaises(error_type) as caught:
                self.client(opener).get_movie(101)
            self.assertNotIn("secret", str(caught.exception))
            self.assertEqual(opener.call_count, 1)
            self.assertEqual(len(list(self.path.glob("*.json"))), 0)

    def test_invalid_json_partial_response_and_mismatched_identity_not_cached(self):
        incomplete = payload()
        del incomplete["keywords"]
        for result in (io.BytesIO(b"<html>bad response</html>"), response(incomplete), response(payload(999))):
            with self.assertRaises(TMDBError):
                self.client(Mock(return_value=result)).get_movie(101)
            self.assertEqual(len(list(self.path.glob("*.json"))), 0)

    def test_refresh_and_corrupt_cache_refetch(self):
        (self.path / "101-en-US.json").write_text("broken JSON")
        changed = payload()
        changed["overview"] = "Updated overview"
        opener = Mock(side_effect=[response(), response(changed)])
        client = self.client(opener)
        self.assertEqual(client.get_movie(101).payload["overview"], payload()["overview"])
        self.assertEqual(client.get_movie(101, refresh=True).payload["overview"], "Updated overview")
        self.assertEqual(client.get_movie(101).payload["overview"], "Updated overview")
        self.assertEqual(opener.call_count, 2)

    def test_language_has_separate_cache_and_invalid_ids_rejected(self):
        client = self.client(Mock(return_value=response()))
        client.get_movie(101)
        french = TMDBClient(cache_dir=self.path, language="fr-FR")
        self.assertIsNone(french.cached_movie(101))
        for invalid in (0, -1, True, "101", "../101"):
            with self.assertRaises(ValueError):
                client.get_movie(invalid)
        for kwargs in ({"language": "../../env"}, {"timeout": 0}, {"max_retries": 10},
                       {"min_interval": float("nan")}):
            with self.assertRaises(ValueError):
                TMDBClient(**kwargs)

    def test_minimum_request_interval(self):
        sleep = Mock()
        client = TMDBClient(token="token", cache_dir=self.path,
                            opener=Mock(side_effect=[response(), response(payload(102))]),
                            sleep=sleep, clock=Mock(side_effect=[0., .1, .25]))
        client.get_movie(101)
        client.get_movie(102)
        sleep.assert_called_once_with(.15)

    def test_redirects_cannot_forward_auth_to_another_host(self):
        with self.assertRaisesRegex(TMDBError, "not forwarded"):
            _NoRedirects().redirect_request(None, None, 302, "", {}, "https://elsewhere.invalid/")

    def test_dotenv_loading_environment_precedence_and_no_interpolation(self):
        env = self.path / ".env"
        env.write_text('export TMDB_API_READ_ACCESS_TOKEN="file-token" # comment\nUNRELATED=ignore\n')
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(read_credentials(env), {"TMDB_API_READ_ACCESS_TOKEN": "file-token"})
            client = TMDBClient.from_environment(env_file=env, cache_dir=self.path)
            self.assertEqual(client._token, "file-token")
        with patch.dict(os.environ, {"TMDB_API_KEY": "env-key"}, clear=True):
            self.assertEqual(read_credentials(env), {"TMDB_API_KEY": "env-key"})
        env.write_text("TMDB_API_KEY=$(do-not-execute)\n")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(read_credentials(env)["TMDB_API_KEY"], "$(do-not-execute)")

    def test_malformed_dotenv_does_not_expose_credential(self):
        env = self.path / ".env"
        env.write_text('TMDB_API_KEY="secret-key\n')
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(TMDBAuthError) as caught:
            read_credentials(env)
        self.assertNotIn("secret-key", str(caught.exception))


class EnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.records = tuple(dict(movieId=str(i), title=f"Movie {i} (1995)", genres="Adventure") for i in (1, 2, 3))

    def test_mapping_keeps_identity_year_genres_and_excludes_popularity(self):
        record = enrich_record(self.records[0], MovieResponse(payload(), "2026-10-06T00:00:00+00:00"), "en-US")
        movie = movie_metadata_from_record(record)
        self.assertEqual((movie.movie_id, movie.title, movie.genres, movie.release_year),
                         (1, "Movie 1 (1995)", ("Adventure",), 1995))
        self.assertEqual(movie.directors, ("Director One",))
        self.assertEqual(movie.cast, ("First Actor", "Second Actor"))
        self.assertEqual(movie.keywords, ("Space", "Robots"))
        self.assertEqual((movie.runtime_minutes, movie.language, movie.plot), (105, "en", payload()["overview"]))
        self.assertEqual(record["metadata_temporality"], "retrospective")
        for field in ("vote_count", "vote_average", "popularity"):
            self.assertNotIn(field, record)

    def test_missing_fields_are_neutral_and_unknown_year_can_be_filled(self):
        data = payload()
        data.update(runtime=0, overview=None, original_language=None)
        data["credits"] = dict(cast=[], crew=[])
        data["keywords"] = dict(keywords=[])
        base = dict(movieId="1", title="Undated", genres="")
        movie = movie_metadata_from_record(enrich_record(base, MovieResponse(data, "2026-10-06T00:00:00+00:00"), "en-US"))
        self.assertEqual((movie.runtime_minutes, movie.plot, movie.language, movie.cast, movie.keywords),
                         (None, None, None, (), ()))
        self.assertEqual(movie.release_year, 2005)

    def test_cast_is_limited_and_dirty_optional_names_are_cleaned(self):
        data = payload()
        data["credits"]["cast"] = [{"name": f"Actor {i}", "order": i} for i in range(30, -1, -1)]
        data["keywords"]["keywords"] += [None, {}, {"name": ""}, {"name": " pipe|value "}]
        movie = movie_metadata_from_record(enrich_record(self.records[0], MovieResponse(data, "now"), "en-US"))
        self.assertEqual(len(movie.cast), 10)
        self.assertEqual(movie.cast[0], "Actor 0")
        self.assertIn("pipe value", movie.keywords)

    def test_csv_roundtrip_preserves_unicode_multiline_plot_and_model_input(self):
        data = payload()
        data["overview"] += " Café."
        row = enrich_record(self.records[0], MovieResponse(data, "2026-10-06T00:00:00+00:00"), "en-US")
        file = self.path / "enriched.csv"
        write_catalog(file, (row, self.records[1]))
        movies = movie_metadata_from_records(read_records(file))
        self.assertEqual(movies[0].plot, data["overview"])
        model = load_selected_model((UserRating(1, 1, 5),), movies)
        self.assertEqual(model.recommend(1)[0].movie_id, 2)

    def test_limit_resumes_next_uncached_movie_and_preserves_catalog(self):
        opener = Mock(side_effect=[response(), response(payload(102))])
        client = TMDBClient(token="token", cache_dir=self.path, opener=opener, min_interval=0)
        links = {1: 101, 2: 102, 3: None}
        rows, first = enrich_catalog(self.records, links, client, limit=1)
        self.assertEqual((len(rows), first["fetched"], first["deferred"], first["unmapped"]), (3, 1, 1, 1))
        rows, second = enrich_catalog(self.records, links, client, limit=1)
        self.assertEqual((second["cache_hits"], second["fetched"], second["deferred"]), (1, 1, 0))
        self.assertEqual([r["tmdb_status"] for r in rows], ["ok", "ok", "unmapped"])
        self.assertEqual(opener.call_count, 2)
        self.assertEqual(second["coverage"]["plot"], 2)

    def test_offline_and_zero_limit_never_fetch(self):
        opener = Mock(side_effect=AssertionError("network"))
        client = TMDBClient(cache_dir=self.path, opener=opener)
        for kwargs in ({"offline": True}, {"limit": 0}):
            rows, report = enrich_catalog(self.records, {1: 101}, client, **kwargs)
            self.assertEqual((report["requests"], report["deferred"], report["unmapped"]), (0, 1, 2))
            self.assertEqual(rows[0]["title"], self.records[0]["title"])
        opener.assert_not_called()

    def test_not_found_does_not_drop_movie_or_abort_other_records(self):
        client = TMDBClient(token="token", cache_dir=self.path, min_interval=0,
                            opener=Mock(side_effect=[http_error(404), response(payload(102))]))
        rows, report = enrich_catalog(self.records, {1: 101, 2: 102}, client)
        self.assertEqual(report["not_found"], 1)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["tmdb_status"], "not_found")
        self.assertEqual(rows[1]["tmdb_status"], "ok")

    def test_link_validation_and_empty_tmdb_id(self):
        file = self.path / "links.csv"
        file.write_text("movieId,imdbId,tmdbId\n1,100,101\n2,200,\n")
        self.assertEqual(read_links(file), {1: 101, 2: None})
        for data in ("1,100,../x", "1,100,-1", "1,100,101\n1,200,102"):
            file.write_text("movieId,imdbId,tmdbId\n" + data + "\n")
            with self.assertRaises(ValueError):
                read_links(file)

    def test_cli_writes_enriched_snapshot_with_coverage_hash_and_provenance(self):
        write_catalog(self.path / "movies.csv", self.records)
        (self.path / "links.csv").write_text("movieId,imdbId,tmdbId\n1,100,101\n2,200,102\n3,300,\n")
        client = TMDBClient(token="token", cache_dir=self.path / "cache", min_interval=0,
                            opener=Mock(side_effect=[response(), response(payload(102))]))
        with patch("src.enrich_movies.TMDBClient.from_environment", return_value=client), redirect_stdout(io.StringIO()):
            main(["--data-dir", str(self.path), "--limit", "2"])
        output = self.path / "movies_enriched.csv"
        report = json.loads(Path(str(output) + ".metadata.json").read_text())
        self.assertEqual((report["enriched_movies"], report["movies"], report["temporality"]), (2, 3, "retrospective"))
        self.assertEqual(len(report["output_sha256"]), 64)
        self.assertEqual(len(read_records(output)), 3)

    def test_cli_auth_failure_keeps_previous_output_and_does_not_print_secrets(self):
        write_catalog(self.path / "movies.csv", self.records)
        (self.path / "links.csv").write_text("movieId,imdbId,tmdbId\n1,100,101\n")
        output = self.path / "movies_enriched.csv"
        output.write_text("previous snapshot")
        stderr = io.StringIO()
        client = TMDBClient(token="secret-token", cache_dir=self.path / "cache",
                            opener=Mock(side_effect=http_error(401)))
        with patch("src.enrich_movies.TMDBClient.from_environment", return_value=client), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                main(["--data-dir", str(self.path)])
        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(output.read_text(), "previous snapshot")
        self.assertNotIn("secret", stderr.getvalue())

    def test_cli_prevents_source_overwrite(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--data-dir", str(self.path), "--output", str(self.path / "movies.csv")])

    def test_explorer_and_demo_accept_enriched_csv_without_network(self):
        from src.content.demo import main as demo
        from src.explore_data import explore

        data = payload()
        rows = (enrich_record(self.records[0], MovieResponse(data, "2026-10-06T00:00:00+00:00"), "en-US"), *self.records[1:])
        output = self.path / "enriched.csv"
        write_catalog(output, rows)
        (self.path / "ratings.csv").write_text("userId,movieId,rating,timestamp\n1,1,5,1\n1,2,1,2\n")
        printed = io.StringIO()
        with redirect_stdout(printed):
            explore(self.path, output)
            demo(1, self.path, 1, movies_file=output)
        self.assertIn("plot: 1/3", printed.getvalue())
        self.assertIn("retrospective snapshot", printed.getvalue())

    def test_evaluator_cli_uses_separate_report_directory_for_enriched_data(self):
        from src.evaluate_content import main as evaluate

        with patch("sys.argv", ["evaluate", "--movies-file", "data/enriched.csv"]), patch("src.evaluate_content.run") as run:
            evaluate()
        run.assert_called_once_with("data", "reports/tmdb-experiment", "data/enriched.csv")
