import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from src.content import TextConfig
from src.evaluate_content import Candidate, read_csv
from src.evaluate_ml32m import (cohort_cases, evaluate, evaluate_reserved, freeze,
                               verify_dataset, verify_frozen)
from src.data_processing.movielens import user_ratings_from_records
from src.prepare_ml32m import (RATING_FIELDS, cohort_key, identity_map, prepare, select_users,
                               validate_archive, write_csv)


def csv_bytes(fields, rows):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode()


class ML32MWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.legacy = self.root / "legacy"
        self.output = self.root / "prepared"
        self.archive = self.root / "ml-32m.zip"
        self.movie_rows = [dict(movieId=i + 1000, title=f"Movie {i} (2000)", genres="Drama")
                           for i in range(1, 26)]
        old_links = [dict(movieId=i, tmdbId=100+i, imdbId=f"{200+i:07d}") for i in range(1, 26)]
        new_links = [{**r, "movieId": int(r["movieId"])+1000} for r in old_links]
        write_csv(self.legacy / "links.csv", ("movieId", "tmdbId", "imdbId"), old_links)
        write_csv(self.legacy / "movies_enriched.csv", ("movieId", "title", "genres", "plot", "keywords"),
                  [dict(movieId=i, title="Old title", genres="Comedy", plot="space travel" if i % 2 else "family drama",
                        keywords="space" if i % 2 else "family") for i in range(1, 26)])
        write_csv(self.legacy / "ratings.csv", RATING_FIELDS,
                  [dict(userId=999, movieId=1, rating=4, timestamp=100)])
        # User 1 exactly overlaps an old event despite a different user/movie ID;
        # user 2 has older activity but no exact overlap. All others are eligible.
        rows = []
        for user in range(1, 41):
            for movie in range(1, 26):
                timestamp = 100 if user == 1 and movie == 1 else (50 if user == 2 and movie == 1 else 200+movie)
                rows.append(dict(userId=user, movieId=1000+movie, rating=4 if movie % 2 else 2, timestamp=timestamp))
        self.parts = {
            "movies.csv": csv_bytes(("movieId", "title", "genres"), self.movie_rows),
            "links.csv": csv_bytes(("movieId", "tmdbId", "imdbId"), new_links),
            "ratings.csv": csv_bytes(RATING_FIELDS, rows),
            "tags.csv": b"userId,movieId,tag,timestamp\n",
        }
        self.checksums = {name: hashlib.md5(value).hexdigest() for name, value in self.parts.items()}
        with zipfile.ZipFile(self.archive, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, value in self.parts.items():
                archive.writestr("ml-32m/"+name, value)
            archive.writestr("ml-32m/README.txt", "Synthetic fixture, not MovieLens data")
        self.preset = self.root / "preset.json"
        self.preset.write_text(json.dumps({"candidate": Candidate("fixture", text=TextConfig()).to_dict()}))

    def prepare(self, output=None):
        return prepare(self.archive, output or self.output, self.legacy, 3, 3, expected_md5=self.checksums)

    def test_import_identity_overlap_disjointness_and_determinism(self):
        manifest = self.prepare()
        other = self.root / "other"
        self.prepare(other)
        self.assertEqual(manifest["counts"]["excluded_exact_event_overlap"], 1)
        self.assertEqual(manifest["counts"]["excluded_recorded_activity_before_legacy_end"], 1)
        self.assertEqual(manifest["catalog_movies"], 25)
        self.assertFalse(manifest["official_checksums_verified"])
        users = {}
        for cohort in ("development", "reserved"):
            path = self.output / cohort / "ratings.csv"
            self.assertEqual(path.read_bytes(), (other / cohort / "ratings.csv").read_bytes())
            rows = read_csv(path)
            users[cohort] = {int(r["userId"]) for r in rows}
            self.assertEqual(len(users[cohort]), 3)
            self.assertFalse(users[cohort] & {1, 2})
            self.assertTrue(all(cohort_key(u, 42)[0] == cohort for u in users[cohort]))
            cases = cohort_cases(user_ratings_from_records(rows))
            self.assertTrue(all(max(r.timestamp for r in c.history) < min(r.timestamp for r in c.outcomes) for c in cases))
        self.assertFalse(users["development"] & users["reserved"])
        movies = read_csv(self.output / "metadata/movies.csv")
        self.assertEqual((movies[0]["movieId"], movies[0]["title"], movies[0]["genres"]),
                         ("1001", "Movie 1 (2000)", "Drama"))
        self.assertEqual(movies[0]["plot"], "space travel")
        availability = {r["movieId"]: int(r["timestamp"]) for r in read_csv(self.output / "metadata/availability.csv")}
        self.assertEqual(availability["1001"], 50)  # Includes excluded source users.

    def test_ambiguous_and_conflicting_external_ids_are_not_joined(self):
        old = [dict(movieId=1, tmdbId=10, imdbId=20)]
        self.assertEqual(identity_map(old, [dict(movieId=2, tmdbId=10, imdbId=30),
                                           dict(movieId=3, tmdbId=40, imdbId=20)]), {})
        self.assertEqual(identity_map(old, [dict(movieId=2, tmdbId=10), dict(movieId=3, tmdbId=10)]), {})
        self.assertEqual(identity_map(old, [dict(movieId=2, tmdbId=999, imdbId=20)]), {})

    def test_bad_checksum_cleans_stage_and_does_not_publish(self):
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            prepare(self.archive, self.output, self.legacy, 3, 3,
                    expected_md5={**self.checksums, "ratings.csv": "incorrect"})
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".ml32m-*")), [])

    def test_zip_traversal_rejected(self):
        with zipfile.ZipFile(self.archive, "a") as archive:
            archive.writestr("../outside", "bad")
        with zipfile.ZipFile(self.archive) as archive:
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                validate_archive(archive, self.checksums)

    def test_existing_dataset_never_overwritten(self):
        self.prepare()
        before = (self.output / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(before, (self.output / "manifest.json").read_bytes())

    def test_insufficient_eligible_users_does_not_silently_shrink_sample(self):
        with self.assertRaisesRegex(ValueError, "eligible"):
            prepare(self.archive, self.output, self.legacy, 100, 100, expected_md5=self.checksums)
        self.assertFalse(self.output.exists())

    def test_unsorted_or_duplicate_events_rejected(self):
        row = dict(userId=1, movieId=1, rating=5, timestamp=200)
        with self.assertRaisesRegex(ValueError, "unique and sorted"):
            select_users([row, row], {1}, set(), 100, {"development": 2, "reserved": 2}, 42)

    def test_development_never_opens_reserved_ratings(self):
        self.prepare()
        reserved = self.output / "reserved/ratings.csv"
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            if path == reserved:
                raise AssertionError("Reserved labels opened during development")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            report = evaluate(self.output, self.preset, self.root / "smoke", limit=2)
        self.assertEqual(report["metrics"]["selected"]["users"], 2)
        self.assertEqual(report["protocol"]["smoke_limit"], 2)

    def test_reserved_end_to_end_and_second_plan_cannot_reopen_cohort(self):
        self.prepare()
        plan_path = self.root / "plan.json"
        freeze(self.output, self.preset, plan_path)
        state = self.root / "state"
        report = evaluate_reserved(plan_path, self.root / "report", state)
        self.assertEqual(report["metrics"]["selected"]["users"], 3)
        self.assertEqual(report["protocol"]["mode"], "ml32m_reserved_offline")
        with self.assertRaises(FileExistsError):
            evaluate_reserved(plan_path, self.root / "again", state)
        second = self.root / "second-plan.json"
        freeze(self.output, self.preset, second, minimum_effect=.02)
        with self.assertRaises(FileExistsError):
            evaluate_reserved(second, self.root / "again", state)

    def test_changed_preset_dataset_source_and_plan_rejected(self):
        self.prepare()
        plan_path = self.root / "plan.json"
        plan = freeze(self.output, self.preset, plan_path)
        verify_frozen(plan)
        with patch("src.evaluate_ml32m.workflow_hashes", return_value={}):
            with self.assertRaisesRegex(ValueError, "Source changed"):
                verify_frozen(plan)
        with self.assertRaisesRegex(ValueError, "plan changed"):
            verify_frozen({**plan, "minimum_effect": 1})
        self.preset.write_text(self.preset.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "Preset changed"):
            verify_frozen(plan)
        path = self.output / "reserved/ratings.csv"
        path.write_text(path.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "dataset changed"):
            verify_dataset(self.output)

    def test_failed_reserved_attempt_remains_claimed(self):
        self.prepare()
        plan_path = self.root / "plan.json"
        freeze(self.output, self.preset, plan_path)
        state = self.root / "state"
        with patch("src.evaluate_ml32m.run_benchmark", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaises(RuntimeError):
                evaluate_reserved(plan_path, self.root / "report", state)
        with self.assertRaises(FileExistsError):
            evaluate_reserved(plan_path, self.root / "again", state)


if __name__ == "__main__":
    unittest.main()
