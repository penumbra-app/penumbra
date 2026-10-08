"""Stream the official MovieLens 32M ZIP into private, bounded user cohorts.

No network access, API keys, ZIP extraction, or model evaluation is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import heapq
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

from src.content.schemas import UserRating
from src.enrich_movies import ENRICHMENT_FIELDS
from src.evaluate_content import read_csv, temporal_split


# Published at https://files.grouplens.org/datasets/movielens/ml-32m-README.html
OFFICIAL_MD5 = {
    "links.csv": "8f033867bcb4e6be8792b21468b4fa6e",
    "movies.csv": "0df90835c19151f9d819d0822e190797",
    "ratings.csv": "cf12b74f9ad4b94a011f079e26d4270a",
    "tags.csv": "963bf4fa4de6b8901868fddd3eb54567",
}
RATING_FIELDS = ("userId", "movieId", "rating", "timestamp")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_rows(archive, name):
    with archive.open("ml-32m/" + name) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8", newline="") as source:
            yield from csv.DictReader(source)


def validate_archive(archive, expected_md5):
    seen = set()
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if (path.is_absolute() or ".." in path.parts or "\\" in member.filename
                or stat.S_ISLNK(member.external_attr >> 16) or member.filename in seen):
            raise ValueError("Unsafe or duplicate ZIP member")
        seen.add(member.filename)
    if sum(m.file_size for m in archive.infolist()) > 8 * 1024**3:
        raise ValueError("Archive exceeds the 8 GiB uncompressed safety limit")
    for name, expected in expected_md5.items():
        digest = hashlib.md5()  # Official integrity checksum; not a security signature.
        with archive.open("ml-32m/" + name) as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Checksum mismatch for {name}; use the official complete 32M ZIP")
        print(f"Verified {name}", flush=True)
    readme = archive.getinfo("ml-32m/README.txt")
    if readme.file_size > 1024 * 1024:
        raise ValueError("Unexpectedly large README")


def identity_map(old_links, new_links):
    """Match external identities uniquely; discard ambiguous/conflicting joins."""
    new_by_id = {int(r["movieId"]): r for r in new_links}
    indices = {}
    for field in ("tmdbId", "imdbId"):
        index = {}
        for row in new_links:
            if row.get(field):
                key = int(row[field])
                index.setdefault(key, set()).add(int(row["movieId"]))
        indices[field] = index
    result = {}
    for row in old_links:
        matches, ambiguous = set(), False
        for field, index in indices.items():
            if row.get(field):
                found = index.get(int(row[field]), set())
                ambiguous |= len(found) > 1
                matches.update(found)
        if not ambiguous and len(matches) == 1:
            match = matches.pop()
            target = new_by_id[match]
            if all(not row.get(f) or not target.get(f) or int(row[f]) == int(target[f])
                   for f in indices):
                result[int(row["movieId"])] = match
    counts = Counter(result.values())
    return {old: new for old, new in result.items() if counts[new] == 1}


def cohort_key(user_id, seed):
    digest = hashlib.sha256(f"ml32m-v1:{seed}:{user_id}".encode()).digest()
    return ("development" if digest[0] % 2 == 0 else "reserved", int.from_bytes(digest[1:], "big"))


def split_history(rows):
    """First ~80% history, final ~20% outcomes, keeping timestamp groups intact."""
    first, middle, last = temporal_split(rows)
    return first + middle, last


def select_users(rows, catalog_ids, old_events, legacy_cutoff, sizes, seed):
    heaps = {name: [] for name in sizes}
    availability, counts = {}, Counter()
    current, history, earliest, overlap = None, [], None, False
    previous_key = None

    def finish():
        if current is None:
            return
        counts["source_users"] += 1
        if overlap:
            counts["excluded_exact_event_overlap"] += 1
            return
        if earliest <= legacy_cutoff:
            counts["excluded_recorded_activity_before_legacy_end"] += 1
            return
        if len(history) < 20:
            counts["excluded_fewer_than_20_catalog_ratings"] += 1
            return
        train, outcomes = split_history(history)
        if len(train) < 5 or len(outcomes) < 2:
            counts["excluded_timestamp_split"] += 1
            return
        name, priority = cohort_key(current, seed)
        counts[f"eligible_{name}"] += 1
        item = (-priority, current, tuple(history))
        heap = heaps[name]
        if len(heap) < sizes[name]:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)

    for row in rows:
        user, movie, rating, timestamp = (int(row["userId"]), int(row["movieId"]),
                                          float(row["rating"]), int(row["timestamp"]))
        key = (user, movie)
        if previous_key is not None and key <= previous_key:
            raise ValueError("Ratings must be unique and sorted by userId then movieId")
        previous_key = key
        if user != current:
            finish()
            current, history, earliest, overlap = user, [], timestamp, False
        earliest = min(earliest, timestamp)
        overlap |= (movie, rating, timestamp) in old_events
        if movie in catalog_ids:
            availability[movie] = min(availability.get(movie, timestamp), timestamp)
            history.append(UserRating(user, movie, rating, timestamp))
        counts["source_ratings"] += 1
        if counts["source_ratings"] % 2_000_000 == 0:
            print(f"Scanned {counts['source_ratings']:,} ratings", flush=True)
    finish()
    for name, heap in heaps.items():
        if len(heap) < sizes[name]:
            raise ValueError(f"Only {len(heap)} eligible {name} users; requested {sizes[name]}. "
                             "No output published. Choose a smaller cohort before evaluating any metrics.")
    cohorts = {name: [rows for _, _, rows in sorted(heap, key=lambda x: x[1])]
               for name, heap in heaps.items()}
    return cohorts, availability, dict(counts)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(zip_path, output, legacy_dir=Path("data"), development_users=1000,
            reserved_users=1000, seed=42, *, expected_md5=None):
    # expected_md5 is dependency injection for tiny synthetic tests, never a CLI bypass.
    zip_path, output, legacy_dir = Path(zip_path), Path(output), Path(legacy_dir)
    if min(development_users, reserved_users) < 2:
        raise ValueError("Each cohort needs at least two users")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing cohort: {output}")
    legacy_paths = [legacy_dir / name for name in ("movies_enriched.csv", "links.csv", "ratings.csv")]
    legacy_hashes = {str(p): file_sha(p) for p in legacy_paths}
    old_movies, old_links, old_ratings = (read_csv(p) for p in legacy_paths)
    if not old_ratings:
        raise ValueError("Legacy ratings required for overlap screening")
    cutoff = max(int(r["timestamp"]) for r in old_ratings)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".ml32m-", dir=output.parent))
    try:
        with zipfile.ZipFile(zip_path) as archive:
            validate_archive(archive, OFFICIAL_MD5 if expected_md5 is None else expected_md5)
            links = tuple(zip_rows(archive, "links.csv"))
            mapping = identity_map(old_links, links)
            reused = {mapping[int(m["movieId"])]: m for m in old_movies if int(m["movieId"]) in mapping}
            movies = []
            for row in zip_rows(archive, "movies.csv"):
                movie = int(row["movieId"])
                if movie in reused:
                    movies.append({**row, **{f: reused[movie].get(f, "") for f in ENRICHMENT_FIELDS}})
            if not movies:
                raise ValueError("No unambiguous movies match the existing enriched catalog")
            ids = {int(m["movieId"]) for m in movies}
            events = {(mapping[int(r["movieId"])], float(r["rating"]), int(r["timestamp"]))
                      for r in old_ratings if int(r["movieId"]) in mapping}
            cohorts, availability, counts = select_users(
                zip_rows(archive, "ratings.csv"), ids, events, cutoff,
                {"development": development_users, "reserved": reserved_users}, seed)
            (stage / "README.txt").write_bytes(archive.read("ml-32m/README.txt"))
        write_csv(stage / "metadata/movies.csv", ("movieId", "title", "genres", *ENRICHMENT_FIELDS), movies)
        write_csv(stage / "metadata/availability.csv", ("movieId", "timestamp"),
                  ({"movieId": m, "timestamp": t} for m, t in sorted(availability.items())))
        for name, groups in cohorts.items():
            write_csv(stage / name / "ratings.csv", RATING_FIELDS,
                      (dict(zip(RATING_FIELDS, (r.user_id, r.movie_id, r.rating, r.timestamp)))
                       for rows in groups for r in rows))
        files = {str(p.relative_to(stage)): file_sha(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        manifest = {
            "schema_version": 1, "dataset": "MovieLens 32M", "zip_sha256": file_sha(zip_path),
            "official_checksums_verified": expected_md5 is None, "legacy_sha256": legacy_hashes,
            "legacy_max_timestamp": cutoff, "seed": seed, "counts": counts,
            "catalog_movies": len(movies), "movies_with_plot": sum(bool(m.get("plot")) for m in movies),
            "cohorts": {name: {"users": len(groups), "ratings": sum(map(len, groups))}
                        for name, groups in cohorts.items()},
            "selection": "SHA256 ml32m-v1:seed:userId; byte0 parity partitions; lowest remaining-byte hashes sampled",
            "eligibility": "first recorded source activity after legacy end; no matching old movie/rating/time event; "
                           ">=20 restricted-catalog ratings; >=5 chronological history and >=2 outcome ratings",
            "split": "first approximately 80% history, final 20% outcomes; timestamp groups intact",
            "catalog": "fixed intersection with existing metadata via unambiguous TMDB/IMDb identities",
            "availability": "earliest interaction in the entire 32M source, used only as an availability proxy",
            "metadata": "retrospective TMDB snapshot; MovieLens community tags intentionally omitted",
            "limitations": "reserved offline cohort, not prospective or causal evidence; event screening and "
                           "newer recorded activity do not prove cross-release person identity disjointness",
            "files": files,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"Output appeared during preparation: {output}")
        stage.rename(output)
        print(f"Prepared {len(movies):,} catalog movies and {development_users:,} development / "
              f"{reserved_users:,} reserved users in {output}. No model metrics computed.", flush=True)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path("data/imports/ml-32m.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/ml32m"))
    parser.add_argument("--legacy-dir", type=Path, default=Path("data"))
    parser.add_argument("--development-users", type=int, default=1000)
    parser.add_argument("--reserved-users", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        prepare(args.zip, args.output, args.legacy_dir, args.development_users, args.reserved_users, args.seed)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
