import argparse
import csv
from pathlib import Path

from src.content.model import ContentModel
from src.content.selection import load_selected_model
from src.data_processing.text_metadata import with_keywords_as_of
from src.data_processing.movielens import (
    movie_metadata_from_records,
    user_ratings_from_records,
)
from src.data_processing.tmdb import ATTRIBUTION
from src.enrich_movies import metadata_summary


def _read_records(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(encoding="utf-8", newline="") as source:
        return tuple(csv.DictReader(source))


def main(
    user_id: int = 1,
    data_directory: str | Path = "data",
    prediction_count: int = 3,
    baseline: bool = False,
    movies_file: str | Path | None = None,
) -> None:
    directory = Path(data_directory)
    rating_rows = _read_records(directory / "ratings.csv")
    movie_rows = _read_records(Path(movies_file) if movies_file is not None else directory / "movies.csv")
    user_rows = tuple(
        row for row in rating_rows if int(row["userId"]) == user_id
    )
    ratings = user_ratings_from_records(user_rows)
    movies = movie_metadata_from_records(movie_rows)
    tags_path = directory / "tags.csv"
    if not baseline and tags_path.exists():
        cutoff = max((r.timestamp for r in ratings if r.timestamp is not None), default=0)
        movies = with_keywords_as_of(movies, _read_records(tags_path), cutoff, user_id)
    model = ContentModel(ratings, movies) if baseline else load_selected_model(ratings, movies)
    profile = model.build_profile(user_id)
    movies_by_id = {movie.movie_id: movie for movie in movies}

    print("PROFILE")
    print(f"Configuration: {'baseline' if baseline else 'Week 6 validation-selected TF-IDF'}")
    metadata = metadata_summary(movie_rows)
    if metadata["enriched_movies"]:
        print(f"TMDB metadata: {metadata['enriched_movies']}/{metadata['movies']} movies (retrospective snapshot)")
        print(ATTRIBUTION)
    print(f"User: {profile.user_id}")
    print(f"Version: {profile.profile_version}")
    print(f"Ratings: {profile.rating_count}")
    print(f"Baseline: {profile.baseline:.3f}")
    print()
    print(f"{'Genre':<20} {'Movies':>6} {'Evidence':>10} {'Mean':>10} {'Shrunk':>10}")

    for preference in sorted(
        profile.genre_preferences,
        key=lambda item: item.total_contribution,
        reverse=True,
    ):
        print(
            f"{preference.genre:<20} "
            f"{preference.movie_count:>6} "
            f"{preference.evidence_weight:>10.3f} "
            f"{preference.mean_contribution:>+10.3f} "
            f"{preference.preference:>+10.3f}"
        )

    predictions = model.recommend(
        user_id,
        limit=prediction_count,
        include_debug=True,
    )

    for prediction in predictions:
        movie = movies_by_id[prediction.movie_id]
        debug = prediction.debug
        if debug is None:
            continue

        print()
        print("MOVIE FEATURES")
        print(f"Movie: {movie.movie_id} - {movie.title}")
        print(f"Genres: {', '.join(debug.movie_genres) or 'None'}")
        print(f"Matched: {', '.join(debug.matched_genres) or 'None'}")
        print(f"Unknown: {', '.join(debug.unknown_genres) or 'None'}")
        print("COMPONENT")
        print(f"Raw genre value: {debug.raw_genre_component:+.3f}")
        print(f"Bounded genre value: {debug.bounded_genre_component:+.3f}")
        print(f"Genre weight: {debug.genre_weight:.3f}")
        print(f"Weighted adjustment: {debug.weighted_genre_adjustment:+.3f}")
        for component in debug.feature_components:
            print(f"{component.feature_type}: "
                  f"{', '.join(component.candidate_values) or 'missing'} "
                  f"→ adjustment {component.weighted_adjustment:+.3f}")
        print("FINAL SCORE")
        print(f"Baseline: {debug.baseline:.3f}")
        print(f"Unclamped: {debug.unclamped_score:.3f}")
        print(f"User {user_id} + unseen movie {movie.movie_id} "
              f"→ predicted {prediction.predicted_score:.3f} / 5")
        print(f"Confidence (evidence support): {prediction.confidence:.3f}")
        print(f"Clamped: {debug.was_clamped}")
        for reason in prediction.reason_signals:
            print(f"Reason: {reason.feature_type}={reason.feature_value}; "
                  f"contribution {reason.score_contribution:+.3f}; "
                  f"evidence {reason.evidence_count:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Flick content recommendations.")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--baseline", action="store_true", help="Use the original content defaults")
    parser.add_argument("--movies-file", help="Optional enriched CSV; defaults to DATA_DIR/movies.csv")
    args = parser.parse_args()
    main(args.user_id, args.data_dir, args.limit, args.baseline, args.movies_file)
