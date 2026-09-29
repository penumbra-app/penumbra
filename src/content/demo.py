import argparse
import csv
import json
from pathlib import Path

from src.content.model import ContentModel
from src.content.reliability import ProfileConfig
from src.content.scoring import ScoringConfig
from src.data_processing.movielens import (
    movie_metadata_from_records,
    user_ratings_from_records,
)


def _read_records(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(encoding="utf-8", newline="") as source:
        return tuple(csv.DictReader(source))


def main(
    user_id: int = 1,
    data_directory: str | Path = "data",
    prediction_count: int = 3,
    movies_path: str | Path | None = None,
    evaluation_report: str | Path | None = None,
    cast_weight: float | None = None,
) -> None:
    if isinstance(prediction_count, bool) or not isinstance(prediction_count, int) or prediction_count < 0:
        raise ValueError("prediction_count must be a non-negative integer")
    directory = Path(data_directory)
    rating_rows = _read_records(directory / "ratings.csv")
    movie_file = Path(movies_path) if movies_path else directory / "movies.csv"
    movie_rows = _read_records(movie_file)
    people_file = movie_file.with_suffix(".people.json")
    people = json.loads(people_file.read_text(encoding="utf-8")) if people_file.exists() else {}
    user_rows = tuple(
        row for row in rating_rows if int(row["userId"]) == user_id
    )
    ratings = user_ratings_from_records(user_rows)
    movies = movie_metadata_from_records(movie_rows)
    scoring_options, profile_options = {}, {}
    if evaluation_report:
        report = json.loads(Path(evaluation_report).read_text(encoding="utf-8"))
        scoring_options = report["selected_config"]
        profile_options = report["profile_config"]
    if cast_weight is not None:
        scoring_options["cast_weight"] = cast_weight
    model = ContentModel(ratings, movies, ScoringConfig(**scoring_options), ProfileConfig(**profile_options))
    profile = model.build_profile(user_id)
    movies_by_id = {movie.movie_id: movie for movie in movies}

    print("PROFILE")
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

    # Score the entire unseen catalog before selecting the highest scores.
    predictions = sorted(model.predict_unseen((user_id,), include_debug=True),
                         key=lambda p: (-p.predicted_score, -p.confidence, p.movie_id))[:prediction_count]

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
                  f"{', '.join(people.get(value, value) for value in component.candidate_values) or 'missing'} "
                  f"-> adjustment {component.weighted_adjustment:+.3f}")
        print("FINAL SCORE")
        print(f"Baseline: {debug.baseline:.3f}")
        print(f"Unclamped: {debug.unclamped_score:.3f}")
        print(f"User {user_id} + unseen movie {movie.movie_id} "
              f"-> predicted {prediction.predicted_score:.3f} / 5")
        print(f"Confidence (evidence support): {prediction.confidence:.3f}")
        print(f"Clamped: {debug.was_clamped}")
        print(f"Fallback: {prediction.fallback_reason or 'none'}")
        print("REASON SIGNALS")
        reasons = prediction.to_dict()["reason_signals"]
        for reason in reasons:
            if reason["feature_value"] in people:
                reason["display_name"] = people[reason["feature_value"]]
        print(json.dumps(reasons, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Content recommendations with supported reason signals")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--data-directory", type=Path, default=Path("data"))
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--movies", type=Path, help="Optional enriched movies CSV")
    parser.add_argument("--evaluation-report", type=Path, help="Load a validation-selected scoring configuration")
    parser.add_argument("--cast-weight", type=float, help="Explicit experimental cast weight override")
    args = parser.parse_args()
    main(args.user_id, args.data_directory, args.count, args.movies, args.evaluation_report, args.cast_weight)
