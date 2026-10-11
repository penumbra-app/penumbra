
"""Train the user-balanced five-signal Hybrid reranker."""

import argparse
from pathlib import Path

from src.hybrid.ml_reranker import (
    build_ablation_training_dataset,
    save_ranker,
    train_ranker,
)
from src.load_data import load_movielens
from src.scripts.week6_ablations import (
    EXPERIMENTS,
    load_content_configuration,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--movies",
        type=Path,
        default=Path("data/movies-enriched.csv"),
    )
    parser.add_argument(
        "--content-report",
        type=Path,
        default=Path("docs/week6-content-repro.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/models/week7_user_balanced.joblib"),
    )

    args = parser.parse_args()

    if not args.movies.is_file():
        parser.error(
            f"Missing enriched movie data: {args.movies}"
        )

    ratings, movies = load_movielens(
        "data",
        movies_path=args.movies,
    )

    scoring, profile, variant = load_content_configuration(
        args.content_report
    )

    features = EXPERIMENTS["plus_personal"]

    X, y, user_count, weights = build_ablation_training_dataset(
        ratings=ratings,
        movies=movies,
        features=features,
        scoring_config=scoring,
        profile_config=profile,
        return_user_weights=True,
    )

    assert X.shape == (len(y), len(features))
    assert weights.shape == (len(y),)

    ranker = train_ranker(
        X,
        y,
        sample_weight=weights,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_ranker(
        ranker,
        path=str(args.output),
    )

    print(f"Content variant: {variant}")
    print(f"Features: {features}")
    print(f"Users trained: {user_count}")
    print(f"Training pairs: {len(y)}")
    print(f"Total weight: {weights.sum():.0f}")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
