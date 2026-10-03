import argparse
import json
from pathlib import Path

import numpy as np

from src.content.profiles import ProfileConfig
from src.content.scoring import ScoringConfig
from src.hybrid.ml_reranker import (
    build_hybrid_training_dataset,
    save_ranker,
    train_ranker,
)
from src.load_data import load_movielens


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--movies",
        type=Path,
        default=Path("data/movies.csv"),
    )

    parser.add_argument(
        "--content-report",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/models/hybrid_v1_reranker.joblib"),
    )

    args = parser.parse_args()

    ratings, movies = load_movielens(
        "data",
        movies_path=args.movies,
    )

    scoring_config = None
    profile_config = None

    if args.content_report is not None:
        report = json.loads(
            args.content_report.read_text(encoding="utf-8")
        )

        scoring_config = ScoringConfig(
            **report["selected_config"]
        )

        profile_config = ProfileConfig(
            **report["profile_config"]
        )

        print(
            "Content configuration:",
            report["selected_variant"],
        )

    print("Building hybrid reranker training dataset...")

    X, y, users_used = build_hybrid_training_dataset(
        ratings=ratings,
        movies=movies,
        scoring_config=scoring_config,
        profile_config=profile_config,
    )

    print(f"Users used:        {users_used:,}")
    print(f"Training examples: {len(y):,}")
    print(f"Feature count:     {X.shape[1]}")
    print(f"Positive labels:   {(y == 1).sum():,}")
    print(f"Negative labels:   {(y == 0).sum():,}")

    print("\nTraining hybrid ranker...")

    ranker = train_ranker(X, y)

    print(
        "Coefficients:",
        ranker.model.coef_[0],
    )

    save_ranker(
        ranker,
        path=args.output,
    )

    print(f"\nSaved model to {args.output}")


if __name__ == "__main__":
    main()