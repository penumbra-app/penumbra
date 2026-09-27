from pathlib import Path

from src.hybrid.ml_reranker import (
    build_training_dataset,
    save_ranker,
    train_ranker,
)
from src.load_data import load_movielens


OUTPUT_PATH = Path("src/models/week5_old_ml_retest.joblib")


def main() -> None:
    ratings, movies = load_movielens("data")

    print("Week 5 Old ML Retraining Sanity Check")
    print("=" * 72)

    print(f"Ratings:      {len(ratings):,}")
    print(f"Users:        {ratings['userId'].nunique():,}")
    print(f"Rated movies: {ratings['movieId'].nunique():,}")
    print(f"Movie rows:   {len(movies):,}")

    print("\nBuilding Personal + Quality + Popularity training dataset...")

    X, y, users_used = build_training_dataset(
        ratings=ratings,
        movies=movies,
    )

    print(f"\nUsers used:        {users_used:,}")
    print(f"Training examples: {len(y):,}")
    print(f"Feature count:     {X.shape[1]}")
    print(f"Positive labels:   {(y == 1).sum():,}")
    print(f"Negative labels:   {(y == 0).sum():,}")

    if X.shape[1] != 3:
        raise AssertionError(
            f"Expected 3 Old ML features, got {X.shape[1]}"
        )

    print("\nTraining fresh Old ML reranker...")

    ranker = train_ranker(X, y)

    feature_names = [
        "personal",
        "quality",
        "popularity",
    ]

    print("\nCoefficients")
    print("-" * 72)

    for feature, coefficient in zip(
        feature_names,
        ranker.model.coef_[0],
    ):
        print(f"{feature:<16} {coefficient:+.6f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    save_ranker(
        ranker,
        path=str(OUTPUT_PATH),
    )

    print(f"\nSaved fresh model to: {OUTPUT_PATH}")
    print("\nFrozen src/models/ml_reranker.joblib was NOT modified.")


if __name__ == "__main__":
    main()