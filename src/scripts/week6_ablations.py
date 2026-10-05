import argparse
import json
from pathlib import Path

from src.content.profiles import ProfileConfig
from src.content.scoring import ScoringConfig
from src.hybrid.ml_reranker import (
    build_ablation_training_dataset,
    save_ranker,
    train_ranker,
)
from src.load_data import load_movielens


EXPERIMENTS = {
    # Current best Week 6 linear hybrid.
    "imdb_baseline": [
        "collaborative",
        "content",
        "quality",
        "popularity",
    ],

    # Test Week 5 finding that Personal is complementary.
    "plus_personal": [
        "collaborative",
        "personal",
        "content",
        "quality",
        "popularity",
    ],

    # Test whether Quality is redundant once MF + Personal exist.
    "personal_no_quality": [
        "collaborative",
        "personal",
        "content",
        "popularity",
    ],

    # Test whether MF is redundant once Personal + Quality exist.
    "personal_no_mf": [
        "personal",
        "content",
        "quality",
        "popularity",
    ],

    # Pure MF-vs-Quality redundancy test without Personal.
    "no_quality": [
        "collaborative",
        "content",
        "popularity",
    ],

    # Remove MF from the current IMDb baseline.
    "no_mf": [
        "content",
        "quality",
        "popularity",
    ],
}


def load_content_configuration(
    report_path: Path,
) -> tuple[ScoringConfig, ProfileConfig, str]:
    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    scoring_config = ScoringConfig(
        **report["selected_config"]
    )

    profile_config = ProfileConfig(
        **report["profile_config"]
    )

    selected_variant = str(
        report["selected_variant"]
    )

    return (
        scoring_config,
        profile_config,
        selected_variant,
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
        "--output-dir",
        type=Path,
        default=Path("src/models/week6_ablations"),
    )

    args = parser.parse_args()

    ratings, movies = load_movielens(
        "data",
        movies_path=args.movies,
    )

    (
        scoring_config,
        profile_config,
        selected_variant,
    ) = load_content_configuration(
        args.content_report
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Week 6 Ablation Training")
    print("=" * 72)

    print(f"Ratings:             {len(ratings):,}")
    print(f"Users:               {ratings['userId'].nunique():,}")
    print(f"Movies:              {len(movies):,}")
    print(f"Content config:      {selected_variant}")
    print(f"Experiments:         {len(EXPERIMENTS)}")
    print()

    for experiment_name, features in EXPERIMENTS.items():
        print("=" * 72)
        print(f"Experiment: {experiment_name}")
        print(f"Features:   {', '.join(features)}")
        print("=" * 72)

        X, y, users_used = build_ablation_training_dataset(
            ratings=ratings,
            movies=movies,
            features=features,
            scoring_config=scoring_config,
            profile_config=profile_config,
        )

        if X.shape[1] != len(features):
            raise AssertionError(
                f"{experiment_name}: expected "
                f"{len(features)} features, "
                f"got {X.shape[1]}"
            )

        print(f"Users used:        {users_used:,}")
        print(f"Training examples: {len(y):,}")
        print(f"Feature count:     {X.shape[1]}")
        print(f"Positive labels:   {(y == 1).sum():,}")
        print(f"Negative labels:   {(y == 0).sum():,}")

        print("\nTraining reranker...")

        ranker = train_ranker(
            X,
            y,
        )

        print("Coefficients:")

        for feature, coefficient in zip(
            features,
            ranker.model.coef_[0],
        ):
            print(
                f"  {feature:<16} "
                f"{coefficient:+.6f}"
            )

        model_path = (
            args.output_dir
            / f"{experiment_name}.joblib"
        )

        save_ranker(
            ranker,
            path=str(model_path),
        )

        print(f"\nSaved: {model_path}")
        print()

    print("=" * 72)
    print(
        f"All {len(EXPERIMENTS)} Week 6 "
        "ablation models trained."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()