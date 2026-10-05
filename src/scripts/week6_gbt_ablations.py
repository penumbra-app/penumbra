import argparse
import json
from pathlib import Path

import numpy as np

from src.content.profiles import ProfileConfig
from src.content.scoring import ScoringConfig
from src.hybrid.gbt_reranker import (
    GBT_FEATURES,
    LINEAR_FEATURES,
    build_gbt_user_splits,
    build_user_gbt_examples,
    examples_to_pairwise_dataset,
    examples_to_pointwise_dataset,
    fit_gbt_collaborative_model,
    save_gbt_ranker,
    train_gbt_ranker,
)
from src.hybrid.ml_reranker import (
    save_ranker,
    train_ranker,
)
from src.load_data import load_movielens


EXPERIMENTS = {
    "full_gbt": [
        "collaborative",
        "personal",
        "content",
        "content_confidence",
        "quality",
        "popularity",
    ],

    "no_confidence": [
        "collaborative",
        "personal",
        "content",
        "quality",
        "popularity",
    ],

    "no_quality": [
        "collaborative",
        "personal",
        "content",
        "content_confidence",
        "popularity",
    ],

    "no_popularity": [
        "collaborative",
        "personal",
        "content",
        "content_confidence",
        "quality",
    ],

    "no_content": [
        "collaborative",
        "personal",
        "quality",
        "popularity",
    ],
}


def load_content_configuration(
    report_path: Path,
) -> tuple[
    ScoringConfig,
    ProfileConfig,
    str,
]:
    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
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
        default=Path(
            "data/movies-enriched.csv"
        ),
    )

    parser.add_argument(
        "--content-report",
        type=Path,
        default=Path(
            "docs/week6-content-repro.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "src/models/"
            "week6_gbt_ablations"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    (
        user_splits,
        global_profile_ratings,
    ) = build_gbt_user_splits(
        ratings
    )

    print("Week 6 GBT Ablation Training")
    print("=" * 76)

    print(
        f"Ratings:             "
        f"{len(ratings):,}"
    )

    print(
        f"Users:               "
        f"{ratings['userId'].nunique():,}"
    )

    print(
        f"Movies:              "
        f"{len(movies):,}"
    )

    print(
        f"Content config:      "
        f"{selected_variant}"
    )

    print(
        f"GBT experiments:     "
        f"{len(EXPERIMENTS)}"
    )

    print(
        "\nTraining shared collaborative "
        "model..."
    )

    collaborative_model = (
        fit_gbt_collaborative_model(
            global_profile_ratings
        )
    )

    # ---------------------------------------------------------
    # Build validation-training examples ONCE.
    #
    # User timeline:
    # first 60% -> profile / upstream models
    # next 10%  -> GBT + Logistic training
    # next 10%  -> validation evaluation
    # final 20% -> untouched
    # ---------------------------------------------------------

    all_examples = []

    examples_by_user = {}

    users_used = 0

    print(
        "\nBuilding shared development "
        "training data..."
    )

    for (
        user_id,
        split,
    ) in user_splits.items():

        examples = (
            build_user_gbt_examples(
                user_id=user_id,
                profile_ratings=(
                    split.profile
                ),
                target_ratings=(
                    split.development_train
                ),
                global_profile_ratings=(
                    global_profile_ratings
                ),
                movies=movies,
                collaborative_model=(
                    collaborative_model
                ),
                scoring_config=(
                    scoring_config
                ),
                profile_config=(
                    profile_config
                ),
            )
        )

        if not examples:
            continue

        examples_by_user[
            user_id
        ] = examples

        all_examples.extend(
            examples
        )

        users_used += 1

    if not all_examples:
        raise ValueError(
            "No GBT training examples "
            "were generated"
        )

    print(
        f"Users used:          "
        f"{users_used:,}"
    )

    print(
        f"Pointwise rows:      "
        f"{len(all_examples):,}"
    )

    ratings_array = np.array(
        [
            example.actual_rating
            for example in all_examples
        ],
        dtype=float,
    )

    print(
        f"Mean rating:         "
        f"{np.mean(ratings_array):.4f}"
    )

    # ---------------------------------------------------------
    # Train ONE apples-to-apples Logistic comparator.
    # ---------------------------------------------------------

    print()
    print("=" * 76)
    print("Validation Logistic Baseline")
    print("=" * 76)

    X_linear, y_linear = (
        examples_to_pairwise_dataset(
            examples_by_user,
            features=LINEAR_FEATURES,
        )
    )

    print(
        f"Features:            "
        f"{', '.join(LINEAR_FEATURES)}"
    )

    print(
        f"Pairwise examples:   "
        f"{len(y_linear):,}"
    )

    print(
        f"Feature count:       "
        f"{X_linear.shape[1]}"
    )

    linear_ranker = train_ranker(
        X_linear,
        y_linear,
    )

    print("\nCoefficients:")

    for (
        feature,
        coefficient,
    ) in zip(
        LINEAR_FEATURES,
        linear_ranker.model.coef_[0],
    ):
        print(
            f"  {feature:<22}"
            f"{coefficient:+.6f}"
        )

    logistic_path = (
        args.output_dir
        / "logistic_baseline.joblib"
    )

    save_ranker(
        linear_ranker,
        path=str(logistic_path),
    )

    print(
        f"\nSaved: {logistic_path}"
    )

    # ---------------------------------------------------------
    # Train all GBT ablations.
    # ---------------------------------------------------------

    for (
        experiment_name,
        features,
    ) in EXPERIMENTS.items():

        print()
        print("=" * 76)
        print(
            f"Experiment: "
            f"{experiment_name}"
        )
        print(
            f"Features:   "
            f"{', '.join(features)}"
        )
        print("=" * 76)

        X, y = (
            examples_to_pointwise_dataset(
                all_examples,
                features=features,
            )
        )

        if X.shape[1] != len(features):
            raise AssertionError(
                f"{experiment_name}: "
                f"expected {len(features)} "
                f"features, got "
                f"{X.shape[1]}"
            )

        print(
            f"Training rows:       "
            f"{len(y):,}"
        )

        print(
            f"Feature count:       "
            f"{X.shape[1]}"
        )

        print("\nTraining GBT...")

        ranker = train_gbt_ranker(
            X,
            y,
            feature_names=features,
        )

        print("\nFeature importances:")

        for (
            feature,
            importance,
        ) in zip(
            ranker.feature_names,
            ranker.model.feature_importances_,
        ):
            print(
                f"  {feature:<22}"
                f"{importance:.6f}"
            )

        model_path = (
            args.output_dir
            / f"{experiment_name}.joblib"
        )

        save_gbt_ranker(
            ranker,
            model_path,
        )

        print(
            f"\nSaved: {model_path}"
        )

    print()
    print("=" * 76)
    print(
        "All Week 6 GBT ablations "
        "trained."
    )
    print("=" * 76)


if __name__ == "__main__":
    main()