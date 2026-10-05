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
        "--stage",
        choices=[
            "validation",
            "final",
        ],
        default="validation",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--linear-output",
        type=Path,
        default=Path(
            "src/models/"
            "week6_logistic_validation.joblib"
        ),
    )

    parser.add_argument(
        "--features",
        nargs="+",
        choices=GBT_FEATURES,
        default=GBT_FEATURES.copy(),
        help="GBT features to include.",
    )

    args = parser.parse_args()

    if args.output is None:
        if args.stage == "validation":
            args.output = Path(
                "src/models/"
                "week6_gbt_validation.joblib"
            )
        else:
            args.output = Path(
                "src/models/"
                "week6_gbt_final.joblib"
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

    print("Week 6 GBT Training")
    print("=" * 72)

    print(
        f"Stage:          {args.stage}"
    )
    print(
        f"Content config: {selected_variant}"
    )
    print(
        "Features:       "
        + ", ".join(args.features)
    )

    print(
        "\nTraining collaborative model..."
    )

    collaborative_model = (
        fit_gbt_collaborative_model(
            global_profile_ratings
        )
    )

    all_examples = []

    examples_by_user = {}

    users_used = 0

    print(
        "\nBuilding GBT training data..."
    )

    for (
        user_id,
        split,
    ) in user_splits.items():

        if args.stage == "validation":
            target_ratings = (
                split.development_train
            )
        else:
            target_ratings = (
                split.middle
            )

        examples = (
            build_user_gbt_examples(
                user_id=user_id,
                profile_ratings=(
                    split.profile
                ),
                target_ratings=(
                    target_ratings
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

    X, y = (
        examples_to_pointwise_dataset(
            all_examples,
            features=args.features,
        )
    )

    print(
        f"Users used:      {users_used:,}"
    )
    print(
        f"Training rows:   {len(y):,}"
    )
    print(
        f"Feature count:   {X.shape[1]}"
    )
    print(
        f"Mean rating:     {np.mean(y):.4f}"
    )

    print("\nTraining GBT...")

    ranker = train_gbt_ranker(
        X,
        y,
        feature_names=args.features,
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

    save_gbt_ranker(
        ranker,
        args.output,
    )

    print(
        f"\nSaved GBT: {args.output}"
    )

    # For validation only, train an
    # apples-to-apples logistic baseline
    # on exactly the same development data.
    if args.stage == "validation":
        print(
            "\nBuilding validation "
            "logistic baseline..."
        )

        X_linear, y_linear = (
            examples_to_pairwise_dataset(
                examples_by_user,
                features=(
                    LINEAR_FEATURES
                ),
            )
        )

        print(
            f"Pairwise examples: "
            f"{len(y_linear):,}"
        )

        linear_ranker = train_ranker(
            X_linear,
            y_linear,
        )

        print(
            "Logistic coefficients:"
        )

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

        save_ranker(
            linear_ranker,
            path=str(
                args.linear_output
            ),
        )

        print(
            "\nSaved validation "
            f"logistic baseline: "
            f"{args.linear_output}"
        )


if __name__ == "__main__":
    main()