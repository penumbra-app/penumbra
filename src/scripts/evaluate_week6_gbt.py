import argparse
import json
from pathlib import Path

import numpy as np

from src.content.profiles import ProfileConfig
from src.content.scoring import ScoringConfig
from src.evaluation.metrics import (
    comparison_credit,
    ndcg_at_k,
)
from src.hybrid.gbt_reranker import (
    LINEAR_FEATURES,
    build_gbt_user_splits,
    build_user_gbt_examples,
    fit_gbt_collaborative_model,
    load_gbt_ranker,
    predict_gbt_score,
)
from src.hybrid.ml_reranker import (
    load_ranker,
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

    return (
        ScoringConfig(
            **report[
                "selected_config"
            ]
        ),
        ProfileConfig(
            **report[
                "profile_config"
            ]
        ),
        str(
            report[
                "selected_variant"
            ]
        ),
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
        "--gbt-model",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--linear-model",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if args.gbt_model is None:
        if args.stage == "validation":
            args.gbt_model = Path(
                "src/models/"
                "week6_gbt_validation.joblib"
            )
        else:
            args.gbt_model = Path(
                "src/models/"
                "week6_gbt_final.joblib"
            )

    if args.linear_model is None:
        if args.stage == "validation":
            args.linear_model = Path(
                "src/models/"
                "week6_logistic_validation.joblib"
            )
        else:
            args.linear_model = Path(
                "src/models/"
                "week6_ablations/"
                "plus_personal.joblib"
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

    print("Week 6 GBT Evaluation")
    print("=" * 76)

    print(
        f"Stage:          {args.stage}"
    )
    print(
        f"Content config: {selected_variant}"
    )

    print(
        "\nTraining evaluation MF..."
    )

    collaborative_model = (
        fit_gbt_collaborative_model(
            global_profile_ratings
        )
    )

    gbt_ranker = load_gbt_ranker(
        args.gbt_model
    )

    linear_ranker = load_ranker(
        path=str(
            args.linear_model
        )
    )

    gbt_user_accuracies = []
    linear_user_accuracies = []

    gbt_user_ndcgs = []
    linear_user_ndcgs = []

    gbt_weighted_correct = 0.0
    linear_weighted_correct = 0.0

    total_pairs = 0
    users_evaluated = 0

    print("\nEvaluating...\n")

    for (
        user_id,
        split,
    ) in user_splits.items():

        if args.stage == "validation":
            target_ratings = (
                split.validation
            )
        else:
            target_ratings = (
                split.test
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

        if len(examples) < 2:
            continue

        movie_data = []

        for example in examples:
            gbt_score = (
                predict_gbt_score(
                    gbt_ranker,
                    example.features,
                )
            )

            linear_raw = (
                example.features
                .as_array(
                    LINEAR_FEATURES
                )
                .reshape(1, -1)
            )

            linear_scaled = (
                linear_ranker
                .scaler
                .transform(
                    linear_raw
                )
            )

            linear_score = float(
                linear_ranker
                .model
                .decision_function(
                    linear_scaled
                )[0]
            )

            movie_data.append({
                "actual": (
                    example.actual_rating
                ),
                "gbt": gbt_score,
                "linear": linear_score,
            })

        gbt_correct = 0.0
        linear_correct = 0.0
        user_pairs = 0

        for i in range(
            len(movie_data)
        ):
            for j in range(
                i + 1,
                len(movie_data),
            ):
                first = movie_data[i]
                second = movie_data[j]

                if (
                    first["actual"]
                    == second["actual"]
                ):
                    continue

                gbt_correct += (
                    comparison_credit(
                        first_score=(
                            first["gbt"]
                        ),
                        second_score=(
                            second["gbt"]
                        ),
                        first_rating=(
                            first["actual"]
                        ),
                        second_rating=(
                            second["actual"]
                        ),
                    )
                )

                linear_correct += (
                    comparison_credit(
                        first_score=(
                            first["linear"]
                        ),
                        second_score=(
                            second["linear"]
                        ),
                        first_rating=(
                            first["actual"]
                        ),
                        second_rating=(
                            second["actual"]
                        ),
                    )
                )

                user_pairs += 1

        if user_pairs == 0:
            continue

        actual_relevance = [
            row["actual"]
            for row in movie_data
        ]

        gbt_scores = [
            row["gbt"]
            for row in movie_data
        ]

        linear_scores = [
            row["linear"]
            for row in movie_data
        ]

        gbt_ndcg = ndcg_at_k(
            actual_relevance=(
                actual_relevance
            ),
            predicted_scores=(
                gbt_scores
            ),
            k=10,
        )

        linear_ndcg = ndcg_at_k(
            actual_relevance=(
                actual_relevance
            ),
            predicted_scores=(
                linear_scores
            ),
            k=10,
        )

        gbt_accuracy = (
            gbt_correct
            / user_pairs
        )

        linear_accuracy = (
            linear_correct
            / user_pairs
        )

        gbt_user_accuracies.append(
            gbt_accuracy
        )

        linear_user_accuracies.append(
            linear_accuracy
        )

        gbt_user_ndcgs.append(
            gbt_ndcg
        )

        linear_user_ndcgs.append(
            linear_ndcg
        )

        gbt_weighted_correct += (
            gbt_correct
        )

        linear_weighted_correct += (
            linear_correct
        )

        total_pairs += user_pairs
        users_evaluated += 1

    if total_pairs == 0:
        raise ValueError(
            "No valid evaluation pairs"
        )

    gbt_macro = float(
        np.mean(
            gbt_user_accuracies
        )
    )

    linear_macro = float(
        np.mean(
            linear_user_accuracies
        )
    )

    gbt_weighted = float(
        gbt_weighted_correct
        / total_pairs
    )

    linear_weighted = float(
        linear_weighted_correct
        / total_pairs
    )

    gbt_ndcg = float(
        np.mean(
            gbt_user_ndcgs
        )
    )

    linear_ndcg = float(
        np.mean(
            linear_user_ndcgs
        )
    )

    print("=" * 76)
    print(
        f"Users evaluated: "
        f"{users_evaluated:,}"
    )
    print(
        f"Pairs evaluated: "
        f"{total_pairs:,}"
    )

    print("\nResults")
    print("=" * 76)

    print(
        f"{'Model':<24}"
        f"{'Macro':>16}"
        f"{'Weighted':>16}"
        f"{'NDCG@10':>16}"
    )

    print("-" * 76)

    print(
        f"{'Logistic':<24}"
        f"{linear_macro:>15.3%}"
        f"{linear_weighted:>15.3%}"
        f"{linear_ndcg:>16.4f}"
    )

    print(
        f"{'GBT':<24}"
        f"{gbt_macro:>15.3%}"
        f"{gbt_weighted:>15.3%}"
        f"{gbt_ndcg:>16.4f}"
    )

    print("-" * 76)

    print("\nGBT minus Logistic")

    print(
        "Macro:    "
        f"{(
            gbt_macro
            - linear_macro
        ) * 100:+.3f} pp"
    )

    print(
        "Weighted: "
        f"{(
            gbt_weighted
            - linear_weighted
        ) * 100:+.3f} pp"
    )

    print(
        "NDCG@10:  "
        f"{(
            gbt_ndcg
            - linear_ndcg
        ):+.4f}"
    )


if __name__ == "__main__":
    main()