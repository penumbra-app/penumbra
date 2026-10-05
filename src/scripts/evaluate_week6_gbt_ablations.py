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
from src.scripts.week6_gbt_ablations import (
    EXPERIMENTS,
)


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
        "--models-dir",
        type=Path,
        default=Path(
            "src/models/"
            "week6_gbt_ablations"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/"
            "week6-gbt-ablation-results.json"
        ),
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

    (
        user_splits,
        global_profile_ratings,
    ) = build_gbt_user_splits(
        ratings
    )

    print("Week 6 GBT Ablation Evaluation")
    print("=" * 88)

    print(
        f"Ratings:        "
        f"{len(ratings):,}"
    )

    print(
        f"Users:          "
        f"{ratings['userId'].nunique():,}"
    )

    print(
        f"Movies:         "
        f"{len(movies):,}"
    )

    print(
        f"Content config: "
        f"{selected_variant}"
    )

    # ---------------------------------------------------------
    # Shared upstream MF.
    # ---------------------------------------------------------

    print(
        "\nTraining shared evaluation MF..."
    )

    collaborative_model = (
        fit_gbt_collaborative_model(
            global_profile_ratings
        )
    )

    # ---------------------------------------------------------
    # Load Logistic comparator.
    # ---------------------------------------------------------

    logistic_path = (
        args.models_dir
        / "logistic_baseline.joblib"
    )

    linear_ranker = load_ranker(
        path=str(logistic_path)
    )

    expected_linear_features = (
        len(LINEAR_FEATURES)
    )

    actual_linear_features = int(
        linear_ranker.scaler.n_features_in_
    )

    if (
        actual_linear_features
        != expected_linear_features
    ):
        raise ValueError(
            "Logistic feature mismatch: "
            f"model expects "
            f"{actual_linear_features}, "
            f"evaluation expects "
            f"{expected_linear_features}"
        )

    # ---------------------------------------------------------
    # Load all GBT models.
    # ---------------------------------------------------------

    gbt_rankers = {}

    for (
        experiment_name,
        expected_features,
    ) in EXPERIMENTS.items():

        model_path = (
            args.models_dir
            / f"{experiment_name}.joblib"
        )

        ranker = load_gbt_ranker(
            model_path
        )

        if (
            ranker.feature_names
            != expected_features
        ):
            raise ValueError(
                f"{experiment_name}: "
                "saved feature list does "
                "not match experiment "
                "definition.\n"
                f"Saved:    "
                f"{ranker.feature_names}\n"
                f"Expected: "
                f"{expected_features}"
            )

        gbt_rankers[
            experiment_name
        ] = ranker

    model_names = [
        "logistic",
        *EXPERIMENTS.keys(),
    ]

    user_accuracies = {
        name: []
        for name in model_names
    }

    user_ndcgs = {
        name: []
        for name in model_names
    }

    weighted_correct = {
        name: 0.0
        for name in model_names
    }

    total_pairs = 0
    users_evaluated = 0

    print(
        "\nEvaluating validation 10%...\n"
    )

    # ---------------------------------------------------------
    # Evaluate all models on exactly the same validation rows.
    # ---------------------------------------------------------

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
                    split.validation
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
            # ---------------------------------------------
            # Logistic score
            # ---------------------------------------------

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

            scores = {
                "logistic": linear_score,
            }

            # ---------------------------------------------
            # Every GBT ablation
            # ---------------------------------------------

            for (
                experiment_name,
                ranker,
            ) in gbt_rankers.items():

                scores[
                    experiment_name
                ] = predict_gbt_score(
                    ranker,
                    example.features,
                )

            movie_data.append({
                "actual": (
                    example.actual_rating
                ),
                "scores": scores,
            })

        correct = {
            name: 0.0
            for name in model_names
        }

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

                for name in model_names:
                    correct[name] += (
                        comparison_credit(
                            first_score=(
                                first["scores"][
                                    name
                                ]
                            ),
                            second_score=(
                                second["scores"][
                                    name
                                ]
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

        users_evaluated += 1
        total_pairs += user_pairs

        actual_relevance = [
            row["actual"]
            for row in movie_data
        ]

        for name in model_names:
            accuracy = (
                correct[name]
                / user_pairs
            )

            predicted_scores = [
                row["scores"][name]
                for row in movie_data
            ]

            ndcg = ndcg_at_k(
                actual_relevance=(
                    actual_relevance
                ),
                predicted_scores=(
                    predicted_scores
                ),
                k=10,
            )

            user_accuracies[
                name
            ].append(
                accuracy
            )

            user_ndcgs[
                name
            ].append(
                ndcg
            )

            weighted_correct[
                name
            ] += correct[name]

    if total_pairs == 0:
        raise ValueError(
            "No valid validation pairs "
            "were generated"
        )

    # ---------------------------------------------------------
    # Collect metrics.
    # ---------------------------------------------------------

    results = {}

    for name in model_names:
        results[name] = {
            "pairwise_macro": float(
                np.mean(
                    user_accuracies[name]
                )
            ),
            "pairwise_weighted": float(
                weighted_correct[name]
                / total_pairs
            ),
            "ndcg_at_10": float(
                np.mean(
                    user_ndcgs[name]
                )
            ),
        }

    logistic = results[
        "logistic"
    ]

    # ---------------------------------------------------------
    # Main result table.
    # ---------------------------------------------------------

    print("=" * 88)

    print(
        f"Users evaluated: "
        f"{users_evaluated:,}"
    )

    print(
        f"Pairs evaluated: "
        f"{total_pairs:,}"
    )

    print("\nResults")
    print("=" * 88)

    print(
        f"{'Model':<30}"
        f"{'Macro':>18}"
        f"{'Weighted':>18}"
        f"{'NDCG@10':>18}"
    )

    print("-" * 88)

    display_names = {
        "logistic": (
            "Logistic"
        ),
        "full_gbt": (
            "Full GBT"
        ),
        "no_confidence": (
            "GBT - Confidence"
        ),
        "no_quality": (
            "GBT - Quality"
        ),
        "no_popularity": (
            "GBT - Popularity"
        ),
        "no_content": (
            "GBT - Content"
        ),
    }

    for name in model_names:
        result = results[name]

        print(
            f"{display_names[name]:<30}"
            f"{result['pairwise_macro']:>17.3%}"
            f"{result['pairwise_weighted']:>17.3%}"
            f"{result['ndcg_at_10']:>18.4f}"
        )

    # ---------------------------------------------------------
    # Deltas against Logistic.
    # ---------------------------------------------------------

    print()
    print("GBT delta vs Logistic")
    print("=" * 88)

    print(
        f"{'Model':<30}"
        f"{'Macro':>18}"
        f"{'Weighted':>18}"
        f"{'NDCG@10':>18}"
    )

    print("-" * 88)

    for name in EXPERIMENTS:
        result = results[name]

        macro_delta = (
            result["pairwise_macro"]
            - logistic["pairwise_macro"]
        ) * 100

        weighted_delta = (
            result["pairwise_weighted"]
            - logistic[
                "pairwise_weighted"
            ]
        ) * 100

        ndcg_delta = (
            result["ndcg_at_10"]
            - logistic["ndcg_at_10"]
        )

        print(
            f"{display_names[name]:<30}"
            f"{macro_delta:>+17.3f} pp"
            f"{weighted_delta:>+17.3f} pp"
            f"{ndcg_delta:>+18.4f}"
        )

    # ---------------------------------------------------------
    # Best GBT by each metric.
    # We deliberately do not pretend one metric is the
    # sole objective.
    # ---------------------------------------------------------

    gbt_names = list(
        EXPERIMENTS.keys()
    )

    best_macro = max(
        gbt_names,
        key=lambda name:
        results[name][
            "pairwise_macro"
        ],
    )

    best_weighted = max(
        gbt_names,
        key=lambda name:
        results[name][
            "pairwise_weighted"
        ],
    )

    best_ndcg = max(
        gbt_names,
        key=lambda name:
        results[name][
            "ndcg_at_10"
        ],
    )

    print()
    print("Best GBT by Metric")
    print("=" * 88)

    print(
        "Macro:    "
        f"{display_names[best_macro]} "
        f"({results[best_macro]['pairwise_macro']:.3%})"
    )

    print(
        "Weighted: "
        f"{display_names[best_weighted]} "
        f"({results[best_weighted]['pairwise_weighted']:.3%})"
    )

    print(
        "NDCG@10:  "
        f"{display_names[best_ndcg]} "
        f"({results[best_ndcg]['ndcg_at_10']:.4f})"
    )

    # ---------------------------------------------------------
    # Save reproducible result report.
    # ---------------------------------------------------------

    output = {
        "stage": "validation",
        "movies": str(
            args.movies
        ),
        "content_report": str(
            args.content_report
        ),
        "content_variant": (
            selected_variant
        ),
        "users_evaluated": (
            users_evaluated
        ),
        "pairs_evaluated": (
            total_pairs
        ),
        "experiments": (
            EXPERIMENTS
        ),
        "results": results,
        "best_gbt_by_metric": {
            "pairwise_macro": (
                best_macro
            ),
            "pairwise_weighted": (
                best_weighted
            ),
            "ndcg_at_10": (
                best_ndcg
            ),
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved results: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()