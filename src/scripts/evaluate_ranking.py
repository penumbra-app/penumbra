import argparse
import json
from pathlib import Path

from src.content.profiles import ProfileConfig
from src.content.scoring import ScoringConfig
from src.evaluation.ranking import evaluate_pairwise_accuracy
from src.evaluation.reports import print_week5_metrics
from src.load_data import load_movielens


def weighted_accuracy(user_results, field_name: str) -> float:
    total_pairs = sum(
        user.pairs_evaluated
        for user in user_results
    )

    total_correct = sum(
        getattr(user, field_name) * user.pairs_evaluated
        for user in user_results
    )

    return total_correct / total_pairs


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--movies",
        type=Path,
        default=Path("data/movies.csv"),
    )

    parser.add_argument(
        "--hybrid-model",
        type=Path,
        default=Path("src/models/hybrid_v1_reranker.joblib"),
    )

    parser.add_argument(
        "--content-report",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--content-confidence",
        action="store_true",
        help="Evaluate a hybrid model trained with content confidence.",
    )

    parser.add_argument(
        "--personal",
        action="store_true",
        help="Evaluate a hybrid model trained with personal preference.",
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

    print(
        "Content confidence:",
        "enabled" if args.content_confidence else "disabled",
    )

    print("Dataset")
    print("=" * 72)

    print(f"Ratings:      {len(ratings):,}")
    print(f"Users:        {ratings['userId'].nunique():,}")
    print(f"Rated movies: {ratings['movieId'].nunique():,}")
    print(f"Movie rows:   {len(movies):,}")

    print("\nRunning final 20% evaluation...\n")

    result = evaluate_pairwise_accuracy(
        ratings=ratings,
        movies=movies,
        hybrid_ranker_path=str(args.hybrid_model),
        content_scoring_config=scoring_config,
        content_profile_config=profile_config,
        include_content_confidence=args.content_confidence,
        include_personal=args.personal,
    )

    print("Pairwise Ranking Evaluation")
    print("=" * 72)

    print(f"Users evaluated: {result.users_evaluated:,}")
    print(f"Test pairs:      {result.pairs_evaluated:,}")

    print("\nMacro Accuracy")
    print("-" * 72)

    print(f"Baseline:   {result.baseline_accuracy:.3%}")
    print(f"Heuristic:  {result.heuristic_accuracy:.3%}")
    print(f"Old ML:     {result.ml_accuracy:.3%}")
    print(f"MF:         {result.matrix_factorization_accuracy:.3%}")
    print(f"Content:    {result.content_accuracy:.3%}")
    print(f"Hybrid:     {result.hybrid_accuracy:.3%}")

    print("\nHybrid Improvement")
    print("-" * 72)

    print(
        f"vs baseline: "
        f"{(result.hybrid_accuracy - result.baseline_accuracy) * 100:+.3f} pp"
    )
    print(
        f"vs old ML:   "
        f"{(result.hybrid_accuracy - result.ml_accuracy) * 100:+.3f} pp"
    )
    print(
        f"vs MF:       "
        f"{(result.hybrid_accuracy - result.matrix_factorization_accuracy) * 100:+.3f} pp"
    )
    print(
        f"vs content:  "
        f"{(result.hybrid_accuracy - result.content_accuracy) * 100:+.3f} pp"
    )

    user_results = result.user_results

    print_week5_metrics(user_results)

    baseline_weighted = weighted_accuracy(
        user_results,
        "baseline_accuracy",
    )

    heuristic_weighted = weighted_accuracy(
        user_results,
        "heuristic_accuracy",
    )

    ml_weighted = weighted_accuracy(
        user_results,
        "ml_accuracy",
    )

    mf_weighted = weighted_accuracy(
        user_results,
        "matrix_factorization_accuracy",
    )

    content_weighted = weighted_accuracy(
        user_results,
        "content_accuracy",
    )

    hybrid_weighted = weighted_accuracy(
        user_results,
        "hybrid_accuracy",
    )

    print("\nPair-Weighted Accuracy")
    print("-" * 72)

    print(f"Baseline:   {baseline_weighted:.3%}")
    print(f"Heuristic:  {heuristic_weighted:.3%}")
    print(f"Old ML:     {ml_weighted:.3%}")
    print(f"MF:         {mf_weighted:.3%}")
    print(f"Content:    {content_weighted:.3%}")
    print(f"Hybrid:     {hybrid_weighted:.3%}")

    print("\nHybrid Pair-Weighted Improvement")
    print("-" * 72)

    print(
        f"vs baseline: "
        f"{(hybrid_weighted - baseline_weighted) * 100:+.3f} pp"
    )
    print(
        f"vs old ML:   "
        f"{(hybrid_weighted - ml_weighted) * 100:+.3f} pp"
    )
    print(
        f"vs MF:       "
        f"{(hybrid_weighted - mf_weighted) * 100:+.3f} pp"
    )
    print(
        f"vs content:  "
        f"{(hybrid_weighted - content_weighted) * 100:+.3f} pp"
    )


if __name__ == "__main__":
    main()