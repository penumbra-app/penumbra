
"""Evaluate Week 6 Hybrid ablations and the Week 7 user-balanced Hybrid."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.collaborative.matrix_factorization import BiasedMatrixFactorization
from src.content.profiles import ProfileConfig
from src.content.scoring import ScoringConfig
from src.evaluation.metrics import comparison_credit, ndcg_at_k
from src.evaluation.splits import build_user_evaluation_split
from src.hybrid.content_adapter import build_content_model_from_frames
from src.hybrid.genre_recommender import score_movies_by_genre
from src.hybrid.ml_reranker import (
    AblationCandidate,
    calculate_movie_popularity,
    load_ranker,
)
from src.load_data import load_movielens
from src.scripts.week6_ablations import EXPERIMENTS


BENCHMARKS = {
    "old_ml": {
        "pairwise_macro": 0.63876,
        "pairwise_weighted": 0.68516,
        "ndcg_at_10": 0.9051,
    },
    "imdb_hybrid": {
        "pairwise_macro": 0.63498,
        "pairwise_weighted": 0.68267,
        "ndcg_at_10": 0.9069,
    },
}


def parse_arguments():
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
        "--models-dir",
        type=Path,
        default=Path("src/models/week6_ablations"),
    )
    parser.add_argument(
        "--balanced-model",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/week6-ablation-results.json"),
    )

    return parser.parse_args()


def load_content_configuration(report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))

    return (
        ScoringConfig(**report["selected_config"]),
        ProfileConfig(**report["profile_config"]),
        str(report["selected_variant"]),
    )


def prepare_evaluation_data(ratings):
    """
    Construct the same chronological 60/20/20 splits used in Week 6.

    Profile ratings provide upstream model evidence.
    Final test ratings are reserved for evaluation.
    """
    profiles = []
    test_users = {}

    for user_id, user_ratings in ratings.groupby(
        "userId", sort=False
    ):
        if len(user_ratings) < 10:
            continue

        profile, _, test = build_user_evaluation_split(
            user_ratings.copy()
        )

        profiles.append(profile)

        if len(test) >= 2:
            test_users[int(user_id)] = (profile, test)

    if not profiles:
        raise ValueError("No eligible users found")

    population_ratings = pd.concat(
        profiles, ignore_index=True
    )

    return population_ratings, test_users


def train_evaluation_mf(population_ratings):
    """
    Reproduce the Week 6 evaluation MF configuration.
    """
    model = BiasedMatrixFactorization(
        n_factors=20,
        learning_rate=0.005,
        regularization=0.02,
        n_epochs=40,
        prior_strength=5.0,
        random_state=42,
        shrink_latent=True,
    )

    model.fit(population_ratings)

    return model


def load_experiments(args):
    """
    Load the original Week 6 experiments and optionally register
    the Week 7 balanced model.

    The balanced model uses exactly the same five features
    as the Week 6 plus_personal experiment.
    """
    configurations = {
        name: list(features)
        for name, features in EXPERIMENTS.items()
    }

    model_paths = {
        name: args.models_dir / f"{name}.joblib"
        for name in configurations
    }

    if args.balanced_model is not None:
        name = "week7_user_balanced"

        configurations[name] = list(
            EXPERIMENTS["plus_personal"]
        )
        model_paths[name] = args.balanced_model

    rankers = {}

    for name, features in configurations.items():
        path = model_paths[name]

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing model for {name}: {path}"
            )

        ranker = load_ranker(path=str(path))

        expected = len(features)
        actual = int(ranker.scaler.n_features_in_)

        if expected != actual:
            raise ValueError(
                f"{name}: expected {expected} features, "
                f"but saved model uses {actual}"
            )

        rankers[name] = ranker

    return configurations, rankers


def score_user_movies(
    user_id,
    profile,
    test,
    movies,
    population_ratings,
    collaborative_model,
    scoring_config,
    profile_config,
    configurations,
    rankers,
):
    """
    Build the five underlying signals for each held-out movie,
    then score that movie using every registered experiment.
    """
    movie_ids = test["movieId"].astype(int).tolist()

    mf_scores = (
        collaborative_model.predict(
            user_ids=[user_id] * len(movie_ids),
            movie_ids=movie_ids,
        )
        .set_index("movie_id")["predicted_score"]
        .to_dict()
    )

    content_model = build_content_model_from_frames(
        profile_ratings=profile,
        movies=movies,
        scoring_config=scoring_config,
        profile_config=profile_config,
    )

    reference_ratings = population_ratings.loc[
        population_ratings["userId"] != user_id
    ].copy()

    popularity = calculate_movie_popularity(
        reference_ratings
    )

    genre_scores = score_movies_by_genre(
        user_id=user_id,
        user_history=profile,
        reference_ratings=reference_ratings,
        movies=movies,
        movie_ids=movie_ids,
    )

    personal = {
        candidate.movie_id: candidate.personal_score
        for candidate in genre_scores
    }

    quality = {
        candidate.movie_id: candidate.quality_score
        for candidate in genre_scores
    }

    actual_ratings = {}

    for movie_id, rating in zip(
        test["movieId"], test["rating"]
    ):
        actual_ratings.setdefault(
            int(movie_id), float(rating)
        )

    scored = {}

    for movie_id in movie_ids:
        if movie_id in scored:
            continue

        if not all(
            movie_id in source
            for source in (
                mf_scores,
                popularity,
                personal,
                quality,
                actual_ratings,
            )
        ):
            continue

        content_prediction = content_model.predict_one(
            user_id=user_id,
            movie_id=movie_id,
        )

        candidate = AblationCandidate(
            movie_id=movie_id,
            collaborative_score=float(mf_scores[movie_id]),
            personal_score=float(personal[movie_id]),
            content_score=float(
                content_prediction.predicted_score
            ),
            quality_score=float(quality[movie_id]),
            popularity=float(popularity[movie_id]),
        )

        predictions = {}

        for name, features in configurations.items():
            ranker = rankers[name]

            vector = candidate.as_array(features).reshape(
                1, -1
            )

            standardized = ranker.scaler.transform(vector)

            predictions[name] = float(
                ranker.model.decision_function(
                    standardized
                )[0]
            )

        scored[movie_id] = (
            actual_ratings[movie_id],
            predictions,
        )

    return scored


def evaluate_user(scored, experiment_names):
    """
    Calculate accuracy for each model using the same
    valid movie pairs, excluding equal-rating pairs.
    """
    movie_records = list(scored.values())

    correct = {
        name: 0.0 for name in experiment_names
    }

    valid_pairs = 0

    for i, first in enumerate(movie_records):
        first_rating, first_scores = first

        for second_rating, second_scores in movie_records[i + 1:]:
            if first_rating == second_rating:
                continue

            valid_pairs += 1

            for name in experiment_names:
                correct[name] += comparison_credit(
                    first_score=first_scores[name],
                    second_score=second_scores[name],
                    first_rating=first_rating,
                    second_rating=second_rating,
                )

    if valid_pairs == 0:
        return None

    actual_relevance = [
        record[0] for record in movie_records
    ]

    metrics = {}

    for name in experiment_names:
        predicted_scores = [
            record[1][name] for record in movie_records
        ]

        metrics[name] = {
            "accuracy": correct[name] / valid_pairs,
            "correct": correct[name],
            "ndcg": ndcg_at_k(
                actual_relevance=actual_relevance,
                predicted_scores=predicted_scores,
                k=10,
            ),
        }

    return valid_pairs, metrics


def evaluate_all_users(
    test_users,
    movies,
    population_ratings,
    collaborative_model,
    scoring_config,
    profile_config,
    configurations,
    rankers,
):
    """
    Accumulate macro, pair-weighted, and NDCG@10 results.
    Every model is evaluated over the same set of users and pairs.
    """
    names = list(configurations)

    accumulators = {
        name: {
            "user_accuracy": [],
            "user_ndcg": [],
            "correct_pairs": 0.0,
        }
        for name in names
    }

    total_pairs = 0
    evaluated_users = 0

    for user_id, (profile, test) in test_users.items():
        scored = score_user_movies(
            user_id=user_id,
            profile=profile,
            test=test,
            movies=movies,
            population_ratings=population_ratings,
            collaborative_model=collaborative_model,
            scoring_config=scoring_config,
            profile_config=profile_config,
            configurations=configurations,
            rankers=rankers,
        )

        evaluation = evaluate_user(scored, names)

        if evaluation is None:
            continue

        pair_count, user_metrics = evaluation

        evaluated_users += 1
        total_pairs += pair_count

        for name in names:
            metrics = user_metrics[name]
            acc = accumulators[name]

            acc["user_accuracy"].append(
                metrics["accuracy"]
            )
            acc["user_ndcg"].append(metrics["ndcg"])
            acc["correct_pairs"] += metrics["correct"]

    if total_pairs == 0:
        raise ValueError("No valid test pairs were generated")

    results = []

    for name, features in configurations.items():
        acc = accumulators[name]

        results.append({
            "name": name,
            "features": features,
            "pairwise_macro": float(
                np.mean(acc["user_accuracy"])
            ),
            "pairwise_weighted": float(
                acc["correct_pairs"] / total_pairs
            ),
            "ndcg_at_10": float(
                np.mean(acc["user_ndcg"])
            ),
        })

    results.sort(
        key=lambda result: (
            result["pairwise_macro"],
            result["pairwise_weighted"],
            result["ndcg_at_10"],
        ),
        reverse=True,
    )

    return results, evaluated_users, total_pairs


def display_results(results, user_count, pair_count):
    print("=" * 84)
    print(f"Users evaluated: {user_count:,}")
    print(f"Test pairs:      {pair_count:,}")

    print("\nResults")
    print("=" * 84)

    print(
        f"{'Experiment':<28}"
        f"{'Pairwise':>14}"
        f"{'Weighted':>14}"
        f"{'NDCG@10':>14}"
    )
    print("-" * 84)

    for result in results:
        print(
            f"{result['name']:<28}"
            f"{result['pairwise_macro']:>13.3%}"
            f"{result['pairwise_weighted']:>13.3%}"
            f"{result['ndcg_at_10']:>14.4f}"
        )

    print("-" * 84)

    for name, label in (
        ("old_ml", "Old ML benchmark"),
        ("imdb_hybrid", "IMDb Hybrid baseline"),
    ):
        benchmark = BENCHMARKS[name]

        print(
            f"{label:<28}"
            f"{benchmark['pairwise_macro']:>13.3%}"
            f"{benchmark['pairwise_weighted']:>13.3%}"
            f"{benchmark['ndcg_at_10']:>14.4f}"
        )

    best = results[0]
    old = BENCHMARKS["old_ml"]

    print("\nBest Evaluated Ablation")
    print("=" * 84)
    print(f"Model:    {best['name']}")
    print(f"Features: {', '.join(best['features'])}")
    print(f"Macro:    {best['pairwise_macro']:.3%}")
    print(f"Weighted: {best['pairwise_weighted']:.3%}")
    print(f"NDCG@10:  {best['ndcg_at_10']:.4f}")

    print("\nVs Old ML:")
    print(
        f"Macro:    "
        f"{(best['pairwise_macro'] - old['pairwise_macro']) * 100:+.3f} pp"
    )
    print(
        f"Weighted: "
        f"{(best['pairwise_weighted'] - old['pairwise_weighted']) * 100:+.3f} pp"
    )
    print(
        f"NDCG@10:  "
        f"{best['ndcg_at_10'] - old['ndcg_at_10']:+.4f}"
    )


def main():
    args = parse_arguments()

    ratings, movies = load_movielens(
        "data", movies_path=args.movies
    )

    (
        scoring_config,
        profile_config,
        content_variant,
    ) = load_content_configuration(args.content_report)

    print("Week 6 / Week 7 Hybrid Evaluation")
    print("=" * 84)
    print(f"Ratings:        {len(ratings):,}")
    print(f"Users:          {ratings['userId'].nunique():,}")
    print(f"Movies:         {len(movies):,}")
    print(f"Content config: {content_variant}")

    population_ratings, test_users = prepare_evaluation_data(
        ratings
    )

    configurations, rankers = load_experiments(args)

    print("\nModels loaded:")
    for name in configurations:
        print(f"  - {name}")

    print("\nTraining evaluation MF...")
    collaborative_model = train_evaluation_mf(
        population_ratings
    )

    print("\nEvaluating final 20%...\n")

    results, user_count, pair_count = evaluate_all_users(
        test_users=test_users,
        movies=movies,
        population_ratings=population_ratings,
        collaborative_model=collaborative_model,
        scoring_config=scoring_config,
        profile_config=profile_config,
        configurations=configurations,
        rankers=rankers,
    )

    display_results(results, user_count, pair_count)

    output = {
        "movies": str(args.movies),
        "content_report": str(args.content_report),
        "content_variant": content_variant,
        "users_evaluated": user_count,
        "test_pairs": pair_count,
        "benchmarks": BENCHMARKS,
        "results": results,
        "best_by_macro": results[0]["name"],
    }

    args.output.parent.mkdir(
        parents=True, exist_ok=True
    )

    args.output.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved results: {args.output}")


if __name__ == "__main__":
    main()
