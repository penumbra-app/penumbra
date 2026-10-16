
"""Train matched Week 7 Hybrid candidates without modifying Week 6 artifacts.

Run from the repository root with:
    python -m src.scripts.train_week7_tfidf
"""
from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
from pathlib import Path

from src.hybrid.ml_reranker import (
    build_ablation_training_dataset,
    save_ranker,
    train_ranker,
)
from src.load_data import load_movielens
from src.scripts.week6_ablations import load_content_configuration


FEATURES = [
    "collaborative",
    "personal",
    "content",
    "quality",
    "popularity",
]

DEFAULT_MOVIES = Path(
    "handoff_inputs/data/movies_enriched.csv"
)
DEFAULT_PRESET = Path(
    "handoff_inputs/reports/tmdb-full/selected_config.json"
)
DEFAULT_REPORT = Path(
    "docs/week6-content-repro.json"
)
DEFAULT_OUTPUT = Path("src/models/week7_tfidf_split")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--movies",
        type=Path,
        default=DEFAULT_MOVIES,
    )
    parser.add_argument(
        "--selected-content-config",
        type=Path,
        default=DEFAULT_PRESET,
    )
    parser.add_argument(
        "--content-report",
        type=Path,
        default=DEFAULT_REPORT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--variant",
        choices=["both", "baseline", "tfidf"],
        default="both",
    )
    parser.add_argument(
        "--user-balanced", action="store_true",
        help="Equal total user weight in scaler and logistic fit; use a new output directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly overwrite Week 7 artifacts",
    )

    args = parser.parse_args()

    if not args.movies.is_file():
        parser.error(
            f"Movie catalog not found: {args.movies}"
        )

    if (
        args.variant in ("both", "tfidf")
        and not args.selected_content_config.is_file()
    ):
        parser.error(
            "Selected TF-IDF preset not found: "
            f"{args.selected_content_config}"
        )

    if (
        args.variant in ("both", "baseline")
        and not args.content_report.is_file()
    ):
        parser.error(
            "Week 6 Content report not found: "
            f"{args.content_report}"
        )

    ratings, movies = load_movielens(
        "data",
        movies_path=args.movies,
    )

    if not {"plot", "keywords"}.issubset(
        movies.columns
    ):
        parser.error(
            "Movie catalog must contain "
            "'plot' and 'keywords' columns"
        )

    if movies["plot"].notna().sum() == 0:
        parser.error(
            "All movie plots are missing; "
            "TF-IDF comparison would be invalid"
        )

    scoring = None
    profile = None
    selected_name = "not used"

    if args.variant in ("both", "baseline"):
        (
            scoring,
            profile,
            selected_name,
        ) = load_content_configuration(
            args.content_report
        )

    chosen = (
        ["baseline", "tfidf"]
        if args.variant == "both"
        else [args.variant]
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Movies: {len(movies):,}; "
        f"ratings: {len(ratings):,}; "
        f"plots: {movies['plot'].notna().sum():,}"
    )
    print(
        "Comparison uses the same catalog, "
        "MF settings, and chronological splits."
    )

    for variant in chosen:
        output = (
            args.output_dir
            / f"{variant}_plus_personal.joblib"
        )
        manifest_path = (
            args.output_dir
            / f"{variant}_plus_personal.json"
        )

        if not args.force and (
            output.exists()
            or manifest_path.exists()
        ):
            parser.error(
                "Refusing to overwrite "
                f"{output} or {manifest_path}; "
                "use --force if intentional"
            )

        use_selected = variant == "tfidf"

        print(
            f"\nTraining {variant} ...",
            flush=True,
        )

        started = perf_counter()
        dataset = (
            build_ablation_training_dataset(
                ratings=ratings,
                movies=movies,
                features=FEATURES,
                scoring_config=(
                    None if use_selected else scoring
                ),
                profile_config=(
                    None if use_selected else profile
                ),
                use_selected_content=use_selected,
                selected_config_path=(
                    args.selected_content_config
                    if use_selected
                    else None
                ),
                train_fraction=0.10,
                return_user_weights=args.user_balanced,
            )
        )
        X, y, users = dataset[:3]
        weights = dataset[3] if args.user_balanced else None

        if X.shape[1] != len(FEATURES):
            raise AssertionError(
                f"Expected {len(FEATURES)} "
                f"features, got {X.shape[1]}"
            )

        ranker = train_ranker(X, y, sample_weight=weights)

        save_ranker(
            ranker,
            path=str(output),
        )

        manifest = {
            "variant": variant,
            "movies": str(args.movies),
            "content_report": (
                str(args.content_report)
                if not use_selected
                else None
            ),
            "content_variant": (
                selected_name
                if not use_selected
                else None
            ),
            "selected_content_config": (
                str(args.selected_content_config)
                if use_selected
                else None
            ),
            "features": FEATURES,
            "train_users": int(users),
            "training_examples": int(len(y)),
            "training_objective": (
                "pairwise logistic; equal total user weight in scaler and model"
                if args.user_balanced else "existing pairwise logistic; standard pair weighting"
            ),
            "training_seconds": perf_counter() - started,
            "split": {"profile": 0.60, "train": 0.10, "validation": 0.10, "test": 0.20},
            "random_state": 42,
            "model": str(output),
        }
        if weights is not None:
            # A user's rows each have weight N / (U * count).
            counts, row_totals = np.unique(
                np.rint(len(y) / (users * weights)).astype(int), return_counts=True,
            )
            per_user_counts = np.repeat(counts, row_totals // counts)
            if len(per_user_counts) != users:
                raise AssertionError("Could not reconstruct user example counts")
            manifest["examples_per_user"] = {
                "min": int(per_user_counts.min()), "max": int(per_user_counts.max()),
                "mean": float(per_user_counts.mean()),
                "quantiles_25_50_75": np.quantile(per_user_counts, [.25, .5, .75]).tolist(),
                "distribution": {str(int(c)): int(n // c) for c, n in zip(counts, row_totals)},
            }

        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        print(
            f"Saved {output} "
            f"({users} users, "
            f"{len(y):,} training examples)"
        )

    print(
        "\nTraining complete."
    )
    print(
        "Split: 60% profile, 10% reranker training, 10% held-out validation, "
        "20% final test. Do not tune on final test."
    )


if __name__ == "__main__":
    main()
