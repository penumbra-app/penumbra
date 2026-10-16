"""Package the frozen TF-IDF reranker and profile-only MF for explicit serving."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.hybrid.ml_reranker import chronological_split, load_ranker
from src.hybrid.week7 import DEFAULT_BUNDLE, FEATURES, file_digest, validate_ranker
from src.load_data import load_movielens
from src.scripts.evaluate_week6_ablations import train_evaluation_mf
from src.scripts.train_week7_tfidf import DEFAULT_MOVIES, DEFAULT_OUTPUT, DEFAULT_PRESET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    model_path = DEFAULT_OUTPUT / "tfidf_plus_personal.joblib"
    manifest = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    ranker = load_ranker(str(model_path))
    validate_ranker(ranker, manifest["features"])
    if manifest["variant"] != "tfidf" or Path(manifest["selected_content_config"]) != DEFAULT_PRESET:
        raise ValueError("Frozen TF-IDF manifest/preset mismatch")
    ratings, movies = load_movielens("data", movies_path=DEFAULT_MOVIES)
    profiles = [chronological_split(rows, train_fraction=.10)[0]
                for _, rows in ratings.groupby("userId", sort=False) if len(rows) >= 10]
    references = pd.concat(profiles, ignore_index=True)
    mf = train_evaluation_mf(references)
    bundle = {
        "schema_version": 1, "features": FEATURES, "ranker": ranker,
        "movies": movies, "reference_ratings": references, "collaborative_model": mf,
        "selected_config_path": str(DEFAULT_PRESET),
        "selected_config_sha256": file_digest(DEFAULT_PRESET),
        "ranker_sha256": file_digest(model_path), "catalog_sha256": file_digest(DEFAULT_MOVIES),
        "ratings_sha256": file_digest("data/ratings.csv"),
        "mf_provenance": "Only each eligible user's chronological first 60%; no downstream target ratings",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output)
    print(f"Saved {args.output}; MF profile rows={len(references)}; catalog={len(movies)}")


if __name__ == "__main__":
    main()
