"""Run with python -m src.scripts.evaluate_content."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.evaluation.content import evaluate_content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=Path("data/ratings.csv"))
    parser.add_argument("--movies", type=Path, default=Path("data/movies.csv"))
    parser.add_argument("--output", type=Path, default=Path("docs/content-evaluation-results.json"))
    parser.add_argument("--splits-output", type=Path)
    args = parser.parse_args()
    report, splits = evaluate_content(pd.read_csv(args.ratings), pd.read_csv(args.movies))
    report["input_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in (("ratings", args.ratings), ("movies", args.movies))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.splits_output:
        args.splits_output.parent.mkdir(parents=True, exist_ok=True)
        splits.to_csv(args.splits_output, index=False, lineterminator="\n")
    print(f"Selected on validation: {report['selected_variant']}")
    print(json.dumps(report["results"]["test"][report["selected_variant"]], indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
