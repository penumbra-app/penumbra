"""Load the selected model and score supplied pairs."""
from pathlib import Path
import sys
import argparse
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.collaborative import BiasedMatrixFactorization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "artifacts/selected_mf.joblib")
    parser.add_argument("--user", type=int, default=23)
    parser.add_argument("--movies", nargs="+", type=int, default=[1, 50, 999999999])
    args = parser.parse_args()
    model = BiasedMatrixFactorization.load(args.model)
    print(model.predict([args.user] * len(args.movies), args.movies).to_string(index=False))


if __name__ == "__main__":
    main()
