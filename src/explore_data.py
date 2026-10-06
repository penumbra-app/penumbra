import argparse

if __package__:
    from src.load_data import load_movielens
else:
    from load_data import load_movielens


def explore(data_directory="data", movies_file=None) -> None:
    ratings, movies = load_movielens(data_directory, movies_file=movies_file)

    rated_movie_ids = ratings["movieId"].nunique()
    total_movie_ids = movies["movieId"].nunique()

    ratings_per_movie = ratings.groupby("movieId").size()

    movies_with_one_rating = (ratings_per_movie == 1).sum()
    movies_with_no_ratings = total_movie_ids - rated_movie_ids

    print(f"Movies in movies.csv: {total_movie_ids:,}")
    print(f"Movies with at least one rating: {rated_movie_ids:,}")
    print(f"Movies with no ratings: {movies_with_no_ratings:,}")
    print(f"Movies with exactly one rating: {movies_with_one_rating:,}")

    fields = [field for field in ("plot", "keywords", "directors", "cast", "runtime_minutes", "language")
              if field in movies.columns]
    if fields:
        print("\nMovie metadata coverage:")
        for field in fields:
            present = movies[field].fillna("").astype(str).str.strip().ne("")
            print(f"{field}: {present.sum():,}/{len(movies):,} ({present.mean():.1%})")
        if "metadata_temporality" in movies and movies["metadata_temporality"].eq("retrospective").any():
            print("TMDB metadata is a current snapshot; historical availability is unverified.")

    print(ratings["rating"].value_counts().sort_index())

    print("\nMost-rated movies:")

    movie_stats = (
        ratings.groupby("movieId")
        .agg(
            average_rating=("rating", "mean"),
            number_of_ratings=("rating", "count"),
        )
        .reset_index()
        .merge(movies, on="movieId")
    )

    print(
        movie_stats
        .sort_values("number_of_ratings", ascending=False)
        [["title", "number_of_ratings", "average_rating"]]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect MovieLens ratings and metadata coverage.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--movies-file", help="Optional enriched CSV; defaults to DATA_DIR/movies.csv")
    args = parser.parse_args()
    explore(args.data_dir, args.movies_file)
