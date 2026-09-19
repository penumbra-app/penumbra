from src.load_data import load_movielens
from src.hybrid.recommender import recommend_for_user


def main() -> None:
    ratings, movies = load_movielens("data")

    user_id = 23

    recommendations = recommend_for_user(
        user_id=user_id,
        ratings=ratings,
        movies=movies,
        limit=10,
    )

    print(f"Final Penumbra recommendations for User {user_id}:\n")

    for position, movie in enumerate(
        recommendations,
        start=1,
    ):
        print(
            f"{position}. {movie.title}\n"
            f"   Genres: {movie.genres}\n"
            f"   Score: {movie.score:.3f}\n"
        )


if __name__ == "__main__":
    main()