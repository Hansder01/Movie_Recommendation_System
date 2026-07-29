import pandas as pd

from feature_builder import FeatureBuilder
from embedding_generator import EmbeddingGenerator

from config import *


class MovieRecommender:

    def __init__(self):

        self.data = None
        self.embeddings = None
        self.nn = None
        self.title_lookup = None

    ########################################################

    def load(self):

        if self.data is not None:
            return

        print("Loading Dataset...")

        self.data = pd.read_csv(
            DATASET,
            usecols=[
                "title",
                "listed_in",
                "cast",
                "director",
                "country",
                "rating",
                "type",
                "release_year",
                "description"
            ]
        )

        self.data = FeatureBuilder.build(self.data)

        # Fast title lookup
        self.title_lookup = {
            str(title).lower().strip(): idx
            for idx, title in enumerate(self.data["title"])
        }

        generator = EmbeddingGenerator(
            
            MODEL_NAME,

            EMBEDDINGS,

            NN_MODEL,

            batch_size=BATCH_SIZE

        )

        self.embeddings, self.nn = generator.load_or_create(
            self.data["content"].tolist()
        )

    ########################################################

    @staticmethod
    def _safe_set(value):

        if pd.isna(value):
            return set()

        return {
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        }

    ########################################################

    def recommend(self, title, n=10):

        self.load()

        title = title.lower().strip()

        idx = self.title_lookup.get(title)

        if idx is None:

            # Partial match fallback
            matches = self.data[
                self.data["title"]
                .str.lower()
                .str.contains(title, na=False)
            ]

            if matches.empty:
                return None, []

            idx = matches.index[0]

        source = self.data.loc[idx]

        neighbours = max(20, n * 3)

        distances, indices = self.nn.kneighbors(
            self.embeddings[idx].reshape(1, -1),
            n_neighbors=neighbours
        )

        source_genres = self._safe_set(source["listed_in"])
        source_cast = self._safe_set(source["cast"])

        source_year = pd.to_numeric(
            source["release_year"],
            errors="coerce"
        )

        recommendations = []

        for dist, i in zip(distances[0][1:], indices[0][1:]):

            movie = self.data.iloc[i]

            if movie["title"] == source["title"]:
                continue

            score = 1 - dist

            ################################################
            # Genre Bonus
            ################################################

            genres = self._safe_set(movie["listed_in"])

            score += (
                len(source_genres & genres) * 0.03
            )

            ################################################
            # Director Bonus
            ################################################

            if (
                pd.notna(movie["director"])
                and pd.notna(source["director"])
                and movie["director"] == source["director"]
            ):
                score += 0.08

            ################################################
            # Cast Bonus
            ################################################

            cast = self._safe_set(movie["cast"])

            score += (
                len(source_cast & cast) * 0.02
            )

            ################################################
            # Release Year Bonus
            ################################################

            movie_year = pd.to_numeric(
                movie["release_year"],
                errors="coerce"
            )

            if (
                pd.notna(source_year)
                and pd.notna(movie_year)
                and abs(source_year - movie_year) <= 5
            ):
                score += 0.02

            recommendations.append({

                "Title": movie["title"],
                "Genre": movie["listed_in"],
                "Director": movie["director"],
                "Cast": movie["cast"],
                "Country": movie["country"],
                "Rating": movie["rating"],
                "Type": movie["type"],
                "Release Year": (
                    int(movie_year)
                    if pd.notna(movie_year)
                    else None
                ),
                "Description": movie["description"],
                "Similarity": round(score * 100, 2)

            })

        source_movie = {
            "Title": source["title"],
            "Genre": source["listed_in"],
            "Director": source["director"],
            "Cast": source["cast"],
            "Country": source["country"],
            "Rating": source["rating"],
            "Type": source["type"],
            "Release Year": int(source_year) if pd.notna(source_year) else None,
            "Description": source["description"],
        }

        return (
            source_movie,
            pd.DataFrame(recommendations)
            .drop_duplicates("Title")
            .sort_values(
                "Similarity",
                ascending=False
            )
            .head(n)
            .to_dict("records")
        )


########################################################

recommender = MovieRecommender()

def load():
    recommender.load()

def recommend(title, n=10):

    return recommender.recommend(title, n)