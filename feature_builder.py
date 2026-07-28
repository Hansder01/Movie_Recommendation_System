import pandas as pd


class FeatureBuilder:

    @staticmethod
    def build(df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        for col in [
            "listed_in",
            "cast",
            "director",
            "country",
            "rating",
            "type",
            "description"
        ]:
            if col not in df:
                df[col] = ""

        if "release_year" not in df:
            df["release_year"] = 0

        df["content"] = (

            (df["listed_in"].fillna("") + " ") * 4 +

            (df["cast"].fillna("") + " ") * 3 +

            (df["director"].fillna("") + " ") * 2 +

            (df["country"].fillna("") + " ") * 2 +

            (df["rating"].fillna("").astype(str) + " ") +

            (df["type"].fillna("") + " ") +

            (df["release_year"].astype(str) + " ") +

            df["description"].fillna("")

        )

        return df