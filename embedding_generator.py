import os
import gc
import joblib
import numpy as np

from pathlib import Path

from sentence_transformers import SentenceTransformer

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


class EmbeddingGenerator:

    def __init__(
        self,
        model_name,
        embedding_file,
        nn_file,
        batch_size=5000
    ):

        self.model_name = model_name
        self.embedding_file = Path(embedding_file)
        self.nn_file = Path(nn_file)
        self.batch_size = batch_size

        self.model = None

    ############################################################

    def load_model(self):

        if self.model is None:

            print("Loading SentenceTransformer...")

            self.model = SentenceTransformer(
                self.model_name
            )

    ############################################################

    def load_or_create(self, contents):

        ########################################################
        # CASE 1
        # Embeddings + NN already exist
        ########################################################

        if self.embedding_file.exists() and self.nn_file.exists():

            print("Loading cached embeddings...")

            embeddings = np.load(
                self.embedding_file,
                mmap_mode="r"
            )

            nn = joblib.load(self.nn_file)

            print("Cached model loaded.")

            return embeddings, nn

        ########################################################
        # CASE 2
        # Embeddings exist but NN model doesn't
        ########################################################

        if self.embedding_file.exists():

            print("Embeddings found.")
            print("Loading embeddings...")

            embeddings = np.load(
                self.embedding_file,
                mmap_mode="r"
            )

            print("Building NearestNeighbors...")

            nn = NearestNeighbors(
                metric="cosine",
                algorithm="brute",
                n_jobs=-1
            )

            nn.fit(embeddings)

            print("Saving NN model...")

            joblib.dump(
                nn,
                self.nn_file,
                compress=3
            )

            print("NearestNeighbors model saved.")

            return embeddings, nn

        ########################################################
        # CASE 3
        # Generate everything
        ########################################################

        self.embedding_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.nn_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.load_model()

        total = len(contents)

        print(f"Generating embeddings for {total:,} movies...")

        embeddings = np.empty(
            (total, 384),
            dtype=np.float32
        )

        for start in range(0, total, self.batch_size):

            end = min(start + self.batch_size, total)

            print(f"Encoding {start:,} - {end:,}")

            batch = contents[start:end]

            batch_embeddings = self.model.encode(
                batch,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype(np.float32)

            embeddings[start:end] = batch_embeddings

            del batch
            del batch_embeddings

            gc.collect()

        print("Building NearestNeighbors...")

        nn = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_jobs=-1
        )

        nn.fit(embeddings)

        print("Saving embeddings...")

        np.save(
            self.embedding_file,
            embeddings
        )

        print("Saving NN model...")

        joblib.dump(
            nn,
            self.nn_file,
            compress=3
        )

        print("Model generation completed.")

        return embeddings, nn