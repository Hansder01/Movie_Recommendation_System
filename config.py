from pathlib import Path

######################################################
# Project Root
######################################################

BASE_DIR = Path(__file__).resolve().parent


######################################################
# Dataset
######################################################

DATASET_OLD = BASE_DIR / "data" / "movies_dataset.parquet"
DATASET = BASE_DIR / "data" / "movies_dataset_new.parquet"

######################################################
# Models
######################################################

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

EMBEDDINGS = MODEL_DIR / "embeddings.npy"

NN_MODEL = MODEL_DIR / "nn_model.pkl"

######################################################
# Sentence Transformer
######################################################

MODEL_NAME = "all-MiniLM-L6-v2"

######################################################
# Recommendation Settings
######################################################

DEFAULT_RECOMMENDATIONS = 10

NN_CANDIDATES = 50

BATCH_SIZE = 3000

ENCODE_BATCH_SIZE = 64

EMBEDDING_DIM = 384

######################################################
# TMBD Api Connection 
######################################################

TMDB_API_KEY = "b811e19e6b4f378c5baba4a4e8dfd78a"

TMDB_BASE = "https://api.themoviedb.org/3"