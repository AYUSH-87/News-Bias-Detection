"""
Configuration Module for News Bias Detection.

Provides centralized paths, model architecture specifications, preprocessing
options, and training hyperparameters.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# File Paths
DEFAULT_DATASET_PATH = DATA_DIR / "news_bias_dataset.csv"
DEFAULT_GLOVE_PATH = EMBEDDINGS_DIR / "glove_mini.50d.txt"
SAMPLE_ARTICLE_PATH = DATA_DIR / "sample_article.txt"

# Saved Model & Artifact Paths
MODEL_PATH = ARTIFACTS_DIR / "fnn_bias_model.keras"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.joblib"
PREPROCESSOR_CONFIG_PATH = ARTIFACTS_DIR / "preprocessor_config.json"
TFIDF_WEIGHTS_PATH = ARTIFACTS_DIR / "tfidf_weights.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
TRAINING_PLOT_PATH = ARTIFACTS_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"

# Dataset Column Definitions
TEXT_COLUMN = "text"
LABEL_COLUMN = "bias"

# Text Preprocessing Settings
LOWERCASE = True
REMOVE_STOPWORDS = False   # Set to True if aggressive stopword removal is desired
REMOVE_PUNCTUATION = True
MIN_WORD_LEN = 2

# Embedding Settings
EMBEDDING_DIM = 50         # Matches GloVe dimension (e.g., 50, 100, 200, 300)
POOLING_STRATEGY = "mean"  # "mean" word pooling for fixed-size document vector

# FNN Architecture & Training Hyperparameters
HIDDEN_UNITS = [128, 64]
DROPOUT_RATES = [0.3, 0.2]
ACTIVATION = "relu"
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3

# Data Splitting
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15
RANDOM_SEED = 42

# Ensure required directories exist
for directory in [DATA_DIR, ARTIFACTS_DIR, EMBEDDINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
