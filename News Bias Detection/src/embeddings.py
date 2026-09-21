"""
GloVe Embeddings Module for News Bias Detection.

Loads pretrained GloVe vector files, handles out-of-vocabulary (OOV) tokens,
and aggregates word vectors into fixed-size document-level representations
via mean pooling for feeding into the Feedforward Neural Network.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import DEFAULT_GLOVE_PATH, EMBEDDING_DIM
from src.preprocessor import TextPreprocessor


class GloVeEmbeddingManager:
    """
    Manages loading of GloVe vectors and converting text documents into
    fixed-size numerical representations.
    """

    def __init__(self, glove_path: Path = DEFAULT_GLOVE_PATH, expected_dim: Optional[int] = EMBEDDING_DIM):
        self.glove_path = Path(glove_path)
        self.expected_dim = expected_dim
        self.embeddings_dict: Dict[str, np.ndarray] = {}
        self.embedding_dim: int = expected_dim or 50

    def load_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Parses the GloVe text file line by line into a word-to-vector lookup table.
        Format: word float1 float2 ... floatD
        """
        if not self.glove_path.exists():
            raise FileNotFoundError(
                f"GloVe embeddings file not found at '{self.glove_path}'.\n"
                f"Please ensure a GloVe file is placed at this path or specify --glove-path.\n"
                f"You can also run: python main.py download-glove --dim 50"
            )

        print(f"Loading GloVe embeddings from: {self.glove_path}")
        embeddings = {}
        detected_dim = None

        with open(self.glove_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_idx, line in enumerate(f, start=1):
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                word = parts[0]
                try:
                    vector = np.asarray(parts[1:], dtype=np.float32)
                except ValueError:
                    # Sometimes special characters or tokens have weird splits; skip malformed lines
                    continue

                if detected_dim is None:
                    detected_dim = vector.shape[0]
                elif vector.shape[0] != detected_dim:
                    continue  # Inconsistent vector length; skip

                embeddings[word.lower()] = vector

        if not embeddings:
            raise ValueError(f"No valid GloVe vectors found in '{self.glove_path}'.")

        self.embeddings_dict = embeddings
        self.embedding_dim = detected_dim

        if self.expected_dim is not None and self.embedding_dim != self.expected_dim:
            print(
                f"Notice: Configured dimension ({self.expected_dim}) differs from loaded "
                f"GloVe dimension ({self.embedding_dim}). Using detected dimension {self.embedding_dim}."
            )

        print(f"Successfully loaded {len(self.embeddings_dict):,} word vectors (dimension: {self.embedding_dim}d).")
        return self.embeddings_dict

    def fit_tfidf(self, texts: List[str], preprocessor: TextPreprocessor) -> None:
        """Fits a TF-IDF vectorizer over the training corpus to compute word importance weights."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=preprocessor.tokenize,
            token_pattern=None,
            lowercase=False,
            min_df=1
        )
        self.tfidf_vectorizer.fit(texts)
        self.tfidf_weights = dict(zip(
            self.tfidf_vectorizer.get_feature_names_out(),
            self.tfidf_vectorizer.idf_
        ))

    def text_to_vector(
        self,
        tokens: List[str],
        use_tfidf: bool = True
    ) -> Tuple[np.ndarray, int, int]:
        """
        Computes the document representation via weighted pooling across recognized word vectors.
        Uses TF-IDF importance weights if fitted; otherwise falls back to uniform mean pooling.
        """
        if not self.embeddings_dict:
            raise RuntimeError("Embeddings dictionary is empty. Call load_embeddings() first.")

        vectors = []
        weights = []
        oov_count = 0

        tfidf_map = getattr(self, "tfidf_weights", {}) if use_tfidf else {}

        for token in tokens:
            vec = self.embeddings_dict.get(token)
            if vec is not None:
                vectors.append(vec)
                # Word weight from TF-IDF IDF table or default 1.0
                w = tfidf_map.get(token, 1.0)
                weights.append(w)
            else:
                oov_count += 1

        in_vocab_count = len(vectors)

        if in_vocab_count > 0:
            weights_arr = np.asarray(weights, dtype=np.float32)
            weights_sum = np.sum(weights_arr)
            if weights_sum > 0:
                doc_vector = np.average(vectors, axis=0, weights=weights_arr)
            else:
                doc_vector = np.mean(vectors, axis=0)
        else:
            doc_vector = np.zeros((self.embedding_dim,), dtype=np.float32)

        return doc_vector, in_vocab_count, oov_count

    def vectorize_corpus(
        self,
        texts: List[str],
        preprocessor: TextPreprocessor,
        use_tfidf: bool = True
    ) -> np.ndarray:
        """
        Transforms an entire corpus of raw texts into a 2D matrix of shape
        (num_samples, embedding_dim).
        """
        matrix = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        total_tokens = 0
        total_oov = 0

        for i, text in enumerate(texts):
            tokens = preprocessor.tokenize(text)
            vec, in_vocab, oov = self.text_to_vector(tokens, use_tfidf=use_tfidf)
            matrix[i] = vec
            total_tokens += (in_vocab + oov)
            total_oov += oov

        oov_rate = (total_oov / total_tokens * 100) if total_tokens > 0 else 0.0
        print(f"Corpus vectorized: {len(texts)} articles | Total words: {total_tokens:,} | OOV: {oov_rate:.2f}%")
        return matrix
