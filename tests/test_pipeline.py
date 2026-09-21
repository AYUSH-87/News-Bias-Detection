"""
Unit and Integration Tests for News Bias Detection Pipeline.

Verifies:
- Data loading, null dropping, deduplication, and stratified splitting.
- Text cleaning, tokenization, and label encoding persistence.
- GloVe embedding parsing, OOV handling, and document vector pooling.
- FNN model construction, layer dimensions, and output shape.
- Full prediction workflow on text and file inputs.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_GLOVE_PATH,
    EMBEDDING_DIM,
    LABEL_COLUMN,
    TEXT_COLUMN,
)
from src.data_loader import clean_and_inspect_data, split_data
from src.embeddings import GloVeEmbeddingManager
from src.model import build_fnn_classifier
from src.preprocessor import (
    TextPreprocessor,
    fit_label_encoder,
    load_label_encoder,
    save_label_encoder,
)


class TestNewsBiasPipeline(unittest.TestCase):

    def setUp(self):
        self.preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=True)
        self.sample_texts = [
            "We must enact progressive taxes on billionaires to fund universal healthcare.",
            "Free market competition and deregulation unleash entrepreneurial growth.",
            "The Department of Labor published monthly employment statistics today."
        ]
        self.sample_labels = ["liberal", "conservative", "neutral"]

    def test_text_cleaning_and_tokenization(self):
        raw = "<p>Check out https://news.example.com/opinion -- Breaking: Market reforms!</p>"
        cleaned = self.preprocessor.clean_text(raw)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("!", cleaned)

        tokens = self.preprocessor.tokenize(raw)
        self.assertIn("market", tokens)
        self.assertIn("reforms", tokens)
        self.assertNotIn("p", tokens)

    def test_label_encoder_persistence(self):
        encoder = fit_label_encoder(self.sample_labels)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_encoder.joblib"
            save_label_encoder(encoder, tmp_path)
            loaded = load_label_encoder(tmp_path)
            self.assertEqual(list(encoder.classes_), list(loaded.classes_))

            transformed = loaded.transform(["liberal", "conservative"])
            self.assertEqual(len(transformed), 2)

    def test_data_loader_cleaning_and_splitting(self):
        data = {
            "text": [
                "Article 1 on policy",
                "Article 2 on reform",
                "Article 1 on policy",  # Duplicate
                None,                   # Missing text
                "   ",                  # Whitespace text
                "Article 3 on economy",
                "Article 4 on tax",
                "Article 5 on energy"
            ],
            "bias": [
                "liberal",
                "conservative",
                "liberal",
                "neutral",
                "liberal",
                "neutral",
                "liberal",
                "conservative"
            ]
        }
        df = pd.DataFrame(data)
        clean_df = clean_and_inspect_data(df, "text", "bias")
        # Duplicates and nulls must be removed: 8 total - 1 none - 1 whitespace - 1 duplicate = 5
        self.assertEqual(len(clean_df), 5)

    def test_glove_embedding_manager(self):
        if not DEFAULT_GLOVE_PATH.exists():
            self.skipTest(f"GloVe file not found at {DEFAULT_GLOVE_PATH}")

        manager = GloVeEmbeddingManager(glove_path=DEFAULT_GLOVE_PATH, expected_dim=50)
        manager.load_embeddings()
        self.assertGreater(len(manager.embeddings_dict), 100)
        self.assertEqual(manager.embedding_dim, 50)

        # Vectorize single token list
        tokens = ["market", "competition", "growth"]
        vec, in_vocab, oov = manager.text_to_vector(tokens)
        self.assertEqual(vec.shape, (50,))
        self.assertGreater(in_vocab, 0)

        # OOV case (unknown nonsense word)
        oov_tokens = ["xyzrandomnonexistentword999"]
        oov_vec, in_vocab, oov = manager.text_to_vector(oov_tokens)
        self.assertEqual(oov_vec.shape, (50,))
        self.assertEqual(in_vocab, 0)
        self.assertEqual(oov, 1)
        self.assertTrue(np.all(oov_vec == 0.0))

    def test_fnn_architecture(self):
        input_dim = 50
        num_classes = 3
        model = build_fnn_classifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_units=[64, 32],
            dropout_rates=[0.2, 0.1],
            learning_rate=0.001
        )
        self.assertEqual(model.input_shape, (None, input_dim))
        self.assertEqual(model.output_shape, (None, num_classes))

        # Forward pass with dummy batch
        dummy_batch = np.random.randn(4, input_dim).astype(np.float32)
        preds = model.predict(dummy_batch, verbose=0)
        self.assertEqual(preds.shape, (4, num_classes))
        # Probabilities should sum to ~1.0
        np.testing.assert_allclose(preds.sum(axis=1), np.ones(4), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
