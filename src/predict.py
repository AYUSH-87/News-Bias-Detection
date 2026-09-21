"""
Inference Module for News Bias Detection.

Provides terminal prediction for raw article text or text files, reporting
the predicted political stance, confidence score, and full class probability
distribution alongside necessary epistemic disclaimers.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from config import (
    DEFAULT_GLOVE_PATH,
    EMBEDDING_DIM,
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    PREPROCESSOR_CONFIG_PATH,
    TFIDF_WEIGHTS_PATH,
)
import joblib
from src.embeddings import GloVeEmbeddingManager
from src.model import load_trained_model
from src.preprocessor import load_label_encoder, load_preprocessor
from src.utils import print_banner


class BiasPredictor:
    """
    Inference engine that encapsulates model, preprocessor, and GloVe embeddings
    to provide terminal predictions.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        label_encoder_path: Path = LABEL_ENCODER_PATH,
        preprocessor_config_path: Path = PREPROCESSOR_CONFIG_PATH,
        tfidf_weights_path: Path = TFIDF_WEIGHTS_PATH,
        glove_path: Path = DEFAULT_GLOVE_PATH
    ):
        self.model = load_trained_model(model_path)
        self.label_encoder = load_label_encoder(label_encoder_path)
        self.preprocessor = load_preprocessor(preprocessor_config_path)
        self.embedding_manager = GloVeEmbeddingManager(glove_path=glove_path, expected_dim=EMBEDDING_DIM)
        self.embedding_manager.load_embeddings()
        if Path(tfidf_weights_path).exists():
            self.embedding_manager.tfidf_weights = joblib.load(tfidf_weights_path)
        self.classes = list(self.label_encoder.classes_)

    def predict_text(self, text: str) -> Dict[str, Any]:
        """
        Processes a raw input text string and returns prediction probabilities.
        """
        tokens = self.preprocessor.tokenize(text)
        vector, in_vocab, oov = self.embedding_manager.text_to_vector(tokens, use_tfidf=True)

        # Batch dimension: (1, embedding_dim)
        input_tensor = np.expand_dims(vector, axis=0)
        probabilities = self.model.predict(input_tensor, verbose=0)[0]

        pred_idx = int(np.argmax(probabilities))
        pred_class = self.classes[pred_idx]
        confidence = float(probabilities[pred_idx])

        prob_dict = {
            cls_name: float(prob)
            for cls_name, prob in zip(self.classes, probabilities)
        }

        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "token_count": len(tokens),
            "in_vocab_count": in_vocab,
            "oov_count": oov,
        }

    def print_prediction_result(self, result: Dict[str, Any], source_desc: str = "Input Text") -> None:
        """
        Prints a formatted terminal summary of the prediction result.
        """
        print_banner("News Bias Prediction Result")
        print(f"Source: {source_desc}")
        print(f"Tokens analyzed: {result['token_count']} (Recognized in GloVe: {result['in_vocab_count']}, OOV: {result['oov_count']})")
        print("\n" + "=" * 55)
        print(f"  PREDICTED STANCE:   {result['predicted_class'].upper()}")
        print(f"  CONFIDENCE:         {result['confidence'] * 100:.2f}%")
        print("=" * 55)

        print("\nProbability Distribution:")
        print(f"{'Class':<15} {'Probability':<12} {'Visual Distribution':<25}")
        print("-" * 55)
        for stance, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            bar_len = int(prob * 20)
            bar = "#" * bar_len
            print(f"{stance:<15} {prob:>6.2%}       |{bar:<20}|")
        print("-" * 55)

        print("\n[ETHICAL / EPISTEMIC DISCLAIMER]")
        print("  This prediction reflects statistical patterns learned strictly from the")
        print("  annotated training dataset. It does not constitute an objective truth")
        print("  determination of political ideology, nor does it replace nuanced human editorial judgment.")
        print("-" * 55 + "\n")


def predict_from_text(
    text: str,
    glove_path: Path = DEFAULT_GLOVE_PATH,
    model_path: Path = MODEL_PATH
) -> Dict[str, Any]:
    """CLI helper to run inference on a raw text string."""
    predictor = BiasPredictor(model_path=model_path, glove_path=glove_path)
    snippet = (text[:80] + "...") if len(text) > 80 else text
    result = predictor.predict_text(text)
    predictor.print_prediction_result(result, source_desc=f'Raw Text ("{snippet}")')
    return result


def predict_from_file(
    file_path: Path,
    glove_path: Path = DEFAULT_GLOVE_PATH,
    model_path: Path = MODEL_PATH
) -> Dict[str, Any]:
    """CLI helper to run inference on an article text file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found at '{file_path}'.")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    predictor = BiasPredictor(model_path=model_path, glove_path=glove_path)
    result = predictor.predict_text(content)
    predictor.print_prediction_result(result, source_desc=f"File: {file_path.name}")
    return result
