"""
Preprocessor Module for News Bias Detection.

Provides text cleaning, normalization, punctuation removal, tokenization,
stopword filtering with safe offline fallback, and LabelEncoder persistence.
"""

import json
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import joblib
from sklearn.preprocessing import LabelEncoder

from config import (
    LABEL_ENCODER_PATH,
    LOWERCASE,
    MIN_WORD_LEN,
    PREPROCESSOR_CONFIG_PATH,
    REMOVE_PUNCTUATION,
    REMOVE_STOPWORDS,
)

# Robust fallback English stopwords if NLTK data is unavailable offline
FALLBACK_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


class TextPreprocessor:
    """
    Standard text normalization and tokenization engine for journalistic articles.
    Ensures identical transformations are applied at train, eval, and inference time.
    """

    def __init__(
        self,
        lowercase: bool = LOWERCASE,
        remove_stopwords: bool = REMOVE_STOPWORDS,
        remove_punctuation: bool = REMOVE_PUNCTUATION,
        min_word_len: int = MIN_WORD_LEN
    ):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.min_word_len = min_word_len
        self.stopwords = self._init_stopwords() if remove_stopwords else set()

    def _init_stopwords(self) -> Set[str]:
        """Loads NLTK English stopwords or falls back to built-in set."""
        try:
            import nltk
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
        except Exception:
            return set(FALLBACK_STOPWORDS)

    def clean_text(self, text: str) -> str:
        """
        Cleans raw article text: strips HTML, URLs, non-ASCII/odd control chars,
        and regularizes whitespace.
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # Normalize whitespace and newlines
        text = re.sub(r"\s+", " ", text).strip()

        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            # Replace punctuation with spaces to avoid joining words
            for char in string.punctuation:
                text = text.replace(char, " ")
            text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Converts text into cleaned, normalized word tokens suitable for GloVe lookup.
        """
        cleaned = self.clean_text(text)
        tokens = cleaned.split()

        filtered_tokens = []
        for token in tokens:
            if len(token) < self.min_word_len:
                continue
            if self.remove_stopwords and token in self.stopwords:
                continue
            # Keep only alphanumeric tokens
            if not token.isalnum():
                token = re.sub(r"[^\w]", "", token)
                if len(token) < self.min_word_len:
                    continue
            filtered_tokens.append(token)

        return filtered_tokens

    def to_config(self) -> Dict[str, Any]:
        """Serializes preprocessor configuration to a dictionary."""
        return {
            "lowercase": self.lowercase,
            "remove_stopwords": self.remove_stopwords,
            "remove_punctuation": self.remove_punctuation,
            "min_word_len": self.min_word_len,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TextPreprocessor":
        """Instantiates preprocessor from a configuration dictionary."""
        return cls(
            lowercase=config.get("lowercase", True),
            remove_stopwords=config.get("remove_stopwords", False),
            remove_punctuation=config.get("remove_punctuation", True),
            min_word_len=config.get("min_word_len", 2),
        )


def fit_label_encoder(labels: List[str]) -> LabelEncoder:
    """Fits a LabelEncoder on the target classes."""
    encoder = LabelEncoder()
    encoder.fit(labels)
    return encoder


def save_label_encoder(encoder: LabelEncoder, filepath: Path = LABEL_ENCODER_PATH) -> None:
    """Serializes LabelEncoder to disk using joblib."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, filepath)
    print(f"Saved LabelEncoder to: {filepath}")


def load_label_encoder(filepath: Path = LABEL_ENCODER_PATH) -> LabelEncoder:
    """Loads a serialized LabelEncoder from disk."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"LabelEncoder artifact not found at '{filepath}'. Please train the model first.")
    return joblib.load(filepath)


def save_preprocessor_config(
    preprocessor: TextPreprocessor,
    filepath: Path = PREPROCESSOR_CONFIG_PATH
) -> None:
    """Saves preprocessor hyperparameters to a JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(preprocessor.to_config(), f, indent=2)
    print(f"Saved preprocessor configuration to: {filepath}")


def load_preprocessor(filepath: Path = PREPROCESSOR_CONFIG_PATH) -> TextPreprocessor:
    """Loads TextPreprocessor settings from a JSON file."""
    filepath = Path(filepath)
    if not filepath.exists():
        # Fall back to default settings if config not found
        return TextPreprocessor()
    with open(filepath, "r", encoding="utf-8") as f:
        config = json.load(f)
    return TextPreprocessor.from_config(config)
