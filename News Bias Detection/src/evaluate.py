"""
Evaluation Module for News Bias Detection.

Computes comprehensive classification metrics (accuracy, macro/weighted precision,
recall, F1-score), prints classification reports and confusion matrices to the
terminal, and saves confusion matrix visualizations.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config import (
    CONFUSION_MATRIX_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_GLOVE_PATH,
    EMBEDDING_DIM,
    LABEL_COLUMN,
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    PREPROCESSOR_CONFIG_PATH,
    RANDOM_SEED,
    TEXT_COLUMN,
    TFIDF_WEIGHTS_PATH,
)
import joblib
from src.data_loader import clean_and_inspect_data, load_raw_data, split_data
from src.embeddings import GloVeEmbeddingManager
from src.model import load_trained_model
from src.preprocessor import load_label_encoder, load_preprocessor
from src.utils import (
    plot_confusion_matrix,
    print_banner,
    print_section,
    set_seed,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    dataset_split_name: str = "Test Set",
    save_cm_plot: bool = True
) -> Dict[str, float]:
    """
    Computes classification metrics, displays a formatted summary in the terminal,
    and optionally saves a confusion matrix visualization.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print(f"\n=======================================================")
    print(f"       PERFORMANCE METRICS ({dataset_split_name.upper()})")
    print(f"=======================================================")
    print(f"  Accuracy:           {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Macro Precision:    {prec_macro:.4f}")
    print(f"  Macro Recall:       {rec_macro:.4f}")
    print(f"  Macro F1-Score:     {f1_macro:.4f}")
    print(f"  Weighted Precision: {prec_weighted:.4f}")
    print(f"  Weighted Recall:    {rec_weighted:.4f}")
    print(f"  Weighted F1-Score:  {f1_weighted:.4f}")
    print(f"=======================================================\n")

    # Scikit-learn Classification Report
    print("Classification Report:")
    print("-" * 55)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (Rows: Actual, Columns: Predicted):")
    header = f"{'':<12}" + "".join(f"{name:>12}" for name in class_names)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:<12}" + "".join(f"{val:>12d}" for val in row)
        print(row_str)

    if save_cm_plot:
        plot_confusion_matrix(cm, class_names, CONFUSION_MATRIX_PATH)

    return {
        "accuracy": float(acc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(prec_weighted),
        "weighted_recall": float(rec_weighted),
        "weighted_f1": float(f1_weighted),
    }


def evaluate_pipeline(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    glove_path: Path = DEFAULT_GLOVE_PATH,
    model_path: Path = MODEL_PATH
) -> Dict[str, float]:
    """
    Loads saved model, re-extracts the held-out test split, and evaluates performance.
    """
    print_banner("News Bias Detection: Standalone Evaluation")
    set_seed(RANDOM_SEED)

    # Load artifacts
    model = load_trained_model(model_path)
    label_encoder = load_label_encoder(LABEL_ENCODER_PATH)
    preprocessor = load_preprocessor(PREPROCESSOR_CONFIG_PATH)
    class_names = list(label_encoder.classes_)

    # Recreate test split
    raw_df = load_raw_data(dataset_path)
    clean_df = clean_and_inspect_data(raw_df, text_col=TEXT_COLUMN, label_col=LABEL_COLUMN)
    _, _, test_df = split_data(clean_df, text_col=TEXT_COLUMN, label_col=LABEL_COLUMN)

    # Vectorize test subset
    print_section("Vectorizing Test Set via GloVe & TF-IDF")
    manager = GloVeEmbeddingManager(glove_path=glove_path, expected_dim=EMBEDDING_DIM)
    manager.load_embeddings()
    if TFIDF_WEIGHTS_PATH.exists():
        manager.tfidf_weights = joblib.load(TFIDF_WEIGHTS_PATH)
    X_test = manager.vectorize_corpus(test_df[TEXT_COLUMN].tolist(), preprocessor, use_tfidf=True)
    y_test = label_encoder.transform(test_df[LABEL_COLUMN])

    # Run predictions
    print_section("Predicting on Test Subset")
    y_probs = model.predict(X_test, verbose=0)
    y_pred = y_probs.argmax(axis=1)

    return evaluate_predictions(y_test, y_pred, class_names, "Held-out Test Set")
