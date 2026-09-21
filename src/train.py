"""
Training Pipeline for News Bias Detection.

Coordinates data loading, cleaning, stratified splitting, GloVe vectorization,
FNN compilation, model training with regularization/callbacks, evaluation on
the held-out test set, artifact persistence, and training curve visualization.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import keras
from sklearn.metrics import accuracy_score, f1_score

from config import (
    BATCH_SIZE,
    DEFAULT_DATASET_PATH,
    DEFAULT_GLOVE_PATH,
    EARLY_STOPPING_PATIENCE,
    EMBEDDING_DIM,
    EPOCHS,
    LABEL_COLUMN,
    LEARNING_RATE,
    METADATA_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    REDUCE_LR_PATIENCE,
    TEXT_COLUMN,
    TFIDF_WEIGHTS_PATH,
    TRAINING_PLOT_PATH,
)
from src.data_loader import clean_and_inspect_data, load_raw_data, split_data
from src.embeddings import GloVeEmbeddingManager
from src.evaluate import evaluate_predictions
from src.model import build_fnn_classifier, save_model
from src.preprocessor import (
    TextPreprocessor,
    fit_label_encoder,
    save_label_encoder,
    save_preprocessor_config,
)
from src.utils import (
    plot_training_history,
    print_banner,
    print_section,
    set_seed,
)


def train_pipeline(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    glove_path: Path = DEFAULT_GLOVE_PATH,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    hidden_units: Optional[List[int]] = None,
    dropout_rates: Optional[List[float]] = None,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
) -> Dict[str, float]:
    """
    Executes the full end-to-end training and test evaluation pipeline.
    """
    print_banner("News Bias Detection: Training Pipeline")
    set_seed(RANDOM_SEED)

    # 1. Load, clean, and analyze data
    raw_df = load_raw_data(dataset_path)
    clean_df = clean_and_inspect_data(raw_df, text_col=TEXT_COLUMN, label_col=LABEL_COLUMN)

    # 2. Stratified train/val/test split
    train_df, val_df, test_df = split_data(clean_df, text_col=TEXT_COLUMN, label_col=LABEL_COLUMN)

    # 3. Fit preprocessing and label encoder
    preprocessor = TextPreprocessor()
    save_preprocessor_config(preprocessor)

    label_encoder = fit_label_encoder(train_df[LABEL_COLUMN].tolist())
    save_label_encoder(label_encoder)

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    print(f"Target stance classes ({num_classes}): {class_names}")

    # Encode target labels
    y_train = label_encoder.transform(train_df[LABEL_COLUMN])
    y_val = label_encoder.transform(val_df[LABEL_COLUMN])
    y_test = label_encoder.transform(test_df[LABEL_COLUMN])

    # 4. Load GloVe embeddings and fit TF-IDF weighter
    print_section("Loading GloVe Embeddings & Computing TF-IDF Importance")
    embedding_manager = GloVeEmbeddingManager(glove_path=glove_path, expected_dim=EMBEDDING_DIM)
    embedding_manager.load_embeddings()
    dim = embedding_manager.embedding_dim

    # Fit TF-IDF on training corpus and save
    embedding_manager.fit_tfidf(train_df[TEXT_COLUMN].tolist(), preprocessor)
    import joblib
    joblib.dump(embedding_manager.tfidf_weights, TFIDF_WEIGHTS_PATH)
    print(f"Saved TF-IDF importance weights ({len(embedding_manager.tfidf_weights)} features) to: {TFIDF_WEIGHTS_PATH}")

    # 5. Vectorize datasets via TF-IDF weighted pooling
    print_section("Vectorizing Text Data via TF-IDF Weighted GloVe Pooling")
    print("Vectorizing training subset...")
    X_train = embedding_manager.vectorize_corpus(train_df[TEXT_COLUMN].tolist(), preprocessor, use_tfidf=True)

    print("Vectorizing validation subset...")
    X_val = embedding_manager.vectorize_corpus(val_df[TEXT_COLUMN].tolist(), preprocessor, use_tfidf=True)

    print("Vectorizing test subset...")
    X_test = embedding_manager.vectorize_corpus(test_df[TEXT_COLUMN].tolist(), preprocessor, use_tfidf=True)

    # 6. Build FNN Architecture
    print_section("Building Feedforward Neural Network (FNN)")
    model = build_fnn_classifier(
        input_dim=dim,
        num_classes=num_classes,
        hidden_units=hidden_units,
        dropout_rates=dropout_rates,
        learning_rate=learning_rate
    )
    model.summary()

    # 7. Configure Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # 8. Train the model
    print_section(f"Training Model ({epochs} max epochs, batch size {batch_size})")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 9. Save trained model
    print_section("Saving Model Artifacts")
    save_model(model, MODEL_PATH)

    # 10. Generate and save training loss & accuracy plot
    plot_training_history(history, TRAINING_PLOT_PATH)

    # 11. Evaluate on test set
    print_section("Evaluating Model on Held-out Test Set")
    test_probs = model.predict(X_test, verbose=0)
    test_preds = test_probs.argmax(axis=1)

    eval_metrics = evaluate_predictions(
        y_true=y_test,
        y_pred=test_preds,
        class_names=class_names,
        dataset_split_name="Test Set"
    )

    # 12. Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "dataset_path": str(dataset_path),
        "glove_path": str(glove_path),
        "embedding_dim": dim,
        "classes": class_names,
        "num_classes": num_classes,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_units": hidden_units or [128, 64],
            "dropout_rates": dropout_rates or [0.3, 0.2]
        },
        "test_metrics": eval_metrics
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved run metadata to: {METADATA_PATH}")

    print_banner("Training Pipeline Completed Successfully")
    return eval_metrics
