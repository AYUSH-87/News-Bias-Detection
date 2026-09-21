"""
Feedforward Neural Network (FNN) Architecture for News Bias Detection.

Defines the multi-layer perceptron (MLP) accepting fixed-size GloVe document
vectors, utilizing Dense layers with ReLU activations, Dropout regularization,
and a Softmax classification head.
"""

from pathlib import Path
from typing import List, Optional

import keras
from keras import layers

from config import (
    ACTIVATION,
    DROPOUT_RATES,
    HIDDEN_UNITS,
    LEARNING_RATE,
    MODEL_PATH,
)


def build_fnn_classifier(
    input_dim: int,
    num_classes: int,
    hidden_units: Optional[List[int]] = None,
    dropout_rates: Optional[List[float]] = None,
    activation: str = ACTIVATION,
    learning_rate: float = LEARNING_RATE
) -> keras.Model:
    """
    Constructs and compiles a Feedforward Neural Network (FNN/MLP).

    Args:
        input_dim: Dimensionality of the pooled GloVe vector (e.g., 50, 100, 300).
        num_classes: Number of political stance classes (e.g., 3 for left/center/right).
        hidden_units: List of unit counts for hidden dense layers.
        dropout_rates: List of dropout rates corresponding to each hidden layer.
        activation: Activation function for hidden layers ('relu').
        learning_rate: Initial learning rate for Adam optimizer.

    Returns:
        Compiled Keras Model instance.
    """
    if hidden_units is None:
        hidden_units = HIDDEN_UNITS
    if dropout_rates is None:
        dropout_rates = DROPOUT_RATES

    if len(hidden_units) != len(dropout_rates):
        raise ValueError(
            f"Length of hidden_units ({len(hidden_units)}) must match "
            f"dropout_rates ({len(dropout_rates)})."
        )

    # Input layer matching GloVe representation size
    inputs = layers.Input(shape=(input_dim,), name="glove_input_vector")
    x = inputs

    # Fully connected hidden layers with ReLU, LayerNorm, and Dropout
    for idx, (units, rate) in enumerate(zip(hidden_units, dropout_rates), start=1):
        x = layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=keras.regularizers.l2(1e-4),
            name=f"dense_{idx}_{units}"
        )(x)
        x = layers.LayerNormalization(name=f"layer_norm_{idx}")(x)
        if rate > 0:
            x = layers.Dropout(rate, name=f"dropout_{idx}_{rate}")(x)

    # Classification output head
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="stance_probabilities"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="NewsBias_FNN_Classifier")

    # Compile with Adam and sparse categorical crossentropy
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_model(model: keras.Model, filepath: Path = MODEL_PATH) -> None:
    """Saves the trained Keras model to disk (.keras format)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    model.save(filepath)
    print(f"Saved trained FNN model to: {filepath}")


def load_trained_model(filepath: Path = MODEL_PATH) -> keras.Model:
    """Loads a saved Keras model from disk."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Model file not found at '{filepath}'. Please train the model first using 'python main.py train'."
        )
    return keras.models.load_model(filepath)
