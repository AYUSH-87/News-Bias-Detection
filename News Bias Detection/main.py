"""
News Bias Detection: Feedforward Neural Network with GloVe Embeddings.

Main terminal CLI entrypoint for training, evaluating, and predicting
political stance in journalistic opinion pieces.

Available Commands:
    python main.py train [options]
    python main.py evaluate [options]
    python main.py predict --text "article text"
    python main.py predict --file path/to/article.txt
    python main.py download-glove [--dim 50]
"""

import argparse
import sys
from pathlib import Path

from config import (
    BATCH_SIZE,
    DEFAULT_DATASET_PATH,
    DEFAULT_GLOVE_PATH,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
)
from src.evaluate import evaluate_pipeline
from src.predict import predict_from_file, predict_from_text
from src.train import train_pipeline
from src.utils import download_official_glove


def parse_args():
    parser = argparse.ArgumentParser(
        description="News Bias Detection CLI: Classify political stance in journalistic opinion pieces using GloVe & FNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # --- TRAIN SUBCOMMAND ---
    train_parser = subparsers.add_parser("train", help="Train the FNN classifier on the labeled news dataset")
    train_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to labeled CSV dataset"
    )
    train_parser.add_argument(
        "--glove-path",
        type=Path,
        default=DEFAULT_GLOVE_PATH,
        help="Path to pretrained GloVe .txt file"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Maximum training epochs"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Mini-batch size for training"
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help="Initial learning rate for Adam optimizer"
    )
    train_parser.add_argument(
        "--hidden-units",
        type=str,
        default="128,64",
        help="Comma-separated list of hidden layer dimensions (e.g. '128,64')"
    )
    train_parser.add_argument(
        "--dropouts",
        type=str,
        default="0.3,0.2",
        help="Comma-separated list of dropout rates (e.g. '0.3,0.2')"
    )
    train_parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Early stopping patience in epochs"
    )

    # --- EVALUATE SUBCOMMAND ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the saved model on the held-out test set")
    eval_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to labeled CSV dataset"
    )
    eval_parser.add_argument(
        "--glove-path",
        type=Path,
        default=DEFAULT_GLOVE_PATH,
        help="Path to pretrained GloVe .txt file"
    )
    eval_parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Path to saved .keras model file"
    )

    # --- PREDICT SUBCOMMAND ---
    predict_parser = subparsers.add_parser("predict", help="Predict political stance for a raw text or text file")
    input_group = predict_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        type=str,
        help="Inline text string of the opinion piece to classify"
    )
    input_group.add_argument(
        "--file",
        type=Path,
        help="Path to a .txt file containing the article text"
    )
    predict_parser.add_argument(
        "--glove-path",
        type=Path,
        default=DEFAULT_GLOVE_PATH,
        help="Path to GloVe embeddings file"
    )
    predict_parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Path to trained model .keras file"
    )

    # --- DOWNLOAD-GLOVE SUBCOMMAND ---
    download_parser = subparsers.add_parser("download-glove", help="Download official Stanford GloVe 6B embeddings")
    download_parser.add_argument(
        "--dim",
        type=int,
        default=50,
        choices=[50, 100, 200, 300],
        help="Embedding dimension to extract from glove.6B.zip"
    )
    download_parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data/embeddings"),
        help="Directory to save the extracted GloVe file"
    )

    return parser


def main():
    parser = parse_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.command == "train":
        try:
            hidden_units = [int(u.strip()) for u in args.hidden_units.split(",") if u.strip()]
            dropout_rates = [float(d.strip()) for d in args.dropouts.split(",") if d.strip()]
        except ValueError:
            print("Error: --hidden-units and --dropouts must be comma-separated numeric values.")
            sys.exit(1)

        train_pipeline(
            dataset_path=args.dataset,
            glove_path=args.glove_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            hidden_units=hidden_units,
            dropout_rates=dropout_rates,
            early_stopping_patience=args.patience
        )

    elif args.command == "evaluate":
        evaluate_pipeline(
            dataset_path=args.dataset,
            glove_path=args.glove_path,
            model_path=args.model_path
        )

    elif args.command == "predict":
        if args.text:
            predict_from_text(
                text=args.text,
                glove_path=args.glove_path,
                model_path=args.model_path
            )
        elif args.file:
            predict_from_file(
                file_path=args.file,
                glove_path=args.glove_path,
                model_path=args.model_path
            )

    elif args.command == "download-glove":
        download_official_glove(dimension=args.dim, target_dir=args.target_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
