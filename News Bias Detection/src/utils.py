"""
Utilities module for News Bias Detection.

Provides reproducible seeding, visualization helpers (loss/accuracy curves,
confusion matrix heatmap), formatted terminal logging, and optional GloVe downloading.
"""

import os
import random
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / terminal execution
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and TensorFlow for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def print_banner(title: str) -> None:
    """Prints a styled terminal banner."""
    line = "=" * 70
    print(f"\n{line}")
    print(f"  {title.upper()}")
    print(f"{line}\n")


def print_section(title: str) -> None:
    """Prints a section header in the terminal."""
    print(f"\n--- {title} ---")


def plot_training_history(history: tf.keras.callbacks.History, save_path: Path) -> None:
    """
    Plots training and validation loss and accuracy curves and saves the figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#FAFAFA")

    # Accuracy Plot
    ax1.set_facecolor("#FFFFFF")
    ax1.plot(epochs_range, acc, label="Training Accuracy", color="#2563EB", lw=2, marker="o", markersize=4)
    ax1.plot(epochs_range, val_acc, label="Validation Accuracy", color="#10B981", lw=2, marker="s", markersize=4)
    ax1.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=True)

    # Loss Plot
    ax2.set_facecolor("#FFFFFF")
    ax2.plot(epochs_range, loss, label="Training Loss", color="#DC2626", lw=2, marker="o", markersize=4)
    ax2.plot(epochs_range, val_loss, label="Validation Loss", color="#F59E0B", lw=2, marker="s", markersize=4)
    ax2.set_title("Training & Validation Loss", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved training history curves to: {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix"
) -> None:
    """
    Plots a confusion matrix heatmap with cell counts and saves to disk.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot to: {save_path}")


def download_official_glove(dimension: int = 50, target_dir: Optional[Path] = None) -> Path:
    """
    Downloads and extracts Stanford's official GloVe 6B word embeddings archive
    (glove.6B.zip, ~822MB compressed, ~2.2GB uncompressed).
    """
    if target_dir is None:
        target_dir = Path("data/embeddings")
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    extracted_file = target_dir / f"glove.6B.{dimension}d.txt"
    if extracted_file.exists():
        print(f"GloVe file already exists at: {extracted_file}")
        return extracted_file

    zip_path = target_dir / "glove.6B.zip"
    url = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"

    print(f"Downloading Stanford GloVe 6B embeddings from {url}...")
    print(f"Destination archive: {zip_path}")
    print("Note: The archive is ~822MB. Download may take several minutes.")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\rProgress: {percent:.1f}% ({mb:.1f}MB / {total_mb:.1f}MB)", end="")
        else:
            print(f"\rDownloaded {downloaded / (1024 * 1024):.1f}MB", end="")

    urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
    print("\nDownload complete. Extracting selected dimension file...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        target_member = f"glove.6B.{dimension}d.txt"
        zip_ref.extract(target_member, target_dir)

    print(f"Extracted: {extracted_file}")
    if zip_path.exists():
        try:
            zip_path.unlink()
            print("Removed temporary zip archive.")
        except Exception:
            pass

    return extracted_file
