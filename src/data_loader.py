"""
Data Loader Module for News Bias Detection.

Handles CSV dataset ingestion, null/missing value cleaning, deduplication,
class distribution reporting, and stratified train/validation/test splitting.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DEFAULT_DATASET_PATH,
    LABEL_COLUMN,
    RANDOM_SEED,
    TEST_SPLIT,
    TEXT_COLUMN,
    VAL_SPLIT,
)
from src.utils import print_section


def load_raw_data(filepath: Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """
    Reads a CSV dataset from disk. Raises FileNotFoundError with helpful guidance
    if the file cannot be located.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. Please check the path or ensure "
            f"data/news_bias_dataset.csv is present."
        )
    df = pd.read_csv(filepath)
    return df


def clean_and_inspect_data(
    df: pd.DataFrame,
    text_col: str = TEXT_COLUMN,
    label_col: str = LABEL_COLUMN
) -> pd.DataFrame:
    """
    Cleans raw DataFrame by removing nulls, empty strings, and duplicates.
    Displays class distribution metrics in the terminal.
    """
    print_section("Data Inspection & Cleaning")
    initial_count = len(df)
    print(f"Total raw records loaded: {initial_count}")

    # Validate required columns
    for col in [text_col, label_col]:
        if col not in df.columns:
            raise KeyError(
                f"Required column '{col}' missing from dataset. Available columns: {list(df.columns)}"
            )

    # Missing value handling
    missing_text = df[text_col].isna().sum()
    missing_label = df[label_col].isna().sum()
    if missing_text > 0 or missing_label > 0:
        print(f"Dropping missing values: {missing_text} missing '{text_col}', {missing_label} missing '{label_col}'")
        df = df.dropna(subset=[text_col, label_col])

    # Remove blank / whitespace-only entries
    df = df[df[text_col].astype(str).str.strip().str.len() > 0]

    # Normalize labels to lower-case stripped strings
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    # Deduplication
    duplicate_count = df.duplicated(subset=[text_col]).sum()
    if duplicate_count > 0:
        print(f"Removing {duplicate_count} duplicate article text entries.")
        df = df.drop_duplicates(subset=[text_col])

    clean_count = len(df)
    print(f"Cleaned dataset count: {clean_count} records (dropped {initial_count - clean_count} records)")

    # Class distribution analysis
    distribution = df[label_col].value_counts()
    percentages = df[label_col].value_counts(normalize=True) * 100

    print("\nClass Distribution Analysis:")
    print(f"{'Class':<15} {'Count':<10} {'Percentage':<12}")
    print("-" * 37)
    for stance, count in distribution.items():
        pct = percentages[stance]
        print(f"{stance:<15} {count:<10} {pct:>6.2f}%")
    print("-" * 37)

    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    text_col: str = TEXT_COLUMN,
    label_col: str = LABEL_COLUMN,
    test_size: float = TEST_SPLIT,
    val_size: float = VAL_SPLIT,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a stratified two-stage split into Train, Validation, and Test sets.
    Preserves class balance across all subsets.
    """
    print_section("Dataset Splitting")

    # Stage 1: Separate Test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=seed
    )

    # Stage 2: Separate Validation from remaining Train
    # Adjust validation fraction relative to train_val subset
    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=train_val_df[label_col],
        random_state=seed
    )

    print(f"Train subset:      {len(train_df):>5} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation subset: {len(val_df):>5} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test subset:       {len(test_df):>5} samples ({len(test_df)/len(df)*100:.1f}%)")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )
