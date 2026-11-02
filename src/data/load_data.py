"""Reusable data-loading and splitting utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Class"
RANDOM_SEED = 42

DEFAULT_SPLIT_CONFIGURATION = {
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "test_fraction": 0.15,
}


def load_raw_data(
    data_path: Path,
) -> pd.DataFrame:
    """Load the raw credit-card transaction dataset."""
    if not data_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {data_path}")

    if data_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV dataset, received: {data_path.suffix}"
        )

    dataframe = pd.read_csv(data_path)

    if dataframe.empty:
        raise ValueError("Raw dataset is empty.")

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' is missing."
        )

    invalid_labels = set(
        dataframe[TARGET_COLUMN].unique(),
    ) - {0, 1}

    if invalid_labels:
        raise ValueError(
            f"Target column contains unexpected labels: {sorted(invalid_labels)}"
        )

    logger.info(
        "Loaded raw dataset with shape: %s",
        dataframe.shape,
    )

    return dataframe


def separate_features_and_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate input features from the fraud target column."""
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' is missing."
        )

    features = dataframe.drop(
        columns=[TARGET_COLUMN],
    )

    target = dataframe[TARGET_COLUMN]

    if features.empty:
        raise ValueError("Feature matrix is empty.")

    logger.info(
        "Separated %s features and %s target values.",
        features.shape[1],
        len(target),
    )

    return features, target


def create_train_validation_test_splits(
    features: pd.DataFrame,
    target: pd.Series,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_seed: int = RANDOM_SEED,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """Create reproducible stratified train, validation, and test splits."""
    split_total = (
        train_fraction
        + validation_fraction
        + test_fraction
    )

    if not abs(split_total - 1.0) < 1e-9:
        raise ValueError(
            "Train, validation, and test fractions must sum to 1.0."
        )

    if train_fraction <= 0 or validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError(
            "All split fractions must be positive."
        )

    X_train, X_temporary, y_train, y_temporary = train_test_split(
        features,
        target,
        test_size=(validation_fraction + test_fraction),
        stratify=target,
        random_state=random_seed,
    )

    validation_share_of_temporary = (
        validation_fraction
        / (validation_fraction + test_fraction)
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temporary,
        y_temporary,
        test_size=(1 - validation_share_of_temporary),
        stratify=y_temporary,
        random_state=random_seed,
    )

    logger.info(
        "Created splits: train=%s, validation=%s, test=%s",
        X_train.shape,
        X_validation.shape,
        X_test.shape,
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


def save_split_data(
    output_directory: Path,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    split_configuration: dict | None = None,
) -> None:
    """Save feature and target Parquet files for each split."""
    if split_configuration is None:
        split_configuration = DEFAULT_SPLIT_CONFIGURATION

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_paths = {
        "train": output_directory / "train",
        "validation": output_directory / "validation",
        "test": output_directory / "test",
    }

    for split_directory in split_paths.values():
        split_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    X_train.to_parquet(
        split_paths["train"] / "X_train.parquet",
        index=False,
    )
    y_train.to_frame().to_parquet(
        split_paths["train"] / "y_train.parquet",
        index=False,
    )

    X_validation.to_parquet(
        split_paths["validation"] / "X_validation.parquet",
        index=False,
    )
    y_validation.to_frame().to_parquet(
        split_paths["validation"] / "y_validation.parquet",
        index=False,
    )