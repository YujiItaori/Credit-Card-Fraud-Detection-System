"""Reusable schema and data-quality validation utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Class"

EXPECTED_RAW_COLUMNS = [
    "Time",
    *[f"V{number}" for number in range(1, 29)],
    "Amount",
    TARGET_COLUMN,
]

EXPECTED_PREDICTION_COLUMNS = [
    "Time",
    *[f"V{number}" for number in range(1, 29)],
    "Amount",
]

VALID_TARGET_LABELS = {0, 1}


class DataValidationError(Exception):
    """Raised when input data fails schema or quality validation."""


def validate_dataframe_not_empty(
    dataframe: pd.DataFrame,
    dataset_name: str = "Input dataset",
) -> None:
    """Validate that a DataFrame is non-empty."""
    if not isinstance(dataframe, pd.DataFrame):
        raise DataValidationError(
            f"{dataset_name} must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise DataValidationError(
            f"{dataset_name} is empty."
        )


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str = "Input dataset",
) -> None:
    """Validate that all required columns are present."""
    missing_columns = set(required_columns) - set(dataframe.columns)

    if missing_columns:
        raise DataValidationError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    logger.info(
        "Required columns validated for %s.",
        dataset_name,
    )


def validate_target_column(
    dataframe: pd.DataFrame,
) -> None:
    """Validate that the target column exists and contains valid labels."""
    if TARGET_COLUMN not in dataframe.columns:
        raise DataValidationError(
            f"Required target column '{TARGET_COLUMN}' is missing."
        )

    target_values = set(
        dataframe[TARGET_COLUMN].unique(),
    )

    invalid_labels = target_values - VALID_TARGET_LABELS

    if invalid_labels:
        raise DataValidationError(
            f"Target column contains unexpected labels: "
            f"{sorted(invalid_labels)}"
        )

    logger.info(
        "Target column validated with class counts: %s",
        dataframe[TARGET_COLUMN].value_counts().sort_index().to_dict(),
    )


def validate_numeric_columns(
    dataframe: pd.DataFrame,
    dataset_name: str = "Input dataset",
) -> None:
    """Validate that all expected feature columns are numeric."""
    feature_columns = [
        column
        for column in dataframe.columns
        if column != TARGET_COLUMN
    ]

    non_numeric_columns = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    if non_numeric_columns:
        raise DataValidationError(
            f"{dataset_name} contains non-numeric feature columns: "
            f"{non_numeric_columns}"
        )

    logger.info(
        "Numeric feature columns validated for %s.",
        dataset_name,
    )


def validate_missing_values(
    dataframe: pd.DataFrame,
    dataset_name: str = "Input dataset",
) -> dict[str, int]:
    """Validate missing values and return a missing-value summary."""
    missing_counts = dataframe.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if not missing_counts.empty:
        raise DataValidationError(
            f"{dataset_name} contains missing values: "
            f"{missing_counts.to_dict()}"
        )

    logger.info(
        "No missing values found in %s.",
        dataset_name,
    )

    return {}


def validate_non_finite_values(
    dataframe: pd.DataFrame,
    dataset_name: str = "Input dataset",
) -> None:
    """Validate that numeric features do not contain infinite values."""
    numeric_dataframe = dataframe.select_dtypes(
        include=[np.number],
    )

    if numeric_dataframe.empty:
        return

    non_finite_mask = np.isinf(
        numeric_dataframe.to_numpy(dtype=float),
    )

    if non_finite_mask.any():
        affected_columns = numeric_dataframe.columns[
            non_finite_mask.any(axis=0)
        ].tolist()

        raise DataValidationError(
            f"{dataset_name} contains infinite values in columns: "
            f"{affected_columns}"
        )

    logger.info(
        "No infinite values found in %s.",
        dataset_name,
    )


def validate_negative_amounts(
    dataframe: pd.DataFrame,
    dataset_name: str = "Input dataset",
) -> None:
    """Validate that transaction amounts are non-negative."""
    if "Amount" not in dataframe.columns:
        raise DataValidationError(
            "Required column 'Amount' is missing."
        )

    if (dataframe["Amount"] < 0).any():
        raise DataValidationError(
            f"{dataset_name} contains negative Amount values."
        )

    logger.info(
        "Transaction amounts validated for %s.",
        dataset_name,
    )


def validate_raw_dataset(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Validate the complete raw dataset schema and quality."""
    validate_dataframe_not_empty(
        dataframe=dataframe,
        dataset_name="Raw dataset",
    )

    validate_required_columns(
        dataframe=dataframe,
        required_columns=EXPECTED_RAW_COLUMNS,
        dataset_name="Raw dataset",
    )

    validate_target_column(
        dataframe=dataframe,
    )

    validate_numeric_columns(
        dataframe=dataframe,
        dataset_name="Raw dataset",
    )

    missing_summary = validate_missing_values(
        dataframe=dataframe,
        dataset_name="Raw dataset",
    )

    validate_non_finite_values(
        dataframe=dataframe,
        dataset_name="Raw dataset",
    )

    validate_negative_amounts(
        dataframe=dataframe,
        dataset_name="Raw dataset",
    )

    duplicate_count = int(
        dataframe.duplicated().sum(),
    )

    validation_summary = {
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "missing_values": missing_summary,
        "duplicate_rows": duplicate_count,
        "fraud_count": int(dataframe[TARGET_COLUMN].sum()),
        "legitimate_count": int(
            (dataframe[TARGET_COLUMN] == 0).sum(),
        ),
    }

    logger.info(
        "Raw dataset validation completed: %s",
        validation_summary,
    )

    return validation_summary


def validate_prediction_dataset(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Validate a prediction input dataset before model inference."""
    validate_dataframe_not_empty(
        dataframe=dataframe,
        dataset_name="Prediction dataset",
    )

    validate_required_columns(
        dataframe=dataframe,
        required_columns=EXPECTED_PREDICTION_COLUMNS,
        dataset_name="Prediction dataset",
    )

    validate_numeric_columns(
        dataframe=dataframe,
        dataset_name="Prediction dataset",
    )

    validate_missing_values(
        dataframe=dataframe,
        dataset_name="Prediction dataset",
    )

    validate_non_finite_values(
        dataframe=dataframe,
        dataset_name="Prediction dataset",
    )

    validate_negative_amounts(
        dataframe=dataframe,
        dataset_name="Prediction dataset",
    )

    validation_summary = {
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
    }

    logger.info(
        "Prediction dataset validation completed: %s",
        validation_summary,
    )

    return validation_summary


def load_and_validate_raw_dataset(
    data_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a raw CSV dataset and validate its schema and quality."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {data_path}"
        )

    dataframe = pd.read_csv(data_path)

    validation_summary = validate_raw_dataset(
        dataframe=dataframe,
    )

    return dataframe, validation_summary