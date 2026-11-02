"""Reusable data-cleaning utilities for the fraud-detection dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.validate_data import (
    DataValidationError,
    validate_raw_dataset,
)

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Class"


def count_missing_values(
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    """Return the number of missing values for each column."""
    missing_counts = dataframe.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]

    return missing_counts.astype(int).to_dict()


def count_duplicate_rows(
    dataframe: pd.DataFrame,
) -> int:
    """Return the number of fully duplicated rows."""
    return int(dataframe.duplicated().sum())


def remove_duplicate_rows(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Remove fully duplicated rows and return the cleaned DataFrame."""
    duplicate_count = count_duplicate_rows(
        dataframe=dataframe,
    )

    if duplicate_count == 0:
        logger.info(
            "No duplicate rows found; cleaning not required.",
        )

        return dataframe.copy(), 0

    cleaned_dataframe = dataframe.drop_duplicates().copy()

    logger.info(
        "Removed %s duplicate rows.",
        duplicate_count,
    )

    return cleaned_dataframe, duplicate_count


def validate_no_negative_amounts(
    dataframe: pd.DataFrame,
) -> None:
    """Ensure transaction amounts are non-negative."""
    if "Amount" not in dataframe.columns:
        raise DataValidationError(
            "Required column 'Amount' is missing."
        )

    if (dataframe[TARGET_COLUMN] == 1).any() is False and dataframe.empty:
        raise DataValidationError(
            "Cannot validate amounts because the dataset is empty."
        )

    if (dataframe["Amount"] < 0).any():
        raise DataValidationError(
            "Dataset contains negative Amount values."
        )

    logger.info(
        "Transaction amounts validated successfully.",
    )


def clean_raw_dataset(
    dataframe: pd.DataFrame,
    remove_duplicates: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Clean the raw dataset using an explicit, documented policy."""
    validation_summary = validate_raw_dataset(
        dataframe=dataframe,
    )

    cleaned_dataframe = dataframe.copy()

    duplicate_count = count_duplicate_rows(
        dataframe=cleaned_dataframe,
    )

    removed_duplicate_count = 0

    if remove_duplicates:
        cleaned_dataframe, removed_duplicate_count = remove_duplicate_rows(
            dataframe=cleaned_dataframe,
        )
    else:
        logger.info(
            "Duplicate-row removal is disabled; %s duplicate rows retained.",
            duplicate_count,
        )

    validate_no_negative_amounts(
        dataframe=cleaned_dataframe,
    )

    cleaning_summary = {
        "input_row_count": validation_summary["row_count"],
        "output_row_count": len(cleaned_dataframe),
        "duplicate_rows_found": duplicate_count,
        "duplicate_rows_removed": removed_duplicate_count,
        "duplicate_removal_enabled": remove_duplicates,
        "missing_values": validation_summary["missing_values"],
        "fraud_count": int(cleaned_dataframe[TARGET_COLUMN].sum()),
        "legitimate_count": int(
            (cleaned_dataframe[TARGET_COLUMN] == 0).sum(),
        ),
    }

    logger.info(
        "Raw dataset cleaning completed: %s",
        cleaning_summary,
    )

    return cleaned_dataframe, cleaning_summary


def save_cleaned_dataset(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save the cleaned dataset to Parquet format."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_parquet(
        output_path,
        index=False,
    )

    logger.info(
        "Cleaned dataset saved to: %s",
        output_path,
    )