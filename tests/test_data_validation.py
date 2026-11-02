"""Tests for prediction-data validation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.validate_data import (
    DataValidationError,
    validate_prediction_dataset,
)

REQUIRED_COLUMNS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]


@pytest.fixture
def valid_prediction_dataframe() -> pd.DataFrame:
    """Create a small valid prediction dataset."""
    row_count = 5

    dataframe_data = {
        "Time": np.linspace(0, 100, row_count),
        "Amount": np.linspace(10, 100, row_count),
    }

    for feature_number in range(1, 29):
        dataframe_data[f"V{feature_number}"] = np.random.default_rng(
            feature_number
        ).normal(
            loc=0,
            scale=1,
            size=row_count,
        )

    return pd.DataFrame(dataframe_data)


def test_valid_prediction_dataset_passes(
    valid_prediction_dataframe: pd.DataFrame,
) -> None:
    """Verify that a valid prediction dataset passes validation."""
    validate_prediction_dataset(
        dataframe=valid_prediction_dataframe,
    )


def test_missing_required_column_raises_error(
    valid_prediction_dataframe: pd.DataFrame,
) -> None:
    """Verify that a missing required column raises an error."""
    invalid_dataframe = valid_prediction_dataframe.drop(
        columns=["Amount"],
    )

    with pytest.raises(DataValidationError):
        validate_prediction_dataset(
            dataframe=invalid_dataframe,
        )


def test_non_numeric_feature_raises_error(
    valid_prediction_dataframe: pd.DataFrame,
) -> None:
    """Verify that a non-numeric feature column raises an error."""
    invalid_dataframe = valid_prediction_dataframe.copy()
    invalid_dataframe["V1"] = "not_numeric"

    with pytest.raises(DataValidationError):
        validate_prediction_dataset(
            dataframe=invalid_dataframe,
        )


def test_missing_values_raise_error(
    valid_prediction_dataframe: pd.DataFrame,
) -> None:
    """Verify that missing values raise an error."""
    invalid_dataframe = valid_prediction_dataframe.copy()
    invalid_dataframe.loc[0, "Amount"] = np.nan

    with pytest.raises(DataValidationError):
        validate_prediction_dataset(
            dataframe=invalid_dataframe,
        )


def test_infinite_values_raise_error(
    valid_prediction_dataframe: pd.DataFrame,
) -> None:
    """Verify that infinite values raise an error."""
    invalid_dataframe = valid_prediction_dataframe.copy()
    invalid_dataframe.loc[0, "V2"] = np.inf

    with pytest.raises(DataValidationError):
        validate_prediction_dataset(
            dataframe=invalid_dataframe,
        )


def test_negative_amount_raises_error(
    valid_prediction_dataframe: pd.DataFrame,
) -> None:
    """Verify that a negative transaction amount raises an error."""
    invalid_dataframe = valid_prediction_dataframe.copy()
    invalid_dataframe.loc[0, "Amount"] = -1

    with pytest.raises(DataValidationError):
        validate_prediction_dataset(
            dataframe=invalid_dataframe,
        )


def test_empty_dataset_raises_error() -> None:
    """Verify that an empty dataset raises an error."""
    empty_dataframe = pd.DataFrame(
        columns=REQUIRED_COLUMNS,
    )

    with pytest.raises(DataValidationError):
        validate_prediction_dataset(
            dataframe=empty_dataframe,
        )