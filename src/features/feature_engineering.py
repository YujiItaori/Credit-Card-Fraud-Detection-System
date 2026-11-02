"""Reusable feature-engineering utilities for transaction data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Class"

FEATURE_CONFIGURATION_NAME = "log_amount_and_time_hours"


class FeatureEngineeringError(Exception):
    """Raised when feature engineering cannot be applied safely."""


def validate_feature_engineering_input(
    features: pd.DataFrame,
) -> None:
    """Validate that the input contains the required original columns."""
    if not isinstance(features, pd.DataFrame):
        raise FeatureEngineeringError(
            "Feature-engineering input must be a pandas DataFrame."
        )

    if features.empty:
        raise FeatureEngineeringError(
            "Feature-engineering input is empty."
        )

    required_columns = {
        "Time",
        "Amount",
    }

    missing_columns = required_columns - set(features.columns)

    if missing_columns:
        raise FeatureEngineeringError(
            f"Feature-engineering input is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if not pd.api.types.is_numeric_dtype(features["Time"]):
        raise FeatureEngineeringError(
            "Column 'Time' must be numeric."
        )

    if not pd.api.types.is_numeric_dtype(features["Amount"]):
        raise FeatureEngineeringError(
            "Column 'Amount' must be numeric."
        )

    if (features["Amount"] < 0).any():
        raise FeatureEngineeringError(
            "Column 'Amount' contains negative values; log1p cannot be applied."
        )

    if np.isinf(
        features[["Time", "Amount"]].to_numpy(dtype=float),
    ).any():
        raise FeatureEngineeringError(
            "Feature-engineering input contains infinite values."
        )

    logger.info(
        "Feature-engineering input validated for %s rows.",
        len(features),
    )


def add_log_amount_feature(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Add a zero-safe logarithmic transformation of transaction amount."""
    transformed_features = features.copy()

    transformed_features["Amount_log1p"] = np.log1p(
        transformed_features["Amount"],
    )

    logger.info(
        "Added feature: Amount_log1p",
    )

    return transformed_features


def add_elapsed_time_hours_feature(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Add elapsed transaction time expressed in hours."""
    transformed_features = features.copy()

    transformed_features["Time_hours"] = (
        transformed_features["Time"] / 3600.0
    )

    logger.info(
        "Added feature: Time_hours",
    )

    return transformed_features


def apply_feature_engineering(
    features: pd.DataFrame,
    configuration_name: str = FEATURE_CONFIGURATION_NAME,
) -> pd.DataFrame:
    """Apply the selected feature-engineering configuration."""
    if configuration_name != FEATURE_CONFIGURATION_NAME:
        raise FeatureEngineeringError(
            f"Unsupported feature configuration: {configuration_name}"
        )

    validate_feature_engineering_input(
        features=features,
    )

    transformed_features = add_log_amount_feature(
        features=features,
    )

    transformed_features = add_elapsed_time_hours_feature(
        features=transformed_features,
    )

    if TARGET_COLUMN in transformed_features.columns:
        raise FeatureEngineeringError(
            "Target leakage detected: target column must not be included "
            "in feature-engineering input."
        )

    expected_engineered_columns = {
        "Time",
        "Amount",
        "Amount_log1p",
        "Time_hours",
    }

    missing_engineered_columns = (
        expected_engineered_columns
        - set(transformed_features.columns)
    )

    if missing_engineered_columns:
        raise FeatureEngineeringError(
            f"Feature engineering produced missing columns: "
            f"{sorted(missing_engineered_columns)}"
        )

    logger.info(
        "Feature engineering completed using configuration: %s",
        configuration_name,
    )

    return transformed_features


def get_feature_configuration_name() -> str:
    """Return the project's selected feature-engineering configuration name."""
    return FEATURE_CONFIGURATION_NAME