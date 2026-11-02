"""Reusable training-data reference profiling for monitoring."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.feature_engineering import apply_feature_engineering

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Class"

REFERENCE_PROFILE_VERSION = "1.0"


class ReferenceProfileError(Exception):
    """Raised when a reference profile cannot be created or loaded."""


def calculate_numeric_feature_profile(
    features: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate reference statistics for numeric features."""
    numeric_features = features.select_dtypes(
        include=[np.number],
    )

    if numeric_features.empty:
        raise ReferenceProfileError(
            "Reference data does not contain numeric features."
        )

    feature_profiles: dict[str, Any] = {}

    for column in numeric_features.columns:
        column_values = numeric_features[column]

        feature_profiles[column] = {
            "count": int(column_values.count()),
            "missing_count": int(column_values.isnull().sum()),
            "mean": float(column_values.mean()),
            "standard_deviation": float(column_values.std()),
            "minimum": float(column_values.min()),
            "maximum": float(column_values.max()),
            "median": float(column_values.median()),
            "percentile_25": float(column_values.quantile(0.25)),
            "percentile_75": float(column_values.quantile(0.75)),
        }

    return feature_profiles


def calculate_categorical_feature_profile(
    features: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate reference value counts for categorical features."""
    categorical_features = features.select_dtypes(
        include=["object", "category"],
    )

    feature_profiles: dict[str, Any] = {}

    for column in categorical_features.columns:
        value_counts = features[column].value_counts(dropna=False)

        feature_profiles[column] = {
            "missing_count": int(features[column].isnull().sum()),
            "unique_count": int(features[column].nunique(dropna=False)),
            "value_counts": {
                str(key): int(value)
                for key, value in value_counts.items()
            },
        }

    return feature_profiles


def create_reference_profile(
    training_features: pd.DataFrame,
    training_target: pd.Series | None = None,
) -> dict[str, Any]:
    """Create a reference profile from training features."""
    if not isinstance(training_features, pd.DataFrame):
        raise ReferenceProfileError(
            "Training features must be a pandas DataFrame."
        )

    if training_features.empty:
        raise ReferenceProfileError(
            "Training features are empty."
        )

    if TARGET_COLUMN in training_features.columns:
        raise ReferenceProfileError(
            "Target leakage detected: target column must not be included "
            "in reference-profile features."
        )

    engineered_features = apply_feature_engineering(
        features=training_features,
    )

    numeric_profile = calculate_numeric_feature_profile(
        features=engineered_features,
    )

    categorical_profile = calculate_categorical_feature_profile(
        features=engineered_features,
    )

    reference_profile = {
        "profile_version": REFERENCE_PROFILE_VERSION,
        "feature_configuration": "log_amount_and_time_hours",
        "reference_row_count": len(engineered_features),
        "reference_column_count": len(engineered_features.columns),
        "feature_columns": list(engineered_features.columns),
        "numeric_features": numeric_profile,
        "categorical_features": categorical_profile,
        "target_column": TARGET_COLUMN,
        "target_used_for_profile": False,
    }

    if training_target is not None:
        reference_profile["training_fraud_count"] = int(
            training_target.sum(),
        )
        reference_profile["training_legitimate_count"] = int(
            (training_target == 0).sum(),
        )

    logger.info(
        "Reference profile created for %s rows and %s features.",
        len(engineered_features),
        len(engineered_features.columns),
    )

    return reference_profile


def save_reference_profile(
    reference_profile: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a reference profile to JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            reference_profile,
            file,
            indent=2,
            default=str,
        )

    logger.info(
        "Reference profile saved to: %s",
        output_path,
    )


def load_reference_profile(
    profile_path: Path,
) -> dict[str, Any]:
    """Load and validate a saved reference profile."""
    if not profile_path.exists():
        raise ReferenceProfileError(
            f"Reference profile not found: {profile_path}"
        )

    try:
        with profile_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            reference_profile = json.load(file)
    except json.JSONDecodeError as error:
        raise ReferenceProfileError(
            f"Reference profile is not valid JSON: {profile_path}"
        ) from error

    required_fields = {
        "profile_version",
        "feature_configuration",
        "reference_row_count",
        "feature_columns",
        "numeric_features",
    }

    missing_fields = required_fields - set(reference_profile.keys())

    if missing_fields:
        raise ReferenceProfileError(
            f"Reference profile is missing fields: {sorted(missing_fields)}"
        )

    logger.info(
        "Reference profile loaded from: %s",
        profile_path,
    )

    return reference_profile