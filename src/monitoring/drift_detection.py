"""Reusable data-drift detection for transaction monitoring."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.feature_engineering import apply_feature_engineering
from src.monitoring.reference_profile import load_reference_profile

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Class"

DEFAULT_NUMERIC_DRIFT_THRESHOLD = 0.2
DEFAULT_MISSING_VALUE_DRIFT_THRESHOLD = 0.01


class DriftDetectionError(Exception):
    """Raised when drift detection cannot be performed."""


def validate_drift_inputs(
    reference_profile: dict[str, Any],
    incoming_features: pd.DataFrame,
) -> None:
    """Validate reference-profile and incoming-data inputs."""
    if not isinstance(reference_profile, dict):
        raise DriftDetectionError(
            "Reference profile must be a dictionary."
        )

    if not isinstance(incoming_features, pd.DataFrame):
        raise DriftDetectionError(
            "Incoming features must be a pandas DataFrame."
        )

    if incoming_features.empty:
        raise DriftDetectionError(
            "Incoming feature data is empty."
        )

    if TARGET_COLUMN in incoming_features.columns:
        raise DriftDetectionError(
            "Target leakage detected: target column must not be included "
            "in incoming monitoring features."
        )

    expected_columns = reference_profile.get("feature_columns")

    if not isinstance(expected_columns, list):
        raise DriftDetectionError(
            "Reference profile does not contain a valid feature-columns list."
        )

    missing_columns = set(expected_columns) - set(
        incoming_features.columns
    )

    if missing_columns:
        raise DriftDetectionError(
            f"Incoming data is missing required features: "
            f"{sorted(missing_columns)}"
        )

    logger.info(
        "Drift-detection inputs validated for %s incoming rows.",
        len(incoming_features),
    )


def calculate_population_stability_index(
    reference_values: np.ndarray,
    incoming_values: np.ndarray,
    number_of_bins: int = 10,
) -> float:
    """Calculate Population Stability Index between two numeric distributions."""
    if len(reference_values) == 0 or len(incoming_values) == 0:
        raise DriftDetectionError(
            "Population Stability Index requires non-empty reference and "
            "incoming values."
        )

    combined_minimum = min(
        np.min(reference_values),
        np.min(incoming_values),
    )

    combined_maximum = max(
        np.max(reference_values),
        np.max(incoming_values),
    )

    if combined_minimum == combined_maximum:
        return 0.0

    bin_edges = np.linspace(
        combined_minimum,
        combined_maximum,
        number_of_bins + 1,
    )

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference_values,
        bins=bin_edges,
    )

    incoming_counts, _ = np.histogram(
        incoming_values,
        bins=bin_edges,
    )

    reference_proportions = reference_counts / len(reference_values)
    incoming_proportions = incoming_counts / len(incoming_values)

    epsilon = 1e-6

    reference_proportions = np.clip(
        reference_proportions,
        epsilon,
        None,
    )

    incoming_proportions = np.clip(
        incoming_proportions,
        epsilon,
        None,
    )

    psi = np.sum(
        (incoming_proportions - reference_proportions)
        * np.log(
            incoming_proportions / reference_proportions,
        )
    )

    return float(psi)


def detect_numeric_drift(
    reference_profile: dict[str, Any],
    incoming_features: pd.DataFrame,
    numeric_drift_threshold: float,
) -> list[dict[str, Any]]:
    """Detect drift in numeric features using PSI."""
    numeric_profiles = reference_profile.get(
        "numeric_features",
        {},
    )

    drift_results: list[dict[str, Any]] = []

    for feature_name, profile in numeric_profiles.items():
        if feature_name not in incoming_features.columns:
            continue

        incoming_values = incoming_features[feature_name].dropna().to_numpy(
            dtype=float,
        )

        reference_mean = profile["mean"]
        reference_standard_deviation = profile["standard_deviation"]

        if reference_standard_deviation == 0:
            reference_values = np.array(
                [reference_mean],
                dtype=float,
            )
        else:
            reference_values = np.random.default_rng(42).normal(
                loc=reference_mean,
                scale=reference_standard_deviation,
                size=10000,
            )

        psi = calculate_population_stability_index(
            reference_values=reference_values,
            incoming_values=incoming_values,
        )

        drift_detected = psi >= numeric_drift_threshold

        drift_results.append(
            {
                "feature_name": feature_name,
                "drift_metric": "population_stability_index",
                "reference_mean": reference_mean,
                "incoming_mean": float(
                    incoming_features[feature_name].mean(),
                ),
                "psi": psi,
                "drift_threshold": numeric_drift_threshold,
                "drift_detected": bool(drift_detected),
            }
        )

    return drift_results


def detect_missing_value_drift(
    reference_profile: dict[str, Any],
    incoming_features: pd.DataFrame,
    missing_value_drift_threshold: float,
) -> list[dict[str, Any]]:
    """Detect changes in missing-value rates."""
    numeric_profiles = reference_profile.get(
        "numeric_features",
        {},
    )

    drift_results: list[dict[str, Any]] = []

    for feature_name, profile in numeric_profiles.items():
        if feature_name not in incoming_features.columns:
            continue

        reference_missing_count = profile.get(
            "missing_count",
            0,
        )

        reference_missing_rate = (
            reference_missing_count
            / reference_profile["reference_row_count"]
        )

        incoming_missing_rate = float(
            incoming_features[feature_name].isnull().mean(),
        )

        missing_value_change = abs(
            incoming_missing_rate - reference_missing_rate,
        )

        drift_detected = (
            missing_value_change >= missing_value_drift_threshold
        )

        drift_results.append(
            {
                "feature_name": feature_name,
                "drift_metric": "missing_value_rate_change",
                "reference_missing_rate": float(reference_missing_rate),
                "incoming_missing_rate": incoming_missing_rate,
                "absolute_change": float(missing_value_change),
                "drift_threshold": missing_value_drift_threshold,
                "drift_detected": bool(drift_detected),
            }
        )

    return drift_results


def detect_data_drift(
    incoming_features: pd.DataFrame,
    reference_profile_path: Path,
    numeric_drift_threshold: float = DEFAULT_NUMERIC_DRIFT_THRESHOLD,
    missing_value_drift_threshold: float = (
        DEFAULT_MISSING_VALUE_DRIFT_THRESHOLD
    ),
) -> dict[str, Any]:
    """Compare incoming transaction data with the saved reference profile."""
    reference_profile = load_reference_profile(
        profile_path=reference_profile_path,
    )

    validate_drift_inputs(
        reference_profile=reference_profile,
        incoming_features=incoming_features,
    )

    engineered_incoming_features = apply_feature_engineering(
        features=incoming_features,
    )

    numeric_drift_results = detect_numeric_drift(
        reference_profile=reference_profile,
        incoming_features=engineered_incoming_features,
        numeric_drift_threshold=numeric_drift_threshold,
    )

    missing_value_drift_results = detect_missing_value_drift(
        reference_profile=reference_profile,
        incoming_features=engineered_incoming_features,
        missing_value_drift_threshold=missing_value_drift_threshold,
    )

    all_drift_results = numeric_drift_results + missing_value_drift_results

    drifted_features = [
        result["feature_name"]
        for result in all_drift_results
        if result["drift_detected"]
    ]

    drift_report = {
        "report_timestamp_utc": datetime.now(
            timezone.utc,
        ).isoformat(),
        "reference_profile_version": reference_profile.get(
            "profile_version",
        ),
        "incoming_row_count": len(incoming_features),
        "numeric_drift_threshold": numeric_drift_threshold,
        "missing_value_drift_threshold": missing_value_drift_threshold,
        "drifted_feature_count": len(set(drifted_features)),
        "drifted_features": sorted(set(drifted_features)),
        "numeric_drift_results": numeric_drift_results,
        "missing_value_drift_results": missing_value_drift_results,
        "drift_detected": bool(drifted_features),
        "interpretation": (
            "Drift indicates a potential change in input-data distribution. "
            "It does not prove that model performance has deteriorated or "
            "that fraud has increased."
        ),
    }

    logger.info(
        "Data-drift detection completed. Drift detected: %s",
        drift_report["drift_detected"],
    )

    return drift_report


def save_drift_report(
    drift_report: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a data-drift report to JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        import json

        json.dump(
            drift_report,
            file,
            indent=2,
            default=str,
        )

    logger.info(
        "Drift report saved to: %s",
        output_path,
    )