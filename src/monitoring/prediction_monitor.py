"""Reusable prediction-behavior monitoring utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Class"


class PredictionMonitoringError(Exception):
    """Raised when prediction monitoring cannot be performed."""


def validate_prediction_monitoring_input(
    prediction_results: pd.DataFrame,
) -> None:
    """Validate that prediction results contain the required monitoring columns."""
    if not isinstance(prediction_results, pd.DataFrame):
        raise PredictionMonitoringError(
            "Prediction results must be a pandas DataFrame."
        )

    if prediction_results.empty:
        raise PredictionMonitoringError(
            "Prediction results are empty."
        )

    required_columns = {
        "fraud_score",
        "fraud_prediction",
        "review_flag",
    }

    missing_columns = required_columns - set(
        prediction_results.columns,
    )

    if missing_columns:
        raise PredictionMonitoringError(
            f"Prediction results are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    logger.info(
        "Prediction-monitoring input validated for %s transactions.",
        len(prediction_results),
    )


def calculate_prediction_distribution(
    prediction_scores: pd.Series,
) -> dict[str, float]:
    """Calculate summary statistics for supervised fraud scores."""
    if prediction_scores.empty:
        raise PredictionMonitoringError(
            "Prediction scores are empty."
        )

    return {
        "minimum_score": float(prediction_scores.min()),
        "maximum_score": float(prediction_scores.max()),
        "mean_score": float(prediction_scores.mean()),
        "median_score": float(prediction_scores.median()),
        "standard_deviation_score": float(prediction_scores.std()),
        "percentile_25": float(prediction_scores.quantile(0.25)),
        "percentile_75": float(prediction_scores.quantile(0.75)),
        "percentile_90": float(prediction_scores.quantile(0.90)),
        "percentile_95": float(prediction_scores.quantile(0.95)),
        "percentile_99": float(prediction_scores.quantile(0.99)),
    }


def monitor_supervised_predictions(
    prediction_results: pd.DataFrame,
    model_version: str,
    decision_threshold: float,
) -> dict[str, Any]:
    """Monitor supervised prediction behavior for one transaction batch."""
    validate_prediction_monitoring_input(
        prediction_results=prediction_results,
    )

    prediction_scores = prediction_results["fraud_score"]
    review_flags = prediction_results["review_flag"]

    flagged_count = int(review_flags.sum())
    flagged_percentage = float(
        review_flags.mean(),
    )

    prediction_distribution = calculate_prediction_distribution(
        prediction_scores=prediction_scores,
    )

    monitoring_report = {
        "report_timestamp_utc": datetime.now(
            timezone.utc,
        ).isoformat(),
        "model_version": model_version,
        "decision_threshold": float(decision_threshold),
        "total_transactions_processed": len(prediction_results),
        "transactions_flagged_for_review": flagged_count,
        "percentage_flagged_for_review": flagged_percentage,
        "prediction_score_distribution": prediction_distribution,
        "monitoring_note": (
            "Flagged transactions are model alerts requiring review. "
            "They are not confirmed fraud outcomes."
        ),
    }

    logger.info(
        "Supervised prediction monitoring completed: %s of %s transactions flagged.",
        flagged_count,
        len(prediction_results),
    )

    return monitoring_report


def monitor_anomaly_predictions(
    prediction_results: pd.DataFrame,
    model_version: str,
) -> dict[str, Any] | None:
    """Monitor anomaly-score behavior when anomaly results are available."""
    required_columns = {
        "anomaly_score",
        "anomaly_prediction",
    }

    if not required_columns.issubset(prediction_results.columns):
        logger.info(
            "Anomaly predictions are not available; anomaly monitoring skipped.",
        )

        return None

    anomaly_scores = prediction_results["anomaly_score"]
    anomaly_predictions = prediction_results["anomaly_prediction"]

    anomaly_flagged_count = int(anomaly_predictions.sum())
    anomaly_flagged_percentage = float(
        anomaly_predictions.mean(),
    )

    anomaly_monitoring_report = {
        "report_timestamp_utc": datetime.now(
            timezone.utc,
        ).isoformat(),
        "model_version": model_version,
        "total_transactions_processed": len(prediction_results),
        "transactions_flagged_as_anomalous": anomaly_flagged_count,
        "percentage_flagged_as_anomalous": anomaly_flagged_percentage,
        "minimum_anomaly_score": float(anomaly_scores.min()),
        "maximum_anomaly_score": float(anomaly_scores.max()),
        "mean_anomaly_score": float(anomaly_scores.mean()),
        "median_anomaly_score": float(anomaly_scores.median()),
        "monitoring_note": (
            "Anomaly scores identify unusual transactions. They are not "
            "fraud probabilities or confirmed fraud outcomes."
        ),
    }

    logger.info(
        "Anomaly prediction monitoring completed: %s of %s transactions flagged.",
        anomaly_flagged_count,
        len(prediction_results),
    )

    return anomaly_monitoring_report


def monitor_prediction_batch(
    prediction_results: pd.DataFrame,
    model_version: str,
    decision_threshold: float,
    include_anomaly_monitoring: bool = False,
) -> dict[str, Any]:
    """Monitor one processed prediction batch."""
    supervised_report = monitor_supervised_predictions(
        prediction_results=prediction_results,
        model_version=model_version,
        decision_threshold=decision_threshold,
    )

    batch_monitoring_report = {
        "supervised_monitoring": supervised_report,
        "anomaly_monitoring": None,
    }

    if include_anomaly_monitoring:
        anomaly_report = monitor_anomaly_predictions(
            prediction_results=prediction_results,
            model_version=model_version,
        )

        batch_monitoring_report["anomaly_monitoring"] = anomaly_report

    logger.info(
        "Prediction-batch monitoring completed.",
    )

    return batch_monitoring_report


def save_prediction_monitoring_report(
    monitoring_report: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a prediction-monitoring report to JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            monitoring_report,
            file,
            indent=2,
            default=str,
        )

    logger.info(
        "Prediction-monitoring report saved to: %s",
        output_path,
    )