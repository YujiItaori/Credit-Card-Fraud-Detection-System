"""Validation-based classification threshold optimization."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

ThresholdStrategy = Literal[
    "max_f1",
    "max_recall_with_min_precision",
    "max_precision_with_min_recall",
    "fixed_review_capacity",
]


def load_validation_data(
    feature_path: Path,
    target_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load validation features and labels from Parquet files."""
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Validation feature file not found: {feature_path}"
        )

    if not target_path.exists():
        raise FileNotFoundError(
            f"Validation target file not found: {target_path}"
        )

    features = pd.read_parquet(feature_path)
    target_frame = pd.read_parquet(target_path)

    if features.empty or target_frame.empty:
        raise ValueError("Validation data is empty.")

    target_column = target_frame.columns[0]
    target = target_frame[target_column]

    if len(features) != len(target):
        raise ValueError(
            "Validation feature and target row counts do not match."
        )

    return features, target


def apply_feature_engineering(features: pd.DataFrame) -> pd.DataFrame:
    """Apply the project's selected feature-engineering configuration."""
    if "Amount" not in features.columns:
        raise ValueError("Required column 'Amount' is missing.")

    if "Time" not in features.columns:
        raise ValueError("Required column 'Time' is missing.")

    if (features["Amount"] < 0).any():
        raise ValueError(
            "Negative Amount values are not supported by log1p."
        )

    engineered = features.copy()
    engineered["Amount_log1p"] = np.log1p(engineered["Amount"])
    engineered["Time_hours"] = engineered["Time"] / 3600.0

    if "Class" in engineered.columns:
        raise ValueError(
            "Target leakage detected in validation features."
        )

    return engineered


def calculate_threshold_metrics(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate fraud-detection metrics at one classification threshold."""
    predictions = (y_score >= threshold).astype(int)

    confusion = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )
    true_negative, false_positive, false_negative, true_positive = (
        confusion.ravel()
    )

    false_positive_rate = (
        false_positive / (false_positive + true_negative)
        if (false_positive + true_negative) > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                y_score,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_score,
            )
        ),
        "false_positive_rate": float(false_positive_rate),
        "true_positives": int(true_positive),
        "false_positives": int(false_positive),
        "true_negatives": int(true_negative),
        "false_negatives": int(false_negative),
        "transactions_flagged": int(true_positive + false_positive),
    }


def evaluate_threshold_candidates(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold_start: float = 0.01,
    threshold_end: float = 0.99,
    threshold_step: float = 0.01,
) -> pd.DataFrame:
    """Evaluate a range of classification thresholds on validation data."""
    if threshold_start <= 0 or threshold_end >= 1:
        raise ValueError(
            "Thresholds must be between 0 and 1, exclusive."
        )

    if threshold_step <= 0:
        raise ValueError("Threshold step must be positive.")

    thresholds = np.arange(
        threshold_start,
        threshold_end + threshold_step,
        threshold_step,
    )

    results = [
        calculate_threshold_metrics(
            y_true=y_true,
            y_score=y_score,
            threshold=float(threshold),
        )
        for threshold in thresholds
    ]

    return pd.DataFrame(results)


def select_threshold(
    threshold_results: pd.DataFrame,
    strategy: ThresholdStrategy,
    minimum_precision: float | None = None,
    minimum_recall: float | None = None,
    review_capacity: int | None = None,
) -> tuple[float, str]:
    """Select an operating threshold using a documented validation objective."""
    if threshold_results.empty:
        raise ValueError("No threshold candidates were provided.")

    if strategy == "max_f1":
        selected = threshold_results.loc[
            threshold_results["f1_score"].idxmax()
        ]
        rationale = (
            "Selected the threshold with the highest validation F1-score."
        )

    elif strategy == "max_recall_with_min_precision":
        if minimum_precision is None:
            raise ValueError(
                "minimum_precision is required for "
                "max_recall_with_min_precision."
            )

        eligible = threshold_results[
            threshold_results["precision"] >= minimum_precision
        ]

        if eligible.empty:
            raise ValueError(
                "No threshold satisfies the minimum precision constraint."
            )

        selected = eligible.loc[eligible["recall"].idxmax()]
        rationale = (
            "Selected the threshold with the highest validation recall "
            f"subject to precision >= {minimum_precision:.4f}."
        )

    elif strategy == "max_precision_with_min_recall":
        if minimum_recall is None:
            raise ValueError(
                "minimum_recall is required for "
                "max_precision_with_min_recall."
            )

        eligible = threshold_results[
            threshold_results["recall"] >= minimum_recall
        ]

        if eligible.empty:
            raise ValueError(
                "No threshold satisfies the minimum recall constraint."
            )

        selected = eligible.loc[eligible["precision"].idxmax()]
        rationale = (
            "Selected the threshold with the highest validation precision "
            f"subject to recall >= {minimum_recall:.4f}."
        )

    elif strategy == "fixed_review_capacity":
        if review_capacity is None or review_capacity <= 0:
            raise ValueError(
                "A positive review_capacity is required for "
                "fixed_review_capacity."
            )

        eligible = threshold_results[
            threshold_results["transactions_flagged"] <= review_capacity
        ]

        if eligible.empty:
            raise ValueError(
                "No threshold satisfies the review-capacity constraint."
            )

        selected = eligible.loc[eligible["recall"].idxmax()]
        rationale = (
            "Selected the threshold with the highest validation recall "
            f"while flagging at most {review_capacity} transactions."
        )

    else:
        raise ValueError(f"Unsupported threshold strategy: {strategy}")

    return float(selected["threshold"]), rationale


def save_threshold_results(
    threshold_results: pd.DataFrame,
    selected_threshold: float,
    rationale: str,
    output_csv_path: Path,
    output_json_path: Path,
) -> None:
    """Save threshold-analysis results and the selected operating threshold."""
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    threshold_results.to_csv(output_csv_path, index=False)

    selected_row = threshold_results[
        np.isclose(
            threshold_results["threshold"],
            selected_threshold,
        )
    ].iloc[0]

    threshold_metadata = {
        "selected_threshold": float(selected_threshold),
        "selection_rationale": rationale,
        "validation_metrics": selected_row.to_dict(),
        "test_set_used": False,
    }

    with output_json_path.open("w", encoding="utf-8") as file:
        json.dump(
            threshold_metadata,
            file,
            indent=2,
        )

    logger.info(
        "Threshold results saved to: %s",
        output_csv_path,
    )
    logger.info(
        "Selected threshold metadata saved to: %s",
        output_json_path,
    )


def run_threshold_optimization(
    model_path: Path,
    validation_feature_path: Path,
    validation_target_path: Path,
    output_csv_path: Path,
    output_json_path: Path,
    strategy: ThresholdStrategy = "max_f1",
    minimum_precision: float | None = None,
    minimum_recall: float | None = None,
    review_capacity: int | None = None,
) -> dict:
    """Run the complete validation-based threshold-optimization workflow."""
    pipeline = joblib.load(model_path)

    X_validation, y_validation = load_validation_data(
        feature_path=validation_feature_path,
        target_path=validation_target_path,
    )

    X_validation_engineered = apply_feature_engineering(
        X_validation,
    )

    validation_scores = pipeline.predict_proba(
        X_validation_engineered,
    )[:, 1]

    threshold_results = evaluate_threshold_candidates(
        y_true=y_validation,
        y_score=validation_scores,
    )

    selected_threshold, rationale = select_threshold(
        threshold_results=threshold_results,
        strategy=strategy,
        minimum_precision=minimum_precision,
        minimum_recall=minimum_recall,
        review_capacity=review_capacity,
    )

    save_threshold_results(
        threshold_results=threshold_results,
        selected_threshold=selected_threshold,
        rationale=rationale,
        output_csv_path=output_csv_path,
        output_json_path=output_json_path,
    )

    selected_metrics = calculate_threshold_metrics(
        y_true=y_validation,
        y_score=validation_scores,
        threshold=selected_threshold,
    )

    logger.info(
        "Selected threshold: %.2f | Strategy: %s",
        selected_threshold,
        strategy,
    )
    logger.info(
        "Validation precision: %.4f | recall: %.4f | F1: %.4f",
        selected_metrics["precision"],
        selected_metrics["recall"],
        selected_metrics["f1_score"],
    )

    return {
        "selected_threshold": selected_threshold,
        "selection_rationale": rationale,
        "validation_metrics": selected_metrics,
    }