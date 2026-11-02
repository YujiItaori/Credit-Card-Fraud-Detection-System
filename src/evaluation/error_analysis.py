"""Reusable error-analysis utilities for fraud-detection models."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def calculate_confusion_counts(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, int]:
    """Calculate true-positive, false-positive, true-negative, and false-negative counts."""
    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    true_negative, false_positive, false_negative, true_positive = (
        confusion.ravel()
    )

    return {
        "true_positives": int(true_positive),
        "false_positives": int(false_positive),
        "true_negatives": int(true_negative),
        "false_negatives": int(false_negative),
    }


def analyze_supervised_errors(
    y_true: pd.Series,
    y_score: np.ndarray,
    X_validation: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Analyze false positives and false negatives for one supervised model."""
    if len(y_true) != len(y_score):
        raise ValueError(
            "Validation labels and prediction scores must have equal length."
        )

    predictions = (y_score >= threshold).astype(int)

    false_positive_mask = (y_true == 0) & (predictions == 1)
    false_negative_mask = (y_true == 1) & (predictions == 0)

    if "Amount" not in X_validation.columns:
        raise ValueError(
            "Validation features must contain an 'Amount' column."
        )

    false_positive_amounts = X_validation.loc[
        false_positive_mask,
        "Amount",
    ]

    false_negative_amounts = X_validation.loc[
        false_negative_mask,
        "Amount",
    ]

    error_summary = pd.DataFrame(
        [
            {
                "error_type": "false_positive",
                "transaction_count": int(false_positive_mask.sum()),
                "fraud_count": int(
                    y_true[false_positive_mask].sum()
                ),
                "median_amount": (
                    float(false_positive_amounts.median())
                    if len(false_positive_amounts) > 0
                    else np.nan
                ),
                "minimum_amount": (
                    float(false_positive_amounts.min())
                    if len(false_positive_amounts) > 0
                    else np.nan
                ),
                "maximum_amount": (
                    float(false_positive_amounts.max())
                    if len(false_positive_amounts) > 0
                    else np.nan
                ),
            },
            {
                "error_type": "false_negative",
                "transaction_count": int(false_negative_mask.sum()),
                "fraud_count": int(
                    y_true[false_negative_mask].sum()
                ),
                "median_amount": (
                    float(false_negative_amounts.median())
                    if len(false_negative_amounts) > 0
                    else np.nan
                ),
                "minimum_amount": (
                    float(false_negative_amounts.min())
                    if len(false_negative_amounts) > 0
                    else np.nan
                ),
                "maximum_amount": (
                    float(false_negative_amounts.max())
                    if len(false_negative_amounts) > 0
                    else np.nan
                ),
            },
        ]
    )

    logger.info(
        "Supervised error analysis completed: %s false positives, %s false negatives.",
        int(false_positive_mask.sum()),
        int(false_negative_mask.sum()),
    )

    return error_summary


def analyze_model_agreement(
    y_true: pd.Series,
    supervised_predictions: np.ndarray,
    anomaly_predictions: np.ndarray,
    X_validation: pd.DataFrame,
) -> pd.DataFrame:
    """Categorize validation transactions by supervised and anomaly-model agreement."""
    if len(y_true) != len(supervised_predictions):
        raise ValueError(
            "Validation labels and supervised predictions must have equal length."
        )

    if len(y_true) != len(anomaly_predictions):
        raise ValueError(
            "Validation labels and anomaly predictions must have equal length."
        )

    if "Amount" not in X_validation.columns:
        raise ValueError(
            "Validation features must contain an 'Amount' column."
        )

    agreement_records = []

    for position in range(len(y_true)):
        supervised_flag = int(supervised_predictions[position])
        anomaly_flag = int(anomaly_predictions[position])
        actual_label = int(y_true.iloc[position])

        if supervised_flag == 1 and anomaly_flag == 1:
            agreement_category = "A: Both models flag"
        elif supervised_flag == 1 and anomaly_flag == 0:
            agreement_category = "B: Supervised model only"
        elif supervised_flag == 0 and anomaly_flag == 1:
            agreement_category = "C: Anomaly detector only"
        else:
            agreement_category = "D: Neither model flags"

        agreement_records.append(
            {
                "agreement_category": agreement_category,
                "actual_class": actual_label,
                "amount": float(
                    X_validation.iloc[position]["Amount"]
                ),
            }
        )

    agreement_dataframe = pd.DataFrame(agreement_records)

    agreement_summary = (
        agreement_dataframe
        .groupby("agreement_category")
        .agg(
            transaction_count=("actual_class", "size"),
            fraud_count=("actual_class", "sum"),
            median_amount=("amount", "median"),
            minimum_amount=("amount", "min"),
            maximum_amount=("amount", "max"),
        )
        .reset_index()
        .sort_values("agreement_category")
    )

    agreement_summary["fraud_rate"] = (
        agreement_summary["fraud_count"]
        / agreement_summary["transaction_count"]
    )

    logger.info(
        "Model-agreement analysis completed for %s validation transactions.",
        len(y_true),
    )

    return agreement_summary


def save_error_analysis(
    error_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save supervised error-analysis results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    error_summary.to_csv(output_path, index=False)

    logger.info(
        "Error-analysis results saved to: %s",
        output_path,
    )


def save_model_agreement(
    agreement_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save model-agreement analysis results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    agreement_summary.to_csv(output_path, index=False)

    logger.info(
        "Model-agreement results saved to: %s",
        output_path,
    )