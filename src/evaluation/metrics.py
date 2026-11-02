"""Reusable evaluation metrics for imbalanced fraud detection."""

from __future__ import annotations

import logging
from typing import Any

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


def calculate_confusion_counts(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, int]:
    """Calculate confusion-matrix counts using the project's label convention."""
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


def calculate_classification_metrics(
    y_true: pd.Series,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Calculate metrics appropriate for highly imbalanced fraud detection."""
    if len(y_true) != len(y_score):
        raise ValueError(
            "True labels and prediction scores must have equal length."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "True labels and predicted labels must have equal length."
        )

    confusion_counts = calculate_confusion_counts(
        y_true=y_true,
        y_pred=y_pred,
    )

    true_positive = confusion_counts["true_positives"]
    false_positive = confusion_counts["false_positives"]
    true_negative = confusion_counts["true_negatives"]

    false_positive_rate = (
        false_positive / (false_positive + true_negative)
        if (false_positive + true_negative) > 0
        else 0.0
    )

    has_both_classes = len(np.unique(y_true)) > 1

    metrics: dict[str, Any] = {
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "pr_auc": (
            float(
                average_precision_score(
                    y_true,
                    y_score,
                )
            )
            if has_both_classes
            else np.nan
        ),
        "roc_auc": (
            float(
                roc_auc_score(
                    y_true,
                    y_score,
                )
            )
            if has_both_classes
            else np.nan
        ),
        "false_positive_rate": float(false_positive_rate),
        "transactions_flagged": int(
            true_positive + false_positive,
        ),
        **confusion_counts,
    }

    if threshold is not None:
        metrics["threshold"] = float(threshold)

    logger.info(
        "Classification metrics calculated: precision=%.4f, recall=%.4f, "
        "F1=%.4f, PR-AUC=%.4f",
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["pr_auc"],
    )

    return metrics


def evaluate_thresholded_predictions(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Convert scores to binary predictions using a threshold and evaluate them."""
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1, inclusive.")

    predictions = (y_score >= threshold).astype(int)

    return calculate_classification_metrics(
        y_true=y_true,
        y_score=y_score,
        y_pred=predictions,
        threshold=threshold,
    )


def calculate_ranking_metrics(
    y_true: pd.Series,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Calculate threshold-independent ranking metrics."""
    if len(y_true) != len(y_score):
        raise ValueError(
            "True labels and prediction scores must have equal length."
        )

    has_both_classes = len(np.unique(y_true)) > 1

    if not has_both_classes:
        raise ValueError(
            "Ranking metrics require both positive and negative classes."
        )

    ranking_metrics = {
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
    }

    logger.info(
        "Ranking metrics calculated: PR-AUC=%.4f, ROC-AUC=%.4f",
        ranking_metrics["pr_auc"],
        ranking_metrics["roc_auc"],
    )

    return ranking_metrics


def calculate_metrics_at_review_capacity(
    y_true: pd.Series,
    y_score: np.ndarray,
    review_capacity: int,
) -> dict[str, Any]:
    """Evaluate the highest-scoring transactions at a fixed review capacity."""
    if review_capacity <= 0:
        raise ValueError("Review capacity must be positive.")

    if review_capacity > len(y_score):
        raise ValueError(
            "Review capacity cannot exceed the number of transactions."
        )

    score_order = np.argsort(y_score)[::-1]
    selected_positions = score_order[:review_capacity]

    selected_scores = y_score[selected_positions]
    selected_predictions = np.ones(
        len(selected_positions),
        dtype=int,
    )

    selected_true_labels = y_true.iloc[selected_positions]

    metrics = calculate_classification_metrics(
        y_true=selected_true_labels,
        y_score=selected_scores,
        y_pred=selected_predictions,
    )

    metrics["review_capacity"] = int(review_capacity)

    logger.info(
        "Metrics calculated at review capacity: %s",
        review_capacity,
    )

    return metrics