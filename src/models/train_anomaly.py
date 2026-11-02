"""Reusable anomaly-detection model-training utilities."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.feature_engineering import apply_feature_engineering
from src.evaluation.metrics import calculate_classification_metrics

logger = logging.getLogger(__name__)

RANDOM_SEED = 42

AnomalyModelName = Literal[
    "isolation_forest",
    "local_outlier_factor",
]


def create_isolation_forest_pipeline(
    contamination: float = 0.001,
) -> Pipeline:
    """Create an Isolation Forest anomaly-detection pipeline."""
    return Pipeline(
        steps=[
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=200,
                    max_samples="auto",
                    contamination=contamination,
                    max_features=1.0,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def create_local_outlier_factor_pipeline(
    contamination: float = 0.001,
    n_neighbors: int = 20,
) -> Pipeline:
    """Create a Local Outlier Factor novelty-detection pipeline."""
    return Pipeline(
        steps=[
            ("standard_scaler", StandardScaler()),
            (
                "local_outlier_factor",
                LocalOutlierFactor(
                    n_neighbors=n_neighbors,
                    metric="minkowski",
                    contamination=contamination,
                    novelty=True,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def create_anomaly_pipeline(
    model_name: AnomalyModelName,
    contamination: float = 0.001,
    n_neighbors: int = 20,
) -> Pipeline:
    """Create an anomaly-detection pipeline for a supported algorithm."""
    if model_name == "isolation_forest":
        return create_isolation_forest_pipeline(
            contamination=contamination,
        )

    if model_name == "local_outlier_factor":
        return create_local_outlier_factor_pipeline(
            contamination=contamination,
            n_neighbors=n_neighbors,
        )

    raise ValueError(f"Unsupported anomaly-detection model: {model_name}")


def validate_anomaly_training_input(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """Ensure fraud labels are available separately but not included in features."""
    if X_train.empty:
        raise ValueError("Anomaly-training feature data is empty.")

    if "Class" in X_train.columns:
        raise ValueError(
            "Target leakage detected: 'Class' must not be included in "
            "anomaly-training features."
        )

    if y_train is None:
        raise ValueError(
            "Fraud labels must be retained separately for later evaluation."
        )

    logger.info(
        "Anomaly-training input validated for %s rows.",
        len(X_train),
    )


def train_anomaly_model(
    model_name: AnomalyModelName,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    contamination: float = 0.001,
    n_neighbors: int = 20,
) -> tuple[Pipeline, dict]:
    """Train and evaluate an anomaly detector using validation data."""
    validate_anomaly_training_input(
        X_train=X_train,
        y_train=y_train,
    )

    X_train_engineered = apply_feature_engineering(
        features=X_train,
    )

    X_validation_engineered = apply_feature_engineering(
        features=X_validation,
    )

    if list(X_train_engineered.columns) != list(
        X_validation_engineered.columns
    ):
        raise ValueError(
            "Training and validation feature columns do not match."
        )

    pipeline = create_anomaly_pipeline(
        model_name=model_name,
        contamination=contamination,
        n_neighbors=n_neighbors,
    )

    training_start_time = time.time()

    pipeline.fit(X_train_engineered)

    training_duration = time.time() - training_start_time

    raw_validation_scores = pipeline.decision_function(
        X_validation_engineered,
    )

    anomaly_scores = -raw_validation_scores

    anomaly_predictions = pipeline.predict(
        X_validation_engineered,
    )

    anomaly_predictions = np.where(
        anomaly_predictions == -1,
        1,
        0,
    )

    metrics = calculate_classification_metrics(
        y_true=y_validation,
        y_score=anomaly_scores,
        y_pred=anomaly_predictions,
    )

    training_metadata = {
        "model_name": model_name,
        "model_type": "unsupervised_anomaly_detector",
        "feature_count": X_train_engineered.shape[1],
        "training_duration_seconds": training_duration,
        "contamination": contamination,
        "n_neighbors": (
            n_neighbors if model_name == "local_outlier_factor" else None
        ),
        "score_direction": "higher_score_means_more_anomalous",
        "target_used_during_training": False,
        "validation_metrics": metrics,
        "test_set_used": False,
    }

    logger.info(
        "Trained %s. Validation PR-AUC: %.4f",
        model_name,
        metrics["pr_auc"],
    )

    return pipeline, training_metadata


def save_anomaly_model(
    pipeline: Pipeline,
    training_metadata: dict,
    feature_columns: list[str],
    output_directory: Path,
) -> dict[str, str]:
    """Save a trained anomaly detector and its metadata."""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = output_directory / "model.joblib"
    metadata_path = output_directory / "metadata.json"

    joblib.dump(
        pipeline,
        model_path,
    )

    artifact_metadata = {
        **training_metadata,
        "feature_columns": feature_columns,
        "target_column": "Class",
        "feature_configuration": "log_amount_and_time_hours",
        "score_interpretation": (
            "Higher anomaly score means more anomalous; it is not a fraud probability."
        ),
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        import json

        json.dump(
            artifact_metadata,
            file,
            indent=2,
            default=str,
        )

    logger.info(
        "Anomaly model saved to: %s",
        model_path,
    )

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
    }