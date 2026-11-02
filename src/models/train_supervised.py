"""Reusable supervised model-training utilities."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline

from src.features.feature_engineering import apply_feature_engineering
from src.evaluation.metrics import evaluate_thresholded_predictions

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
DEFAULT_THRESHOLD = 0.5

SupervisedModelName = Literal[
    "logistic_regression",
    "random_forest",
    "lightgbm",
]

ImbalanceStrategy = Literal[
    "none",
    "class_weighting",
    "smote",
]


def create_logistic_regression_pipeline(
    imbalance_strategy: ImbalanceStrategy,
) -> Pipeline | ImbalancedPipeline:
    """Create a Logistic Regression pipeline for the selected imbalance strategy."""
    if imbalance_strategy == "none":
        return Pipeline(
            steps=[
                ("standard_scaler", StandardScaler()),
                (
                    "logistic_regression",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=RANDOM_SEED,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    if imbalance_strategy == "class_weighting":
        return Pipeline(
            steps=[
                ("standard_scaler", StandardScaler()),
                (
                    "logistic_regression",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    if imbalance_strategy == "smote":
        return ImbalancedPipeline(
            steps=[
                ("standard_scaler", StandardScaler()),
                (
                    "smote",
                    SMOTE(
                        sampling_strategy="auto",
                        random_state=RANDOM_SEED,
                    ),
                ),
                (
                    "logistic_regression",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=RANDOM_SEED,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    raise ValueError(
        f"Unsupported imbalance strategy: {imbalance_strategy}"
    )


def create_random_forest_pipeline(
    imbalance_strategy: ImbalanceStrategy,
) -> Pipeline | ImbalancedPipeline:
    """Create a Random Forest pipeline for the selected imbalance strategy."""
    if imbalance_strategy == "none":
        return Pipeline(
            steps=[
                (
                    "random_forest",
                    RandomForestClassifier(
                        n_estimators=200,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )

    if imbalance_strategy == "class_weighting":
        return Pipeline(
            steps=[
                (
                    "random_forest",
                    RandomForestClassifier(
                        n_estimators=200,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )

    if imbalance_strategy == "smote":
        return ImbalancedPipeline(
            steps=[
                (
                    "smote",
                    SMOTE(
                        sampling_strategy="auto",
                        random_state=RANDOM_SEED,
                    ),
                ),
                (
                    "random_forest",
                    RandomForestClassifier(
                        n_estimators=200,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )

    raise ValueError(
        f"Unsupported imbalance strategy: {imbalance_strategy}"
    )


def create_lightgbm_pipeline(
    imbalance_strategy: ImbalanceStrategy,
    positive_class_weight: float,
) -> Pipeline:
    """Create a LightGBM pipeline for the selected imbalance strategy."""
    if imbalance_strategy not in {"none", "class_weighting"}:
        raise ValueError(
            "LightGBM supports only 'none' or 'class_weighting' in this module."
        )

    return Pipeline(
        steps=[
            (
                "lightgbm",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=20,
                    scale_pos_weight=(
                        positive_class_weight
                        if imbalance_strategy == "class_weighting"
                        else 1.0
                    ),
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )


def create_supervised_pipeline(
    model_name: SupervisedModelName,
    imbalance_strategy: ImbalanceStrategy,
    positive_class_weight: float = 1.0,
) -> Pipeline | ImbalancedPipeline:
    """Create a supervised pipeline for a supported model and imbalance strategy."""
    if model_name == "logistic_regression":
        return create_logistic_regression_pipeline(
            imbalance_strategy=imbalance_strategy,
        )

    if model_name == "random_forest":
        return create_random_forest_pipeline(
            imbalance_strategy=imbalance_strategy,
        )

    if model_name == "lightgbm":
        return create_lightgbm_pipeline(
            imbalance_strategy=imbalance_strategy,
            positive_class_weight=positive_class_weight,
        )

    raise ValueError(f"Unsupported supervised model: {model_name}")


def calculate_positive_class_weight(
    y_train: pd.Series,
) -> float:
    """Calculate the positive-class weight for LightGBM."""
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0:
        raise ValueError(
            "Training data contains no positive fraud examples."
        )

    return negative_count / positive_count


def train_supervised_model(
    model_name: SupervisedModelName,
    imbalance_strategy: ImbalanceStrategy,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[Pipeline | ImbalancedPipeline, dict]:
    """Train and evaluate one supervised model using validation data."""
    positive_class_weight = calculate_positive_class_weight(
        y_train=y_train,
    )

    pipeline = create_supervised_pipeline(
        model_name=model_name,
        imbalance_strategy=imbalance_strategy,
        positive_class_weight=positive_class_weight,
    )

    training_start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_duration = time.time() - training_start_time

    validation_scores = pipeline.predict_proba(
        X_validation,
    )[:, 1]

    metrics = evaluate_thresholded_predictions(
        y_true=y_validation,
        y_score=validation_scores,
        threshold=threshold,
    )

    training_metadata = {
        "model_name": model_name,
        "imbalance_strategy": imbalance_strategy,
        "feature_count": X_train.shape[1],
        "training_duration_seconds": training_duration,
        "positive_class_weight": positive_class_weight,
        "decision_threshold": threshold,
        "validation_metrics": metrics,
        "test_set_used": False,
    }

    logger.info(
        "Trained %s with %s strategy. Validation PR-AUC: %.4f",
        model_name,
        imbalance_strategy,
        metrics["pr_auc"],
    )

    return pipeline, training_metadata


def save_supervised_model(
    pipeline: Pipeline | ImbalancedPipeline,
    training_metadata: dict,
    feature_columns: list[str],
    output_directory: Path,
) -> dict[str, str]:
    """Save a trained supervised pipeline and its metadata."""
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
        "Supervised model saved to: %s",
        model_path,
    )

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
    }


def prepare_supervised_training_features(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the project's selected feature engineering to training and validation data."""
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

    return X_train_engineered, X_validation_engineered