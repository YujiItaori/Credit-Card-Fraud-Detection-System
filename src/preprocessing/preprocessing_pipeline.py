"""Reusable preprocessing pipelines for supervised and anomaly models."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.feature_engineering import (
    FeatureEngineeringError,
    apply_feature_engineering,
)

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Class"


class PreprocessingError(Exception):
    """Raised when preprocessing cannot be applied safely."""


def validate_preprocessing_input(
    features: pd.DataFrame,
) -> None:
    """Validate that preprocessing input is suitable."""
    if not isinstance(features, pd.DataFrame):
        raise PreprocessingError(
            "Preprocessing input must be a pandas DataFrame."
        )

    if features.empty:
        raise PreprocessingError(
            "Preprocessing input is empty."
        )

    if TARGET_COLUMN in features.columns:
        raise PreprocessingError(
            "Target leakage detected: target column must not be included "
            "in preprocessing input."
        )

    numeric_features = features.select_dtypes(
        include=[np.number],
    )

    if numeric_features.empty:
        raise PreprocessingError(
            "Preprocessing input does not contain numeric features."
        )

    if features.isnull().any().any():
        raise PreprocessingError(
            "Preprocessing input contains missing values."
        )

    if np.isinf(
        numeric_features.to_numpy(dtype=float),
    ).any():
        raise PreprocessingError(
            "Preprocessing input contains infinite values."
        )

    logger.info(
        "Preprocessing input validated for %s rows.",
        len(features),
    )


def create_supervised_preprocessing_pipeline(
    apply_scaling: bool = True,
) -> Pipeline:
    """Create a preprocessing pipeline for supervised models."""
    steps = []

    if apply_scaling:
        steps.append(
            (
                "standard_scaler",
                StandardScaler(),
            )
        )

    pipeline = Pipeline(
        steps=steps,
    )

    logger.info(
        "Created supervised preprocessing pipeline with scaling=%s.",
        apply_scaling,
    )

    return pipeline


def create_anomaly_preprocessing_pipeline(
    apply_scaling: bool = True,
) -> Pipeline:
    """Create a preprocessing pipeline for anomaly-detection models."""
    steps = []

    if apply_scaling:
        steps.append(
            (
                "standard_scaler",
                StandardScaler(),
            )
        )

    pipeline = Pipeline(
        steps=steps,
    )

    logger.info(
        "Created anomaly preprocessing pipeline with scaling=%s.",
        apply_scaling,
    )

    return pipeline


def prepare_supervised_features(
    features: pd.DataFrame,
    apply_scaling: bool = True,
) -> tuple[pd.DataFrame, Pipeline | None]:
    """Prepare supervised features by applying feature engineering and scaling."""
    validate_preprocessing_input(
        features=features,
    )

    engineered_features = apply_feature_engineering(
        features=features,
    )

    if not apply_scaling:
        logger.info(
            "Scaling disabled for supervised features.",
        )

        return engineered_features, None

    scaler = StandardScaler()

    scaled_features = pd.DataFrame(
        scaler.fit_transform(engineered_features),
        columns=engineered_features.columns,
        index=engineered_features.index,
    )

    logger.info(
        "Supervised features scaled using training-fitted scaler.",
    )

    return scaled_features, scaler


def prepare_anomaly_features(
    features: pd.DataFrame,
    apply_scaling: bool = True,
) -> tuple[pd.DataFrame, Pipeline | None]:
    """Prepare anomaly-detection features by applying feature engineering and scaling."""
    validate_preprocessing_input(
        features=features,
    )

    engineered_features = apply_feature_engineering(
        features=features,
    )

    if not apply_scaling:
        logger.info(
            "Scaling disabled for anomaly features.",
        )

        return engineered_features, None

    scaler = StandardScaler()

    scaled_features = pd.DataFrame(
        scaler.fit_transform(engineered_features),
        columns=engineered_features.columns,
        index=engineered_features.index,
    )

    logger.info(
        "Anomaly features scaled using training-fitted scaler.",
    )

    return scaled_features, scaler


def apply_fitted_scaler(
    features: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Apply an already-fitted scaler to validation, test, or inference data."""
    if scaler is None:
        raise PreprocessingError(
            "A fitted scaler is required to transform new data."
        )

    validate_preprocessing_input(
        features=features,
    )

    engineered_features = apply_feature_engineering(
        features=features,
    )

    if list(engineered_features.columns) != list(scaler.feature_names_in_):
        raise PreprocessingError(
            "Feature columns do not match the fitted scaler schema."
        )

    scaled_features = pd.DataFrame(
        scaler.transform(engineered_features),
        columns=engineered_features.columns,
        index=engineered_features.index,
    )

    logger.info(
        "Fitted scaler applied to %s rows.",
        len(scaled_features),
    )

    return scaled_features