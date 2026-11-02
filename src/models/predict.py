"""Reusable prediction pipeline for supervised and anomaly models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.model_loader import (
    ModelArtifactError,
    load_anomaly_model_artifacts,
    load_supervised_model_artifacts,
    prepare_model_input,
)

logger = logging.getLogger(__name__)


class PredictionInputError(Exception):
    """Raised when prediction input is invalid or incompatible."""


def validate_prediction_input(
    input_features: pd.DataFrame,
) -> None:
    """Validate the basic structure and values of prediction input."""
    if not isinstance(input_features, pd.DataFrame):
        raise PredictionInputError(
            "Prediction input must be a pandas DataFrame."
        )

    if input_features.empty:
        raise PredictionInputError(
            "Prediction input contains no transactions."
        )

    required_original_columns = {
        "Time",
        "Amount",
    }

    missing_original_columns = (
        required_original_columns - set(input_features.columns)
    )

    if missing_original_columns:
        raise PredictionInputError(
            f"Prediction input is missing required columns: "
            f"{sorted(missing_original_columns)}"
        )

    numeric_columns = input_features.select_dtypes(
        include=[np.number],
    ).columns

    if len(numeric_columns) == 0:
        raise PredictionInputError(
            "Prediction input does not contain numeric features."
        )

    if input_features[numeric_columns].isnull().any().any():
        raise PredictionInputError(
            "Prediction input contains missing numeric values."
        )

    if np.isinf(
        input_features[numeric_columns].to_numpy(dtype=float)
    ).any():
        raise PredictionInputError(
            "Prediction input contains infinite numeric values."
        )

    if (input_features["Amount"] < 0).any():
        raise PredictionInputError(
            "Prediction input contains negative Amount values."
        )

    logger.info(
        "Prediction input validated for %s transactions.",
        len(input_features),
    )


def apply_prediction_feature_engineering(
    input_features: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the established feature-engineering configuration."""
    engineered_features = input_features.copy()

    engineered_features["Amount_log1p"] = np.log1p(
        engineered_features["Amount"],
    )

    engineered_features["Time_hours"] = (
        engineered_features["Time"] / 3600.0
    )

    if "Class" in engineered_features.columns:
        engineered_features = engineered_features.drop(
            columns=["Class"],
        )

    logger.info(
        "Prediction feature engineering applied.",
    )

    return engineered_features


class FraudPredictionService:
    """Reusable prediction service for supervised and anomaly models."""

    def __init__(
        self,
        supervised_model_directory: Path,
        anomaly_model_directory: Path | None = None,
    ) -> None:
        """Load and validate saved model artifacts."""
        (
            self.supervised_pipeline,
            self.supervised_metadata,
        ) = load_supervised_model_artifacts(
            model_directory=supervised_model_directory,
        )

        self.anomaly_pipeline = None
        self.anomaly_metadata = None

        if anomaly_model_directory is not None:
            (
                self.anomaly_pipeline,
                self.anomaly_metadata,
            ) = load_anomaly_model_artifacts(
                model_directory=anomaly_model_directory,
            )

        self.supervised_feature_columns = list(
            self.supervised_metadata["feature_columns"],
        )

        self.supervised_threshold = float(
            self.supervised_metadata["decision_threshold"],
        )

        logger.info(
            "FraudPredictionService initialized with supervised model: %s",
            self.supervised_metadata["model_name"],
        )

    def predict(
        self,
        input_features: pd.DataFrame,
        include_anomaly_scores: bool = False,
    ) -> pd.DataFrame:
        """Generate supervised predictions and optional anomaly scores."""
        validate_prediction_input(
            input_features=input_features,
        )

        engineered_features = apply_prediction_feature_engineering(
            input_features=input_features,
        )

        supervised_input = prepare_model_input(
            input_features=engineered_features,
            expected_feature_columns=self.supervised_feature_columns,
        )

        supervised_scores = self.supervised_pipeline.predict_proba(
            supervised_input,
        )[:, 1]

        supervised_predictions = (
            supervised_scores >= self.supervised_threshold
        ).astype(int)

        result_dataframe = input_features.copy()

        result_dataframe["fraud_score"] = supervised_scores
        result_dataframe["fraud_prediction"] = supervised_predictions
        result_dataframe["decision_threshold"] = self.supervised_threshold
        result_dataframe["review_flag"] = supervised_predictions

        if include_anomaly_scores:
            if self.anomaly_pipeline is None or self.anomaly_metadata is None:
                raise ModelArtifactError(
                    "Anomaly model artifacts were not loaded."
                )

            anomaly_feature_columns = list(
                self.anomaly_metadata["feature_columns"],
            )

            anomaly_input = prepare_model_input(
                input_features=engineered_features,
                expected_feature_columns=anomaly_feature_columns,
            )

            raw_anomaly_scores = self.anomaly_pipeline.decision_function(
                anomaly_input,
            )

            anomaly_scores = -raw_anomaly_scores
            anomaly_predictions = self.anomaly_pipeline.predict(
                anomaly_input,
            )

            anomaly_predictions = np.where(
                anomaly_predictions == -1,
                1,
                0,
            )

            result_dataframe["anomaly_score"] = anomaly_scores
            result_dataframe["anomaly_prediction"] = anomaly_predictions

        logger.info(
            "Generated predictions for %s transactions.",
            len(result_dataframe),
        )

        return result_dataframe


def create_prediction_service(
    supervised_model_directory: Path,
    anomaly_model_directory: Path | None = None,
) -> FraudPredictionService:
    """Create a reusable prediction service from saved model artifacts."""
    return FraudPredictionService(
        supervised_model_directory=supervised_model_directory,
        anomaly_model_directory=anomaly_model_directory,
    )