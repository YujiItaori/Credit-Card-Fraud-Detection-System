"""Reusable loading and validation of saved model artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class ModelArtifactError(Exception):
    """Raised when a saved model artifact is missing or incompatible."""


def load_json_metadata(
    metadata_path: Path,
) -> dict[str, Any]:
    """Load and validate model metadata from a JSON file."""
    if not metadata_path.exists():
        raise ModelArtifactError(
            f"Model metadata file not found: {metadata_path}"
        )

    try:
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)
    except json.JSONDecodeError as error:
        raise ModelArtifactError(
            f"Model metadata file is not valid JSON: {metadata_path}"
        ) from error

    if not isinstance(metadata, dict):
        raise ModelArtifactError(
            f"Model metadata must contain a JSON object: {metadata_path}"
        )

    logger.info(
        "Model metadata loaded from: %s",
        metadata_path,
    )

    return metadata


def load_model_pipeline(
    model_path: Path,
) -> object:
    """Load a saved model pipeline from a joblib file."""
    if not model_path.exists():
        raise ModelArtifactError(
            f"Model artifact file not found: {model_path}"
        )

    try:
        pipeline = joblib.load(model_path)
    except Exception as error:
        raise ModelArtifactError(
            f"Failed to load model artifact: {model_path}"
        ) from error

    if not hasattr(pipeline, "predict"):
        raise ModelArtifactError(
            f"Loaded artifact does not support prediction: {model_path}"
        )

    logger.info(
        "Model pipeline loaded from: %s",
        model_path,
    )

    return pipeline


def validate_feature_schema(
    input_features: pd.DataFrame,
    expected_feature_columns: list[str],
) -> None:
    """Validate that input features match the expected model schema."""
    if input_features.empty:
        raise ModelArtifactError(
            "Input feature data is empty."
        )

    missing_columns = set(expected_feature_columns) - set(
        input_features.columns
    )

    if missing_columns:
        raise ModelArtifactError(
            f"Input data is missing required features: "
            f"{sorted(missing_columns)}"
        )

    extra_columns = set(input_features.columns) - set(
        expected_feature_columns
    )

    if extra_columns:
        logger.warning(
            "Input data contains unexpected columns; they will be ignored: %s",
            sorted(extra_columns),
        )

    expected_feature_order = expected_feature_columns
    input_feature_order = list(input_features.columns)

    if input_feature_order != expected_feature_order:
        logger.info(
            "Reordering input features to match the saved model schema."
        )

    logger.info(
        "Input feature schema validated for %s rows.",
        len(input_features),
    )


def load_supervised_model_artifacts(
    model_directory: Path,
) -> tuple[object, dict[str, Any]]:
    """Load a supervised model pipeline and its metadata."""
    model_path = model_directory / "model.joblib"
    metadata_path = model_directory / "metadata.json"

    pipeline = load_model_pipeline(
        model_path=model_path,
    )

    metadata = load_json_metadata(
        metadata_path=metadata_path,
    )

    required_metadata_fields = {
        "model_name",
        "feature_configuration",
        "feature_columns",
        "decision_threshold",
    }

    missing_fields = required_metadata_fields - set(metadata.keys())

    if missing_fields:
        raise ModelArtifactError(
            f"Supervised model metadata is missing fields: "
            f"{sorted(missing_fields)}"
        )

    if not isinstance(metadata["feature_columns"], list):
        raise ModelArtifactError(
            "Supervised model metadata field 'feature_columns' must be a list."
        )

    if not isinstance(metadata["decision_threshold"], (int, float)):
        raise ModelArtifactError(
            "Supervised model metadata field 'decision_threshold' must be numeric."
        )

    logger.info(
        "Supervised model artifacts loaded from: %s",
        model_directory,
    )

    return pipeline, metadata


def load_anomaly_model_artifacts(
    model_directory: Path,
) -> tuple[object, dict[str, Any]]:
    """Load an anomaly-detection pipeline and its metadata."""
    model_path = model_directory / "model.joblib"
    metadata_path = model_directory / "metadata.json"

    pipeline = load_model_pipeline(
        model_path=model_path,
    )

    metadata = load_json_metadata(
        metadata_path=metadata_path,
    )

    required_metadata_fields = {
        "model_name",
        "feature_configuration",
        "feature_columns",
        "score_direction",
    }

    missing_fields = required_metadata_fields - set(metadata.keys())

    if missing_fields:
        raise ModelArtifactError(
            f"Anomaly model metadata is missing fields: "
            f"{sorted(missing_fields)}"
        )

    if not isinstance(metadata["feature_columns"], list):
        raise ModelArtifactError(
            "Anomaly model metadata field 'feature_columns' must be a list."
        )

    if metadata.get("target_used_during_training") is not False:
        raise ModelArtifactError(
            "Anomaly model metadata must confirm that fraud labels were "
            "not used during training."
        )

    logger.info(
        "Anomaly model artifacts loaded from: %s",
        model_directory,
    )

    return pipeline, metadata


def prepare_model_input(
    input_features: pd.DataFrame,
    expected_feature_columns: list[str],
) -> pd.DataFrame:
    """Prepare a model input matrix with the expected feature order."""
    validate_feature_schema(
        input_features=input_features,
        expected_feature_columns=expected_feature_columns,
    )

    return input_features[expected_feature_columns].copy()