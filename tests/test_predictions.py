"""Tests for the fraud prediction workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.predict import (
    FraudPredictionService,
    PredictionInputError,
    create_prediction_service,
)

SUPERVISED_MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "supervised"
    / "final_supervised_model"
)

ANOMALY_MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "anomaly"
    / "baseline_anomaly_models"
    / "isolation_forest_baseline"
)

REQUIRED_COLUMNS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]

EXPECTED_PREDICTION_COLUMNS = [
    "fraud_score",
    "fraud_prediction",
    "decision_threshold",
    "review_flag",
]

EXPECTED_ANOMALY_COLUMNS = [
    "anomaly_score",
    "anomaly_prediction",
]


@pytest.fixture(scope="module")
def prediction_service() -> FraudPredictionService:
    """Load the final supervised prediction service."""
    return create_prediction_service(
        supervised_model_directory=SUPERVISED_MODEL_DIRECTORY,
        anomaly_model_directory=None,
    )


@pytest.fixture(scope="module")
def anomaly_service() -> FraudPredictionService:
    """Load the prediction service with anomaly scoring enabled."""
    return create_prediction_service(
        supervised_model_directory=SUPERVISED_MODEL_DIRECTORY,
        anomaly_model_directory=ANOMALY_MODEL_DIRECTORY,
    )


@pytest.fixture(scope="module")
def valid_input_dataframe() -> pd.DataFrame:
    """Create a small valid prediction input DataFrame."""
    row_count = 10

    input_data = {
        "Time": np.linspace(0, 1000, row_count),
        "Amount": np.linspace(10, 500, row_count),
    }

    for feature_number in range(1, 29):
        input_data[f"V{feature_number}"] = np.random.default_rng(
            feature_number
        ).normal(
            loc=0,
            scale=1,
            size=row_count,
        )

    return pd.DataFrame(input_data)


def test_model_artifacts_exist() -> None:
    """Verify that final supervised model artifacts exist."""
    assert (SUPERVISED_MODEL_DIRECTORY / "model.joblib").exists()
    assert (SUPERVISED_MODEL_DIRECTORY / "metadata.json").exists()


def test_prediction_service_loads(
    prediction_service: FraudPredictionService,
    valid_input_dataframe: pd.DataFrame,
) -> None:
    """Verify that the prediction service loads and generates predictions."""
    assert prediction_service is not None

    prediction_results = prediction_service.predict(
        input_features=valid_input_dataframe,
        include_anomaly_scores=False,
    )

    assert len(prediction_results) == len(valid_input_dataframe)
    assert "fraud_score" in prediction_results.columns
    assert "decision_threshold" in prediction_results.columns

    decision_threshold = prediction_results["decision_threshold"].iloc[0]

    assert 0 < decision_threshold < 1


def test_prediction_output_schema(
    prediction_service: FraudPredictionService,
    valid_input_dataframe: pd.DataFrame,
) -> None:
    """Verify that predictions contain the required output columns."""
    prediction_results = prediction_service.predict(
        input_features=valid_input_dataframe,
        include_anomaly_scores=False,
    )

    for column in EXPECTED_PREDICTION_COLUMNS:
        assert column in prediction_results.columns

    assert len(prediction_results) == len(valid_input_dataframe)


def test_prediction_values_are_valid(
    prediction_service: FraudPredictionService,
    valid_input_dataframe: pd.DataFrame,
) -> None:
    """Verify that prediction values are valid."""
    prediction_results = prediction_service.predict(
        input_features=valid_input_dataframe,
        include_anomaly_scores=False,
    )

    assert prediction_results["fraud_score"].between(0, 1).all()
    assert prediction_results["fraud_prediction"].isin([0, 1]).all()
    assert prediction_results["review_flag"].isin([0, 1]).all()


def test_review_flag_matches_threshold(
    prediction_service: FraudPredictionService,
    valid_input_dataframe: pd.DataFrame,
) -> None:
    """Verify that review flags follow the decision threshold."""
    prediction_results = prediction_service.predict(
        input_features=valid_input_dataframe,
        include_anomaly_scores=False,
    )

    expected_flags = (
        prediction_results["fraud_score"]
        >= prediction_results["decision_threshold"]
    ).astype(int)

    assert (
        prediction_results["review_flag"] == expected_flags
    ).all()


def test_anomaly_prediction_output_schema(
    anomaly_service: FraudPredictionService,
    valid_input_dataframe: pd.DataFrame,
) -> None:
    """Verify that anomaly predictions contain required columns."""
    prediction_results = anomaly_service.predict(
        input_features=valid_input_dataframe,
        include_anomaly_scores=True,
    )

    for column in EXPECTED_PREDICTION_COLUMNS:
        assert column in prediction_results.columns

    for column in EXPECTED_ANOMALY_COLUMNS:
        assert column in prediction_results.columns


def test_missing_column_raises_error(
    prediction_service: FraudPredictionService,
    valid_input_dataframe: pd.DataFrame,
) -> None:
    """Verify that a missing required column raises an error."""
    invalid_input = valid_input_dataframe.drop(columns=["Amount"])

    with pytest.raises(PredictionInputError):
        prediction_service.predict(
            input_features=invalid_input,
            include_anomaly_scores=False,
        )


def test_empty_input_raises_error(
    prediction_service: FraudPredictionService,
) -> None:
    """Verify that an empty input DataFrame raises an error."""
    empty_input = pd.DataFrame(columns=REQUIRED_COLUMNS)

    with pytest.raises(PredictionInputError):
        prediction_service.predict(
            input_features=empty_input,
            include_anomaly_scores=False,
        )


def test_missing_value_raises_error(
    prediction_service: FraudPredictionService,
    valid_input_dataframe: pd.DataFrame,
) -> None:
    """Verify that missing values raise an error."""
    invalid_input = valid_input_dataframe.copy()
    invalid_input.loc[0, "Amount"] = np.nan

    with pytest.raises(PredictionInputError):
        prediction_service.predict(
            input_features=invalid_input,
            include_anomaly_scores=False,
        )


def test_negative_amount_raises_error(
    prediction_service: FraudPredictionService,
    valid_input_dataframe: pd.DataFrame,
) -> None:
    """Verify that a negative transaction amount raises an error."""
    invalid_input = valid_input_dataframe.copy()
    invalid_input.loc[0, "Amount"] = -10

    with pytest.raises(PredictionInputError):
        prediction_service.predict(
            input_features=invalid_input,
            include_anomaly_scores=False,
        )