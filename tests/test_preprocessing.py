"""Tests for preprocessing pipelines and feature preparation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.preprocessing_pipeline import (
    PreprocessingError,
    apply_fitted_scaler,
    create_anomaly_preprocessing_pipeline,
    create_supervised_preprocessing_pipeline,
    prepare_anomaly_features,
    prepare_supervised_features,
    validate_preprocessing_input,
)


@pytest.fixture
def valid_features() -> pd.DataFrame:
    """Create a small valid feature DataFrame."""
    row_count = 10

    feature_data = {
        "Time": np.linspace(0, 1000, row_count),
        "Amount": np.linspace(10, 500, row_count),
    }

    for feature_number in range(1, 29):
        feature_data[f"V{feature_number}"] = np.random.default_rng(
            feature_number
        ).normal(
            loc=0,
            scale=1,
            size=row_count,
        )

    return pd.DataFrame(feature_data)


def test_validate_preprocessing_input_passes(
    valid_features: pd.DataFrame,
) -> None:
    """Verify that valid preprocessing input passes validation."""
    validate_preprocessing_input(
        features=valid_features,
    )


def test_validate_preprocessing_input_rejects_non_dataframe() -> None:
    """Verify that non-DataFrame input raises an error."""
    with pytest.raises(PreprocessingError):
        validate_preprocessing_input(
            features="not_a_dataframe",
        )


def test_validate_preprocessing_input_rejects_empty_dataframe() -> None:
    """Verify that an empty DataFrame raises an error."""
    empty_dataframe = pd.DataFrame()

    with pytest.raises(PreprocessingError):
        validate_preprocessing_input(
            features=empty_dataframe,
        )


def test_validate_preprocessing_input_rejects_target_column(
    valid_features: pd.DataFrame,
) -> None:
    """Verify that target leakage raises an error."""
    invalid_features = valid_features.copy()
    invalid_features["Class"] = 0

    with pytest.raises(PreprocessingError):
        validate_preprocessing_input(
            features=invalid_features,
        )


def test_validate_preprocessing_input_rejects_missing_values(
    valid_features: pd.DataFrame,
) -> None:
    """Verify that missing values raise an error."""
    invalid_features = valid_features.copy()
    invalid_features.loc[0, "Amount"] = np.nan

    with pytest.raises(PreprocessingError):
        validate_preprocessing_input(
            features=invalid_features,
        )


def test_validate_preprocessing_input_rejects_infinite_values(
    valid_features: pd.DataFrame,
) -> None:
    """Verify that infinite values raise an error."""
    invalid_features = valid_features.copy()
    invalid_features.loc[0, "V1"] = np.inf

    with pytest.raises(PreprocessingError):
        validate_preprocessing_input(
            features=invalid_features,
        )


def test_create_supervised_preprocessing_pipeline_with_scaling() -> None:
    """Verify that the supervised pipeline includes a scaler."""
    pipeline = create_supervised_preprocessing_pipeline(
        apply_scaling=True,
    )

    assert pipeline is not None
    assert "standard_scaler" in pipeline.named_steps


def test_create_supervised_preprocessing_pipeline_without_scaling() -> None:
    """Verify that the supervised pipeline can exclude scaling."""
    pipeline = create_supervised_preprocessing_pipeline(
        apply_scaling=False,
    )

    assert pipeline is not None
    assert "standard_scaler" not in pipeline.named_steps


def test_create_anomaly_preprocessing_pipeline_with_scaling() -> None:
    """Verify that the anomaly pipeline includes a scaler."""
    pipeline = create_anomaly_preprocessing_pipeline(
        apply_scaling=True,
    )

    assert pipeline is not None
    assert "standard_scaler" in pipeline.named_steps


def test_prepare_supervised_features_with_scaling(
    valid_features: pd.DataFrame,
) -> None:
    """Verify supervised feature preparation with scaling."""
    prepared_features, scaler = prepare_supervised_features(
        features=valid_features,
        apply_scaling=True,
    )

    assert isinstance(prepared_features, pd.DataFrame)
    assert scaler is not None
    assert len(prepared_features) == len(valid_features)
    assert prepared_features.isnull().sum().sum() == 0

    expected_mean = np.zeros(prepared_features.shape[1])

    np.testing.assert_allclose(
        prepared_features.mean().to_numpy(),
        expected_mean,
        atol=1e-10,
    )


def test_prepare_supervised_features_without_scaling(
    valid_features: pd.DataFrame,
) -> None:
    """Verify supervised feature preparation without scaling."""
    prepared_features, scaler = prepare_supervised_features(
        features=valid_features,
        apply_scaling=False,
    )

    assert isinstance(prepared_features, pd.DataFrame)
    assert scaler is None
    assert len(prepared_features) == len(valid_features)


def test_prepare_anomaly_features_with_scaling(
    valid_features: pd.DataFrame,
) -> None:
    """Verify anomaly feature preparation with scaling."""
    prepared_features, scaler = prepare_anomaly_features(
        features=valid_features,
        apply_scaling=True,
    )

    assert isinstance(prepared_features, pd.DataFrame)
    assert scaler is not None
    assert len(prepared_features) == len(valid_features)
    assert prepared_features.isnull().sum().sum() == 0


def test_prepare_anomaly_features_without_scaling(
    valid_features: pd.DataFrame,
) -> None:
    """Verify anomaly feature preparation without scaling."""
    prepared_features, scaler = prepare_anomaly_features(
        features=valid_features,
        apply_scaling=False,
    )

    assert isinstance(prepared_features, pd.DataFrame)
    assert scaler is None
    assert len(prepared_features) == len(valid_features)


def test_apply_fitted_scaler(
    valid_features: pd.DataFrame,
) -> None:
    """Verify that a fitted scaler can transform new data."""
    prepared_features, scaler = prepare_supervised_features(
        features=valid_features,
        apply_scaling=True,
    )

    scaled_features = apply_fitted_scaler(
        features=valid_features,
        scaler=scaler,
    )

    assert isinstance(scaled_features, pd.DataFrame)
    assert list(scaled_features.columns) == list(
        prepared_features.columns
    )
    assert len(scaled_features) == len(valid_features)


def test_apply_fitted_scaler_requires_scaler(
    valid_features: pd.DataFrame,
) -> None:
    """Verify that applying a missing scaler raises an error."""
    with pytest.raises(PreprocessingError):
        apply_fitted_scaler(
            features=valid_features,
            scaler=None,
        )


def test_apply_fitted_scaler_rejects_schema_mismatch(
    valid_features: pd.DataFrame,
) -> None:
    """Verify that a feature-schema mismatch raises an error."""
    _, scaler = prepare_supervised_features(
        features=valid_features,
        apply_scaling=True,
    )

    mismatched_features = valid_features.drop(
        columns=["V1"],
    )

    with pytest.raises(PreprocessingError):
        apply_fitted_scaler(
            features=mismatched_features,
            scaler=scaler,
        )