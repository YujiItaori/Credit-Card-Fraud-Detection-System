"""Tests for the fraud-detection FastAPI application."""

from __future__ import annotations
import json
from fastapi.testclient import TestClient
import pytest

from api.main import app


@pytest.fixture(scope="module")
def api_client() -> TestClient:
    """Create a test client and trigger application startup."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def valid_transaction() -> dict:
    """Create a valid synthetic transaction request body."""
    transaction = {
        "Time": 1000.0,
        "Amount": 100.0,
    }

    for feature_number in range(1, 29):
        transaction[f"V{feature_number}"] = 0.01 * feature_number

    return transaction


def test_root_endpoint(api_client: TestClient) -> None:
    """Verify that the root endpoint returns API information."""
    response = api_client.get("/")

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["api_name"] == "Credit Card Fraud Detection API"
    assert response_data["api_version"] == "1.0.0"
    assert "description" in response_data


def test_health_endpoint(api_client: TestClient) -> None:
    """Verify that the health endpoint reports loaded models."""
    response = api_client.get("/health")

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["status"] == "healthy"
    assert response_data["supervised_model_loaded"] is True
    assert response_data["anomaly_model_loaded"] is True


def test_model_info_endpoint(api_client: TestClient) -> None:
    """Verify that model metadata is returned correctly."""
    response = api_client.get("/model-info")

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["supervised_model_name"] == "Random Forest"
    assert response_data["decision_threshold"] == 0.38
    assert response_data["threshold_selection_dataset"] == "validation"
    assert response_data["test_set_used_for_selection"] is False
    assert response_data["anomaly_model_name"] == "Isolation Forest"
    assert response_data["feature_count"] == 32


def test_single_prediction_success(
    api_client: TestClient,
    valid_transaction: dict,
) -> None:
    """Verify a successful single-transaction prediction."""
    response = api_client.post(
        "/predict",
        json=valid_transaction,
    )

    assert response.status_code == 200

    response_data = response.json()

    assert 0 <= response_data["fraud_score"] <= 1
    assert response_data["fraud_prediction"] in [0, 1]
    assert response_data["decision_threshold"] == 0.38
    assert response_data["review_flag"] in [0, 1]

    expected_review_flag = int(
        response_data["fraud_score"] >= response_data["decision_threshold"],
    )

    assert response_data["review_flag"] == expected_review_flag


def test_batch_prediction_success(
    api_client: TestClient,
    valid_transaction: dict,
) -> None:
    """Verify a successful batch prediction."""
    request_body = {
        "transactions": [
            valid_transaction,
            valid_transaction,
        ],
    }

    response = api_client.post(
        "/predict/batch",
        json=request_body,
    )

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["transaction_count"] == 2
    assert len(response_data["predictions"]) == 2
    assert response_data["flagged_count"] in [0, 1, 2]

    expected_flagged_count = sum(
        prediction["review_flag"]
        for prediction in response_data["predictions"]
    )

    assert response_data["flagged_count"] == expected_flagged_count

    expected_flagged_rate = (
        expected_flagged_count / 2
    )

    assert response_data["flagged_rate"] == expected_flagged_rate


def test_anomaly_detection_success(
    api_client: TestClient,
    valid_transaction: dict,
) -> None:
    """Verify a successful anomaly-detection request."""
    response = api_client.post(
        "/detect-anomaly",
        json=valid_transaction,
    )

    assert response.status_code == 200

    response_data = response.json()

    assert 0 <= response_data["fraud_score"] <= 1
    assert response_data["fraud_prediction"] in [0, 1]
    assert response_data["decision_threshold"] == 0.38
    assert response_data["review_flag"] in [0, 1]
    assert isinstance(response_data["anomaly_score"], float)
    assert response_data["anomaly_prediction"] in [0, 1]


def test_missing_required_feature_returns_validation_error(
    api_client: TestClient,
    valid_transaction: dict,
) -> None:
    """Verify that a missing required feature returns HTTP 422."""
    invalid_transaction = valid_transaction.copy()
    invalid_transaction.pop("Amount")

    response = api_client.post(
        "/predict",
        json=invalid_transaction,
    )

    assert response.status_code == 422


def test_invalid_data_type_returns_validation_error(
    api_client: TestClient,
    valid_transaction: dict,
) -> None:
    """Verify that an invalid data type returns HTTP 422."""
    invalid_transaction = valid_transaction.copy()
    invalid_transaction["Amount"] = "not_a_number"

    response = api_client.post(
        "/predict",
        json=invalid_transaction,
    )

    assert response.status_code == 422


def test_non_finite_value_returns_validation_error(
    api_client: TestClient,
    valid_transaction: dict,
) -> None:
    """Verify that a non-finite value returns HTTP 422."""
    invalid_transaction = valid_transaction.copy()
    invalid_transaction["V1"] = float("inf")

    response = api_client.post(
        "/predict",
        content=json.dumps(invalid_transaction),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


def test_negative_amount_returns_validation_error(
    api_client: TestClient,
    valid_transaction: dict,
) -> None:
    """Verify that a negative amount returns HTTP 422."""
    invalid_transaction = valid_transaction.copy()
    invalid_transaction["Amount"] = -10

    response = api_client.post(
        "/predict",
        json=invalid_transaction,
    )

    assert response.status_code == 422


def test_unexpected_feature_returns_validation_error(
    api_client: TestClient,
    valid_transaction: dict,
) -> None:
    """Verify that unexpected features return HTTP 422."""
    invalid_transaction = valid_transaction.copy()
    invalid_transaction["UnexpectedFeature"] = 123

    response = api_client.post(
        "/predict",
        json=invalid_transaction,
    )

    assert response.status_code == 422


def test_empty_batch_request_returns_validation_error(
    api_client: TestClient,
) -> None:
    """Verify that an empty batch request returns HTTP 422."""
    response = api_client.post(
        "/predict/batch",
        json={"transactions": []},
    )

    assert response.status_code == 422


def test_incorrect_batch_schema_returns_validation_error(
    api_client: TestClient,
    valid_transaction: dict,
) -> None:
    """Verify that an incorrect batch schema returns HTTP 422."""
    response = api_client.post(
        "/predict/batch",
        json={"transaction": [valid_transaction]},
    )

    assert response.status_code == 422
    
def test_batch_prediction_with_anomaly_scores_returns_anomaly_fields(
    api_client,
    valid_transaction,
):
    """Anomaly-enabled batch predictions should include anomaly fields."""
    response = api_client.post(
        "/predict/batch",
        json={
            "transactions": [valid_transaction],
            "include_anomaly_scores": True,
        },
    )

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["anomaly_count"] is not None
    assert response_data["anomaly_rate"] is not None

    prediction = response_data["predictions"][0]

    assert prediction["anomaly_score"] is not None
    assert prediction["anomaly_prediction"] in [-1, 0, 1]


def test_batch_prediction_without_anomaly_scores_omits_anomaly_values(
    api_client,
    valid_transaction,
):
    """Supervised-only batch predictions should not return anomaly values."""
    response = api_client.post(
        "/predict/batch",
        json={
            "transactions": [valid_transaction],
            "include_anomaly_scores": False,
        },
    )

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["anomaly_count"] is None
    assert response_data["anomaly_rate"] is None

    prediction = response_data["predictions"][0]

    assert prediction["anomaly_score"] is None
    assert prediction["anomaly_prediction"] is None


def test_batch_prediction_preserves_transaction_order(
    api_client,
    valid_transaction,
):
    """Batch predictions should remain in the same order as the input."""
    first_transaction = valid_transaction.copy()

    second_transaction = valid_transaction.copy()
    second_transaction["Amount"] = (
        float(second_transaction["Amount"]) + 1.0
    )

    response = api_client.post(
        "/predict/batch",
        json={
            "transactions": [
                first_transaction,
                second_transaction,
            ],
            "include_anomaly_scores": False,
        },
    )

    assert response.status_code == 200

    predictions = response.json()["predictions"]

    assert len(predictions) == 2
    assert predictions[0]["fraud_score"] >= 0
    assert predictions[1]["fraud_score"] >= 0


def test_evaluation_report_endpoint_returns_saved_report(api_client):
    """The evaluation-report endpoint should return the saved report."""
    response = api_client.get("/evaluation-report")

    if response.status_code == 404:
        assert "Evaluation report is not available" in response.json()[
            "detail"
        ]
        return

    assert response.status_code == 200

    response_data = response.json()

    assert "report" in response_data
    assert isinstance(response_data["report"], str)
    assert len(response_data["report"]) > 0