"""Tests for the shared Streamlit API client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.api_client import (
    ApiClientError,
    detect_anomaly,
    get_api_base_url,
    get_api_health,
    get_evaluation_report,
    get_api_request_timeout,
    handle_api_response,
    predict_batch_transactions,
    predict_single_transaction,
)


def test_default_api_base_url():
    """The client should use the local API URL by default."""
    with patch.dict(
        "os.environ",
        {},
        clear=True,
    ):
        assert get_api_base_url() == "http://127.0.0.1:8000"


def test_api_base_url_can_be_configured():
    """The client should read the API base URL from the environment."""
    with patch.dict(
        "os.environ",
        {
            "FRAUD_API_BASE_URL": "http://localhost:9000/"
        },
    ):
        assert get_api_base_url() == "http://localhost:9000"


def test_default_request_timeout():
    """The client should use a 30-second timeout by default."""
    with patch.dict(
        "os.environ",
        {},
        clear=True,
    ):
        assert get_api_request_timeout() == 30.0


def test_invalid_request_timeout_raises_error():
    """Non-numeric timeout configuration should raise a clear error."""
    with patch.dict(
        "os.environ",
        {
            "FRAUD_API_TIMEOUT_SECONDS": "not-a-number"
        },
    ):
        with pytest.raises(
            ApiClientError,
            match="must be a numeric value",
        ):
            get_api_request_timeout()


def test_zero_request_timeout_raises_error():
    """Zero or negative timeout configuration should raise an error."""
    with patch.dict(
        "os.environ",
        {
            "FRAUD_API_TIMEOUT_SECONDS": "0"
        },
    ):
        with pytest.raises(
            ApiClientError,
            match="must be greater than zero",
        ):
            get_api_request_timeout()


def test_get_api_health_success():
    """The health function should return the API JSON response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "healthy",
        "supervised_model_loaded": True,
        "anomaly_model_loaded": True,
    }

    with patch(
        "app.api_client.requests.get",
        return_value=mock_response,
    ) as mock_get:
        health = get_api_health()

    mock_get.assert_called_once_with(
        "http://127.0.0.1:8000/health",
        timeout=30.0,
    )

    assert health["status"] == "healthy"
    assert health["supervised_model_loaded"] is True
    assert health["anomaly_model_loaded"] is True


def test_get_api_health_connection_error():
    """Connection failures should raise a user-friendly API client error."""
    with patch(
        "app.api_client.requests.get",
        side_effect=requests.ConnectionError(),
    ):
        with pytest.raises(
            ApiClientError,
            match="Unable to connect to the fraud-detection API",
        ):
            get_api_health()


def test_get_api_health_timeout_error():
    """Timeout failures should raise a user-friendly API client error."""
    with patch(
        "app.api_client.requests.get",
        side_effect=requests.Timeout(),
    ):
        with pytest.raises(
            ApiClientError,
            match="health request timed out",
        ):
            get_api_health()


def test_predict_single_transaction_success():
    """Single prediction should POST the transaction to /predict."""
    transaction = {
        "Time": 0.0,
        "Amount": 100.0,
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "fraud_score": 0.12,
        "fraud_prediction": 0,
        "decision_threshold": 0.38,
        "review_flag": 0,
    }

    with patch(
        "app.api_client.requests.post",
        return_value=mock_response,
    ) as mock_post:
        prediction = predict_single_transaction(
            transaction=transaction,
        )

    mock_post.assert_called_once_with(
        "http://127.0.0.1:8000/predict",
        json=transaction,
        timeout=30.0,
    )

    assert prediction["fraud_score"] == 0.12
    assert prediction["fraud_prediction"] == 0
    assert prediction["decision_threshold"] == 0.38
    assert prediction["review_flag"] == 0


def test_predict_batch_transactions_without_anomaly_scores():
    """Batch prediction should send the configured anomaly-score flag."""
    transactions = [
        {
            "Time": 0.0,
            "Amount": 100.0,
        },
        {
            "Time": 1.0,
            "Amount": 200.0,
        },
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "transaction_count": 2,
        "flagged_count": 0,
        "flagged_rate": 0.0,
        "anomaly_count": None,
        "anomaly_rate": None,
        "predictions": [],
    }

    expected_request_body = {
        "transactions": transactions,
        "include_anomaly_scores": False,
    }

    with patch(
        "app.api_client.requests.post",
        return_value=mock_response,
    ) as mock_post:
        response = predict_batch_transactions(
            transactions=transactions,
            include_anomaly_scores=False,
        )

    mock_post.assert_called_once_with(
        "http://127.0.0.1:8000/predict/batch",
        json=expected_request_body,
        timeout=30.0,
    )

    assert response["transaction_count"] == 2
    assert response["anomaly_count"] is None
    assert response["anomaly_rate"] is None


def test_predict_batch_transactions_with_anomaly_scores():
    """Batch prediction should request anomaly scores when enabled."""
    transactions = [
        {
            "Time": 0.0,
            "Amount": 100.0,
        },
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "transaction_count": 1,
        "flagged_count": 0,
        "flagged_rate": 0.0,
        "anomaly_count": 1,
        "anomaly_rate": 1.0,
        "predictions": [],
    }

    expected_request_body = {
        "transactions": transactions,
        "include_anomaly_scores": True,
    }

    with patch(
        "app.api_client.requests.post",
        return_value=mock_response,
    ) as mock_post:
        response = predict_batch_transactions(
            transactions=transactions,
            include_anomaly_scores=True,
        )

    mock_post.assert_called_once_with(
        "http://127.0.0.1:8000/predict/batch",
        json=expected_request_body,
        timeout=30.0,
    )

    assert response["anomaly_count"] == 1
    assert response["anomaly_rate"] == 1.0


def test_predict_batch_connection_error():
    """Batch connection failures should raise a clear API client error."""
    with patch(
        "app.api_client.requests.post",
        side_effect=requests.ConnectionError(),
    ):
        with pytest.raises(
            ApiClientError,
            match="Unable to connect to the fraud-detection API",
        ):
            predict_batch_transactions(
                transactions=[],
            )


def test_detect_anomaly_success():
    """Anomaly detection should POST the transaction to /detect-anomaly."""
    transaction = {
        "Time": 0.0,
        "Amount": 100.0,
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "fraud_score": 0.42,
        "fraud_prediction": 1,
        "decision_threshold": 0.38,
        "review_flag": 1,
        "anomaly_score": -0.65,
        "anomaly_prediction": -1,
    }

    with patch(
        "app.api_client.requests.post",
        return_value=mock_response,
    ) as mock_post:
        prediction = detect_anomaly(
            transaction=transaction,
        )

    mock_post.assert_called_once_with(
        "http://127.0.0.1:8000/detect-anomaly",
        json=transaction,
        timeout=30.0,
    )

    assert prediction["fraud_prediction"] == 1
    assert prediction["review_flag"] == 1
    assert prediction["anomaly_prediction"] == -1


def test_handle_api_response_validation_error():
    """HTTP 422 responses should expose the API validation details."""
    mock_response = MagicMock()
    mock_response.status_code = 422
    mock_response.json.return_value = {
        "detail": [
            {
                "loc": ["body", "Amount"],
                "msg": "Input should be greater than or equal to 0",
            }
        ]
    }

    with pytest.raises(
        ApiClientError,
        match="input data is invalid",
    ):
        handle_api_response(
            response=mock_response,
            error_context="Batch prediction",
        )


def test_handle_api_response_bad_request_error():
    """HTTP 400 responses should expose the API error details."""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {
        "detail": "Input contains missing values."
    }

    with pytest.raises(
        ApiClientError,
        match="Input contains missing values",
    ):
        handle_api_response(
            response=mock_response,
            error_context="Single-transaction prediction",
        )


def test_handle_api_response_service_unavailable_error():
    """HTTP 503 responses should indicate that the service is unavailable."""
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.json.return_value = {
        "detail": "Anomaly model is not loaded."
    }

    with pytest.raises(
        ApiClientError,
        match="Anomaly model is not loaded",
    ):
        handle_api_response(
            response=mock_response,
            error_context="Anomaly detection",
        )


def test_handle_api_response_internal_server_error():
    """HTTP 5xx responses should raise an internal-server-error message."""
    mock_response = MagicMock()
    mock_response.status_code = 500

    with pytest.raises(
        ApiClientError,
        match="internal server error",
    ):
        handle_api_response(
            response=mock_response,
            error_context="Batch prediction",
        )


def test_handle_api_response_unexpected_status_error():
    """Unexpected non-200 statuses should raise a clear error."""
    mock_response = MagicMock()
    mock_response.status_code = 404

    with pytest.raises(
        ApiClientError,
        match="unexpected status 404",
    ):
        handle_api_response(
            response=mock_response,
            error_context="Single-transaction prediction",
        )
    
def test_get_evaluation_report_success():
    """The client should call and parse the evaluation-report endpoint."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "report": "# Supervised vs Anomaly Detection Evaluation"
    }

    with patch(
        "app.api_client.requests.get",
        return_value=mock_response,
    ) as mock_get:
        response = get_evaluation_report()

    mock_get.assert_called_once_with(
        "http://127.0.0.1:8000/evaluation-report",
        timeout=30.0,
    )

    assert response["report"].startswith(
        "# Supervised vs Anomaly Detection Evaluation"
    )