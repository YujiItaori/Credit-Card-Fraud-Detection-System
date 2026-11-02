"""Shared HTTP client for Streamlit-to-FastAPI communication."""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"

DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0

API_BASE_URL_ENVIRONMENT_VARIABLE = "FRAUD_API_BASE_URL"


class ApiClientError(Exception):
    """Raised when the FastAPI service cannot be reached or returns an error."""


def get_api_base_url() -> str:
    """Return the configured FastAPI base URL."""
    return os.getenv(
        API_BASE_URL_ENVIRONMENT_VARIABLE,
        DEFAULT_API_BASE_URL,
    ).rstrip("/")


def get_api_request_timeout() -> float:
    """Return the configured request timeout in seconds."""
    timeout_value = os.getenv(
        "FRAUD_API_TIMEOUT_SECONDS",
        str(DEFAULT_REQUEST_TIMEOUT_SECONDS),
    )

    try:
        timeout = float(timeout_value)
    except ValueError as error:
        raise ApiClientError(
            "FRAUD_API_TIMEOUT_SECONDS must be a numeric value.",
        ) from error

    if timeout <= 0:
        raise ApiClientError(
            "FRAUD_API_TIMEOUT_SECONDS must be greater than zero.",
        )

    return timeout


def get_api_health() -> dict[str, Any]:
    """Return health information from the FastAPI service."""
    url = f"{get_api_base_url()}/health"

    try:
        response = requests.get(
            url,
            timeout=get_api_request_timeout(),
        )
    except requests.ConnectionError as error:
        raise ApiClientError(
            "Unable to connect to the fraud-detection API. "
            "Ensure it is running with: uvicorn api.main:app --reload"
        ) from error
    except requests.Timeout as error:
        raise ApiClientError(
            "The fraud-detection API health request timed out."
        ) from error

    if response.status_code != 200:
        raise ApiClientError(
            f"The fraud-detection API health check failed with status "
            f"{response.status_code}."
        )

    return response.json()


def predict_single_transaction(
    transaction: dict[str, Any],
) -> dict[str, Any]:
    """Send one transaction to the supervised prediction endpoint."""
    url = f"{get_api_base_url()}/predict"

    try:
        response = requests.post(
            url,
            json=transaction,
            timeout=get_api_request_timeout(),
        )
    except requests.ConnectionError as error:
        raise ApiClientError(
            "Unable to connect to the fraud-detection API. "
            "Ensure it is running with: uvicorn api.main:app --reload"
        ) from error
    except requests.Timeout as error:
        raise ApiClientError(
            "The single-transaction prediction request timed out."
        ) from error

    return handle_api_response(
        response=response,
        error_context="Single-transaction prediction",
    )


def predict_batch_transactions(
    transactions: list[dict[str, Any]],
    include_anomaly_scores: bool = False,
) -> dict[str, Any]:
    """Send multiple transactions to the batch-prediction endpoint."""
    url = f"{get_api_base_url()}/predict/batch"

    request_body = {
        "transactions": transactions,
        "include_anomaly_scores": include_anomaly_scores,
    }

    try:
        response = requests.post(
            url,
            json=request_body,
            timeout=get_api_request_timeout(),
        )
    except requests.ConnectionError as error:
        raise ApiClientError(
            "Unable to connect to the fraud-detection API. "
            "Ensure it is running with: uvicorn api.main:app --reload"
        ) from error
    except requests.Timeout as error:
        raise ApiClientError(
            "The batch-prediction request timed out."
        ) from error

    return handle_api_response(
        response=response,
        error_context="Batch prediction",
    )


def detect_anomaly(
    transaction: dict[str, Any],
) -> dict[str, Any]:
    """Send one transaction to the anomaly-detection endpoint."""
    url = f"{get_api_base_url()}/detect-anomaly"

    try:
        response = requests.post(
            url,
            json=transaction,
            timeout=get_api_request_timeout(),
        )
    except requests.ConnectionError as error:
        raise ApiClientError(
            "Unable to connect to the fraud-detection API. "
            "Ensure it is running with: uvicorn api.main:app --reload"
        ) from error
    except requests.Timeout as error:
        raise ApiClientError(
            "The anomaly-detection request timed out."
        ) from error

    return handle_api_response(
        response=response,
        error_context="Anomaly detection",
    )


def handle_api_response(
    response: requests.Response,
    error_context: str,
) -> dict[str, Any]:
    """Validate an API response and return its JSON payload."""
    if response.status_code == 422:
        detail = extract_error_detail(response)

        raise ApiClientError(
            f"{error_context} request was rejected because the input data "
            f"is invalid: {detail}"
        )

    if response.status_code == 400:
        detail = extract_error_detail(response)

        raise ApiClientError(
            f"{error_context} request failed because the input data is "
            f"invalid: {detail}"
        )

    if response.status_code == 503:
        detail = extract_error_detail(response)

        raise ApiClientError(
            f"{error_context} is unavailable: {detail}"
        )

    if response.status_code >= 500:
        raise ApiClientError(
            f"{error_context} failed because the fraud-detection API "
            f"returned an internal server error "
            f"(status {response.status_code})."
        )

    if response.status_code != 200:
        raise ApiClientError(
            f"{error_context} failed with unexpected status "
            f"{response.status_code}."
        )

    return response.json()


def extract_error_detail(
    response: requests.Response,
) -> str:
    """Extract a readable error message from an API error response."""
    try:
        response_data = response.json()
    except ValueError:
        return response.text

    detail = response_data.get(
        "detail",
        response_data,
    )

    if isinstance(detail, list):
        return "; ".join(
            str(item)
            for item in detail
        )

    return str(detail)

def get_evaluation_report() -> dict[str, Any]:
    """Return the saved model-evaluation report from the API."""
    url = f"{get_api_base_url()}/evaluation-report"

    try:
        response = requests.get(
            url,
            timeout=get_api_request_timeout(),
        )
    except requests.ConnectionError as error:
        raise ApiClientError(
            "Unable to connect to the fraud-detection API. "
            "Ensure it is running with: uvicorn api.main:app --reload"
        ) from error
    except requests.Timeout as error:
        raise ApiClientError(
            "The evaluation-report request timed out."
        ) from error

    return handle_api_response(
        response=response,
        error_context="Model evaluation report",
    )