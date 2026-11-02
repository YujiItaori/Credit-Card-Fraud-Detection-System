"""API routes for the fraud-detection service."""


from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status

from src.models.model_loader import ModelArtifactError
from src.models.predict import PredictionInputError

from api.schemas import (
    AnomalyDetectionResponse,
    ApiInformationResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInformationResponse,
    PredictionResponse,
    TransactionFeatures,
)

logger = logging.getLogger(__name__)

router = APIRouter()

API_NAME = "Credit Card Fraud Detection API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "REST API for supervised credit-card fraud prediction and Isolation "
    "Forest anomaly detection."
)
EVALUATION_REPORT_PATH = Path(
    "reports",
    "model_evaluation_report.md",
)

def get_prediction_service(request: Request):
    """Return the prediction service loaded during application startup."""
    prediction_service = getattr(
        request.app.state,
        "prediction_service",
        None,
    )

    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not loaded.",
        )

    return prediction_service


def transaction_to_dataframe(
    transaction: TransactionFeatures,
) -> pd.DataFrame:
    """Convert one validated transaction into a one-row DataFrame."""
    return pd.DataFrame(
        [transaction.model_dump()],
    )


def transactions_to_dataframe(
    transactions: list[TransactionFeatures],
) -> pd.DataFrame:
    """Convert validated transactions into a DataFrame."""
    return pd.DataFrame(
        [transaction.model_dump() for transaction in transactions],
    )


@router.get(
    "/",
    response_model=ApiInformationResponse,
    summary="Get API information",
)
def get_api_information() -> ApiInformationResponse:
    """Return basic information about the API."""
    return ApiInformationResponse(
        api_name=API_NAME,
        api_version=API_VERSION,
        description=API_DESCRIPTION,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check application health",
)
def get_health(request: Request) -> HealthResponse:
    """Return application and model-health information."""
    prediction_service = get_prediction_service(request)

    return HealthResponse(
        status="healthy",
        supervised_model_loaded=(
            prediction_service.supervised_pipeline is not None
        ),
        anomaly_model_loaded=(
            prediction_service.anomaly_pipeline is not None
        ),
    )


@router.get(
    "/model-info",
    response_model=ModelInformationResponse,
    summary="Get loaded model metadata",
)
def get_model_information(request: Request) -> ModelInformationResponse:
    """Return metadata for the loaded supervised and anomaly models."""
    prediction_service = get_prediction_service(request)

    supervised_metadata = prediction_service.supervised_metadata
    anomaly_metadata = prediction_service.anomaly_metadata

    return ModelInformationResponse(
        supervised_model_name=supervised_metadata["model_name"],
        supervised_model_type=supervised_metadata["model_type"],
        feature_configuration=supervised_metadata["feature_configuration"],
        feature_count=len(
            prediction_service.supervised_feature_columns,
        ),
        decision_threshold=(
            prediction_service.supervised_threshold
        ),
        threshold_selection_dataset=supervised_metadata[
            "threshold_selection_dataset"
        ],
        test_set_used_for_selection=supervised_metadata[
            "test_set_used_for_selection"
        ],
        anomaly_model_name=(
            anomaly_metadata["model_name"]
            if anomaly_metadata is not None
            else None
        ),
        anomaly_score_interpretation=(
            anomaly_metadata["score_interpretation"]
            if anomaly_metadata is not None
            else None
        ),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict fraud for one transaction",
)
def predict_single_transaction(
    request: Request,
    transaction: TransactionFeatures,
) -> PredictionResponse:
    """Generate a supervised fraud prediction for one transaction."""
    prediction_service = get_prediction_service(request)

    input_dataframe = transaction_to_dataframe(
        transaction=transaction,
    )

    try:
        prediction_results = prediction_service.predict(
            input_features=input_dataframe,
            include_anomaly_scores=False,
        )
    except PredictionInputError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except ModelArtifactError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error

    prediction_result = prediction_results.iloc[0]

    return PredictionResponse(
        fraud_score=float(
            prediction_result["fraud_score"],
        ),
        fraud_prediction=int(
            prediction_result["fraud_prediction"],
        ),
        decision_threshold=float(
            prediction_result["decision_threshold"],
        ),
        review_flag=int(
            prediction_result["review_flag"],
        ),
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict fraud for multiple transactions",
)
def predict_batch_transactions(
    request: Request,
    batch_request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Generate supervised fraud predictions for multiple transactions."""
    prediction_service = get_prediction_service(request)

    input_dataframe = transactions_to_dataframe(
        transactions=batch_request.transactions,
    )

    include_anomaly_scores = batch_request.include_anomaly_scores

    try:
        prediction_results = prediction_service.predict(
            input_features=input_dataframe,
            include_anomaly_scores=include_anomaly_scores,
        )
    except PredictionInputError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except ModelArtifactError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error

    predictions = []

    for _, prediction_result in prediction_results.iterrows():
        prediction_response = PredictionResponse(
            fraud_score=float(
                prediction_result["fraud_score"],
            ),
            fraud_prediction=int(
                prediction_result["fraud_prediction"],
            ),
            decision_threshold=float(
                prediction_result["decision_threshold"],
            ),
            review_flag=int(
                prediction_result["review_flag"],
            ),
        )

        if include_anomaly_scores:
            prediction_response = PredictionResponse(
                fraud_score=float(
                    prediction_result["fraud_score"],
                ),
                fraud_prediction=int(
                    prediction_result["fraud_prediction"],
                ),
                decision_threshold=float(
                    prediction_result["decision_threshold"],
                ),
                review_flag=int(
                    prediction_result["review_flag"],
                ),
                anomaly_score=float(
                    prediction_result["anomaly_score"],
                ),
                anomaly_prediction=int(
                    prediction_result["anomaly_prediction"],
                ),
            )

        predictions.append(
            prediction_response,
        )

    flagged_count = int(
        prediction_results["review_flag"].sum(),
    )

    transaction_count = len(prediction_results)

    anomaly_count = None
    anomaly_rate = None

    if include_anomaly_scores:
        anomaly_count = int(
            prediction_results["anomaly_prediction"].sum(),
        )

        anomaly_rate = (
            anomaly_count / transaction_count
            if transaction_count > 0
            else 0.0
        )

    return BatchPredictionResponse(
        transaction_count=transaction_count,
        flagged_count=flagged_count,
        flagged_rate=(
            flagged_count / transaction_count
            if transaction_count > 0
            else 0.0
        ),
        anomaly_count=anomaly_count,
        anomaly_rate=anomaly_rate,
        predictions=predictions,
    )


@router.post(
    "/detect-anomaly",
    response_model=AnomalyDetectionResponse,
    summary="Detect anomalies for one transaction",
)
def detect_anomaly(
    request: Request,
    transaction: TransactionFeatures,
) -> AnomalyDetectionResponse:
    """Generate supervised and Isolation Forest anomaly results."""
    prediction_service = get_prediction_service(request)

    if prediction_service.anomaly_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Anomaly model is not loaded.",
        )

    input_dataframe = transaction_to_dataframe(
        transaction=transaction,
    )

    try:
        prediction_results = prediction_service.predict(
            input_features=input_dataframe,
            include_anomaly_scores=True,
        )
    except PredictionInputError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except ModelArtifactError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error

    prediction_result = prediction_results.iloc[0]

    return AnomalyDetectionResponse(
        fraud_score=float(
            prediction_result["fraud_score"],
        ),
        fraud_prediction=int(
            prediction_result["fraud_prediction"],
        ),
        decision_threshold=float(
            prediction_result["decision_threshold"],
        ),
        review_flag=int(
            prediction_result["review_flag"],
        ),
        anomaly_score=float(
            prediction_result["anomaly_score"],
        ),
        anomaly_prediction=int(
            prediction_result["anomaly_prediction"],
        ),
    )

@router.get(
    "/evaluation-report",
    summary="Get the saved model-evaluation report",
)
def get_evaluation_report() -> dict[str, str]:
    """Return the saved model-evaluation report for dashboard display."""
    if not EVALUATION_REPORT_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Evaluation report is not available. Expected file: "
                f"{EVALUATION_REPORT_PATH}."
            ),
        )

    try:
        report_content = EVALUATION_REPORT_PATH.read_text(
            encoding="utf-8",
        )
    except OSError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evaluation report could not be read.",
        ) from error

    return {
        "report": report_content,
    }