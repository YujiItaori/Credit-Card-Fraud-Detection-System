"""Pydantic schemas for the fraud-detection API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TransactionFeatures(BaseModel):
    """Features for one credit-card transaction."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    Time: float = Field(
        ...,
        description="Seconds elapsed between this transaction and the first transaction.",
    )

    V1: float = Field(
        ...,
        description="Anonymized PCA feature V1.",
    )
    V2: float = Field(
        ...,
        description="Anonymized PCA feature V2.",
    )
    V3: float = Field(
        ...,
        description="Anonymized PCA feature V3.",
    )
    V4: float = Field(
        ...,
        description="Anonymized PCA feature V4.",
    )
    V5: float = Field(
        ...,
        description="Anonymized PCA feature V5.",
    )
    V6: float = Field(
        ...,
        description="Anonymized PCA feature V6.",
    )
    V7: float = Field(
        ...,
        description="Anonymized PCA feature V7.",
    )
    V8: float = Field(
        ...,
        description="Anonymized PCA feature V8.",
    )
    V9: float = Field(
        ...,
        description="Anonymized PCA feature V9.",
    )
    V10: float = Field(
        ...,
        description="Anonymized PCA feature V10.",
    )
    V11: float = Field(
        ...,
        description="Anonymized PCA feature V11.",
    )
    V12: float = Field(
        ...,
        description="Anonymized PCA feature V12.",
    )
    V13: float = Field(
        ...,
        description="Anonymized PCA feature V13.",
    )
    V14: float = Field(
        ...,
        description="Anonymized PCA feature V14.",
    )
    V15: float = Field(
        ...,
        description="Anonymized PCA feature V15.",
    )
    V16: float = Field(
        ...,
        description="Anonymized PCA feature V16.",
    )
    V17: float = Field(
        ...,
        description="Anonymized PCA feature V17.",
    )
    V18: float = Field(
        ...,
        description="Anonymized PCA feature V18.",
    )
    V19: float = Field(
        ...,
        description="Anonymized PCA feature V19.",
    )
    V20: float = Field(
        ...,
        description="Anonymized PCA feature V20.",
    )
    V21: float = Field(
        ...,
        description="Anonymized PCA feature V21.",
    )
    V22: float = Field(
        ...,
        description="Anonymized PCA feature V22.",
    )
    V23: float = Field(
        ...,
        description="Anonymized PCA feature V23.",
    )
    V24: float = Field(
        ...,
        description="Anonymized PCA feature V24.",
    )
    V25: float = Field(
        ...,
        description="Anonymized PCA feature V25.",
    )
    V26: float = Field(
        ...,
        description="Anonymized PCA feature V26.",
    )
    V27: float = Field(
        ...,
        description="Anonymized PCA feature V27.",
    )
    V28: float = Field(
        ...,
        description="Anonymized PCA feature V28.",
    )

    Amount: float = Field(
        ...,
        ge=0,
        description="Transaction amount. Must be non-negative.",
    )

class PredictionResponse(BaseModel):
    fraud_score: float
    fraud_prediction: int
    decision_threshold: float
    review_flag: int
    anomaly_score: float | None = None
    anomaly_prediction: int | None = None


class BatchPredictionRequest(BaseModel):
    """Request containing multiple transactions."""

    transactions: list[TransactionFeatures] = Field(
        ...,
        min_length=1,
        description="List of transactions to classify.",
    )

    include_anomaly_scores: bool = Field(
        default=False,
        description=(
            "Whether to include Isolation Forest anomaly scores in the "
            "batch response."
        ),
    )


class BatchPredictionResponse(BaseModel):
    """Response containing predictions for multiple transactions."""

    transaction_count: int = Field(
        ...,
        description="Number of transactions processed.",
    )

    flagged_count: int = Field(
        ...,
        description="Number of transactions flagged for review.",
    )

    flagged_rate: float = Field(
        ...,
        description="Proportion of transactions flagged for review.",
    )

    anomaly_count: int | None = Field(
        default=None,
        description=(
            "Number of transactions classified as anomalies. This is "
            "returned only when anomaly scoring is requested."
        ),
    )

    anomaly_rate: float | None = Field(
        default=None,
        description=(
            "Proportion of transactions classified as anomalies. This is "
            "returned only when anomaly scoring is requested."
        ),
    )

    predictions: list[PredictionResponse] = Field(
        ...,
        description="Prediction result for each transaction.",
    )


class AnomalyDetectionResponse(BaseModel):
    """Response for supervised prediction and anomaly detection."""

    fraud_score: float = Field(
        ...,
        description="Supervised fraud probability.",
    )

    fraud_prediction: int = Field(
        ...,
        description="Supervised fraud prediction.",
    )

    decision_threshold: float = Field(
        ...,
        description="Supervised decision threshold.",
    )

    review_flag: int = Field(
        ...,
        description="Supervised review flag.",
    )

    anomaly_score: float = Field(
        ...,
        description=(
            "Isolation Forest anomaly score. Higher values indicate more "
            "anomalous transactions."
        ),
    )

    anomaly_prediction: int = Field(
        ...,
        description="1 when the transaction is classified as an anomaly.",
    )


class ModelInformationResponse(BaseModel):
    """Metadata for the loaded supervised and anomaly models."""

    supervised_model_name: str = Field(
        ...,
        description="Name of the loaded supervised model.",
    )

    supervised_model_type: str = Field(
        ...,
        description="Type of the loaded supervised model.",
    )

    feature_configuration: str = Field(
        ...,
        description="Feature configuration used by the supervised model.",
    )

    feature_count: int = Field(
        ...,
        description="Number of model input features.",
    )

    decision_threshold: float = Field(
        ...,
        description="Supervised decision threshold.",
    )

    threshold_selection_dataset: str = Field(
        ...,
        description="Dataset used to select the decision threshold.",
    )

    test_set_used_for_selection: bool = Field(
        ...,
        description="Whether the test set was used for threshold selection.",
    )

    anomaly_model_name: str | None = Field(
        default=None,
        description="Name of the loaded anomaly model, if available.",
    )

    anomaly_score_interpretation: str | None = Field(
        default=None,
        description="Interpretation of the anomaly score, if available.",
    )


class HealthResponse(BaseModel):
    """Application and model-health response."""

    status: str = Field(
        ...,
        description="Overall application health status.",
    )

    supervised_model_loaded: bool = Field(
        ...,
        description="Whether the supervised model is loaded.",
    )

    anomaly_model_loaded: bool = Field(
        ...,
        description="Whether the anomaly model is loaded.",
    )


class ApiInformationResponse(BaseModel):
    """Basic API information."""

    api_name: str = Field(
        ...,
        description="Name of the API.",
    )

    api_version: str = Field(
        ...,
        description="API version.",
    )

    description: str = Field(
        ...,
        description="Short description of the API.",
    )