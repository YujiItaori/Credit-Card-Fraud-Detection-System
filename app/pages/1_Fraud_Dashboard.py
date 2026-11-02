"""Fraud Dashboard page for the Streamlit application."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.validate_data import (
    DataValidationError,
    validate_prediction_dataset,
)
from src.models.predict import (
    FraudPredictionService,
    PredictionInputError,
    create_prediction_service,
)

st.set_page_config(
    page_title="Fraud Dashboard",
    page_icon="📊",
    layout="wide",
)

DEFAULT_SUPERVISED_MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "supervised"
    / "final_supervised_model"
)

DEFAULT_ANOMALY_MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "anomaly"
    / "baseline_anomaly_models"
    / "isolation_forest_baseline"
)


def initialize_page() -> None:
    """Display the page title and description."""
    st.title("📊 Fraud Dashboard")
    st.markdown(
        """
        Upload transaction data to generate fraud predictions and review
        the overall fraud-risk profile.
        """
    )


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_prediction_service(
    include_anomaly_scores: bool,
) -> FraudPredictionService:
    """Load and cache the prediction service."""
    anomaly_model_directory = (
        DEFAULT_ANOMALY_MODEL_DIRECTORY
        if include_anomaly_scores
        else None
    )

    return create_prediction_service(
        supervised_model_directory=(
            DEFAULT_SUPERVISED_MODEL_DIRECTORY
        ),
        anomaly_model_directory=anomaly_model_directory,
    )


def read_and_validate_input(
    uploaded_file,
) -> pd.DataFrame:
    """Read and validate the uploaded CSV file."""
    if uploaded_file is None:
        st.info(
            "Upload a CSV file containing transaction data to begin."
        )
        st.stop()

    if not uploaded_file.name.lower().endswith(".csv"):
        st.error("Please upload a CSV file.")
        st.stop()

    try:
        input_dataframe = pd.read_csv(uploaded_file)
        validate_prediction_dataset(
            dataframe=input_dataframe,
        )
    except (
        ValueError,
        DataValidationError,
        PredictionInputError,
    ) as error:
        st.error(f"Input validation failed: {error}")
        st.stop()

    return input_dataframe


def generate_predictions(
    input_dataframe: pd.DataFrame,
    include_anomaly_scores: bool,
) -> pd.DataFrame:
    """Generate predictions using the prediction service."""
    prediction_service = load_prediction_service(
        include_anomaly_scores=include_anomaly_scores,
    )

    try:
        with st.spinner("Generating fraud predictions..."):
            prediction_results = prediction_service.predict(
                input_features=input_dataframe,
                include_anomaly_scores=include_anomaly_scores,
            )
    except (
        ValueError,
        DataValidationError,
        PredictionInputError,
    ) as error:
        st.error(f"Prediction failed: {error}")
        st.stop()

    return prediction_results


def display_key_metrics(
    prediction_results: pd.DataFrame,
) -> None:
    """Display high-level prediction metrics."""
    total_transactions = len(prediction_results)

    flagged_transactions = int(
        prediction_results["review_flag"].sum(),
    )

    flagged_percentage = (
        flagged_transactions / total_transactions
        if total_transactions > 0
        else 0.0
    )

    average_fraud_score = float(
        prediction_results["fraud_score"].mean(),
    )

    maximum_fraud_score = float(
        prediction_results["fraud_score"].max(),
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label="Transactions processed",
        value=f"{total_transactions:,}",
    )

    col2.metric(
        label="Transactions flagged",
        value=f"{flagged_transactions:,}",
    )

    col3.metric(
        label="Review-flag rate",
        value=f"{flagged_percentage:.4%}",
    )

    col4.metric(
        label="Average fraud score",
        value=f"{average_fraud_score:.4f}",
    )

    st.caption(
        f"Maximum fraud score: {maximum_fraud_score:.4f}"
    )


def display_score_distribution(
    prediction_results: pd.DataFrame,
) -> None:
    """Display the fraud-score distribution."""
    st.subheader("Fraud-score distribution")

    score_histogram = px.histogram(
        prediction_results,
        x="fraud_score",
        nbins=50,
        title="Distribution of predicted fraud scores",
        labels={
            "fraud_score": "Predicted fraud score",
            "count": "Transaction count",
        },
    )

    score_histogram.update_layout(
        showlegend=False,
    )

    st.plotly_chart(
        score_histogram,
        use_container_width=True,
    )


def display_flagged_transactions(
    prediction_results: pd.DataFrame,
) -> None:
    """Display transactions recommended for review."""
    st.subheader("Transactions recommended for review")

    flagged_results = prediction_results[
        prediction_results["review_flag"] == 1
    ].copy()

    if flagged_results.empty:
        st.success(
            "No transactions were flagged for review."
        )
        return

    display_columns = [
        "fraud_score",
        "fraud_prediction",
        "decision_threshold",
        "review_flag",
    ]

    if "anomaly_score" in flagged_results.columns:
        display_columns.extend(
            [
                "anomaly_score",
                "anomaly_prediction",
            ]
        )

    st.dataframe(
        flagged_results[display_columns],
        use_container_width=True,
    )

    csv_data = flagged_results.to_csv(
        index=False,
    ).encode("utf-8")

    st.download_button(
        label="Download flagged transactions",
        data=csv_data,
        file_name="flagged_transactions.csv",
        mime="text/csv",
    )


def display_download_option(
    prediction_results: pd.DataFrame,
) -> None:
    """Provide a download option for all prediction results."""
    st.subheader("Download all predictions")

    csv_data = prediction_results.to_csv(
        index=False,
    ).encode("utf-8")

    st.download_button(
        label="Download complete prediction results",
        data=csv_data,
        file_name="fraud_dashboard_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    """Run the Fraud Dashboard page."""
    initialize_page()

    include_anomaly_scores = st.checkbox(
        "Include Isolation Forest anomaly scores",
        value=False,
    )

    uploaded_file = st.file_uploader(
        label="Upload transaction CSV",
        type=["csv"],
    )

    input_dataframe = read_and_validate_input(
        uploaded_file=uploaded_file,
    )

    prediction_results = generate_predictions(
        input_dataframe=input_dataframe,
        include_anomaly_scores=include_anomaly_scores,
    )

    display_key_metrics(
        prediction_results=prediction_results,
    )

    display_score_distribution(
        prediction_results=prediction_results,
    )

    display_flagged_transactions(
        prediction_results=prediction_results,
    )

    display_download_option(
        prediction_results=prediction_results,
    )


if __name__ == "__main__":
    main()