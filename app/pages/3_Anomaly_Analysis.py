"""Anomaly Analysis page for the Streamlit application."""

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
    page_title="Anomaly Analysis",
    page_icon="🚨",
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
    st.title("🚨 Anomaly Analysis")
    st.markdown(
        """
        Upload transaction data to generate Isolation Forest anomaly scores
        and inspect transactions identified as statistical anomalies.
        """
    )


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_prediction_service() -> FraudPredictionService:
    """Load and cache the prediction service with anomaly model."""
    return create_prediction_service(
        supervised_model_directory=(
            DEFAULT_SUPERVISED_MODEL_DIRECTORY
        ),
        anomaly_model_directory=(
            DEFAULT_ANOMALY_MODEL_DIRECTORY
        ),
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


def generate_anomaly_results(
    input_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Generate supervised and anomaly prediction results."""
    prediction_service = load_prediction_service()

    try:
        with st.spinner("Generating anomaly scores..."):
            prediction_results = prediction_service.predict(
                input_features=input_dataframe,
                include_anomaly_scores=True,
            )
    except (
        ValueError,
        DataValidationError,
        PredictionInputError,
    ) as error:
        st.error(f"Anomaly analysis failed: {error}")
        st.stop()

    return prediction_results


def display_anomaly_metrics(
    prediction_results: pd.DataFrame,
) -> None:
    """Display anomaly summary metrics."""
    total_transactions = len(prediction_results)

    anomaly_count = int(
        prediction_results["anomaly_prediction"].sum(),
    )

    anomaly_percentage = (
        anomaly_count / total_transactions
        if total_transactions > 0
        else 0.0
    )

    average_anomaly_score = float(
        prediction_results["anomaly_score"].mean(),
    )

    maximum_anomaly_score = float(
        prediction_results["anomaly_score"].max(),
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label="Transactions analyzed",
        value=f"{total_transactions:,}",
    )

    col2.metric(
        label="Anomalies detected",
        value=f"{anomaly_count:,}",
    )

    col3.metric(
        label="Anomaly rate",
        value=f"{anomaly_percentage:.4%}",
    )

    col4.metric(
        label="Average anomaly score",
        value=f"{average_anomaly_score:.4f}",
    )

    st.caption(
        f"Maximum anomaly score: {maximum_anomaly_score:.4f}"
    )


def display_anomaly_distribution(
    prediction_results: pd.DataFrame,
) -> None:
    """Display the anomaly-score distribution."""
    st.subheader("Anomaly-score distribution")

    anomaly_histogram = px.histogram(
        prediction_results,
        x="anomaly_score",
        color="anomaly_prediction",
        nbins=50,
        title="Distribution of Isolation Forest anomaly scores",
        labels={
            "anomaly_score": "Anomaly score",
            "count": "Transaction count",
            "anomaly_prediction": "Anomaly prediction",
        },
    )

    st.plotly_chart(
        anomaly_histogram,
        use_container_width=True,
    )


def display_anomaly_table(
    prediction_results: pd.DataFrame,
) -> None:
    """Display transactions identified as anomalies."""
    st.subheader("Transactions identified as anomalies")

    anomaly_results = prediction_results[
        prediction_results["anomaly_prediction"] == 1
    ].copy()

    if anomaly_results.empty:
        st.success(
            "No transactions were identified as anomalies."
        )
        return

    display_columns = [
        "anomaly_score",
        "anomaly_prediction",
        "fraud_score",
        "fraud_prediction",
        "review_flag",
    ]

    st.dataframe(
        anomaly_results[display_columns],
        use_container_width=True,
    )

    csv_data = anomaly_results.to_csv(
        index=False,
    ).encode("utf-8")

    st.download_button(
        label="Download anomaly transactions",
        data=csv_data,
        file_name="anomaly_transactions.csv",
        mime="text/csv",
    )


def display_download_option(
    prediction_results: pd.DataFrame,
) -> None:
    """Provide a download option for all anomaly-analysis results."""
    st.subheader("Download all results")

    csv_data = prediction_results.to_csv(
        index=False,
    ).encode("utf-8")

    st.download_button(
        label="Download complete anomaly-analysis results",
        data=csv_data,
        file_name="anomaly_analysis_results.csv",
        mime="text/csv",
    )


def main() -> None:
    """Run the Anomaly Analysis page."""
    initialize_page()

    uploaded_file = st.file_uploader(
        label="Upload transaction CSV",
        type=["csv"],
    )

    input_dataframe = read_and_validate_input(
        uploaded_file=uploaded_file,
    )

    prediction_results = generate_anomaly_results(
        input_dataframe=input_dataframe,
    )

    display_anomaly_metrics(
        prediction_results=prediction_results,
    )

    display_anomaly_distribution(
        prediction_results=prediction_results,
    )

    display_anomaly_table(
        prediction_results=prediction_results,
    )

    display_download_option(
        prediction_results=prediction_results,
    )


if __name__ == "__main__":
    main()