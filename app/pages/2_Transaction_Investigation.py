"""Transaction Investigation page for the Streamlit application."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

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
    page_title="Transaction Investigation",
    page_icon="🔍",
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

PREDICTION_RESULT_COLUMNS = [
    "fraud_score",
    "fraud_prediction",
    "decision_threshold",
    "review_flag",
]


def initialize_page() -> None:
    """Display the page title and description."""
    st.title("🔍 Transaction Investigation")
    st.markdown(
        """
        Upload raw transaction data or an existing prediction-results CSV,
        then filter and inspect individual transactions.
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


def detect_prediction_results(
    dataframe: pd.DataFrame,
) -> bool:
    """Check whether an uploaded CSV already contains prediction results."""
    return all(
        column in dataframe.columns
        for column in PREDICTION_RESULT_COLUMNS
    )


def read_uploaded_csv(
    uploaded_file,
) -> pd.DataFrame:
    """Read an uploaded CSV file."""
    if uploaded_file is None:
        st.info(
            "Upload a raw transaction CSV or a prediction-results CSV."
        )
        st.stop()

    if not uploaded_file.name.lower().endswith(".csv"):
        st.error("Please upload a CSV file.")
        st.stop()

    try:
        return pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Unable to read the uploaded CSV file: {error}")
        st.stop()


def prepare_raw_predictions(
    input_dataframe: pd.DataFrame,
    include_anomaly_scores: bool,
) -> pd.DataFrame:
    """Validate raw transactions and generate prediction results."""
    try:
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

    prediction_service = load_prediction_service(
        include_anomaly_scores=include_anomaly_scores,
    )

    try:
        with st.spinner("Generating predictions..."):
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


def display_filter_controls(
    prediction_results: pd.DataFrame,
) -> pd.DataFrame:
    """Display filters and return the filtered prediction results."""
    st.subheader("Filter transactions")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    minimum_score = float(
        prediction_results["fraud_score"].min(),
    )

    maximum_score = float(
        prediction_results["fraud_score"].max(),
    )

    score_range = filter_col1.slider(
        label="Fraud-score range",
        min_value=minimum_score,
        max_value=maximum_score,
        value=(
            minimum_score,
            maximum_score,
        ),
        step=0.01,
    )

    review_filter = filter_col2.selectbox(
        label="Review flag",
        options=[
            "All transactions",
            "Flagged for review",
            "Not flagged",
        ],
    )

    if "anomaly_prediction" in prediction_results.columns:
        anomaly_filter = filter_col3.selectbox(
            label="Anomaly prediction",
            options=[
                "All transactions",
                "Anomalies",
                "Normal transactions",
            ],
        )
    else:
        anomaly_filter = "All transactions"

    filtered_results = prediction_results[
        (prediction_results["fraud_score"] >= score_range[0])
        & (prediction_results["fraud_score"] <= score_range[1])
    ].copy()

    if review_filter == "Flagged for review":
        filtered_results = filtered_results[
            filtered_results["review_flag"] == 1
        ]
    elif review_filter == "Not flagged":
        filtered_results = filtered_results[
            filtered_results["review_flag"] == 0
        ]

    if anomaly_filter == "Anomalies":
        filtered_results = filtered_results[
            filtered_results["anomaly_prediction"] == 1
        ]
    elif anomaly_filter == "Normal transactions":
        filtered_results = filtered_results[
            filtered_results["anomaly_prediction"] == 0
        ]

    st.caption(
        f"Showing {len(filtered_results):,} of "
        f"{len(prediction_results):,} transactions."
    )

    return filtered_results


def display_transaction_table(
    filtered_results: pd.DataFrame,
) -> None:
    """Display the filtered transaction table."""
    st.subheader("Filtered transactions")

    if filtered_results.empty:
        st.warning(
            "No transactions match the selected filters."
        )
        return

    st.dataframe(
        filtered_results,
        use_container_width=True,
    )


def display_transaction_details(
    filtered_results: pd.DataFrame,
) -> None:
    """Display details for one selected transaction."""
    st.subheader("Transaction details")

    if filtered_results.empty:
        st.info(
            "Select filters that return at least one transaction."
        )
        return

    maximum_rows = min(
        len(filtered_results),
        1000,
    )

    selected_index = st.number_input(
        label="Select transaction row",
        min_value=0,
        max_value=maximum_rows - 1,
        value=0,
        step=1,
    )

    selected_transaction = filtered_results.iloc[
        int(selected_index)
    ]

    st.json(
        selected_transaction.to_dict(),
    )


def display_download_option(
    filtered_results: pd.DataFrame,
) -> None:
    """Provide a download option for filtered results."""
    st.subheader("Download filtered results")

    if filtered_results.empty:
        st.info(
            "No filtered transactions are available to download."
        )
        return

    csv_data = filtered_results.to_csv(
        index=False,
    ).encode("utf-8")

    st.download_button(
        label="Download filtered transactions",
        data=csv_data,
        file_name="filtered_transactions.csv",
        mime="text/csv",
    )


def main() -> None:
    """Run the Transaction Investigation page."""
    initialize_page()

    uploaded_file = st.file_uploader(
        label="Upload transaction CSV",
        type=["csv"],
    )

    input_dataframe = read_uploaded_csv(
        uploaded_file=uploaded_file,
    )

    if detect_prediction_results(
        dataframe=input_dataframe,
    ):
        prediction_results = input_dataframe
        st.success(
            "Prediction-results CSV detected. Skipping model inference."
        )
    else:
        include_anomaly_scores = st.checkbox(
            "Include Isolation Forest anomaly scores",
            value=False,
        )

        prediction_results = prepare_raw_predictions(
            input_dataframe=input_dataframe,
            include_anomaly_scores=include_anomaly_scores,
        )

    filtered_results = display_filter_controls(
        prediction_results=prediction_results,
    )

    display_transaction_table(
        filtered_results=filtered_results,
    )

    display_transaction_details(
        filtered_results=filtered_results,
    )

    display_download_option(
        filtered_results=filtered_results,
    )


if __name__ == "__main__":
    main()