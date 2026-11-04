"""Monitoring page for the Streamlit application."""


from __future__ import annotations


import sys
from pathlib import Path


import pandas as pd
import streamlit as st


import plotly.express as px
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


st.set_page_config(
    page_title="Monitoring",
    page_icon="🛰️",
    layout="wide",
)


REQUIRED_PREDICTION_COLUMNS = [
    "fraud_score",
    "fraud_prediction",
    "decision_threshold",
    "review_flag",
]


MONITORING_SAMPLE_PATH = (
    PROJECT_ROOT / "data" / "monitoring_prediction_results_sample.csv"
)



def initialize_page() -> None:
    """Display the page title and description."""
    st.title("🛰️ Monitoring")
    st.markdown(
        """
        Upload one or more prediction-results CSV files to monitor
        prediction volume, review-flag rate, fraud-score distribution,
        and input-data quality.
        """
    )



def load_monitoring_sample() -> bytes | None:
    """Load the downloadable Monitoring sample CSV, if available."""
    if not MONITORING_SAMPLE_PATH.exists():
        return None

    return MONITORING_SAMPLE_PATH.read_bytes()



def read_uploaded_files(
    uploaded_files,
) -> dict[str, pd.DataFrame]:
    """Read uploaded prediction-results CSV files."""
    if not uploaded_files:
        st.info(
            "Upload one or more prediction-results CSV files to begin."
        )
        st.stop()


    prediction_files = {}


    for uploaded_file in uploaded_files:
        if not uploaded_file.name.lower().endswith(".csv"):
            st.warning(
                f"Skipped non-CSV file: {uploaded_file.name}"
            )
            continue


        try:
            dataframe = pd.read_csv(uploaded_file)
        except Exception as error:
            st.warning(
                f"Unable to read {uploaded_file.name}: {error}"
            )
            continue


        missing_columns = [
            column
            for column in REQUIRED_PREDICTION_COLUMNS
            if column not in dataframe.columns
        ]


        if missing_columns:
            st.warning(
                f"Skipped {uploaded_file.name}. Missing columns: "
                f"{', '.join(missing_columns)}"
            )
            continue


        prediction_files[uploaded_file.name] = dataframe


    if not prediction_files:
        st.error(
            "No valid prediction-results CSV files were uploaded."
        )
        st.stop()


    return prediction_files



def create_monitoring_summary(
    prediction_files: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create a monitoring summary for each uploaded file."""
    summary_rows = []


    for file_name, dataframe in prediction_files.items():
        total_transactions = len(dataframe)


        flagged_transactions = int(
            dataframe["review_flag"].sum(),
        )


        flagged_percentage = (
            flagged_transactions / total_transactions
            if total_transactions > 0
            else 0.0
        )


        average_fraud_score = float(
            dataframe["fraud_score"].mean(),
        )


        maximum_fraud_score = float(
            dataframe["fraud_score"].max(),
        )


        average_threshold = float(
            dataframe["decision_threshold"].mean(),
        )


        missing_value_count = int(
            dataframe[
                REQUIRED_PREDICTION_COLUMNS
            ].isna().sum().sum(),
        )


        summary_rows.append(
            {
                "file_name": file_name,
                "transaction_count": total_transactions,
                "flagged_count": flagged_transactions,
                "flagged_rate": flagged_percentage,
                "average_fraud_score": average_fraud_score,
                "maximum_fraud_score": maximum_fraud_score,
                "average_threshold": average_threshold,
                "missing_prediction_values": missing_value_count,
            }
        )


    return pd.DataFrame(summary_rows)



def display_monitoring_summary(
    monitoring_summary: pd.DataFrame,
) -> None:
    """Display the monitoring summary table."""
    st.subheader("Monitoring summary")


    st.dataframe(
        monitoring_summary,
        use_container_width=True,
    )



def display_volume_and_flag_rate(
    monitoring_summary: pd.DataFrame,
) -> None:
    """Display prediction volume and review-flag rate trends."""
    st.subheader("Prediction volume and review-flag rate")


    volume_figure = go.Figure()


    volume_figure.add_trace(
        go.Bar(
            x=monitoring_summary["file_name"],
            y=monitoring_summary["transaction_count"],
            name="Transactions processed",
        )
    )


    volume_figure.add_trace(
        go.Scatter(
            x=monitoring_summary["file_name"],
            y=monitoring_summary["flagged_count"],
            name="Transactions flagged",
            mode="lines+markers",
            yaxis="y2",
        )
    )


    volume_figure.update_layout(
        title="Prediction volume and flagged transactions",
        xaxis_title="Prediction file",
        yaxis_title="Transaction count",
        yaxis2=dict(
            title="Flagged transaction count",
            overlaying="y",
            side="right",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
        ),
    )


    st.plotly_chart(
        volume_figure,
        use_container_width=True,
    )


    flag_rate_figure = px.line(
        monitoring_summary,
        x="file_name",
        y="flagged_rate",
        markers=True,
        title="Review-flag rate by prediction file",
        labels={
            "file_name": "Prediction file",
            "flagged_rate": "Review-flag rate",
        },
    )


    st.plotly_chart(
        flag_rate_figure,
        use_container_width=True,
    )



def display_score_distribution(
    prediction_files: dict[str, pd.DataFrame],
) -> None:
    """Display fraud-score distributions across uploaded files."""
    st.subheader("Fraud-score distribution")


    score_dataframes = []


    for file_name, dataframe in prediction_files.items():
        file_scores = dataframe[["fraud_score"]].copy()
        file_scores["file_name"] = file_name
        score_dataframes.append(file_scores)


    combined_scores = pd.concat(
        score_dataframes,
        ignore_index=True,
    )


    score_histogram = px.histogram(
        combined_scores,
        x="fraud_score",
        color="file_name",
        nbins=50,
        title="Fraud-score distribution by prediction file",
        labels={
            "fraud_score": "Predicted fraud score",
            "count": "Transaction count",
            "file_name": "Prediction file",
        },
    )


    st.plotly_chart(
        score_histogram,
        use_container_width=True,
    )



def display_anomaly_monitoring(
    prediction_files: dict[str, pd.DataFrame],
) -> None:
    """Display anomaly monitoring if anomaly columns are available."""
    anomaly_available_files = [
        file_name
        for file_name, dataframe in prediction_files.items()
        if "anomaly_prediction" in dataframe.columns
    ]


    if not anomaly_available_files:
        st.info(
            "No anomaly columns were found. Upload prediction results "
            "generated with anomaly scores to monitor anomaly rates."
        )
        return


    st.subheader("Anomaly monitoring")


    anomaly_rows = []


    for file_name in anomaly_available_files:
        dataframe = prediction_files[file_name]


        total_transactions = len(dataframe)


        anomaly_count = int(
            dataframe["anomaly_prediction"].sum(),
        )


        anomaly_rate = (
            anomaly_count / total_transactions
            if total_transactions > 0
            else 0.0
        )


        average_anomaly_score = float(
            dataframe["anomaly_score"].mean(),
        )


        anomaly_rows.append(
            {
                "file_name": file_name,
                "transaction_count": total_transactions,
                "anomaly_count": anomaly_count,
                "anomaly_rate": anomaly_rate,
                "average_anomaly_score": average_anomaly_score,
            }
        )


    anomaly_summary = pd.DataFrame(anomaly_rows)


    st.dataframe(
        anomaly_summary,
        use_container_width=True,
    )


    anomaly_chart = px.line(
        anomaly_summary,
        x="file_name",
        y="anomaly_rate",
        markers=True,
        title="Anomaly rate by prediction file",
        labels={
            "file_name": "Prediction file",
            "anomaly_rate": "Anomaly rate",
        },
    )


    st.plotly_chart(
        anomaly_chart,
        use_container_width=True,
    )



def display_data_quality_checks(
    prediction_files: dict[str, pd.DataFrame],
) -> None:
    """Display basic data-quality checks for prediction files."""
    st.subheader("Data-quality checks")


    quality_rows = []


    for file_name, dataframe in prediction_files.items():
        missing_prediction_values = int(
            dataframe[
                REQUIRED_PREDICTION_COLUMNS
            ].isna().sum().sum(),
        )


        invalid_review_flags = int(
            (~dataframe["review_flag"].isin([0, 1])).sum(),
        )


        invalid_fraud_predictions = int(
            (~dataframe["fraud_prediction"].isin([0, 1])).sum(),
        )


        scores_outside_range = int(
            (
                (dataframe["fraud_score"] < 0)
                | (dataframe["fraud_score"] > 1)
            ).sum(),
        )


        quality_rows.append(
            {
                "file_name": file_name,
                "transaction_count": len(dataframe),
                "missing_prediction_values": (
                    missing_prediction_values
                ),
                "invalid_review_flags": invalid_review_flags,
                "invalid_fraud_predictions": (
                    invalid_fraud_predictions
                ),
                "fraud_scores_outside_0_1": scores_outside_range,
            }
        )


    quality_summary = pd.DataFrame(quality_rows)


    st.dataframe(
        quality_summary,
        use_container_width=True,
    )


    total_issues = int(
        quality_summary[
            [
                "missing_prediction_values",
                "invalid_review_flags",
                "invalid_fraud_predictions",
                "fraud_scores_outside_0_1",
            ]
        ].sum().sum(),
    )


    if total_issues == 0:
        st.success(
            "No data-quality issues were detected in the uploaded "
            "prediction files."
        )
    else:
        st.warning(
            f"{total_issues} potential data-quality issues were detected."
        )



def main() -> None:
    """Run the Monitoring page."""
    initialize_page()


    monitoring_sample_bytes = load_monitoring_sample()


    if monitoring_sample_bytes is not None:
        with st.expander("Download Monitoring sample CSV", expanded=True):
            st.markdown(
                """
                This sample file contains **prediction results**, not original
                transaction input features.


                It includes the prediction-output columns required by this
                Monitoring page:


                - `fraud_score`
                - `fraud_prediction`
                - `review_flag`
                - `decision_threshold`


                Download the file and upload it below to test the Monitoring
                workflow. This file is separate from the transaction-input
                sample used on the Batch Prediction page.
                """
            )


            st.download_button(
                label="Download monitoring_prediction_results_sample.csv",
                data=monitoring_sample_bytes,
                file_name="monitoring_prediction_results_sample.csv",
                mime="text/csv",
            )
    else:
        st.warning(
            "Monitoring sample file not found: "
            f"`{MONITORING_SAMPLE_PATH}`."
        )


    uploaded_files = st.file_uploader(
        label="Upload prediction-results CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )


    prediction_files = read_uploaded_files(
        uploaded_files=uploaded_files,
    )


    monitoring_summary = create_monitoring_summary(
        prediction_files=prediction_files,
    )


    display_monitoring_summary(
        monitoring_summary=monitoring_summary,
    )


    display_volume_and_flag_rate(
        monitoring_summary=monitoring_summary,
    )


    display_score_distribution(
        prediction_files=prediction_files,
    )


    display_anomaly_monitoring(
        prediction_files=prediction_files,
    )


    display_data_quality_checks(
        prediction_files=prediction_files,
    )



if __name__ == "__main__":
    main()