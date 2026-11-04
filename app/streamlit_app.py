"""Streamlit application for the credit-card fraud-detection system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    from app.api_client import (
        ApiClientError,
        detect_anomaly,
        get_api_base_url,
        get_api_health,
        get_evaluation_report,
        predict_batch_transactions,
        predict_single_transaction,
    )
except ModuleNotFoundError:
    from api_client import (
        ApiClientError,
        detect_anomaly,
        get_api_base_url,
        get_api_health,
        get_evaluation_report,
        predict_batch_transactions,
        predict_single_transaction,
    )

APP_TITLE = "Credit Card Fraud Detection System"
APP_DESCRIPTION = (
    "Interactive dashboard for supervised fraud prediction and "
    "Isolation Forest anomaly detection."
)

FEATURE_COLUMNS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]

REQUIRED_FEATURE_COLUMNS = FEATURE_COLUMNS

SAMPLE_PREDICTION_INPUT_PATH = Path("data") / "prediction_input_sample.csv"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_custom_css() -> None:
    """Load the project-wide Streamlit stylesheet if it exists."""
    css_path = Path("app") / "styles.css"

    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def display_page_header(
    title: str,
    description: str,
) -> None:
    """Display a consistent page header."""
    st.markdown(
        f"""
        <div class="page-header">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_metric_card(
    label: str,
    value: str,
    help_text: str | None = None,
) -> None:
    """Display a styled metric card."""
    help_html = ""

    if help_text:
        help_html = f"<p class='metric-help'>{help_text}</p>"

    st.markdown(
        f"""
        <div class="metric-card">
            <p class="metric-label">{label}</p>
            <p class="metric-value">{value}</p>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_prediction_result(
    prediction: dict[str, Any],
) -> None:
    """Display the supervised prediction result for one transaction."""
    fraud_score = float(prediction["fraud_score"])
    fraud_prediction = int(prediction["fraud_prediction"])
    review_flag = int(prediction["review_flag"])
    decision_threshold = float(prediction["decision_threshold"])

    if fraud_prediction == 1:
        result_class = "prediction-fraud"
        result_label = "Fraud predicted"
    else:
        result_class = "prediction-legitimate"
        result_label = "Legitimate predicted"

    if review_flag == 1:
        review_label = "Manual review recommended"
        review_class = "review-required"
    else:
        review_label = "No manual review required"
        review_class = "review-not-required"

    st.markdown(
        f"""
        <div class="prediction-result {result_class}">
            <h2>{result_label}</h2>
            <p class="prediction-score">
                Fraud score: <strong>{fraud_score:.4f}</strong>
            </p>
            <p class="prediction-threshold">
                Decision threshold: <strong>{decision_threshold:.2f}</strong>
            </p>
            <p class="prediction-review {review_class}">
                {review_label}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(
        min(max(fraud_score, 0.0), 1.0),
        text=f"Fraud score: {fraud_score:.4f}",
    )


def render_anomaly_analysis_views(
    prediction_dataframe: pd.DataFrame,
) -> None:
    """Display anomaly-analysis views when anomaly scores are available."""
    required_anomaly_columns = [
        "anomaly_score",
        "anomaly_prediction",
    ]

    if not all(
        column in prediction_dataframe.columns
        for column in required_anomaly_columns
    ):
        return

    anomaly_dataframe = prediction_dataframe.copy()

    anomaly_dataframe = anomaly_dataframe[
        anomaly_dataframe["anomaly_score"].notna()
        & anomaly_dataframe["anomaly_prediction"].notna()
    ]

    if anomaly_dataframe.empty:
        st.info(
            "Anomaly scoring was enabled, but no anomaly results were "
            "returned for this batch."
        )
        return

    anomaly_dataframe["anomaly_score"] = anomaly_dataframe[
        "anomaly_score"
    ].astype(float)

    anomaly_dataframe["anomaly_prediction"] = anomaly_dataframe[
        "anomaly_prediction"
    ].astype(int)

    st.subheader("Anomaly analysis")

    anomaly_count = int(
        (
            anomaly_dataframe["anomaly_prediction"] == -1
        ).sum()
    )

    transaction_count = len(anomaly_dataframe)

    anomaly_rate = (
        anomaly_count / transaction_count
        if transaction_count > 0
        else 0.0
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Anomalies detected",
            value=str(anomaly_count),
        )

    with col2:
        st.metric(
            label="Anomaly rate",
            value=f"{anomaly_rate:.2%}",
        )

    st.subheader("Anomaly-score distribution")

    st.bar_chart(
        anomaly_dataframe["anomaly_score"]
        .value_counts()
        .sort_index()
        .head(50),
        use_container_width=True,
    )

    st.caption(
        "Lower Isolation Forest scores indicate more unusual "
        "transactions. An anomaly score is not a fraud probability."
    )

    st.subheader("Most anomalous transactions")

    ranked_anomalies = anomaly_dataframe.sort_values(
        by="anomaly_score",
        ascending=True,
    ).head(20)

    if ranked_anomalies.empty:
        st.info("No anomaly results are available.")
    else:
        st.dataframe(
            ranked_anomalies,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Supervised and anomaly comparison")

    anomaly_dataframe["detection_category"] = "Neither model flagged"

    anomaly_dataframe.loc[
        (anomaly_dataframe["fraud_prediction"] == 1)
        & (anomaly_dataframe["anomaly_prediction"] == -1),
        "detection_category",
    ] = "Flagged by both models"

    anomaly_dataframe.loc[
        (anomaly_dataframe["fraud_prediction"] == 1)
        & (anomaly_dataframe["anomaly_prediction"] != -1),
        "detection_category",
    ] = "Supervised model only"

    anomaly_dataframe.loc[
        (anomaly_dataframe["fraud_prediction"] != 1)
        & (anomaly_dataframe["anomaly_prediction"] == -1),
        "detection_category",
    ] = "Anomaly model only"

    category_counts = anomaly_dataframe[
        "detection_category"
    ].value_counts()

    comparison_dataframe = pd.DataFrame(
        {
            "Category": category_counts.index,
            "Transactions": category_counts.values,
        }
    )

    st.bar_chart(
        comparison_dataframe.set_index("Category"),
        use_container_width=True,
    )

    st.dataframe(
        comparison_dataframe,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        """
        Anomaly detection identifies unusual transactions. It does not
        confirm fraud. Model disagreement does not prove that either model
        is incorrect.
        """
    )


def render_fraud_dashboard_views(
    prediction_dataframe: pd.DataFrame,
) -> None:
    """Display dashboard views for the latest batch prediction."""
    st.subheader("Fraud dashboard")

    if prediction_dataframe.empty:
        st.info("No prediction results are available.")
        return

    fraud_scores = prediction_dataframe["fraud_score"].astype(float)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Transactions",
            value=str(len(prediction_dataframe)),
        )

    with col2:
        st.metric(
            label="Predicted fraud",
            value=str(
                int(
                    (
                        prediction_dataframe["fraud_prediction"] == 1
                    ).sum()
                )
            ),
        )

    with col3:
        st.metric(
            label="Flagged for review",
            value=str(
                int(
                    (
                        prediction_dataframe["review_flag"] == 1
                    ).sum()
                )
            ),
        )

    st.subheader("Fraud-score distribution")

    st.bar_chart(
        fraud_scores.value_counts()
        .sort_index()
        .head(50),
        use_container_width=True,
    )

    st.caption(
        "This chart shows the number of transactions at each returned "
        "fraud score. Scores are model outputs, not confirmed fraud labels."
    )

    st.subheader("Prediction distribution")

    prediction_counts = prediction_dataframe[
        "fraud_prediction"
    ].value_counts()

    prediction_labels = {
        0: "Legitimate predicted",
        1: "Fraud predicted",
    }

    distribution_dataframe = pd.DataFrame(
        {
            "Prediction": [
                prediction_labels.get(
                    prediction_value,
                    str(prediction_value),
                )
                for prediction_value in prediction_counts.index
            ],
            "Transactions": prediction_counts.values,
        }
    )

    st.bar_chart(
        distribution_dataframe.set_index("Prediction"),
        use_container_width=True,
    )

    st.divider()

    st.subheader("Filter prediction results")

    prediction_filter = st.selectbox(
        "Prediction",
        options=[
            "All predictions",
            "Legitimate predicted",
            "Fraud predicted",
        ],
    )

    review_filter = st.selectbox(
        "Review status",
        options=[
            "All transactions",
            "Review required",
            "No review required",
        ],
    )

    minimum_score, maximum_score = st.slider(
        "Fraud-score range",
        min_value=0.0,
        max_value=1.0,
        value=(
            0.0,
            1.0,
        ),
        step=0.01,
    )

    filtered_dataframe = prediction_dataframe.copy()

    if prediction_filter == "Legitimate predicted":
        filtered_dataframe = filtered_dataframe[
            filtered_dataframe["fraud_prediction"] == 0
        ]

    elif prediction_filter == "Fraud predicted":
        filtered_dataframe = filtered_dataframe[
            filtered_dataframe["fraud_prediction"] == 1
        ]

    if review_filter == "Review required":
        filtered_dataframe = filtered_dataframe[
            filtered_dataframe["review_flag"] == 1
        ]

    elif review_filter == "No review required":
        filtered_dataframe = filtered_dataframe[
            filtered_dataframe["review_flag"] == 0
        ]

    filtered_dataframe = filtered_dataframe[
        (filtered_dataframe["fraud_score"].astype(float) >= minimum_score)
        & (filtered_dataframe["fraud_score"].astype(float) <= maximum_score)
    ]

    st.caption(
        f"Showing {len(filtered_dataframe)} of "
        f"{len(prediction_dataframe)} transactions."
    )

    if filtered_dataframe.empty:
        st.info(
            "No transactions match the selected filters."
        )
    else:
        st.dataframe(
            filtered_dataframe,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Flagged transactions")

    flagged_dataframe = prediction_dataframe[
        prediction_dataframe["review_flag"] == 1
    ]

    if flagged_dataframe.empty:
        st.success(
            "No transactions were flagged for manual review."
        )
    else:
        st.warning(
            "The following transactions meet or exceed the decision "
            "threshold and are recommended for manual review. They are "
            "model predictions, not confirmed fraud."
        )

        st.dataframe(
            flagged_dataframe,
            use_container_width=True,
            hide_index=True,
        )

def display_anomaly_result(
    prediction: dict[str, Any],
) -> None:
    """Display combined supervised and anomaly-detection results."""
    display_prediction_result(
        prediction=prediction,
    )

    anomaly_score = float(prediction["anomaly_score"])
    anomaly_prediction = int(prediction["anomaly_prediction"])

    if anomaly_prediction == -1:
        anomaly_class = "anomaly-detected"
        anomaly_label = "Anomaly detected"
    else:
        anomaly_class = "anomaly-not-detected"
        anomaly_label = "No anomaly detected"

    st.markdown(
        f"""
        <div class="anomaly-result {anomaly_class}">
            <h3>{anomaly_label}</h3>
            <p>
                Isolation Forest anomaly score:
                <strong>{anomaly_score:.4f}</strong>
            </p>
            <p class="anomaly-note">
                Lower scores indicate more unusual transactions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def validate_transaction_input(
    transaction: dict[str, Any],
) -> list[str]:
    """Return a list of validation errors for a transaction."""
    errors = []

    for column in REQUIRED_FEATURE_COLUMNS:
        if column not in transaction:
            errors.append(f"Missing required feature: {column}.")
            continue

        value = transaction[column]

        if not isinstance(value, (int, float)):
            errors.append(f"{column} must be numeric.")
            continue

        if pd.isna(value):
            errors.append(f"{column} cannot be missing.")

    amount = transaction.get("Amount")

    if isinstance(amount, (int, float)) and amount < 0:
        errors.append("Amount cannot be negative.")

    return errors


def render_home_page() -> None:
    """Render the application home page."""
    display_page_header(
        title="Welcome",
        description=APP_DESCRIPTION,
    )

    st.markdown(
        """
        This dashboard sends transactions to a FastAPI fraud-detection
        service. The service performs all feature preparation, supervised
        prediction, thresholding, and Isolation Forest anomaly scoring.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            ### Single prediction
            Enter one transaction manually and receive a fraud score,
            prediction, review recommendation, and decision threshold.
            """
        )

    with col2:
        st.markdown(
            """
            ### Batch prediction
            Upload a CSV file containing multiple transactions and receive
            predictions for every row.
            """
        )

    with col3:
        st.markdown(
            """
            ### Anomaly detection
            Check one transaction using both the supervised fraud model and
            the Isolation Forest anomaly model.
            """
        )

    st.divider()

    st.subheader("API status")

    try:
        health = get_api_health()

        supervised_loaded = health.get(
            "supervised_model_loaded",
            False,
        )

        anomaly_loaded = health.get(
            "anomaly_model_loaded",
            False,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.success("API connected")

        with col2:
            if supervised_loaded:
                st.success("Supervised model loaded")
            else:
                st.error("Supervised model unavailable")

        with col3:
            if anomaly_loaded:
                st.success("Anomaly model loaded")
            else:
                st.warning("Anomaly model unavailable")

    except ApiClientError as error:
        st.error(str(error))


def render_single_prediction_page() -> None:
    """Render the manual single-transaction prediction page."""
    display_page_header(
        title="Single Prediction",
        description=(
            "Enter transaction features to obtain a supervised fraud "
            "prediction."
        ),
    )

    with st.expander(
        "Transaction features",
        expanded=True,
    ):
        transaction: dict[str, Any] = {}

        col1, col2 = st.columns(2)

        with col1:
            transaction["Time"] = st.number_input(
                "Time",
                min_value=0.0,
                value=0.0,
                step=1.0,
                help="Seconds elapsed since the first transaction.",
            )

        with col2:
            transaction["Amount"] = st.number_input(
                "Amount",
                min_value=0.0,
                value=100.0,
                step=0.01,
                help="Transaction amount.",
            )

        st.divider()

        pca_columns = [
            column
            for column in FEATURE_COLUMNS
            if column.startswith("V")
        ]

        number_of_columns = 3

        for start_index in range(
            0,
            len(pca_columns),
            number_of_columns,
        ):
            columns = st.columns(number_of_columns)

            for column_offset, column_name in enumerate(
                pca_columns[
                    start_index:start_index + number_of_columns
                ]
            ):
                with columns[column_offset]:
                    transaction[column_name] = st.number_input(
                        column_name,
                        value=0.0,
                        step=0.01,
                        format="%.6f",
                    )

    validation_errors = validate_transaction_input(
        transaction=transaction,
    )

    if validation_errors:
        for error in validation_errors:
            st.error(error)

    if st.button(
        "Predict fraud",
        type="primary",
        disabled=bool(validation_errors),
    ):
        try:
            with st.spinner("Requesting prediction from API..."):
                prediction = predict_single_transaction(
                    transaction=transaction,
                )

            st.success("Prediction completed.")

            display_prediction_result(
                prediction=prediction,
            )

            with st.expander("Raw API response"):
                st.json(prediction)

        except ApiClientError as error:
            st.error(str(error))


def load_sample_prediction_input() -> bytes | None:
    """Load the downloadable sample prediction-input CSV, if available."""
    if not SAMPLE_PREDICTION_INPUT_PATH.exists():
        return None

    return SAMPLE_PREDICTION_INPUT_PATH.read_bytes()

def render_batch_prediction_page() -> None:
    """Render the CSV batch-prediction page."""
    display_page_header(
        title="Batch Prediction",
        description=(
            "Upload a CSV file containing transaction features to obtain "
            "predictions for every row."
        ),
    )

    include_anomaly_scores = st.toggle(
        "Include Isolation Forest anomaly scores",
        value=False,
        help=(
            "When enabled, the API also returns an anomaly score and "
            "anomaly prediction for every transaction."
        ),
    )

    sample_csv_bytes = load_sample_prediction_input()

    if sample_csv_bytes is not None:
        with st.expander("Download sample prediction input", expanded=True):
            st.markdown(
                """
                This sample file contains **1,000 real transactions** selected
                from the original European Credit Card Fraud Detection dataset.
                It includes both legitimate and fraudulent examples.

                Download the file, upload it using the CSV uploader below, and
                run batch predictions. The file contains the 30 original model
                input features and intentionally does **not** include the
                `Class` target column.

                The sample is curated to include both classes, so its class
                distribution is not representative of real-world transaction
                distributions.
                """
            )

            st.download_button(
                label="Download prediction_input_sample.csv",
                data=sample_csv_bytes,
                file_name="prediction_input_sample.csv",
                mime="text/csv",
            )
    else:
        st.warning(
            "Sample prediction input file not found: "
            f"`{SAMPLE_PREDICTION_INPUT_PATH}`. "
            "Run `python scripts/generate_prediction_input_sample.py` to create it."
        )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
    )

    if uploaded_file is None:
        st.info(
            "Upload a CSV file with the following columns: "
            + ", ".join(REQUIRED_FEATURE_COLUMNS)
        )
        return

    try:
        input_dataframe = pd.read_csv(
            uploaded_file,
        )
    except UnicodeDecodeError:
        st.error(
            "The uploaded file could not be decoded. Please upload a "
            "UTF-8 encoded CSV file."
        )
        return
    except Exception:
        st.error(
            "The uploaded file could not be read as a CSV file. "
            "Please check the file format."
        )
        return

    if input_dataframe.empty:
        st.error("The uploaded CSV file does not contain any rows.")
        return

    missing_columns = [
        column
        for column in REQUIRED_FEATURE_COLUMNS
        if column not in input_dataframe.columns
    ]

    if missing_columns:
        st.error(
            "The uploaded CSV file is missing required columns: "
            + ", ".join(missing_columns)
        )
        return

    unexpected_columns = [
        column
        for column in input_dataframe.columns
        if column not in REQUIRED_FEATURE_COLUMNS
    ]

    if unexpected_columns:
        st.error(
            "The uploaded CSV file contains unexpected columns: "
            + ", ".join(unexpected_columns)
        )
        return

    st.subheader("Uploaded data preview")

    st.dataframe(
        input_dataframe.head(10),
        use_container_width=True,
    )

    records = input_dataframe.to_dict(
        orient="records",
    )

    if st.button(
        "Run batch prediction",
        type="primary",
    ):
        try:
            with st.spinner("Requesting batch predictions from API..."):
                batch_response = predict_batch_transactions(
                    transactions=records,
                    include_anomaly_scores=include_anomaly_scores,
                )

            st.success("Batch prediction completed.")

            prediction_rows = []

            for index, prediction in enumerate(
                batch_response["predictions"],
            ):
                row = {
                    "row_number": index + 1,
                    "fraud_score": prediction["fraud_score"],
                    "fraud_prediction": prediction["fraud_prediction"],
                    "review_flag": prediction["review_flag"],
                    "decision_threshold": prediction["decision_threshold"],
                }

                if include_anomaly_scores:
                    row["anomaly_score"] = prediction.get(
                        "anomaly_score",
                    )

                    row["anomaly_prediction"] = prediction.get(
                        "anomaly_prediction",
                    )

                prediction_rows.append(row)

            prediction_dataframe = pd.DataFrame(
                prediction_rows,
            )

            st.session_state["batch_input_dataframe"] = input_dataframe.copy()
            st.session_state["batch_prediction_dataframe"] = (
                prediction_dataframe.copy()
            )

            transaction_count = int(
                batch_response["transaction_count"],
            )

            flagged_count = int(
                batch_response["flagged_count"],
            )

            flagged_rate = float(
                batch_response["flagged_rate"],
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                display_metric_card(
                    label="Transactions",
                    value=str(transaction_count),
                )

            with col2:
                display_metric_card(
                    label="Flagged for review",
                    value=str(flagged_count),
                )

            with col3:
                display_metric_card(
                    label="Flagged rate",
                    value=f"{flagged_rate:.2%}",
                )

            if include_anomaly_scores:
                anomaly_count = batch_response.get(
                    "anomaly_count",
                )

                anomaly_rate = batch_response.get(
                    "anomaly_rate",
                )

                if anomaly_count is not None and anomaly_rate is not None:
                    col1, col2 = st.columns(2)

                    with col1:
                        display_metric_card(
                            label="Anomalies detected",
                            value=str(int(anomaly_count)),
                        )

                    with col2:
                        display_metric_card(
                            label="Anomaly rate",
                            value=f"{float(anomaly_rate):.2%}",
                        )

            st.subheader("Prediction results")

            st.dataframe(
                prediction_dataframe,
                use_container_width=True,
            )

            render_fraud_dashboard_views(
                prediction_dataframe=prediction_dataframe,
            )

            render_anomaly_analysis_views(
                prediction_dataframe=prediction_dataframe,
            )

            csv_data = prediction_dataframe.to_csv(
                index=False,
            ).encode("utf-8")

            st.download_button(
                label="Download predictions as CSV",
                data=csv_data,
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )

            st.info(
                "The prediction results are now available on the "
                "Transaction Investigation page."
            )

            with st.expander("Raw API response"):
                st.json(batch_response)

        except ApiClientError as error:
            st.error(str(error))


def render_transaction_investigation_page() -> None:
    """Render investigation for a transaction from the latest batch."""
    display_page_header(
        title="Transaction Investigation",
        description=(
            "Select a transaction from the latest batch prediction to "
            "inspect its input features and API-returned prediction."
        ),
    )

    input_dataframe = st.session_state.get(
        "batch_input_dataframe",
    )

    prediction_dataframe = st.session_state.get(
        "batch_prediction_dataframe",
    )

    if input_dataframe is None or prediction_dataframe is None:
        st.warning(
            "No batch prediction is available. Upload a CSV and run a "
            "batch prediction first."
        )
        return

    if input_dataframe.empty or prediction_dataframe.empty:
        st.warning(
            "The available batch prediction does not contain any "
            "transactions."
        )
        return

    if len(input_dataframe) != len(prediction_dataframe):
        st.error(
            "The uploaded transactions and prediction results do not "
            "have the same number of rows."
        )
        return

    investigation_dataframe = prediction_dataframe.copy()

    st.subheader("Select a transaction")

    selected_row = st.dataframe(
        investigation_dataframe,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    selected_rows = selected_row.selection.rows

    if not selected_rows:
        st.info(
            "Select one row in the table above to inspect its prediction."
        )
        return

    selected_index = selected_rows[0]

    input_transaction = input_dataframe.iloc[
        selected_index
    ]

    prediction_result = prediction_dataframe.iloc[
        selected_index
    ]

    row_number = int(
        prediction_result["row_number"],
    )

    st.subheader(
        f"Transaction {row_number}"
    )

    st.caption(
        "The values below are the original input features from the "
        "uploaded CSV file."
    )

    feature_dataframe = input_transaction.to_frame(
        name="Value",
    )

    feature_dataframe.index.name = "Feature"

    st.dataframe(
        feature_dataframe,
        use_container_width=True,
    )

    st.subheader("API prediction")

    display_prediction_result(
        prediction=prediction_result.to_dict(),
    )

    st.warning(
        """
        This is a machine-learning prediction, not confirmed fraud.
        A fraud prediction or review flag indicates that the transaction
        should be investigated; it does not prove that fraud occurred.
        """
    )

def render_model_evaluation_page() -> None:
    """Display the saved model-evaluation report from the API."""
    display_page_header(
        title="Model Evaluation",
        description=(
            "View the saved supervised and anomaly-detection evaluation "
            "results generated during model development."
        ),
    )

    if st.button(
        "Load evaluation report",
        type="primary",
    ):
        try:
            with st.spinner("Loading evaluation report from API..."):
                response = get_evaluation_report()

            st.session_state["evaluation_report"] = response.get(
                "report",
                "",
            )

            st.success("Evaluation report loaded.")

        except ApiClientError as error:
            st.error(str(error))

    report = st.session_state.get(
        "evaluation_report",
    )

    if not report:
        st.info(
            "Click “Load evaluation report” to display the saved "
            "evaluation results."
        )
        return

    st.markdown(
        report,
    )

    st.warning(
        """
        These results come from the saved model-evaluation report. They
        describe historical model performance and do not guarantee future
        performance. Anomaly scores are not fraud probabilities.
        """
    )


def render_anomaly_detection_page() -> None:
    """Render the single-transaction anomaly-detection page."""
    display_page_header(
        title="Anomaly Detection",
        description=(
            "Enter transaction features to obtain both supervised fraud "
            "and Isolation Forest anomaly results."
        ),
    )

    with st.expander(
        "Transaction features",
        expanded=True,
    ):
        transaction: dict[str, Any] = {}

        col1, col2 = st.columns(2)

        with col1:
            transaction["Time"] = st.number_input(
                "Time",
                min_value=0.0,
                value=0.0,
                step=1.0,
                help="Seconds elapsed since the first transaction.",
            )

        with col2:
            transaction["Amount"] = st.number_input(
                "Amount",
                min_value=0.0,
                value=100.0,
                step=0.01,
                help="Transaction amount.",
            )

        st.divider()

        pca_columns = [
            column
            for column in FEATURE_COLUMNS
            if column.startswith("V")
        ]

        number_of_columns = 3

        for start_index in range(
            0,
            len(pca_columns),
            number_of_columns,
        ):
            columns = st.columns(number_of_columns)

            for column_offset, column_name in enumerate(
                pca_columns[
                    start_index:start_index + number_of_columns
                ]
            ):
                with columns[column_offset]:
                    transaction[column_name] = st.number_input(
                        column_name,
                        value=0.0,
                        step=0.01,
                        format="%.6f",
                    )

    validation_errors = validate_transaction_input(
        transaction=transaction,
    )

    if validation_errors:
        for error in validation_errors:
            st.error(error)

    if st.button(
        "Detect anomaly",
        type="primary",
        disabled=bool(validation_errors),
    ):
        try:
            with st.spinner("Requesting anomaly detection from API..."):
                prediction = detect_anomaly(
                    transaction=transaction,
                )

            st.success("Anomaly detection completed.")

            display_anomaly_result(
                prediction=prediction,
            )

            with st.expander("Raw API response"):
                st.json(prediction)

        except ApiClientError as error:
            st.error(str(error))


def main() -> None:
    """Run the Streamlit application."""
    load_custom_css()

    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Select page",
        options=[
            "Home",
            "Single Prediction",
            "Batch Prediction",
            "Transaction Investigation",
            "Anomaly Detection",
            "Model Evaluation",
        ],
    )

    st.sidebar.divider()

    st.sidebar.caption(
        f"API base URL: `{get_api_base_url()}`"
    )

    if page == "Home":
        render_home_page()

    elif page == "Single Prediction":
        render_single_prediction_page()

    elif page == "Batch Prediction":
        render_batch_prediction_page()

    elif page == "Transaction Investigation":
        render_transaction_investigation_page()

    elif page == "Anomaly Detection":
        render_anomaly_detection_page()

    elif page == "Model Evaluation":
        render_model_evaluation_page()


if __name__ == "__main__":
    main()