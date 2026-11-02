"""Model Evaluation page for the Streamlit application."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📈",
    layout="wide",
)

REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
FIGURES_DIRECTORY = REPORTS_DIRECTORY / "figures"

FINAL_TEST_METRICS_PATH = (
    REPORTS_DIRECTORY
    / "final_test_metrics.json"
)

SELECTED_THRESHOLD_PATH = (
    REPORTS_DIRECTORY
    / "selected_threshold.json"
)

VALIDATION_COMPARISON_PATH = (
    REPORTS_DIRECTORY
    / "model_comparison.csv"
)

FINAL_TEST_CONFUSION_MATRIX_PATH = (
    FIGURES_DIRECTORY
    / "final_test_confusion_matrix.png"
)

FINAL_TEST_PR_CURVE_PATH = (
    FIGURES_DIRECTORY
    / "final_test_precision_recall_curve.png"
)

FINAL_TEST_ROC_CURVE_PATH = (
    FIGURES_DIRECTORY
    / "final_test_roc_curve.png"
)


def initialize_page() -> None:
    """Display the page title and description."""
    st.title("📈 Model Evaluation")
    st.markdown(
        """
        Review the saved validation comparison and untouched final test
        evaluation results for the selected fraud-detection model.
        """
    )


@st.cache_data(show_spinner="Loading evaluation artifacts...")
def load_json_file(
    file_path: Path,
) -> dict:
    """Load a JSON evaluation artifact."""
    if not file_path.exists():
        st.error(f"Required evaluation file not found: {file_path}")
        st.stop()

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


@st.cache_data(show_spinner="Loading evaluation artifacts...")
def load_csv_file(
    file_path: Path,
) -> pd.DataFrame:
    """Load a CSV evaluation artifact."""
    if not file_path.exists():
        st.error(f"Required evaluation file not found: {file_path}")
        st.stop()

    return pd.read_csv(file_path)


def display_final_model_configuration(
    final_test_metrics: dict,
    selected_threshold: dict,
) -> None:
    """Display the final model configuration."""
    st.subheader("Final model configuration")

    configuration_col1, configuration_col2, configuration_col3 = (
        st.columns(3)
    )

    configuration_col1.metric(
        label="Model",
        value=final_test_metrics.get(
            "model_name",
            "Random Forest",
        ),
    )

    configuration_col2.metric(
        label="Decision threshold",
        value=f"{final_test_metrics.get('threshold', 0.0):.2f}",
    )

    configuration_col3.metric(
        label="Threshold-selection dataset",
        value=selected_threshold.get(
            "selection_dataset",
            "Validation",
        ),
    )

    st.caption(
        "The final test set was not used for model selection or "
        "threshold optimization."
    )


def display_final_test_metrics(
    final_test_metrics: dict,
) -> None:
    """Display final test evaluation metrics."""
    st.subheader("Final test metrics")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    metric_col1.metric(
        label="Precision",
        value=f"{final_test_metrics['precision']:.4f}",
    )

    metric_col2.metric(
        label="Recall",
        value=f"{final_test_metrics['recall']:.4f}",
    )

    metric_col3.metric(
        label="F1-score",
        value=f"{final_test_metrics['f1_score']:.4f}",
    )

    metric_col4.metric(
        label="PR-AUC",
        value=f"{final_test_metrics['pr_auc']:.4f}",
    )

    metric_col5, metric_col6, metric_col7, metric_col8 = st.columns(4)

    metric_col5.metric(
        label="ROC-AUC",
        value=f"{final_test_metrics['roc_auc']:.4f}",
    )

    metric_col6.metric(
        label="False-positive rate",
        value=f"{final_test_metrics['false_positive_rate']:.6f}",
    )

    metric_col7.metric(
        label="Transactions flagged",
        value=f"{final_test_metrics['transactions_flagged']:,}",
    )

    metric_col8.metric(
        label="Decision threshold",
        value=f"{final_test_metrics['threshold']:.2f}",
    )


def display_confusion_counts(
    final_test_metrics: dict,
) -> None:
    """Display final test confusion-matrix counts."""
    st.subheader("Final test confusion matrix")

    confusion_col1, confusion_col2, confusion_col3, confusion_col4 = (
        st.columns(4)
    )

    confusion_col1.metric(
        label="True positives",
        value=f"{final_test_metrics['true_positives']:,}",
    )

    confusion_col2.metric(
        label="False positives",
        value=f"{final_test_metrics['false_positives']:,}",
    )

    confusion_col3.metric(
        label="True negatives",
        value=f"{final_test_metrics['true_negatives']:,}",
    )

    confusion_col4.metric(
        label="False negatives",
        value=f"{final_test_metrics['false_negatives']:,}",
    )

    if FINAL_TEST_CONFUSION_MATRIX_PATH.exists():
        st.image(
            str(FINAL_TEST_CONFUSION_MATRIX_PATH),
            caption="Final test confusion matrix",
            use_container_width=True,
        )
    else:
        st.warning(
            "Final test confusion-matrix image was not found."
        )


def display_evaluation_curves(
) -> None:
    """Display the saved final test evaluation curves."""
    st.subheader("Evaluation curves")

    pr_col, roc_col = st.columns(2)

    if FINAL_TEST_PR_CURVE_PATH.exists():
        pr_col.image(
            str(FINAL_TEST_PR_CURVE_PATH),
            caption="Final test precision-recall curve",
            use_container_width=True,
        )
    else:
        pr_col.warning(
            "Final test precision-recall curve image was not found."
        )

    if FINAL_TEST_ROC_CURVE_PATH.exists():
        roc_col.image(
            str(FINAL_TEST_ROC_CURVE_PATH),
            caption="Final test ROC curve",
            use_container_width=True,
        )
    else:
        roc_col.warning(
            "Final test ROC curve image was not found."
        )


def display_validation_comparison(
    validation_results: pd.DataFrame,
) -> None:
    """Display the validation model-comparison results."""
    st.subheader("Validation model comparison")

    st.dataframe(
        validation_results,
        use_container_width=True,
    )

    if "pr_auc" in validation_results.columns:
        validation_chart = px.bar(
            validation_results,
            x="model",
            y="pr_auc",
            color="feature_configuration",
            barmode="group",
            title="Validation PR-AUC by model and feature configuration",
            labels={
                "model": "Model",
                "pr_auc": "PR-AUC",
                "feature_configuration": "Feature configuration",
            },
        )

        validation_chart.update_layout(
            showlegend=False,
        )

        st.plotly_chart(
            validation_chart,
            use_container_width=True,
        )


def main() -> None:
    """Run the Model Evaluation page."""
    initialize_page()

    final_test_metrics = load_json_file(
        file_path=FINAL_TEST_METRICS_PATH,
    )

    selected_threshold = load_json_file(
        file_path=SELECTED_THRESHOLD_PATH,
    )

    display_final_model_configuration(
        final_test_metrics=final_test_metrics,
        selected_threshold=selected_threshold,
    )

    display_final_test_metrics(
        final_test_metrics=final_test_metrics,
    )

    display_confusion_counts(
        final_test_metrics=final_test_metrics,
    )

    display_evaluation_curves()

    if VALIDATION_COMPARISON_PATH.exists():
        validation_results = load_csv_file(
            file_path=VALIDATION_COMPARISON_PATH,
        )

        display_validation_comparison(
            validation_results=validation_results,
        )
    else:
        st.info(
            "Validation model-comparison results were not found."
        )


if __name__ == "__main__":
    main()