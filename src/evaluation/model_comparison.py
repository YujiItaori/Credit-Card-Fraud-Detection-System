"""Reusable model-comparison utilities for fraud-detection experiments."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COMPARISON_COLUMNS = {
    "experiment_name",
    "model",
    "precision",
    "recall",
    "f1_score",
    "pr_auc",
    "roc_auc",
    "false_positive_rate",
}

CORE_METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1_score",
    "pr_auc",
    "roc_auc",
    "false_positive_rate",
]

DISPLAY_COLUMNS = [
    "experiment_name",
    "model",
    "feature_configuration",
    "imbalance_strategy",
    "precision",
    "recall",
    "f1_score",
    "pr_auc",
    "roc_auc",
    "false_positive_rate",
    "transactions_flagged",
    "false_positives",
    "false_negatives",
]


def load_experiment_results(
    comparison_path: Path,
) -> pd.DataFrame:
    """Load and validate the consolidated experiment-results CSV."""
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Model comparison report not found: {comparison_path}"
        )

    results = pd.read_csv(comparison_path)

    missing_columns = REQUIRED_COMPARISON_COLUMNS - set(results.columns)

    if missing_columns:
        raise ValueError(
            f"Model comparison report is missing columns: "
            f"{sorted(missing_columns)}"
        )

    valid_results = results[
        results["experiment_name"].notna()
        & results["experiment_name"].astype(str).str.strip().ne("")
    ].copy()

    valid_results["experiment_name"] = valid_results[
        "experiment_name"
    ].astype(str).str.strip()

    if valid_results.empty:
        raise ValueError("No valid experiment results were found.")

    logger.info(
        "Loaded %s valid experiment results from %s",
        len(valid_results),
        comparison_path,
    )

    return valid_results


def rank_experiments_by_pr_auc(
    experiment_results: pd.DataFrame,
) -> pd.DataFrame:
    """Rank experiments by validation PR-AUC and F1-score."""
    if experiment_results.empty:
        raise ValueError("No experiment results were provided.")

    ranked_results = experiment_results.sort_values(
        by=["pr_auc", "f1_score"],
        ascending=[False, False],
    ).copy()

    ranked_results["pr_auc_rank"] = range(
        1,
        len(ranked_results) + 1,
    )

    return ranked_results


def select_candidate_experiments(
    experiment_results: pd.DataFrame,
    shortlist_size: int = 3,
    minimum_precision_for_high_recall: float = 0.50,
    minimum_recall_for_high_recall: float = 0.80,
) -> pd.DataFrame:
    """Select candidate experiments using validation PR-AUC and recall rules."""
    ranked_results = rank_experiments_by_pr_auc(
        experiment_results=experiment_results,
    )

    top_pr_auc_candidates = ranked_results.head(shortlist_size)[
        "experiment_name"
    ].tolist()

    high_recall_candidates = ranked_results[
        (ranked_results["recall"] >= minimum_recall_for_high_recall)
        & (
            ranked_results["precision"]
            >= minimum_precision_for_high_recall
        )
    ]["experiment_name"].tolist()

    selected_experiment_names = list(
        dict.fromkeys(top_pr_auc_candidates + high_recall_candidates)
    )

    candidate_results = ranked_results[
        ranked_results["experiment_name"].isin(selected_experiment_names)
    ].copy()

    candidate_results["selection_reason"] = candidate_results[
        "experiment_name"
    ].map(
        lambda experiment_name: (
            "Top validation PR-AUC candidate"
            if experiment_name in top_pr_auc_candidates
            else ""
        )
        + (
            "; High-recall candidate"
            if experiment_name in high_recall_candidates
            else ""
        )
    )

    logger.info(
        "Selected %s candidate experiments.",
        len(candidate_results),
    )

    return candidate_results


def compare_experiment_results(
    baseline_results: pd.DataFrame,
    tuned_results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare baseline and tuned results for the same experiment names."""
    required_tuned_columns = {
        "experiment_name",
        *CORE_METRIC_COLUMNS,
    }

    missing_columns = required_tuned_columns - set(tuned_results.columns)

    if missing_columns:
        raise ValueError(
            f"Tuned results are missing columns: {sorted(missing_columns)}"
        )

    comparison_columns = [
        "experiment_name",
        *CORE_METRIC_COLUMNS,
    ]

    baseline_subset = baseline_results[
        baseline_results["experiment_name"].isin(
            tuned_results["experiment_name"]
        )
    ][comparison_columns].set_index("experiment_name")

    tuned_subset = tuned_results[comparison_columns].set_index(
        "experiment_name"
    )

    comparison = baseline_subset.join(
        tuned_subset,
        lsuffix="_baseline",
        rsuffix="_tuned",
    )

    for metric_column in CORE_METRIC_COLUMNS:
        baseline_column = f"{metric_column}_baseline"
        tuned_column = f"{metric_column}_tuned"
        difference_column = f"{metric_column}_difference"

        if baseline_column in comparison.columns:
            comparison[difference_column] = (
                comparison[tuned_column] - comparison[baseline_column]
            )

    logger.info(
        "Compared %s baseline experiments with tuned results.",
        len(comparison),
    )

    return comparison.reset_index()


def save_model_comparison(
    comparison_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a model-comparison table to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_results.to_csv(output_path, index=False)

    logger.info(
        "Model-comparison results saved to: %s",
        output_path,
    )


def create_comparison_display_table(
    experiment_results: pd.DataFrame,
) -> pd.DataFrame:
    """Create a concise display table using available columns."""
    available_columns = [
        column
        for column in DISPLAY_COLUMNS
        if column in experiment_results.columns
    ]

    return experiment_results[available_columns]