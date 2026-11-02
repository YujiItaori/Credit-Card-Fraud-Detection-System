"""Reusable SHAP explainability utilities for supervised fraud models."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

POSITIVE_CLASS_INDEX = 1


def load_supervised_pipeline(
    model_path: Path,
) -> object:
    """Load a saved supervised-model pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    pipeline = joblib.load(model_path)

    if not hasattr(pipeline, "named_steps"):
        raise ValueError(
            "The selected artifact is not a scikit-learn pipeline."
        )

    logger.info(
        "Supervised pipeline loaded from: %s",
        model_path,
    )

    return pipeline


def extract_tree_model(
    pipeline: object,
    model_step_name: str = "random_forest",
) -> object:
    """Extract the tree-based estimator from a pipeline."""
    if model_step_name not in pipeline.named_steps:
        raise ValueError(
            f"Pipeline does not contain a '{model_step_name}' step."
        )

    tree_model = pipeline.named_steps[model_step_name]

    if not hasattr(tree_model, "predict_proba"):
        raise ValueError(
            "The selected model does not support probability predictions."
        )

    logger.info(
        "Extracted tree model from pipeline step: %s",
        model_step_name,
    )

    return tree_model


def create_shap_explainer(
    tree_model: object,
) -> shap.TreeExplainer:
    """Create a SHAP TreeExplainer for a tree-based model."""
    explainer = shap.TreeExplainer(tree_model)

    logger.info("SHAP TreeExplainer created successfully.")

    return explainer


def create_shap_sample(
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    validation_scores: pd.Series,
    validation_predictions: pd.Series,
    sample_size: int,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, dict[str, int]]:
    """Create a reproducible SHAP sample that includes key error categories."""
    if sample_size < 3:
        raise ValueError("SHAP sample size must be at least 3.")

    if sample_size > len(X_validation):
        raise ValueError(
            "SHAP sample size cannot exceed the number of validation transactions."
        )

    true_positive_mask = (
        (y_validation == 1)
        & (validation_predictions == 1)
    )
    false_positive_mask = (
        (y_validation == 0)
        & (validation_predictions == 1)
    )
    false_negative_mask = (
        (y_validation == 1)
        & (validation_predictions == 0)
    )

    if not true_positive_mask.any():
        raise ValueError("No true positives found in the validation set.")

    if not false_positive_mask.any():
        raise ValueError("No false positives found in the validation set.")

    if not false_negative_mask.any():
        raise ValueError("No false negatives found in the validation set.")

    true_positive_index = validation_scores[
        true_positive_mask
    ].idxmax()

    false_positive_index = validation_scores[
        false_positive_mask
    ].idxmax()

    false_negative_index = validation_scores[
        false_negative_mask
    ].idxmax()

    required_indices = [
        true_positive_index,
        false_positive_index,
        false_negative_index,
    ]

    remaining_sample_size = sample_size - len(required_indices)

    random_sample_indices = X_validation.drop(
        index=required_indices,
    ).sample(
        n=remaining_sample_size,
        random_state=random_seed,
    ).index

    shap_sample_indices = required_indices + list(random_sample_indices)

    X_shap_sample = X_validation.loc[shap_sample_indices]
    y_shap_sample = y_validation.loc[shap_sample_indices]
    shap_sample_scores = validation_scores.loc[shap_sample_indices]
    shap_sample_predictions = validation_predictions.loc[
        shap_sample_indices
    ]

    explanation_indices = {
        "true_positive": true_positive_index,
        "false_positive": false_positive_index,
        "false_negative": false_negative_index,
    }

    logger.info(
        "Created SHAP sample with shape: %s",
        X_shap_sample.shape,
    )

    return (
        X_shap_sample,
        y_shap_sample,
        shap_sample_scores,
        shap_sample_predictions,
        explanation_indices,
    )


def calculate_shap_values(
    explainer: shap.TreeExplainer,
    X_shap_sample: pd.DataFrame,
) -> np.ndarray:
    """Calculate positive-class SHAP values for a validation sample."""
    shap_values = explainer.shap_values(X_shap_sample)

    if isinstance(shap_values, list):
        shap_values_positive_class = shap_values[POSITIVE_CLASS_INDEX]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values_positive_class = shap_values[
            :,
            :,
            POSITIVE_CLASS_INDEX,
        ]
    else:
        raise ValueError(
            "Unexpected SHAP output format. Expected a list or "
            "three-dimensional array."
        )

    if shap_values_positive_class.shape != X_shap_sample.shape:
        raise ValueError(
            "SHAP values and SHAP sample shapes do not match."
        )

    logger.info(
        "Calculated SHAP values with shape: %s",
        shap_values_positive_class.shape,
    )

    return shap_values_positive_class


def calculate_global_importance(
    shap_values_positive_class: np.ndarray,
    X_shap_sample: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate mean absolute SHAP importance for each feature."""
    mean_absolute_shap_values = np.abs(
        shap_values_positive_class,
    ).mean(axis=0)

    importance_dataframe = pd.DataFrame(
        {
            "feature": X_shap_sample.columns,
            "mean_absolute_shap_value": mean_absolute_shap_values,
        }
    ).sort_values(
        by="mean_absolute_shap_value",
        ascending=False,
    )

    logger.info("Global SHAP importance calculated.")

    return importance_dataframe


def plot_global_importance(
    importance_dataframe: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Plot and save global SHAP feature importance."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_features = importance_dataframe.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(
        top_features["feature"][::-1],
        top_features["mean_absolute_shap_value"][::-1],
    )
    plt.title("Global SHAP Feature Importance")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    logger.info(
        "Global SHAP importance figure saved to: %s",
        output_path,
    )


def plot_shap_summary(
    shap_values_positive_class: np.ndarray,
    X_shap_sample: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot and save a SHAP summary plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(
        shap_values_positive_class,
        X_shap_sample,
        show=False,
    )
    plt.title("SHAP Summary")
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    logger.info(
        "SHAP summary figure saved to: %s",
        output_path,
    )


def get_positive_class_base_value(
    explainer: shap.TreeExplainer,
) -> float:
    """Return the SHAP base value for the positive fraud class."""
    expected_value = explainer.expected_value

    if isinstance(
        expected_value,
        (list, tuple, np.ndarray),
    ):
        if len(expected_value) <= POSITIVE_CLASS_INDEX:
            raise ValueError(
                "SHAP explainer does not contain a positive-class base value."
            )

        return float(expected_value[POSITIVE_CLASS_INDEX])

    return float(expected_value)


def plot_individual_explanation(
    explanation_index: int | np.integer,
    shap_values_positive_class: np.ndarray,
    X_shap_sample: pd.DataFrame,
    explainer: shap.TreeExplainer,
    title: str,
    output_path: Path,
    max_display: int = 15,
) -> None:
    """Plot and save a SHAP waterfall explanation for one transaction."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_position = X_shap_sample.index.get_loc(explanation_index)

    positive_class_base_value = get_positive_class_base_value(
        explainer=explainer,
    )

    explanation = shap.Explanation(
        values=shap_values_positive_class[sample_position],
        base_values=positive_class_base_value,
        data=X_shap_sample.loc[explanation_index],
        feature_names=list(X_shap_sample.columns),
    )

    plt.figure()
    shap.plots.waterfall(
        explanation,
        max_display=max_display,
        show=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    logger.info(
        "Individual SHAP explanation saved to: %s",
        output_path,
    )