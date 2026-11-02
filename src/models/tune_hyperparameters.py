"""Reusable Optuna-based hyperparameter tuning for supervised models."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline

logger = logging.getLogger(__name__)

ModelName = Literal[
    "random_forest_class_weighting",
    "random_forest_smote",
    "lightgbm_positive_class_weighting",
]


def load_split(
    feature_path: Path,
    target_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a feature matrix and target vector from Parquet files."""
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    features = pd.read_parquet(feature_path)
    target_frame = pd.read_parquet(target_path)

    if features.empty or target_frame.empty:
        raise ValueError("Loaded split is empty.")

    target_column = target_frame.columns[0]
    target = target_frame[target_column]

    if len(features) != len(target):
        raise ValueError("Feature and target row counts do not match.")

    return features, target


def apply_feature_engineering(features: pd.DataFrame) -> pd.DataFrame:
    """Apply the project's selected feature-engineering configuration."""
    if "Amount" not in features.columns:
        raise ValueError("Required column 'Amount' is missing.")

    if "Time" not in features.columns:
        raise ValueError("Required column 'Time' is missing.")

    if (features["Amount"] < 0).any():
        raise ValueError("Negative Amount values are not supported by log1p.")

    engineered = features.copy()
    engineered["Amount_log1p"] = np.log1p(engineered["Amount"])
    engineered["Time_hours"] = engineered["Time"] / 3600.0

    if "Class" in engineered.columns:
        raise ValueError("Target leakage detected in model features.")

    return engineered


def create_model_pipeline(
    model_name: ModelName,
    params: dict,
    positive_class_weight: float,
) -> Pipeline | ImbalancedPipeline:
    """Create a model pipeline for one supported tuning target."""
    if model_name == "random_forest_class_weighting":
        return Pipeline(
            steps=[
                (
                    "random_forest",
                    RandomForestClassifier(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        min_samples_split=params["min_samples_split"],
                        min_samples_leaf=params["min_samples_leaf"],
                        max_features=params["max_features"],
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )

    if model_name == "random_forest_smote":
        return ImbalancedPipeline(
            steps=[
                (
                    "smote",
                    SMOTE(
                        sampling_strategy="auto",
                        random_state=42,
                    ),
                ),
                (
                    "random_forest",
                    RandomForestClassifier(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        min_samples_split=params["min_samples_split"],
                        min_samples_leaf=params["min_samples_leaf"],
                        max_features=params["max_features"],
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )

    if model_name == "lightgbm_positive_class_weighting":
        return Pipeline(
            steps=[
                (
                    "lightgbm",
                    LGBMClassifier(
                        n_estimators=params["n_estimators"],
                        learning_rate=params["learning_rate"],
                        num_leaves=params["num_leaves"],
                        max_depth=params["max_depth"],
                        min_child_samples=params["min_child_samples"],
                        subsample=params["subsample"],
                        colsample_bytree=params["colsample_bytree"],
                        reg_alpha=params["reg_alpha"],
                        reg_lambda=params["reg_lambda"],
                        scale_pos_weight=positive_class_weight,
                        random_state=42,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported model for tuning: {model_name}")


def suggest_model_parameters(
    trial: optuna.Trial,
    model_name: ModelName,
) -> dict:
    """Suggest a controlled hyperparameter search space for one model."""
    if model_name == "random_forest_class_weighting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                2,
                20,
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                1,
                10,
            ),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2"],
            ),
        }

    if model_name == "random_forest_smote":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                2,
                20,
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                1,
                10,
            ),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2"],
            ),
        }

    if model_name == "lightgbm_positive_class_weighting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.2,
                log=True,
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                10,
                100,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.7,
                1.0,
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.7,
                1.0,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-8,
                10.0,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-8,
                10.0,
                log=True,
            ),
        }

    raise ValueError(f"Unsupported model for tuning: {model_name}")


def create_objective(
    model_name: ModelName,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    positive_class_weight: float,
):
    """Create an Optuna objective that optimizes validation PR-AUC."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_model_parameters(
            trial=trial,
            model_name=model_name,
        )

        pipeline = create_model_pipeline(
            model_name=model_name,
            params=params,
            positive_class_weight=positive_class_weight,
        )

        pipeline.fit(X_train, y_train)

        validation_scores = pipeline.predict_proba(
            X_validation,
        )[:, 1]

        validation_pr_auc = average_precision_score(
            y_validation,
            validation_scores,
        )

        trial.set_user_attr(
            "validation_pr_auc",
            float(validation_pr_auc),
        )

        return float(validation_pr_auc)

    return objective


def tune_model(
    model_name: ModelName,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    n_trials: int = 30,
    timeout_seconds: int | None = 1800,
) -> tuple[optuna.Study, Pipeline, dict]:
    """Tune one supported supervised model using validation PR-AUC."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0:
        raise ValueError("Training data contains no positive fraud examples.")

    positive_class_weight = negative_count / positive_count

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model_name}_validation_pr_auc",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    objective = create_objective(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        positive_class_weight=positive_class_weight,
    )

    tuning_start_time = time.time()

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
    )

    tuning_duration = time.time() - tuning_start_time

    best_pipeline = create_model_pipeline(
        model_name=model_name,
        params=study.best_params,
        positive_class_weight=positive_class_weight,
    )

    best_pipeline.fit(X_train, y_train)

    tuning_metadata = {
        "model_name": model_name,
        "optimization_objective": "validation_pr_auc",
        "n_trials": n_trials,
        "timeout_seconds": timeout_seconds,
        "best_validation_pr_auc": float(study.best_value),
        "best_params": study.best_params,
        "positive_class_weight": positive_class_weight,
        "tuning_duration_seconds": tuning_duration,
        "test_set_used": False,
    }

    logger.info(
        "Tuning completed for %s. Best validation PR-AUC: %.4f",
        model_name,
        study.best_value,
    )

    return study, best_pipeline, tuning_metadata


def save_tuning_results(
    study: optuna.Study,
    best_pipeline: Pipeline | ImbalancedPipeline,
    tuning_metadata: dict,
    output_directory: Path,
) -> dict:
    """Save the tuned model, study results, and tuning metadata."""
    output_directory.mkdir(parents=True, exist_ok=True)

    model_path = output_directory / "tuned_model.joblib"
    study_path = output_directory / "optuna_study_results.csv"
    metadata_path = output_directory / "tuning_metadata.json"

    joblib.dump(best_pipeline, model_path)

    study_results = study.trials_dataframe()
    study_results.to_csv(study_path, index=False)

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            tuning_metadata,
            file,
            indent=2,
            default=str,
        )

    logger.info("Tuned model saved to: %s", model_path)
    logger.info("Optuna study results saved to: %s", study_path)
    logger.info("Tuning metadata saved to: %s", metadata_path)

    return {
        "model_path": str(model_path),
        "study_path": str(study_path),
        "metadata_path": str(metadata_path),
    }