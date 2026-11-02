"""Command-line batch prediction entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.predict import (
    FraudPredictionService,
    PredictionInputError,
    create_prediction_service,
)
from src.data.validate_data import (
    DataValidationError,
    validate_prediction_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

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


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run batch fraud predictions on a CSV file using the final "
            "supervised model."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input CSV file containing transactions.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output CSV file for prediction results.",
    )

    parser.add_argument(
        "--include-anomaly-scores",
        action="store_true",
        help="Also generate anomaly scores using the Isolation Forest model.",
    )

    return parser.parse_args()


def load_prediction_input(
    input_path: Path,
) -> pd.DataFrame:
    """Load and validate the input CSV file."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV file not found: {input_path}"
        )

    if input_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV input file, received: {input_path.suffix}"
        )

    input_dataframe = pd.read_csv(input_path)

    validate_prediction_dataset(
        dataframe=input_dataframe,
    )

    return input_dataframe


def save_prediction_results(
    prediction_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save prediction results to a CSV file."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_results.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "Prediction results saved to: %s",
        output_path,
    )


def main() -> None:
    """Run the batch-prediction workflow."""
    arguments = parse_arguments()

    try:
        input_dataframe = load_prediction_input(
            input_path=arguments.input,
        )

        anomaly_model_directory = (
            DEFAULT_ANOMALY_MODEL_DIRECTORY
            if arguments.include_anomaly_scores
            else None
        )

        prediction_service: FraudPredictionService = (
            create_prediction_service(
                supervised_model_directory=(
                    DEFAULT_SUPERVISED_MODEL_DIRECTORY
                ),
                anomaly_model_directory=anomaly_model_directory,
            )
        )

        prediction_results = prediction_service.predict(
            input_features=input_dataframe,
            include_anomaly_scores=arguments.include_anomaly_scores,
        )

        save_prediction_results(
            prediction_results=prediction_results,
            output_path=arguments.output,
        )

        flagged_count = int(
            prediction_results["review_flag"].sum(),
        )

        flagged_percentage = float(
            prediction_results["review_flag"].mean(),
        )

        logger.info(
            "Processed %s transactions.",
            len(prediction_results),
        )
        logger.info(
            "Flagged %s transactions for review (%.4f%%).",
            flagged_count,
            flagged_percentage * 100,
        )

        print("Batch prediction completed successfully.")
        print(f"Input file: {arguments.input}")
        print(f"Output file: {arguments.output}")
        print(f"Transactions processed: {len(prediction_results)}")
        print(f"Transactions flagged: {flagged_count}")
        print(f"Percentage flagged: {flagged_percentage:.4%}")

    except (
        FileNotFoundError,
        ValueError,
        DataValidationError,
        PredictionInputError,
    ) as error:
        logger.error("Batch prediction failed: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()