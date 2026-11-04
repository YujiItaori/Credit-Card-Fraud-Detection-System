"""Generate a reproducible sample input file for batch fraud predictions.

The output contains 1,000 real transactions selected from the original
European Credit Card Fraud Detection dataset. It includes both legitimate
and fraudulent transactions, but excludes the Class target column because
the file is intended for prediction input.
"""

from pathlib import Path

import pandas as pd

# Project-root-relative paths.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "prediction_input_sample.csv"

TOTAL_ROWS = 1_000
FRAUD_ROWS = 50
LEGITIMATE_ROWS = TOTAL_ROWS - FRAUD_ROWS
RANDOM_STATE = 42

TARGET_COLUMN = "Class"
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def load_raw_dataset(input_path: Path) -> pd.DataFrame:
    """Load and validate the original credit card fraud dataset."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Original dataset not found at: {input_path}\n"
            "Place creditcard.csv in data/raw/ or pass --input with its path."
        )

    dataframe = pd.read_csv(input_path)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"The dataset is missing required columns: {missing_columns}"
        )

    invalid_labels = set(dataframe[TARGET_COLUMN].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(
            f"The Class column contains unexpected values: {sorted(invalid_labels)}"
        )

    return dataframe


def generate_sample(
    dataframe: pd.DataFrame,
    total_rows: int = TOTAL_ROWS,
    fraud_rows: int = FRAUD_ROWS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Select real legitimate and fraudulent transactions reproducibly."""
    if fraud_rows >= total_rows:
        raise ValueError("fraud_rows must be smaller than total_rows.")

    legitimate_available = int((dataframe[TARGET_COLUMN] == 0).sum())
    fraud_available = int((dataframe[TARGET_COLUMN] == 1).sum())
    legitimate_rows = total_rows - fraud_rows

    if legitimate_available < legitimate_rows:
        raise ValueError(
            f"Need {legitimate_rows} legitimate transactions, "
            f"but only {legitimate_available} are available."
        )

    if fraud_available < fraud_rows:
        raise ValueError(
            f"Need {fraud_rows} fraudulent transactions, "
            f"but only {fraud_available} are available."
        )

    legitimate_sample = dataframe[dataframe[TARGET_COLUMN] == 0].sample(
        n=legitimate_rows,
        random_state=random_state,
    )

    fraud_sample = dataframe[dataframe[TARGET_COLUMN] == 1].sample(
        n=fraud_rows,
        random_state=random_state,
    )

    sample = pd.concat([legitimate_sample, fraud_sample], ignore_index=True)
    sample = sample.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return sample


def main() -> None:
    dataframe = load_raw_dataset(DEFAULT_INPUT_PATH)
    sample = generate_sample(dataframe)

    legitimate_count = int((sample[TARGET_COLUMN] == 0).sum())
    fraud_count = int((sample[TARGET_COLUMN] == 1).sum())

    print("Selected sample summary before removing the target column:")
    print(f"  Total rows: {len(sample)}")
    print(f"  Total columns: {len(sample.columns)}")
    print(f"  Columns: {list(sample.columns)}")
    print(f"  Legitimate transactions (Class=0): {legitimate_count}")
    print(f"  Fraudulent transactions (Class=1): {fraud_count}")

    prediction_input = sample[FEATURE_COLUMNS].copy()

    if TARGET_COLUMN in prediction_input.columns:
        raise AssertionError("Class column must not be included in prediction input.")

    if list(prediction_input.columns) != FEATURE_COLUMNS:
        raise AssertionError("Prediction input columns do not match the API schema.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prediction_input.to_csv(OUTPUT_PATH, index=False)

    print("\nGenerated prediction input file:")
    print(f"  Path: {OUTPUT_PATH}")
    print(f"  Rows: {len(prediction_input)}")
    print(f"  Columns: {len(prediction_input.columns)}")
    print(f"  Class column included: {TARGET_COLUMN in prediction_input.columns}")


if __name__ == "__main__":
    main()