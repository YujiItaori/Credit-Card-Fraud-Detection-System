# Data Quality Report

## Source Dataset

- Source file: `creditcard.csv`
- Raw rows: 284,807
- Raw columns: 31
- Missing values: 0
- Exact duplicate rows involved in duplication groups: 1,854

## Cleaning Decision

Remove exact duplicate rows, retaining the first occurrence.

**Justification:** Rows are identical across every feature and the target label. Retaining them could allow identical transactions to appear in both training and evaluation partitions, weakening split independence.

- Rows before cleaning: 284,807
- Rows after cleaning: 283,726
- Rows removed: 1,081
- Legitimate transactions before cleaning: 284,315
- Legitimate transactions after cleaning: 283,253
- Fraudulent transactions before cleaning: 492
- Fraudulent transactions after cleaning: 473

## Split Integrity

- Training rows: 198,608
- Validation rows: 42,559
- Test rows: 42,559
- Train-validation overlap: 0
- Train-test overlap: 0
- Validation-test overlap: 0

## Preprocessing Policy

- The target column `Class` is excluded from all feature matrices.
- Scaling is not applied in this notebook.
- Any scaler or imputer must be fitted using training data only.
- Validation and test data must be transformed using already-fitted training components.
- No outliers are removed in this phase.

## Limitations

- `V1` to `V28` are anonymized PCA-transformed features and must not be assigned business meanings.
- `Time` is elapsed seconds from the first transaction, not a calendar timestamp.
- Extreme values are retained because they may be legitimate transactions or useful signals.
