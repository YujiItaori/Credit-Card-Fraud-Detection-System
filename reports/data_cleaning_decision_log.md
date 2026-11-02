# Data Cleaning Decision Log

## Exact Duplicate Rows

- Decision: Remove exact duplicate rows, retaining the first occurrence.
- Justification: Rows are identical across every feature and the target label. Retaining them could allow identical transactions to appear in both training and evaluation partitions, weakening split independence.
- Rows removed: 1,081
- Rows remaining: 283,726

## Outliers

- No outliers were removed. Extreme values may represent valid transactions or useful fraud-related signals.

## Missing Values

- No missing values were found; therefore, no imputation was applied.

## Data Types

- All columns were already numeric; no type conversion was necessary.
