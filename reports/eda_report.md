# Exploratory Data Analysis Report

## Dataset Overview

- Rows: 284,807
- Columns: 31
- Numerical columns: 31
- Categorical columns: 0
- Missing values: 0
- Duplicate rows: 1,081 (0.380%)

## Target Distribution

- Legitimate transactions: 284,315
- Fraudulent transactions: 492
- Legitimate percentage: 99.827%
- Fraudulent percentage: 0.173%

The dataset is extremely imbalanced. Accuracy alone is not an appropriate primary evaluation metric.

## Transaction Amount

Fraudulent transactions have a higher mean amount but a lower median amount than legitimate transactions. This indicates skewness and substantial distribution overlap. Transaction amount alone must not be used to classify fraud.

## Time Feature

`Time` represents elapsed seconds from the first transaction and covers approximately 48 hours. It must not be interpreted as a calendar timestamp, hour of day, or day of week.

## Feature Correlations

The strongest absolute Pearson correlations with `Class` are:

- `V17`: -0.3265
- `V14`: -0.3025
- `V12`: -0.2606
- `V10`: -0.2169
- `V16`: -0.1965
- `V3`: -0.1930
- `V7`: -0.1873
- `V11`: 0.1549
- `V4`: 0.1334
- `V18`: -0.1115

These correlations describe linear associations only. They do not establish causation or guarantee predictive value for nonlinear models.

## Potential Outliers

- High-amount threshold (99th percentile): 1017.97
- High-amount transactions: 2,849
- High-amount fraudulent transactions: 9
- High-amount legitimate transactions: 2,840
- PCA z-score outlier candidates: 7,952
- PCA outlier candidates that are fraudulent: 407
- PCA outlier candidates that are legitimate: 7,545

Potential outliers are observations for further investigation. They are not automatically invalid records or confirmed fraud cases.

## Implications for Later Phases

- Duplicate records must be investigated before deciding whether to retain or remove them.
- The extreme class imbalance requires evaluation using precision, recall, F1-score, PR-AUC, and confusion matrices.
- Amount is highly skewed; transformations may be investigated later but must not be applied solely to remove outliers.
- Time is elapsed seconds, not calendar time; no hour-of-day or day-of-week features can be derived reliably.
- PCA features are anonymized and must not be assigned business meanings.
- Potential outliers are candidates for investigation, not confirmed fraud or invalid records.
