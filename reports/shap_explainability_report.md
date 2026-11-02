# SHAP Explainability Report

## Scope

- Selected model: `Random Forest | Class weighting`
- Feature configuration: `log_amount_and_time_hours`
- SHAP sample size: 2,000
- Data source: Validation set
- Positive class: Fraudulent transaction (`Class = 1`)
- Test set: Not used

## Important Limitation

- The features `V1` through `V28` are anonymized transformed variables.
- Their original business meanings are unavailable.
- SHAP explains how the model uses these features; it does not identify the cause of fraud.
- SHAP values do not prove that a transaction is fraudulent.

## Global Feature Importance

- `V12`: 0.074604
- `V14`: 0.071109
- `V4`: 0.051970
- `V10`: 0.051205
- `V3`: 0.049813
- `V11`: 0.043228
- `V17`: 0.034913
- `V16`: 0.020308
- `V7`: 0.012303
- `V9`: 0.010852
- `V1`: 0.008295
- `V18`: 0.007467
- `V2`: 0.006195
- `V8`: 0.005546
- `V5`: 0.005258

## Individual Explanations

- High-confidence true positive validation index: `42936`
- High-confidence false positive validation index: `43281`
- High-confidence false negative validation index: `6971`

## Interpretation

- Positive SHAP values increase the model’s fraud score.
- Negative SHAP values decrease the model’s fraud score.
- The waterfall plots show the feature contributions for one selected transaction.
- Threshold optimization and final test evaluation occur in later phases.