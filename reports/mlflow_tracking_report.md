# MLflow Experiment Tracking Report

## Tracking Configuration

- Tracking URI: `sqlite:///c:/Users/use/Desktop/Credit Card Fraud Detection/mlflow.db`
- Experiment name: `fraud-detection-anomaly-analysis`
- Experiment ID: `1`
- Tracking report generated: 2026-11-01T07:07:52.679108+00:00
- Evaluation dataset: Validation set
- Test set: Not used

## Logged Experiments

### Logistic Regression | Class weighting

- MLflow run ID: `2719890a69db4230877e3ed3bb6d6a76`
- Model: Logistic Regression
- Feature configuration: log_amount_and_time_hours
- Imbalance strategy: None
- Validation PR-AUC: 0.7066
- Validation recall: 0.8873
- Validation precision: 0.0506

### Random Forest | Class weighting

- MLflow run ID: `243f72fe5ebd4625bfad0abff6efd5dd`
- Model: Random Forest
- Feature configuration: log_amount_and_time_hours
- Imbalance strategy: None
- Validation PR-AUC: 0.8494
- Validation recall: 0.7465
- Validation precision: 0.9636

### LightGBM | Positive-class weighting

- MLflow run ID: `5d2e05582a76427590eb675fd09e38fb`
- Model: LightGBM
- Feature configuration: log_amount_and_time_hours
- Imbalance strategy: None
- Validation PR-AUC: 0.0499
- Validation recall: 0.6479
- Validation precision: 0.0762

### Logistic Regression | SMOTE

- MLflow run ID: `d138137788774988b80012767a1b3a40`
- Model: Logistic Regression
- Feature configuration: log_amount_and_time_hours
- Imbalance strategy: None
- Validation PR-AUC: 0.7091
- Validation recall: 0.8873
- Validation precision: 0.0478

### Random Forest | SMOTE

- MLflow run ID: `e0601f75ffa64e7ab31c03b6e5b82fb3`
- Model: Random Forest
- Feature configuration: log_amount_and_time_hours
- Imbalance strategy: None
- Validation PR-AUC: 0.8407
- Validation recall: 0.7887
- Validation precision: 0.8889

### Isolation Forest | Baseline

- MLflow run ID: `7fc3b089c9c84700b8f56ead344b7a0a`
- Model: Isolation Forest
- Feature configuration: log_amount_and_time_hours
- Imbalance strategy: None
- Validation PR-AUC: 0.0796
- Validation recall: 0.1127
- Validation precision: 0.2222

### Local Outlier Factor | Baseline

- MLflow run ID: `b8ba0c7213a54bc58ccbbc592b07a2c2`
- Model: Local Outlier Factor
- Feature configuration: log_amount_and_time_hours
- Imbalance strategy: None
- Validation PR-AUC: 0.0026
- Validation recall: 0.0000
- Validation precision: 0.0000

## Selected Supervised Candidate

- Model: `Random Forest | Class weighting`
- MLflow run ID: `36723fd06c664c01af482203907f7f95`
- Status: Validation candidate; not yet the final production model

## Important Notes

- All logged metrics are validation metrics.
- The test set was not used for model selection or threshold optimization.
- The selected model artifact is logged for reproducibility and inspection.
- Final test evaluation must occur only after threshold selection is complete.