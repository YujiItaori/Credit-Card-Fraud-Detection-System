# Anomaly Detection Experiment Results

## Scope

- Feature configuration: `log_amount_and_time_hours`
- Training data: Training features only
- Evaluation dataset: Validation set
- Fraud labels: Used only for evaluation, not training
- Test set: Not used in this phase
- Score convention: Higher anomaly score means more anomalous
- Anomaly score is not a fraud probability

## Validation Results

### Isolation Forest | Baseline

- Precision: 0.2222
- Recall: 0.1127
- F1-score: 0.1495
- PR-AUC: 0.0796
- ROC-AUC: 0.9332
- False-positive rate: 0.000659
- Transactions flagged: 36
- True positives: 8
- False positives: 28
- False negatives: 63
- Training duration: 9.20 seconds
- Preprocessing: No scaling required

### Local Outlier Factor | Baseline

- Precision: 0.0000
- Recall: 0.0000
- F1-score: 0.0000
- PR-AUC: 0.0026
- ROC-AUC: 0.5149
- False-positive rate: 0.001106
- Transactions flagged: 47
- True positives: 0
- False positives: 47
- False negatives: 71
- Training duration: 135.39 seconds
- Preprocessing: StandardScaler fitted on training data only

## Interpretation

- These are validation results, not final generalization estimates.
- Anomaly detectors identify unusual transactions, not confirmed fraud.
- PR-AUC is the primary ranking metric because fraud is extremely rare.
- The contamination parameter controls the initial anomaly-flagging rate but does not define the true fraud rate.
- The untouched test set remains reserved for final evaluation.