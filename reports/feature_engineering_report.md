# Feature Engineering and Baseline Results

## Scope

- Model: Logistic Regression
- Evaluation dataset: Validation set
- Decision threshold: 0.5
- Test set: Not used in this phase

## Feature Configurations

### original_features

- Precision: 0.8333
- Recall: 0.4930
- F1-score: 0.6195
- PR-AUC: 0.6877
- ROC-AUC: 0.9649
- False-positive rate: 0.000165
- Transactions flagged: 42
- False positives: 7
- False negatives: 36
- Training duration: 0.79 seconds

### original_plus_log_amount

- Precision: 0.8372
- Recall: 0.5070
- F1-score: 0.6316
- PR-AUC: 0.6916
- ROC-AUC: 0.9609
- False-positive rate: 0.000165
- Transactions flagged: 43
- False positives: 7
- False negatives: 35
- Training duration: 1.12 seconds

### log_amount_and_time_hours

- Precision: 0.8372
- Recall: 0.5070
- F1-score: 0.6316
- PR-AUC: 0.6938
- ROC-AUC: 0.9614
- False-positive rate: 0.000165
- Transactions flagged: 43
- False positives: 7
- False negatives: 35
- Training duration: 0.97 seconds

## Interpretation

- These are validation results, not final generalization estimates.
- PR-AUC is the primary ranking metric because fraud is extremely rare.
- A higher recall can increase false positives; both must be considered together.
- The selected baseline is chosen using validation PR-AUC only.
- The untouched test set remains reserved for final evaluation.