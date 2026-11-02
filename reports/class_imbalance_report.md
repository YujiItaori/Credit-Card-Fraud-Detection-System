# Class Imbalance Experiment Results

## Scope

- Feature configuration: `log_amount_and_time_hours`
- Evaluation dataset: Validation set
- Decision threshold: 0.5
- Test set: Not used in this phase
- SMOTE and preprocessing were applied only during training fit

## Validation Results

### Logistic Regression | Class weighting

- Precision: 0.0506
- Recall: 0.8873
- F1-score: 0.0957
- PR-AUC: 0.7066
- ROC-AUC: 0.9681
- False-positive rate: 0.027820
- Transactions flagged: 1,245
- True positives: 63
- False positives: 1,182
- False negatives: 8
- Training duration: 3.01 seconds
- Preprocessing: StandardScaler fitted on training data only

### Random Forest | Class weighting

- Precision: 0.9636
- Recall: 0.7465
- F1-score: 0.8413
- PR-AUC: 0.8494
- ROC-AUC: 0.9456
- False-positive rate: 0.000047
- Transactions flagged: 55
- True positives: 53
- False positives: 2
- False negatives: 18
- Training duration: 52.73 seconds
- Preprocessing: No scaling required

### LightGBM | Positive-class weighting

- Precision: 0.0762
- Recall: 0.6479
- F1-score: 0.1363
- PR-AUC: 0.0499
- ROC-AUC: 0.8174
- False-positive rate: 0.013133
- Transactions flagged: 604
- True positives: 46
- False positives: 558
- False negatives: 25
- Training duration: 8.27 seconds
- Preprocessing: No scaling required

### Logistic Regression | SMOTE

- Precision: 0.0478
- Recall: 0.8873
- F1-score: 0.0907
- PR-AUC: 0.7091
- ROC-AUC: 0.9692
- False-positive rate: 0.029538
- Transactions flagged: 1,318
- True positives: 63
- False positives: 1,255
- False negatives: 8
- Training duration: 4.84 seconds
- Preprocessing: StandardScaler and SMOTE fitted on training data only

### Random Forest | SMOTE

- Precision: 0.8889
- Recall: 0.7887
- F1-score: 0.8358
- PR-AUC: 0.8407
- ROC-AUC: 0.9617
- False-positive rate: 0.000165
- Transactions flagged: 63
- True positives: 56
- False positives: 7
- False negatives: 15
- Training duration: 277.36 seconds
- Preprocessing: SMOTE fitted on training data only

## Interpretation

- These are validation results, not final generalization estimates.
- PR-AUC is the primary ranking metric because fraud is extremely rare.
- Class weighting and SMOTE may increase recall while reducing precision.
- The best configuration must be selected using validation PR-AUC and operational false-positive trade-offs.
- The untouched test set remains reserved for final evaluation.