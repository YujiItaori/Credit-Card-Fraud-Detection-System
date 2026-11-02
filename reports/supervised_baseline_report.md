# Supervised Baseline Model Results

## Scope

- Feature configuration: `log_amount_and_time_hours`
- Evaluation dataset: Validation set
- Decision threshold: 0.5
- Imbalance strategy: None
- Test set: Not used in this phase

## Validation Results

### Logistic Regression

- Precision: 0.8372
- Recall: 0.5070
- F1-score: 0.6316
- PR-AUC: 0.6938
- ROC-AUC: 0.9614
- False-positive rate: 0.000165
- Transactions flagged: 43
- True positives: 36
- False positives: 7
- False negatives: 35
- Training duration: 1.23 seconds
- Preprocessing: StandardScaler fitted on training data only

### Random Forest

- Precision: 0.9815
- Recall: 0.7465
- F1-score: 0.8480
- PR-AUC: 0.8217
- ROC-AUC: 0.9455
- False-positive rate: 0.000024
- Transactions flagged: 54
- True positives: 53
- False positives: 1
- False negatives: 18
- Training duration: 124.48 seconds
- Preprocessing: No scaling required

### LightGBM

- Precision: 0.2734
- Recall: 0.5352
- F1-score: 0.3619
- PR-AUC: 0.3478
- ROC-AUC: 0.7015
- False-positive rate: 0.002377
- Transactions flagged: 139
- True positives: 38
- False positives: 101
- False negatives: 33
- Training duration: 5.79 seconds
- Preprocessing: No scaling required

## Interpretation

- These are validation results, not final generalization estimates.
- PR-AUC is the primary ranking metric because fraud is extremely rare.
- A model with higher recall may produce more false positives.
- Class weighting and resampling will be investigated in the next notebook.
- The untouched test set remains reserved for final evaluation.