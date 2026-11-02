# Supervised vs Anomaly Detection Evaluation

## Scope

- Feature configuration: `log_amount_and_time_hours`
- Evaluation dataset: Validation set
- Supervised decision threshold: 0.5
- Anomaly decision rule: Fitted contamination-based rule
- Test set: Not used

## Validation Results

### Random Forest | Class weighting

- Model type: Supervised
- Decision rule: Probability >= 0.5
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

### Random Forest | SMOTE

- Model type: Supervised
- Decision rule: Probability >= 0.5
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

### Logistic Regression | SMOTE

- Model type: Supervised
- Decision rule: Probability >= 0.5
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

### Isolation Forest | Baseline

- Model type: Unsupervised anomaly detection
- Decision rule: Fitted contamination-based anomaly rule
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

### Local Outlier Factor | Baseline

- Model type: Unsupervised anomaly detection
- Decision rule: Fitted contamination-based anomaly rule
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

## Agreement Analysis

Primary supervised model: Random Forest | Class weighting

Primary anomaly detector: Isolation Forest | Baseline

### A: Both models flag

- Transactions: 8
- Actual frauds: 8
- Fraud rate: 1.0000
- Median amount: 68.84

### B: Supervised model only

- Transactions: 47
- Actual frauds: 45
- Fraud rate: 0.9574
- Median amount: 24.90

### C: Anomaly detector only

- Transactions: 28
- Actual frauds: 0
- Fraud rate: 0.0000
- Median amount: 1557.55

### D: Neither model flags

- Transactions: 42,476
- Actual frauds: 18
- Fraud rate: 0.0004
- Median amount: 21.02

## Interpretation

- Supervised and anomaly-detection metrics are not directly interchangeable because their decision rules differ.
- PR-AUC is the primary ranking metric because fraud is extremely rare.
- Anomaly scores are not fraud probabilities.
- Agreement between models does not prove that a transaction is fraudulent.
- The strongest supervised candidate remains Random Forest | Class weighting based on validation PR-AUC.
- The untouched test set remains reserved for final evaluation.