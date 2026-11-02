# Supervised Candidate Selection Report

## Selection Policy

- Primary ranking metric: Validation PR-AUC
- Shortlist size: 3
- Additional high-recall candidates require recall >= 0.80 and precision >= 0.50
- Test set: Not used

## Selected Candidates

### Random Forest | Class weighting

- Validation PR-AUC: 0.8494
- Validation ROC-AUC: 0.9456
- Precision: 0.9636
- Recall: 0.7465
- F1-score: 0.8413
- False positives: 2
- False negatives: 18
- Selection reason: Top validation PR-AUC candidate

### Random Forest | SMOTE

- Validation PR-AUC: 0.8407
- Validation ROC-AUC: 0.9617
- Precision: 0.8889
- Recall: 0.7887
- F1-score: 0.8358
- False positives: 7
- False negatives: 15
- Selection reason: Top validation PR-AUC candidate

### Logistic Regression | SMOTE

- Validation PR-AUC: 0.7091
- Validation ROC-AUC: 0.9692
- Precision: 0.0478
- Recall: 0.8873
- F1-score: 0.0907
- False positives: 1,255
- False negatives: 8
- Selection reason: Top validation PR-AUC candidate
