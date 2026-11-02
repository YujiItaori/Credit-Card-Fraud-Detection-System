# Supervised Error Analysis Report

## Scope

- Dataset: Validation set
- Decision threshold: 0.5
- Test set: Not used

## Error Summary

### Random Forest | Class weighting

- False positives: 2
- False negatives: 18
- False-positive median amount: 5.00
- False-negative median amount: 2.00

### Random Forest | SMOTE

- False positives: 7
- False negatives: 15
- False-positive median amount: 8.99
- False-negative median amount: 1.18

### Logistic Regression | SMOTE

- False positives: 1,255
- False negatives: 8
- False-positive median amount: 18.73
- False-negative median amount: 1.09

## Interpretation

- False positives are legitimate transactions flagged for review.
- False negatives are fraudulent transactions missed by the model.
- Transaction amount alone does not explain or justify either error type.
- Threshold optimization in a later phase may change these counts.