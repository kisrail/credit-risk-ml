# Loan Default Prediction

Based on the ML in Industry Course @ UniMi

---
## What it does
The model estimates the Probability of Default (PD) for consumer loans.

## How it works
The pipeline includes:
- Exploratory data analysis
- Dataset pre-processing
- Feature engineering
- Model training and comparison
- Cross-validation and out-of-sample evaluation

The models included are:
- Logistic Regression with all features (baseline)
- Logistic Regression with selected features
- Random Forest
- XGBoost

Model performance is evaluated using cross-validated ROC-AUC score

## Output
The final model produces a predicted probability of default for each loan in the scoring dataset (`submission.csv`)