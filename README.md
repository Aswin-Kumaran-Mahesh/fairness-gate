# Fairness Gate CI/CD Pipeline

Automated fairness governance pipeline for tabular ML models.
Runs on every push via GitHub Actions.

## Metrics Evaluated
- Demographic Parity Difference (DPD)
- Equalized Odds Difference
- FPR Gap / FNR Gap
- SHAP Attribution Drift

## Modes
- PASS: all metrics within threshold, exit 0
- FAIL: threshold breach detected, exit 1, deployment blocked
