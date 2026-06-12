# HR Analytics — Salary Prediction & Hiring Classification

**Supervised Machine Learning · Regression · Classification · Python**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Overview

This project builds two end-to-end predictive models to support data-driven HR decision-making:

| Part | Task | Best Model | Key Metric |
|------|------|-----------|------------|
| 1 | Salary prediction | Ridge Regression | R² = 0.651 · RMSE ≈ 5,967 |
| 2 | Hiring classification | Random Forest | Accuracy 91.3% · AUC ≈ 0.97 |

The workflow mirrors quantitative modelling in finance: feature engineering, regularisation, model selection via cross-validation, and rigorous out-of-sample evaluation.

---

## Business Problem

HR departments routinely make compensation and hiring decisions based on intuition rather than evidence. Uncalibrated salary offers drive attrition; inconsistent hiring criteria introduce costly bias. This project answers two concrete questions:

1. **What salary should we expect to pay?** Given a candidate's experience, seniority, and education, what is the expected market salary?
2. **Will this candidate be hired?** Given the same attributes plus the salary, what is the probability of a positive hiring outcome?

Replacing gut-feel benchmarks with model-derived estimates reduces both over-payment and candidate drop-off.

---

## Dataset

| File | Rows | Features | Target |
|------|------|----------|--------|
| `hr_data.csv` | 1,000 | 5 | `Salario` (salary, continuous) |
| `hr_data_validation.csv` | 1,000 | 6 | `Contratado` (hired, binary) |

**Features**

| Column | Type | Description |
|--------|------|-------------|
| `Experiencia` | Continuous | Years of professional experience |
| `Posicion` | Categorical | Job level: Analyst · Coordinator · Manager |
| `Hijos` | Integer | Number of children |
| `Casado` | Binary | Marital status (1 = married) |
| `Educacion` | Categorical | Highest qualification: Bachelor · Licentiate · Postgraduate |
| `Salario` | Continuous | Annual salary (used as feature in Part 2) |

---

## Methodology

### Part 1 — Salary Prediction (Regression)

```
Raw data → One-hot encoding → StandardScaler → 80/20 split
         → OLS · Ridge · Lasso · ElasticNet
         → CV-tuned alpha selection → RMSE / R² evaluation
```

- Categorical features (`Posicion`, `Educacion`) are one-hot encoded with `drop_first=True` to avoid multicollinearity in linear models.
- All features are standardised so that regularisation penalties are applied uniformly.
- `RidgeCV`, `LassoCV`, and `ElasticNetCV` perform cross-validated hyperparameter search over a log-spaced alpha grid — an approach equivalent to walk-forward validation in finance.
- **Ridge outperforms OLS** (lower RMSE, better generalisation) because the mild multicollinearity introduced by correlated dummies inflates OLS coefficient variance.

### Part 2 — Hiring Classification

```
Raw data → One-hot encoding → 70/30 split
         → Logistic Regression (baseline) → Random Forest (1,000 trees)
         → Confusion matrix · ROC-AUC · Classification report
```

- **Logistic Regression** provides a linear, interpretable baseline (accuracy ≈ 65%).
- **Random Forest** captures non-linear feature interactions and is robust to noisy features — analogous to ensemble methods used in systematic trading (accuracy ≈ 91%, AUC ≈ 0.97).
- Models are evaluated on a held-out test set; no metric is reported on training data.

---

## Key Findings

- **Experience is the dominant salary driver** (Pearson r = 0.66), consistent with human capital theory. All other features add marginal predictive power.
- **Ridge Regression** is the recommended salary model: R² = 0.651, RMSE ≈ 5,967. The OLS baseline shows near-identical performance, confirming that regularisation matters most for generalisability, not raw training fit.
- **Random Forest** achieves 91.3% accuracy and AUC ≈ 0.97 on the hiring task — a 26-percentage-point lift over Logistic Regression. The ensemble captures interactive effects (e.g., high salary + low experience = unlikely hire) that a linear model cannot represent.
- The logistic baseline serves as a useful interpretability anchor: its coefficients can be converted to odds ratios for stakeholder communication.

---

## Relevance to Quantitative Finance

The skills demonstrated here map directly to quantitative analyst workflows:

| This Project | Quant Analogy |
|--------------|---------------|
| Ridge / Lasso / ElasticNet regression | Factor model estimation with regularisation |
| Cross-validated alpha selection | Walk-forward hyperparameter optimisation |
| ROC-AUC on binary outcome | Signal quality assessment (IR, precision/recall trade-off) |
| Ensemble (Random Forest) vs. linear baseline | Alpha stacking / model combination |
| Feature correlation analysis | Factor exposure / collinearity diagnostics |
| Standardisation before regularisation | Z-scoring factor exposures |

---

## Project Structure

```
.
├── hr_analytics.py             # Main pipeline script (run this)
├── hr_data.csv                 # Salary dataset (Part 1)
├── hr_data_validation.csv      # Hiring dataset (Part 2)
├── requirements.txt            # Pinned Python dependencies
├── plots/                      # Auto-generated output figures
│   ├── salary_feature_correlations.png
│   ├── ridge_actual_vs_predicted.png
│   ├── salary_vs_hired.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── roc_curve_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   └── roc_curve_random_forest.png
└── .gitignore
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Kathela/HR-Analytics-Salary-Prediction-and-Hiring-Classification
cd HR-Analytics-Salary-Prediction-and-Hiring-Classification

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python hr_analytics.py
```

Plots are written to `./plots/`. Console output summarises all model metrics.

---

## Tech Stack

| Library | Version | Role |
|---------|---------|------|
| Python | 3.10+ | Runtime |
| pandas | 2.2 | Data wrangling |
| NumPy | 1.26 | Numerical computation |
| matplotlib | 3.9 | Plotting |
| seaborn | 0.13 | Statistical visualisation |
| scikit-learn | 1.5 | Modelling, preprocessing, evaluation |

---

## License

MIT — free to use, adapt, and distribute with attribution.
