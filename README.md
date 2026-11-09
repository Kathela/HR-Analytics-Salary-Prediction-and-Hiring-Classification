# HR Analytics — Salary Prediction & Hiring Classification

**Supervised Machine Learning · Regression · Classification · Python**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Overview

This project builds two end-to-end predictive models to support data-driven HR decision-making.

> Built as part of a data science bootcamp, then refactored into a reproducible script. The models deliberately **exclude marital status and number of children** (see *Fairness* below).

| Part | Task | Best Model | Key Metric |
|------|------|-----------|------------|
| 1 | Salary prediction | Ridge Regression | R² = 0.59 · RMSE ≈ 6,440 |
| 2 | Hiring classification | Random Forest | Accuracy 83% · AUC ≈ 0.88 |

The workflow covers feature engineering, regularisation, model selection via cross-validation and out-of-sample evaluation.

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
| `Hijos` | Integer | Number of children (**excluded from models**) |
| `Casado` | Binary | Marital status (**excluded from models**) |
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
- `RidgeCV`, `LassoCV`, and `ElasticNetCV` perform cross-validated hyperparameter search over a log-spaced alpha grid.
- Ridge, Lasso and ElasticNet perform almost the same as OLS here, so the simple linear model is already well conditioned.

### Part 2 — Hiring Classification

```
Raw data → One-hot encoding → 70/30 split
         → Logistic Regression (baseline) → Random Forest (1,000 trees)
         → Confusion matrix · ROC-AUC · Classification report
```

- **Logistic Regression** provides a linear, interpretable baseline (accuracy 76%, AUC 0.75).
- **Random Forest** captures non-linear feature interactions (accuracy 83%, AUC 0.88).
- Models are evaluated on a held-out test set; no metric is reported on training data.

---

## Results

**Salary prediction (test set, 20%)**

| Model | RMSE | R² |
|-------|------|----|
| Linear Regression | 6,442 | 0.593 |
| Ridge | 6,440 | 0.594 |
| Lasso | 6,442 | 0.593 |
| ElasticNet | 6,440 | 0.593 |

**Hiring classification (test set, 30%)**

| Model | Accuracy | ROC-AUC | F1 (hired) |
|-------|----------|---------|------------|
| Logistic Regression | 0.76 | 0.75 | 0.75 |
| Random Forest | 0.83 | 0.88 | 0.82 |

- **Experience is the main salary driver** (Pearson r = 0.66 with salary).
- **Random Forest beats the logistic baseline by 7 points of accuracy**, capturing interactions such as salary relative to experience that a linear model misses.

---

## Fairness

The original bootcamp dataset includes **marital status** and **number of children**. Both carry signal in this data (for example, hire rates differ by number of children), and keeping them raised Random Forest accuracy to about 91%.

They are **excluded on purpose**: using family status to set pay or decide who gets hired would be discriminatory and, in many countries, illegal. The lower metrics above are the honest performance of a model that only uses job-related attributes. In a real deployment I would also audit predictions for disparate impact across groups.

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
