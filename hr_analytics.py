"""
HR Analytics: Salary Prediction and Hiring Classification
==========================================================
Two-part supervised learning pipeline for HR decision support:

  Part 1 — Regression   : predict expected salary from candidate attributes
  Part 2 — Classification: predict hiring outcome (hired / not hired)

Usage:
    python hr_analytics.py

Outputs:
    Console metrics for all models.
    PNG plots written to the ./plots directory.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    RidgeCV,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PLOTS_DIR = "plots"


# ─── UTILITIES ────────────────────────────────────────────────────────────────


def setup_plots_dir() -> None:
    """Create the plots output directory if it does not already exist."""
    os.makedirs(PLOTS_DIR, exist_ok=True)


def _save_figure(filename: str) -> None:
    """Save current figure to the plots directory and close it."""
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── PART 1: SALARY PREDICTION (REGRESSION) ───────────────────────────────────


def load_salary_data(filepath: str = "hr_data.csv") -> pd.DataFrame:
    """
    Load the salary dataset.

    The dataset contains 1,000 HR records with the following columns:
        Experiencia  – years of work experience (continuous)
        Posicion     – job level: Analista, Coordinador, Gerente (categorical)
        Hijos        – number of children (integer)
        Casado       – marital status, 1 = married (binary)
        Educacion    – highest qualification: Bachillerato, Licenciatura,
                       Posgrado (categorical)
        Salario      – annual salary in local currency (target)
    """
    df = pd.read_csv(filepath)
    print(f"Loaded salary dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(df.head().to_string())
    return df


def engineer_salary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical features as binary dummy variables.

    drop_first=True removes one dummy per category group to prevent
    perfect multicollinearity (the dummy variable trap) in linear models.
    """
    return pd.get_dummies(df, columns=["Posicion", "Educacion"], drop_first=True)


def analyze_salary_correlations(df: pd.DataFrame) -> None:
    """
    Report and visualise feature correlations with the salary target.

    Experience dominates, consistent with human capital theory — each
    additional year of experience carries the largest marginal salary premium.
    """
    correlations = df.corr(numeric_only=True)["Salario"].sort_values(ascending=False)
    print("\nFeature correlations with Salary:")
    print(correlations.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    correlations.drop("Salario").plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Feature Correlation with Salary", fontsize=13)
    ax.set_xlabel("Pearson Correlation Coefficient")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    _save_figure("salary_feature_correlations.png")


def train_salary_models(df: pd.DataFrame) -> dict:
    """
    Prepare features, fit four regression models, and return predictions.

    Models trained:
        - Ordinary Least Squares (OLS) — interpretable baseline
        - Ridge (L2)     — shrinks all coefficients; handles multicollinearity
        - Lasso (L1)     — performs implicit feature selection via sparsity
        - ElasticNet     — convex combination of L1 and L2 penalties

    Hyperparameters are chosen by cross-validated grid search over alpha,
    which ensures fair comparison without data leakage from the test set.
    An 80/20 train-test split is used; random_state=42 ensures reproducibility.
    """
    X = df.drop("Salario", axis=1)
    y = df["Salario"]

    # Standardise so regularisation penalties are applied uniformly across features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, train_size=0.80, random_state=42
    )

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    ridge_model = RidgeCV(alphas=np.logspace(-10, 2, 100))
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)

    lasso_model = LassoCV(alphas=np.logspace(-10, 3, 100), cv=10)
    lasso_model.fit(X_train, y_train)
    y_pred_lasso = lasso_model.predict(X_test)

    elastic_model = ElasticNetCV(
        alphas=np.logspace(-10, 3, 100),
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
        cv=10,
    )
    elastic_model.fit(X_train, y_train)
    y_pred_elastic = elastic_model.predict(X_test)

    return {
        "y_test": y_test,
        "lr":      {"model": lr_model,      "y_pred": y_pred_lr},
        "ridge":   {"model": ridge_model,   "y_pred": y_pred_ridge},
        "lasso":   {"model": lasso_model,   "y_pred": y_pred_lasso},
        "elastic": {"model": elastic_model, "y_pred": y_pred_elastic},
    }


def evaluate_salary_models(results: dict) -> None:
    """
    Print a side-by-side model comparison table (RMSE, R²) and plot
    actual vs. predicted salaries for the Ridge model.

    Ridge is the recommended model: its L2 penalty reduces variance
    on this modestly-sized dataset without introducing the sparsity
    artefacts of Lasso when features are genuinely correlated.
    """
    y_test = results["y_test"]

    def _metrics(y_pred):
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        return rmse, r2

    rows = [
        ("Linear Regression", "—",
         *_metrics(results["lr"]["y_pred"])),
        (f"Ridge   (α={results['ridge']['model'].alpha_:.2e})",
         f"{results['ridge']['model'].alpha_:.2e}",
         *_metrics(results["ridge"]["y_pred"])),
        (f"Lasso   (α={results['lasso']['model'].alpha_:.2e})",
         f"{results['lasso']['model'].alpha_:.2e}",
         *_metrics(results["lasso"]["y_pred"])),
        (f"ElasticNet (α={results['elastic']['model'].alpha_:.2e})",
         f"{results['elastic']['model'].alpha_:.2e}",
         *_metrics(results["elastic"]["y_pred"])),
    ]

    header = f"\n{'Model':<38} {'RMSE':>10} {'R²':>8}"
    print("\n── Regression Model Comparison " + "─" * 30)
    print(header)
    print("─" * 58)
    for name, _, rmse, r2 in rows:
        print(f"{name:<38} {rmse:>10.2f} {r2:>8.4f}")

    comparison = pd.DataFrame({
        "Actual":           y_test.values,
        "Predicted_LR":     results["lr"]["y_pred"],
        "Predicted_Ridge":  results["ridge"]["y_pred"],
    })
    print("\nActual vs. Predicted — first 5 rows:")
    print(comparison.head().to_string(index=False))

    # Scatter: perfect-fit diagonal lets us immediately see where the model errs
    y_pred_ridge = results["ridge"]["y_pred"]
    _, r2_ridge = _metrics(y_pred_ridge)
    lo = min(y_test.min(), y_pred_ridge.min())
    hi = max(y_test.max(), y_pred_ridge.max())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred_ridge, alpha=0.45, edgecolors="none", color="steelblue", s=30)
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Salary")
    ax.set_ylabel("Predicted Salary")
    ax.set_title(f"Ridge Regression — Actual vs. Predicted\n(R² = {r2_ridge:.4f})", fontsize=12)
    ax.legend()
    _save_figure("ridge_actual_vs_predicted.png")


# ─── PART 2: HIRING CLASSIFICATION ────────────────────────────────────────────


def load_hiring_data(filepath: str = "hr_data_validation.csv") -> pd.DataFrame:
    """
    Load the hiring outcome dataset.

    Extends the salary dataset with a binary target column:
        Contratado — 1 = hired, 0 = not hired
    """
    df = pd.read_csv(filepath)
    print(f"\nLoaded hiring dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(df.head().to_string())
    return df


def plot_salary_vs_hired(df: pd.DataFrame) -> None:
    """
    Visualise salary distribution stratified by hiring outcome.

    If salary alone were a perfect predictor, the two distributions
    would be fully separated — this plot tests that assumption.
    """
    hired = df[df["Contratado"] == 1]["Salario"]
    not_hired = df[df["Contratado"] == 0]["Salario"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(hired,     [1] * len(hired),     alpha=0.25, color="green", label="Hired",     s=20)
    ax.scatter(not_hired, [0] * len(not_hired), alpha=0.25, color="red",   label="Not Hired", s=20)
    ax.set_xlabel("Salary")
    ax.set_ylabel("Hired (1 = Yes, 0 = No)")
    ax.set_title("Salary vs. Hiring Outcome")
    ax.set_yticks([0, 1])
    ax.legend()
    _save_figure("salary_vs_hired.png")


def engineer_hiring_features(df: pd.DataFrame) -> tuple:
    """
    One-hot encode categorical columns, then split into X and y.

    drop_first is omitted here because Random Forest is a tree-based model
    that is insensitive to the dummy variable collinearity issue.
    """
    df_encoded = pd.get_dummies(df, columns=["Posicion", "Educacion"])
    X = df_encoded.drop(columns="Contratado")
    y = df_encoded["Contratado"]
    return X, y


def train_hiring_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train Logistic Regression and Random Forest on a 70/30 train-test split.

    Logistic Regression: linear decision boundary, fully interpretable via
    odds ratios — a useful sanity-check baseline.

    Random Forest (1,000 trees): captures non-linear feature interactions
    and is robust to irrelevant or noisy features without explicit
    feature selection.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42, shuffle=True
    )

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    y_pred_log   = log_model.predict(X_test)
    y_proba_log  = log_model.predict_proba(X_test)[:, 1]

    rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf    = rf_model.predict(X_test)
    y_proba_rf   = rf_model.predict_proba(X_test)[:, 1]

    return {
        "y_test":        y_test,
        "logistic":      {"model": log_model, "y_pred": y_pred_log,  "y_proba": y_proba_log},
        "random_forest": {"model": rf_model,  "y_pred": y_pred_rf,   "y_proba": y_proba_rf},
    }


def _plot_confusion_matrix(y_test, y_pred, model_name: str, cmap: str) -> None:
    """Save a labelled confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                xticklabels=["Not Hired", "Hired"],
                yticklabels=["Not Hired", "Hired"])
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    slug = model_name.lower().replace(" ", "_")
    _save_figure(f"confusion_matrix_{slug}.png")


def _plot_roc_curve(y_test, y_proba, model_name: str) -> None:
    """Save a ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=12)
    ax.legend(loc="lower right")
    slug = model_name.lower().replace(" ", "_")
    _save_figure(f"roc_curve_{slug}.png")


def evaluate_hiring_models(results: dict) -> None:
    """
    Print accuracy, AUC, and a full classification report for each classifier.
    Persist confusion matrix and ROC curve plots for both models.
    """
    y_test = results["y_test"]

    model_configs = [
        ("Logistic Regression", "logistic",      "Blues"),
        ("Random Forest",       "random_forest", "Greens"),
    ]

    for label, key, cmap in model_configs:
        res = results[key]
        accuracy = (res["y_pred"] == y_test).mean()
        auc      = roc_auc_score(y_test, res["y_proba"])

        print(f"\n── {label} " + "─" * (50 - len(label)))
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(classification_report(y_test, res["y_pred"],
                                    target_names=["Not Hired", "Hired"]))

        _plot_confusion_matrix(y_test, res["y_pred"], label, cmap)
        _plot_roc_curve(y_test, res["y_proba"], label)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────


def main() -> None:
    """Run the full HR analytics pipeline end-to-end."""
    setup_plots_dir()

    print("=" * 60)
    print("PART 1 — SALARY PREDICTION (REGRESSION)")
    print("=" * 60)

    salary_df = load_salary_data("hr_data.csv")
    salary_df = engineer_salary_features(salary_df)
    analyze_salary_correlations(salary_df)
    regression_results = train_salary_models(salary_df)
    evaluate_salary_models(regression_results)

    print("\n" + "=" * 60)
    print("PART 2 — HIRING CLASSIFICATION")
    print("=" * 60)

    hiring_df = load_hiring_data("hr_data_validation.csv")
    plot_salary_vs_hired(hiring_df)
    X, y = engineer_hiring_features(hiring_df)
    classification_results = train_hiring_models(X, y)
    evaluate_hiring_models(classification_results)

    print(f"\nAll plots saved to: ./{PLOTS_DIR}/")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
