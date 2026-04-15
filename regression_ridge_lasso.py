"""
Mamoon's Design Decision — Regression Variant: Ridge & Lasso regularization

Difference from baseline (regression.py):
    Pipeline's final step: LinearRegression  ->  Ridge(alpha=1.0)  AND  Lasso(alpha=1.0)
    Everything else identical (same features, imputation, scaling, CV, seed).

Hypothesis:
    Regularization will NOT meaningfully improve test R^2 (|delta| < 0.02),
    because the baseline is underfitting (weak feature-MAP correlations in EDA)
    rather than overfitting. Lasso may zero out uninformative coefficients,
    giving interpretability even if overall fit doesn't improve.

Acceptance threshold:
    If |Ridge R^2 - Baseline R^2| < 0.02 AND |Lasso R^2 - Baseline R^2| < 0.02
    -> hypothesis supported (regularization is the wrong tool here).
    Otherwise -> hypothesis rejected.

Evidence:
    - Table of RMSE/MAE/R^2 for Baseline vs Ridge vs Lasso (CV + test).
    - Bar chart of standardized coefficient magnitudes across the 3 models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from datasetup import load_data

RANDOM_SEED = 42
N_SPLITS = 5
TARGET = "MAP"
EXCLUDED = ["MAP", "SBP", "DBP", "SepsisLabel", "patient_id"]
RIDGE_ALPHA = 1.0
LASSO_ALPHA = 1.0

np.random.seed(RANDOM_SEED)

train, val, test = load_data()
features = [c for c in train.columns if c not in EXCLUDED]

train = train.dropna(subset=[TARGET]).reset_index(drop=True)
val = val.dropna(subset=[TARGET]).reset_index(drop=True)
test = test.dropna(subset=[TARGET]).reset_index(drop=True)

X_train, y_train, groups_train = train[features], train[TARGET], train["patient_id"]
X_val, y_val = val[features], val[TARGET]
X_test, y_test = test[features], test[TARGET]

print(f"Features ({len(features)}): {features}")
print(f"Train rows: {len(X_train)} ({groups_train.nunique()} patients)")

MODELS = {
    "Baseline (LinearRegression)": LinearRegression(),
    f"Ridge (alpha={RIDGE_ALPHA})": Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED),
    f"Lasso (alpha={LASSO_ALPHA})": Lasso(alpha=LASSO_ALPHA, random_state=RANDOM_SEED, max_iter=10000),
}

gkf = GroupKFold(n_splits=N_SPLITS)
scoring = {"neg_rmse": "neg_root_mean_squared_error",
           "neg_mae": "neg_mean_absolute_error",
           "r2": "r2"}

results = []
coefs = {}

for name, model in MODELS.items():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])
    cv = cross_validate(pipe, X_train, y_train, groups=groups_train,
                        cv=gkf, scoring=scoring, n_jobs=-1)
    pipe.fit(X_train, y_train)
    test_pred = pipe.predict(X_test)

    row = {
        "Model": name,
        "CV RMSE": f"{(-cv['test_neg_rmse']).mean():.3f} ± {cv['test_neg_rmse'].std():.3f}",
        "CV MAE":  f"{(-cv['test_neg_mae']).mean():.3f} ± {cv['test_neg_mae'].std():.3f}",
        "CV R^2":  f"{cv['test_r2'].mean():.4f} ± {cv['test_r2'].std():.4f}",
        "Test RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
        "Test MAE":  mean_absolute_error(y_test, test_pred),
        "Test R^2":  r2_score(y_test, test_pred),
    }
    results.append(row)
    coefs[name] = pd.Series(pipe.named_steps["model"].coef_, index=features)

results_df = pd.DataFrame(results)
print("\n--- Results ---")
print(results_df.to_string(index=False))

# Coefficient comparison table
coef_df = pd.DataFrame(coefs)
print("\n--- Standardized Coefficients ---")
print(coef_df.round(4).to_string())

print(f"\nLasso zeroed-out features: "
      f"{(coef_df[f'Lasso (alpha={LASSO_ALPHA})'].abs() < 1e-10).sum()} / {len(features)}")

# Save results for writeup
results_df.to_csv("mamoon_regression_results.csv", index=False)
coef_df.to_csv("mamoon_regression_coefficients.csv")
print("\nSaved: mamoon_regression_results.csv, mamoon_regression_coefficients.csv")
