"""
Team Baseline — Regression (predict MAP)

Design choices (team-agreed):
- All allowed input features (exclude outputs MAP/SBP/DBP, target leakage SepsisLabel, and patient_id)
- Mean imputation for missing values (fit inside CV to avoid leakage)
- StandardScaler (helps interpretability of coefficients; doesn't change LR predictions)
- Patient-level split (done in datasetup.py)
- 5-fold GroupKFold CV on training set, grouped by patient_id
- Default LinearRegression (no regularization, no tuning)
- Metrics: RMSE, MAE, R^2 — reported as CV mean±std and final held-out test
- Fixed seed for reproducibility
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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

np.random.seed(RANDOM_SEED)

train, val, test = load_data()

features = [c for c in train.columns if c not in EXCLUDED]

# Drop rows with missing target (can't train/evaluate without a label)
train = train.dropna(subset=[TARGET]).reset_index(drop=True)
val = val.dropna(subset=[TARGET]).reset_index(drop=True)
test = test.dropna(subset=[TARGET]).reset_index(drop=True)

X_train, y_train, groups_train = train[features], train[TARGET], train["patient_id"]
X_val, y_val = val[features], val[TARGET]
X_test, y_test = test[features], test[TARGET]

print(f"Features ({len(features)}): {features}")
print(f"Train rows: {len(X_train)} ({groups_train.nunique()} patients)")
print(f"Val rows:   {len(X_val)}")
print(f"Test rows:  {len(X_test)}")

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression()),
])

# 5-fold CV on training (grouped by patient_id — no patient leakage within CV)
gkf = GroupKFold(n_splits=N_SPLITS)
scoring = {
    "neg_rmse": "neg_root_mean_squared_error",
    "neg_mae": "neg_mean_absolute_error",
    "r2": "r2",
}
cv = cross_validate(
    pipeline, X_train, y_train, groups=groups_train,
    cv=gkf, scoring=scoring, n_jobs=-1, return_train_score=False,
)

print(f"\n--- 5-Fold Grouped CV (on training set) ---")
print(f"RMSE: {(-cv['test_neg_rmse']).mean():.3f} ± {cv['test_neg_rmse'].std():.3f} mmHg")
print(f"MAE : {(-cv['test_neg_mae']).mean():.3f} ± {cv['test_neg_mae'].std():.3f} mmHg")
print(f"R^2 : {cv['test_r2'].mean():.4f} ± {cv['test_r2'].std():.4f}")

# Fit on full training set; evaluate on val and test
pipeline.fit(X_train, y_train)

val_preds = pipeline.predict(X_val)
print(f"\n--- Validation (held-out) ---")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, val_preds)):.3f} mmHg")
print(f"MAE : {mean_absolute_error(y_val, val_preds):.3f} mmHg")
print(f"R^2 : {r2_score(y_val, val_preds):.4f}")

test_preds = pipeline.predict(X_test)
print(f"\n--- Test (held-out) ---")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, test_preds)):.3f} mmHg")
print(f"MAE : {mean_absolute_error(y_test, test_preds):.3f} mmHg")
print(f"R^2 : {r2_score(y_test, test_preds):.4f}")
