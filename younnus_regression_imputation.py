"""
Design Decision: Imputation Strategy for the regression model

Besides the imputation implementation, this pipeline is identical to the baseline 
regression (regression_baseline.py) in all other respects (same features, same
scaling, same CV, same seed). 

    Imputation strategy for ICU time-series data:
 
      1. Forward fill (ffill) within each patient — carries the last observed
         value forward.  This is the main strategy.
      2. Backward fill (bfill) within each patient — fills any leading NaNs at
         the start of a patient's record (added for the specific cases where
         ffill wouldn't work).
      3. Column-mean fallback — fills anything still missing (e.g. a feature
         that is entirely absent for some patients).  Mean is computed on the
         training set only and stored in `fit()` to prevent leakage, which is why
         its added.
 
    Hypothesis: This patient-level forward/backward fill strategy will outperform
    simple column-mean imputation by at least 0.02 R^2 on the test set, because it
    maintains the continuity of each patient's trajectory and is more clinically 
    plausible for ICU data. The mean fallback ensures we can still handle edge cases 
    without crashing.

    
    
"""
 
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
 
from datasetup import load_data
from feature_policy import REGRESSION_TARGET as TARGET, regression_features
 
RANDOM_SEED = 42
N_SPLITS = 5
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
 
np.random.seed(RANDOM_SEED)
 

 
class PatientForwardFillImputer(BaseEstimator, TransformerMixin):
 
    def __init__(self, patient_id_col: str = "patient_id"):
        self.patient_id_col = patient_id_col
 
    def fit(self, X: pd.DataFrame, y=None):
        # Store column-level means from training data as the last-resort fallback
        self.col_means_ = X.drop(columns=[self.patient_id_col], errors="ignore").mean()
        self.feature_cols_ = [c for c in X.columns if c != self.patient_id_col]
        return self
 
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        df = X.copy()

        df[self.feature_cols_] = (
            df.groupby(self.patient_id_col)[self.feature_cols_]
            .transform(lambda g: g.ffill().bfill())
        )

        df[self.feature_cols_] = df[self.feature_cols_].fillna(self.col_means_)
 
        return df[self.feature_cols_].to_numpy()
 

 
train, val, test = load_data()
 
features = regression_features(train)
 
train = train.dropna(subset=[TARGET]).reset_index(drop=True)
val   = val.dropna(subset=[TARGET]).reset_index(drop=True)
test  = test.dropna(subset=[TARGET]).reset_index(drop=True)
 
# Include patient_id in X so the imputer can group by it, it is dropped automatically by the imputer's transform output.
X_train = train[features + ["patient_id"]]
y_train = train[TARGET]
groups_train = train["patient_id"]
 
X_val  = val[features + ["patient_id"]]
y_val  = val[TARGET]
 
X_test = test[features + ["patient_id"]]
y_test = test[TARGET]
 
print(f"Features ({len(features)}): {features}")
print(f"Train rows: {len(X_train)} ({groups_train.nunique()} patients)")
print(f"Val rows:   {len(X_val)}")
print(f"Test rows:  {len(X_test)}")
 

pipeline = Pipeline([
    ("imputer", PatientForwardFillImputer(patient_id_col="patient_id")),
    ("scaler",  StandardScaler()),
    ("model",   LinearRegression()),
])
 

gkf = GroupKFold(n_splits=N_SPLITS)
scoring = {
    "neg_rmse": "neg_root_mean_squared_error",
    "neg_mae":  "neg_mean_absolute_error",
    "r2":       "r2",
}
cv = cross_validate(
    pipeline, X_train, y_train, groups=groups_train,
    cv=gkf, scoring=scoring, n_jobs=1, return_train_score=False,
)
 
print(f"\n--- 5-Fold Grouped CV (on training set) ---")
print(f"RMSE: {(-cv['test_neg_rmse']).mean():.3f} ± {cv['test_neg_rmse'].std():.3f} mmHg")
print(f"MAE : {(-cv['test_neg_mae']).mean():.3f} ± {cv['test_neg_mae'].std():.3f} mmHg")
print(f"R^2 : {cv['test_r2'].mean():.4f} ± {cv['test_r2'].std():.4f}")
 

pipeline.fit(X_train, y_train)
 
val_preds  = pipeline.predict(X_val)
print(f"\n--- Validation (held-out) ---")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, val_preds)):.3f} mmHg")
print(f"MAE : {mean_absolute_error(y_val, val_preds):.3f} mmHg")
print(f"R^2 : {r2_score(y_val, val_preds):.4f}")
 
test_preds = pipeline.predict(X_test)
print(f"\n--- Test (held-out) ---")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, test_preds)):.3f} mmHg")
print(f"MAE : {mean_absolute_error(y_test, test_preds):.3f} mmHg")
print(f"R^2 : {r2_score(y_test, test_preds):.4f}")
 

metrics = pd.DataFrame([{
    "split":    "cv_train",
    "rmse":     (-cv["test_neg_rmse"]).mean(),
    "rmse_std": cv["test_neg_rmse"].std(),
    "mae":      (-cv["test_neg_mae"]).mean(),
    "mae_std":  cv["test_neg_mae"].std(),
    "r2":       cv["test_r2"].mean(),
    "r2_std":   cv["test_r2"].std(),
}, {
    "split": "val",
    "rmse":  np.sqrt(mean_squared_error(y_val, val_preds)),
    "mae":   mean_absolute_error(y_val, val_preds),
    "r2":    r2_score(y_val, val_preds),
}, {
    "split": "test",
    "rmse":  np.sqrt(mean_squared_error(y_test, test_preds)),
    "mae":   mean_absolute_error(y_test, test_preds),
    "r2":    r2_score(y_test, test_preds),
}])
 
out_path = os.path.join(RESULTS_DIR, "ffill_regression_metrics.csv")
metrics.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")