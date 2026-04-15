"""
Mamoon's Design Decision — Non-NN Variant: Missingness Indicators

Difference from baseline (desicion_tree_classifier_and_random_forrest.py):
    For every input feature with any NaN in training, add a new column
    `<feature>_missing` = 1 if NaN, 0 if present — BEFORE mean imputation.
    The tree then sees both the (imputed) value AND whether it was measured.
    Everything else identical (same features, imputer, DT params, CV, seed).

Hypothesis:
    Missingness indicators will IMPROVE AUROC (>= +0.03) over the baseline,
    because lab tests on this dataset are ordered when clinically indicated
    (informative missingness). The indicator flags therefore carry clinical
    signal that mean-imputation alone discards.

Acceptance threshold:
    Test AUROC(variant) - AUROC(baseline) >= +0.03  -> hypothesis supported.
    Also examine feature importance: if any *_missing column ranks in top-10
    -> additional evidence that missingness was informative.

Evidence:
    - Table of AUROC/F1/Precision/Recall: baseline vs variant (CV + test).
    - Bar chart of feature importances with *_missing columns highlighted.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report,
)

from datasetup import load_data

RANDOM_SEED = 42
N_SPLITS = 5
TARGET = "SepsisLabel"
EXCLUDED = ["MAP", "SepsisLabel", "patient_id"]

np.random.seed(RANDOM_SEED)


class MissingnessIndicator(BaseEstimator, TransformerMixin):
    """Append a binary *_missing column for every input feature.
    Added BEFORE imputation so the indicator reflects true missingness."""
    def fit(self, X, y=None):
        self.columns_ = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            flags = X.isna().astype(int).add_suffix("_missing")
            return pd.concat([X.reset_index(drop=True), flags.reset_index(drop=True)], axis=1)
        else:
            flags = pd.DataFrame(np.isnan(X).astype(int), columns=[f"{c}_missing" for c in self.columns_])
            base = pd.DataFrame(X, columns=self.columns_)
            return pd.concat([base, flags], axis=1)


train, val, test = load_data()
features = [c for c in train.columns if c not in EXCLUDED]

train = train.dropna(subset=[TARGET]).reset_index(drop=True)
val = val.dropna(subset=[TARGET]).reset_index(drop=True)
test = test.dropna(subset=[TARGET]).reset_index(drop=True)

X_train, y_train, groups_train = train[features], train[TARGET].astype(int), train["patient_id"]
X_val, y_val = val[features], val[TARGET].astype(int)
X_test, y_test = test[features], test[TARGET].astype(int)

print(f"Features ({len(features)}): {features}")
print(f"Train rows: {len(X_train)} ({groups_train.nunique()} patients) — sepsis {y_train.mean()*100:.2f}%")

pipeline = Pipeline([
    ("missingness", MissingnessIndicator()),
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED)),
])

# 5-fold Grouped CV
gkf = GroupKFold(n_splits=N_SPLITS)
scoring = {"auroc": "roc_auc", "f1": "f1", "precision": "precision", "recall": "recall"}
cv = cross_validate(pipeline, X_train, y_train, groups=groups_train,
                    cv=gkf, scoring=scoring, n_jobs=-1)

print(f"\n--- 5-Fold Grouped CV (Variant: DT + Missingness Indicators) ---")
print(f"AUROC    : {cv['test_auroc'].mean():.4f} ± {cv['test_auroc'].std():.4f}")
print(f"F1       : {cv['test_f1'].mean():.4f} ± {cv['test_f1'].std():.4f}")
print(f"Precision: {cv['test_precision'].mean():.4f} ± {cv['test_precision'].std():.4f}")
print(f"Recall   : {cv['test_recall'].mean():.4f} ± {cv['test_recall'].std():.4f}")

pipeline.fit(X_train, y_train)

def report(label, X, y):
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:, 1]
    print(f"\n--- {label} ---")
    print(f"AUROC    : {roc_auc_score(y, probs):.4f}")
    print(f"F1       : {f1_score(y, preds, zero_division=0):.4f}")
    print(f"Precision: {precision_score(y, preds, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y, preds):.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(y, preds)}")
    print(classification_report(y, preds, target_names=['No Sepsis', 'Sepsis'], zero_division=0))

report("Validation", X_val, y_val)
report("Test", X_test, y_test)

# Feature importance — highlight *_missing columns
expanded_features = features + [f"{f}_missing" for f in features]
importances = pipeline.named_steps["model"].feature_importances_
fi = pd.Series(importances, index=expanded_features).sort_values(ascending=False)

print(f"\n--- Top 15 Feature Importances ---")
print(fi.head(15).to_string())

missing_cols_top10 = [c for c in fi.head(10).index if c.endswith("_missing")]
print(f"\n*_missing columns in top-10: {missing_cols_top10}")

fi.to_csv("mamoon_dt_missingness_feature_importance.csv")
print("\nSaved: mamoon_dt_missingness_feature_importance.csv")
