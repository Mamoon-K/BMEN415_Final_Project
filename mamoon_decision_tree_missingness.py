"""
Mamoon's Design Decision — Non-NN Variant: Missingness Indicators

Difference from baseline (decision_tree_baseline.py):
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

import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from feature_policy import CLASSIFICATION_TARGET as TARGET, classification_features

RANDOM_SEED = 42
N_SPLITS = 5
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

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
features = classification_features(train)

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
                    cv=gkf, scoring=scoring, n_jobs=1)

print(f"\n--- 5-Fold Grouped CV (Variant: DT + Missingness Indicators) ---")
print(f"AUROC    : {cv['test_auroc'].mean():.4f} ± {cv['test_auroc'].std():.4f}")
print(f"F1       : {cv['test_f1'].mean():.4f} ± {cv['test_f1'].std():.4f}")
print(f"Precision: {cv['test_precision'].mean():.4f} ± {cv['test_precision'].std():.4f}")
print(f"Recall   : {cv['test_recall'].mean():.4f} ± {cv['test_recall'].std():.4f}")

pipeline.fit(X_train, y_train)

records = [{
    "split": "cv_train",
    "auroc": cv["test_auroc"].mean(),
    "auroc_std": cv["test_auroc"].std(),
    "f1": cv["test_f1"].mean(),
    "precision": cv["test_precision"].mean(),
    "recall": cv["test_recall"].mean(),
}]

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
    records.append({
        "split": label,
        "auroc": roc_auc_score(y, probs),
        "f1": f1_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds),
    })

report("val", X_val, y_val)
report("test", X_test, y_test)

variant_df = pd.DataFrame(records)
metrics_path = os.path.join(RESULTS_DIR, "mamoon_dt_missingness_metrics.csv")
variant_df.to_csv(metrics_path, index=False)

# Baseline-vs-variant comparison table (evidence for the verification plan).
# Run the DT baseline first (via run_all.py) so this file exists.
baseline_path = os.path.join(RESULTS_DIR, "baseline_dt_metrics.csv")
comparison_path = os.path.join(RESULTS_DIR, "mamoon_dt_missingness_vs_baseline.csv")
if os.path.exists(baseline_path):
    baseline_df = pd.read_csv(baseline_path).assign(model="baseline_dt")
    variant_tagged = variant_df.assign(model="dt_missingness")
    comparison = pd.concat([baseline_df, variant_tagged], ignore_index=True)
    comparison = comparison[["model", "split", "auroc", "f1", "precision", "recall"]]
    comparison.to_csv(comparison_path, index=False)
    print(f"\n--- Baseline vs Variant ---\n{comparison.to_string(index=False)}")
else:
    print(f"\n[skip baseline comparison: {baseline_path} not found — run baseline first]")

# Feature importance — highlight *_missing columns
expanded_features = features + [f"{f}_missing" for f in features]
importances = pipeline.named_steps["model"].feature_importances_
fi = pd.Series(importances, index=expanded_features).sort_values(ascending=False)

print(f"\n--- Top 15 Feature Importances ---")
print(fi.head(15).to_string())

missing_cols_top10 = [c for c in fi.head(10).index if c.endswith("_missing")]
print(f"\n*_missing columns in top-10: {missing_cols_top10}")

fi_path = os.path.join(RESULTS_DIR, "mamoon_dt_missingness_feature_importance.csv")
fi.to_csv(fi_path)

# Top-20 feature importance bar chart; _missing columns highlighted
top20 = fi.head(20)
colors = ["#dd8452" if c.endswith("_missing") else "steelblue" for c in top20.index]
fig, ax = plt.subplots(figsize=(9, 7))
top20.iloc[::-1].plot.barh(ax=ax, color=colors[::-1])
ax.set_xlabel("Feature importance (Gini)")
ax.set_title("DT + missingness indicators — top-20 features\n(orange = *_missing flag)")
plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "mamoon_dt_missingness_feature_importance.png")
fig.savefig(fig_path, dpi=150)
plt.close(fig)

print(f"\nSaved:\n  {metrics_path}\n  {fi_path}\n  {fig_path}")
if os.path.exists(baseline_path):
    print(f"  {comparison_path}")
