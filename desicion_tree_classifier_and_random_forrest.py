"""
Team Baseline — Non-Neural Network Classifier (predict SepsisLabel)

Design choices (team-agreed):
- All allowed input features (exclude target SepsisLabel, regression target MAP, and patient_id)
- Mean imputation for missing values (fit inside CV to avoid leakage)
- No StandardScaler (trees are scale-invariant)
- Patient-level split (done in datasetup.py)
- 5-fold GroupKFold CV on training set, grouped by patient_id
- DecisionTreeClassifier with max_depth=5, random_state=42
- NO class_weight (imbalance handling is a team member's design decision)
- Metrics: AUROC, F1, Precision, Recall — reported as CV mean±std and final held-out test
- Fixed seed for reproducibility
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, recall_score, precision_score,
)

from datasetup import load_data

RANDOM_SEED = 42
N_SPLITS = 5
TARGET = "SepsisLabel"
EXCLUDED = ["MAP", "SepsisLabel", "patient_id"]

np.random.seed(RANDOM_SEED)

train, val, test = load_data()

features = [c for c in train.columns if c not in EXCLUDED]

# Drop rows with missing target
train = train.dropna(subset=[TARGET]).reset_index(drop=True)
val = val.dropna(subset=[TARGET]).reset_index(drop=True)
test = test.dropna(subset=[TARGET]).reset_index(drop=True)

X_train, y_train, groups_train = train[features], train[TARGET].astype(int), train["patient_id"]
X_val, y_val = val[features], val[TARGET].astype(int)
X_test, y_test = test[features], test[TARGET].astype(int)

print(f"Features ({len(features)}): {features}")
print(f"Train rows: {len(X_train)} ({groups_train.nunique()} patients) — sepsis prevalence {y_train.mean()*100:.2f}%")
print(f"Val rows:   {len(X_val)} — sepsis prevalence {y_val.mean()*100:.2f}%")
print(f"Test rows:  {len(X_test)} — sepsis prevalence {y_test.mean()*100:.2f}%")

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED)),
])

# 5-fold Grouped CV on training
gkf = GroupKFold(n_splits=N_SPLITS)
scoring = {
    "auroc": "roc_auc",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
}
cv = cross_validate(
    pipeline, X_train, y_train, groups=groups_train,
    cv=gkf, scoring=scoring, n_jobs=-1,
)

print(f"\n--- 5-Fold Grouped CV (on training set) ---")
print(f"AUROC    : {cv['test_auroc'].mean():.4f} ± {cv['test_auroc'].std():.4f}")
print(f"F1       : {cv['test_f1'].mean():.4f} ± {cv['test_f1'].std():.4f}")
print(f"Precision: {cv['test_precision'].mean():.4f} ± {cv['test_precision'].std():.4f}")
print(f"Recall   : {cv['test_recall'].mean():.4f} ± {cv['test_recall'].std():.4f}")

# Fit on full training, evaluate on val and test
pipeline.fit(X_train, y_train)

def report(label, X, y):
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:, 1]
    print(f"\n--- {label} ---")
    print(f"AUROC    : {roc_auc_score(y, probs):.4f}")
    print(f"F1       : {f1_score(y, preds):.4f}")
    print(f"Precision: {precision_score(y, preds, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y, preds):.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(y, preds)}")
    print(classification_report(y, preds, target_names=["No Sepsis", "Sepsis"], zero_division=0))

report("Validation (held-out)", X_val, y_val)
report("Test (held-out)", X_test, y_test)
