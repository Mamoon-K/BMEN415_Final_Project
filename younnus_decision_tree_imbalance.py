"""
Team Experiment — Non-Neural Network Classifier (predict SepsisLabel)

(Still needs to be edited so far scores are quite worse in aspects other than f1 and recall)
 
Decision change vs. baseline:
- CHANGED: No class imbalance handling → BorderlineSMOTE oversampling
  Rationale: Sepsis prevalence is about 2%, meaning the tree can achieve ~98% accuracy
  by predicting "no sepsis" for every row. BorderlineSMOTE generates synthetic
  minority-class samples specifically near the decision boundary — the ambiguous
  region where the tree most needs signal — rather than in clearly-sepsis regions
  where the model already has confidence.
 
  Why BorderlineSMOTE over vanilla SMOTE:
  - Vanilla SMOTE interpolates uniformly across all minority samples, including
    easy cases deep in the sepsis cluster that the tree already handles.
  - BorderlineSMOTE focuses synthesis on the hard boundary cases, which is where
    a shallow tree (max_depth=5) is most likely to misclassify.
 
Hypothesis: Using the Borderline SMOTE to treat the class imbalance may reduce some
scores such as AUROC, precision, and AUPRC. But it will likely increase recall and F1
as the model will reduce false negatives by better learning the minority class. Overall,
This result will likely benefit the model as the clinical cost of missing sepsis cases 
(false negatives) is much higher than the cost of false positives in this context.
 
"""
 
import os
import numpy as np
import pandas as pd
 
# imblearn's Pipeline correctly gates SMOTE to training folds only
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
 
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score,
)
 
from datasetup import load_data
from feature_policy import CLASSIFICATION_TARGET as TARGET, classification_features
 
RANDOM_SEED = 42
N_SPLITS = 5
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
 
np.random.seed(RANDOM_SEED)
 

train, val, test = load_data()
 
features = classification_features(train)
 
train = train.dropna(subset=[TARGET]).reset_index(drop=True)
val   = val.dropna(subset=[TARGET]).reset_index(drop=True)
test  = test.dropna(subset=[TARGET]).reset_index(drop=True)
 
X_train, y_train, groups_train = train[features], train[TARGET].astype(int), train["patient_id"]
X_val,   y_val                 = val[features],   val[TARGET].astype(int)
X_test,  y_test                = test[features],  test[TARGET].astype(int)
 
print(f"Features ({len(features)}): {features}")
print(f"Train rows: {len(X_train)} ({groups_train.nunique()} patients) — sepsis prevalence {y_train.mean()*100:.2f}%")
print(f"Val rows:   {len(X_val)} — sepsis prevalence {y_val.mean()*100:.2f}%")
print(f"Test rows:  {len(X_test)} — sepsis prevalence {y_test.mean()*100:.2f}%")
 
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
print(f"\nClass ratio (neg:pos) in training set: {n_neg}:{n_pos} ({n_neg/n_pos:.1f}:1)")

 
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("smote",   BorderlineSMOTE(
        random_state=RANDOM_SEED,
        k_neighbors=5, 
        m_neighbors=10,
        kind="borderline-1", 
    )),
    ("model",   DecisionTreeClassifier(
        max_depth=5,
        random_state=RANDOM_SEED,
    )),
])
 

# 5-fold grouped CV on training set
# SMOTE is applied inside each fold by imblearn's Pipeline automatically.
# val/test folds are passed through imputer + model only (no oversampling).

 
gkf = GroupKFold(n_splits=N_SPLITS)
scoring = {
    "auroc":     "roc_auc",
    "auprc":     "average_precision",
    "f1":        "f1",
    "precision": "precision",
    "recall":    "recall",
}
cv = cross_validate(
    pipeline, X_train, y_train, groups=groups_train,
    cv=gkf, scoring=scoring, n_jobs=1,
)
 
print(f"\n--- 5-Fold Grouped CV (on training set) ---")
print(f"AUROC    : {cv['test_auroc'].mean():.4f} ± {cv['test_auroc'].std():.4f}")
print(f"AUPRC    : {cv['test_auprc'].mean():.4f} ± {cv['test_auprc'].std():.4f}")
print(f"F1       : {cv['test_f1'].mean():.4f} ± {cv['test_f1'].std():.4f}")
print(f"Precision: {cv['test_precision'].mean():.4f} ± {cv['test_precision'].std():.4f}")
print(f"Recall   : {cv['test_recall'].mean():.4f} ± {cv['test_recall'].std():.4f}")
 

# Final fit on full training set + evaluation on validation and test
# SMOTE runs on the full training set here (no leakage — val/test untouched)
 
pipeline.fit(X_train, y_train)
 
#confirm SMOTE ran and balanced the training set
smote_step = pipeline.named_steps["smote"]
print(f"\nSMOTE resampled training set — "
      f"original minority count: {n_pos}, "
      f"target after resampling: {n_neg} (1:1 ratio)")
 
records = []
 
def report(label, X, y):
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:, 1]
    print(f"\n--- {label} ---")
    print(f"AUROC    : {roc_auc_score(y, probs):.4f}")
    print(f"AUPRC    : {average_precision_score(y, probs):.4f}")
    print(f"F1       : {f1_score(y, preds):.4f}")
    print(f"Precision: {precision_score(y, preds, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y, preds):.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(y, preds)}")
    print(classification_report(y, preds, target_names=["No Sepsis", "Sepsis"], zero_division=0))
    records.append({
        "split":     label,
        "auroc":     roc_auc_score(y, probs),
        "auprc":     average_precision_score(y, probs),
        "f1":        f1_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall":    recall_score(y, preds),
    })
 
records.append({
    "split":      "cv_train",
    "auroc":      cv["test_auroc"].mean(),
    "auroc_std":  cv["test_auroc"].std(),
    "auprc":      cv["test_auprc"].mean(),
    "auprc_std":  cv["test_auprc"].std(),
    "f1":         cv["test_f1"].mean(),
    "f1_std":     cv["test_f1"].std(),
    "precision":  cv["test_precision"].mean(),
    "recall":     cv["test_recall"].mean(),
})
 
report("val",  X_val,  y_val)
report("test", X_test, y_test)

 
out_path = os.path.join(RESULTS_DIR, "borderlinesmote_dt_metrics.csv")
pd.DataFrame(records).to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
 