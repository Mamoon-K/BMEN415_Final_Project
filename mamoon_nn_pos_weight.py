"""
Mamoon's Design Decision — NN Variant: pos_weight for class imbalance

Difference from baseline (neural_network_baseline.py):
    BCEWithLogitsLoss()  ->  BCEWithLogitsLoss(pos_weight = num_neg / num_pos)
    pos_weight is computed from the training fold so the positive (sepsis)
    class contributes proportionally more to the loss.
    Everything else identical (architecture, features, imputer, scaler,
    optimizer, lr, batch_size, epochs, seed, CV folds).

Hypothesis:
    pos_weight will IMPROVE Recall (>= +0.10) AND AUROC (>= +0.03) over the
    baseline, because sepsis prevalence is ~2%; an unweighted BCE loss lets
    the model minimize loss by predicting "no sepsis" almost always.
    Precision is expected to DROP — that trade-off is clinically acceptable
    (missing sepsis is worse than a false alarm).

Acceptance threshold:
    Test Recall(variant) - Recall(baseline) >= +0.10
    AND Test AUROC(variant) - AUROC(baseline) >= +0.03
    -> hypothesis supported.

Evidence:
    - Table of AUROC / F1 / Precision / Recall: baseline vs variant (CV + test).
    - Side-by-side confusion matrices (baseline vs variant) on the test set.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report,
)

from datasetup import load_data
from feature_policy import CLASSIFICATION_TARGET as TARGET, classification_features

RANDOM_SEED = 42
N_SPLITS = 5
BATCH_SIZE = 1024
EPOCHS = 10
LR = 1e-3
HIDDEN1 = 64
HIDDEN2 = 32
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class MLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, HIDDEN1),
            nn.ReLU(),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one_model(X_tr, y_tr, n_features, pos_weight_value):
    model = MLP(n_features).to(DEVICE)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)   # <-- the DD
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        print(f"    epoch {epoch+1:2d}/{EPOCHS}  loss={total_loss/len(ds):.4f}")
    return model


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    logits = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
    return torch.sigmoid(logits).cpu().numpy()


def score(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "auroc": roc_auc_score(y_true, probs),
        "f1": f1_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds),
        "preds": preds,
    }


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
train, val, test = load_data()
features = classification_features(train)

train = train.dropna(subset=[TARGET]).reset_index(drop=True)
val = val.dropna(subset=[TARGET]).reset_index(drop=True)
test = test.dropna(subset=[TARGET]).reset_index(drop=True)

X_train_df = train[features]
y_train = train[TARGET].astype(int).values
groups_train = train["patient_id"].values
X_val_df, y_val = val[features], val[TARGET].astype(int).values
X_test_df, y_test = test[features], test[TARGET].astype(int).values

print(f"Features ({len(features)}): {features}")
print(f"Train sepsis prevalence: {y_train.mean()*100:.2f}%")

# ------------------------------------------------------------------
# 5-fold Grouped CV
# ------------------------------------------------------------------
gkf = GroupKFold(n_splits=N_SPLITS)
cv_scores = {"auroc": [], "f1": [], "precision": [], "recall": []}

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train_df, y_train, groups=groups_train), start=1):
    print(f"\n[Fold {fold}/{N_SPLITS}]")
    X_tr_raw = X_train_df.iloc[tr_idx].values
    X_va_raw = X_train_df.iloc[va_idx].values
    y_tr_fold = y_train[tr_idx]
    y_va_fold = y_train[va_idx]

    imputer = SimpleImputer(strategy="mean").fit(X_tr_raw)
    scaler = StandardScaler().fit(imputer.transform(X_tr_raw))
    X_tr = scaler.transform(imputer.transform(X_tr_raw))
    X_va = scaler.transform(imputer.transform(X_va_raw))

    # pos_weight from training fold only
    n_pos = int((y_tr_fold == 1).sum())
    n_neg = int((y_tr_fold == 0).sum())
    pos_weight_value = n_neg / max(n_pos, 1)
    print(f"    pos_weight = {n_neg}/{n_pos} = {pos_weight_value:.2f}")

    model = train_one_model(X_tr, y_tr_fold, X_tr.shape[1], pos_weight_value)
    probs = predict_proba(model, X_va)
    s = score(y_va_fold, probs)
    print(f"    AUROC={s['auroc']:.4f} F1={s['f1']:.4f} Prec={s['precision']:.4f} Rec={s['recall']:.4f}")
    for k in cv_scores:
        cv_scores[k].append(s[k])

print(f"\n--- 5-Fold Grouped CV Summary (Variant: NN + pos_weight) ---")
for k, vals in cv_scores.items():
    vals = np.array(vals)
    print(f"{k:9s}: {vals.mean():.4f} ± {vals.std():.4f}")

# ------------------------------------------------------------------
# Final model on full training set
# ------------------------------------------------------------------
print("\n[Final model — fit on full training set]")
imputer = SimpleImputer(strategy="mean").fit(X_train_df.values)
scaler = StandardScaler().fit(imputer.transform(X_train_df.values))
X_train_final = scaler.transform(imputer.transform(X_train_df.values))
X_val_final = scaler.transform(imputer.transform(X_val_df.values))
X_test_final = scaler.transform(imputer.transform(X_test_df.values))

n_pos = int((y_train == 1).sum())
n_neg = int((y_train == 0).sum())
pos_weight_value = n_neg / max(n_pos, 1)
print(f"Final pos_weight = {n_neg}/{n_pos} = {pos_weight_value:.2f}")

final_model = train_one_model(X_train_final, y_train, X_train_final.shape[1], pos_weight_value)

records = [{
    "split": "cv_train",
    "auroc": float(np.mean(cv_scores["auroc"])),
    "auroc_std": float(np.std(cv_scores["auroc"])),
    "f1": float(np.mean(cv_scores["f1"])),
    "precision": float(np.mean(cv_scores["precision"])),
    "recall": float(np.mean(cv_scores["recall"])),
}]

confmats = {}
for label, X, y in [("val", X_val_final, y_val), ("test", X_test_final, y_test)]:
    probs = predict_proba(final_model, X)
    s = score(y, probs)
    print(f"\n--- {label} ---")
    print(f"AUROC    : {s['auroc']:.4f}")
    print(f"F1       : {s['f1']:.4f}")
    print(f"Precision: {s['precision']:.4f}")
    print(f"Recall   : {s['recall']:.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(y, s['preds'])}")
    print(classification_report(y, s['preds'], target_names=['No Sepsis', 'Sepsis'], zero_division=0))
    records.append({
        "split": label,
        "auroc": s["auroc"],
        "f1": s["f1"],
        "precision": s["precision"],
        "recall": s["recall"],
    })
    confmats[label] = confusion_matrix(y, s["preds"])

variant_df = pd.DataFrame(records)
metrics_path = os.path.join(RESULTS_DIR, "mamoon_nn_pos_weight_metrics.csv")
variant_df.to_csv(metrics_path, index=False)

# Baseline-vs-variant comparison table.
# Run neural_network_baseline.py first (via run_all.py) so these files exist.
baseline_metrics_path = os.path.join(RESULTS_DIR, "baseline_nn_metrics.csv")
comparison_path = os.path.join(RESULTS_DIR, "mamoon_nn_pos_weight_vs_baseline.csv")
baseline_metrics_available = os.path.exists(baseline_metrics_path)
if baseline_metrics_available:
    baseline_df = pd.read_csv(baseline_metrics_path).assign(model="baseline_nn")
    variant_tagged = variant_df.assign(model="nn_pos_weight")
    comparison = pd.concat([baseline_df, variant_tagged], ignore_index=True)
    comparison = comparison[["model", "split", "auroc", "f1", "precision", "recall"]]
    comparison.to_csv(comparison_path, index=False)
    print(f"\n--- Baseline vs Variant ---\n{comparison.to_string(index=False)}")
else:
    print(f"\n[skip baseline comparison: {baseline_metrics_path} not found — run baseline first]")


def _draw_cm(ax, cm, title):
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Sepsis", "Sepsis"])
    ax.set_yticklabels(["No Sepsis", "Sepsis"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")


# Try to load baseline CMs so the figure is a true side-by-side comparison.
baseline_cms = {}
for label in ("val", "test"):
    p = os.path.join(RESULTS_DIR, f"baseline_nn_confmat_{label}.csv")
    if os.path.exists(p):
        baseline_cms[label] = pd.read_csv(p, index_col=0).values

fig_path = os.path.join(RESULTS_DIR, "mamoon_nn_pos_weight_confmats.png")
if len(baseline_cms) == 2:
    # 2x2 grid: rows = model (baseline, variant), cols = split (val, test)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    _draw_cm(axes[0, 0], baseline_cms["val"],  "Baseline NN — val")
    _draw_cm(axes[0, 1], baseline_cms["test"], "Baseline NN — test")
    _draw_cm(axes[1, 0], confmats["val"],      "NN + pos_weight — val")
    _draw_cm(axes[1, 1], confmats["test"],     "NN + pos_weight — test")
    fig.suptitle("Confusion matrices — baseline vs pos_weight variant")
else:
    # Fall back to variant-only if baseline CMs aren't available yet.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _draw_cm(axes[0], confmats["val"],  "NN + pos_weight — val")
    _draw_cm(axes[1], confmats["test"], "NN + pos_weight — test")
    fig.suptitle("NN + pos_weight — confusion matrices")

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(fig_path, dpi=150)
plt.close(fig)

print(f"\nSaved:\n  {metrics_path}\n  {fig_path}")
if baseline_metrics_available:
    print(f"  {comparison_path}")
