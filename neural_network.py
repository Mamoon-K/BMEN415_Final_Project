"""
Team Baseline — Neural Network Classifier (predict SepsisLabel)

Design choices (team-agreed):
- All allowed input features (exclude target SepsisLabel, regression target MAP, patient_id)
- Mean imputation for missing values (fit per CV fold to avoid leakage)
- StandardScaler (NNs need standardized inputs)
- Patient-level split (done in datasetup.py)
- 5-fold GroupKFold CV on training set, grouped by patient_id
- Simple feed-forward NN: Input -> 64 -> 32 -> 1, ReLU activations, dropout disabled
- BCEWithLogitsLoss (NO pos_weight — imbalance handling is a team member's DD)
- Adam optimizer, lr=1e-3, batch_size=1024, 10 epochs
- Metrics: AUROC, F1, Precision, Recall — CV mean±std and final held-out test
- Fixed seed for reproducibility
"""

import os
import numpy as np
import pandas as pd
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


def train_one_model(X_tr, y_tr, n_features):
    model = MLP(n_features).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()  # no pos_weight — baseline
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
        avg = total_loss / len(ds)
        print(f"    epoch {epoch+1:2d}/{EPOCHS}  loss={avg:.4f}")
    return model


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    logits = model(xb)
    return torch.sigmoid(logits).cpu().numpy()


def score(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "auroc": roc_auc_score(y_true, probs),
        "f1": f1_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "preds": preds,
    }


# ------------------------------------------------------------------
# Load & prepare data
# ------------------------------------------------------------------
train, val, test = load_data()
features = classification_features(train)

train = train.dropna(subset=[TARGET]).reset_index(drop=True)
val = val.dropna(subset=[TARGET]).reset_index(drop=True)
test = test.dropna(subset=[TARGET]).reset_index(drop=True)

X_train_df = train[features]
y_train = train[TARGET].astype(int).values
groups_train = train["patient_id"].values

X_val_df = val[features]
y_val = val[TARGET].astype(int).values

X_test_df = test[features]
y_test = test[TARGET].astype(int).values

print(f"Features ({len(features)}): {features}")
print(f"Train rows: {len(X_train_df)} ({pd.Series(groups_train).nunique()} patients) — sepsis {y_train.mean()*100:.2f}%")
print(f"Val rows:   {len(X_val_df)} — sepsis {y_val.mean()*100:.2f}%")
print(f"Test rows:  {len(X_test_df)} — sepsis {y_test.mean()*100:.2f}%")

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

    # Fit imputer + scaler on train fold only
    imputer = SimpleImputer(strategy="mean").fit(X_tr_raw)
    X_tr_imp = imputer.transform(X_tr_raw)
    X_va_imp = imputer.transform(X_va_raw)

    scaler = StandardScaler().fit(X_tr_imp)
    X_tr_s = scaler.transform(X_tr_imp)
    X_va_s = scaler.transform(X_va_imp)

    model = train_one_model(X_tr_s, y_tr_fold, n_features=X_tr_s.shape[1])

    probs = predict_proba(model, X_va_s)
    s = score(y_va_fold, probs)
    print(f"    AUROC={s['auroc']:.4f} F1={s['f1']:.4f} Prec={s['precision']:.4f} Rec={s['recall']:.4f}")
    for k in cv_scores:
        cv_scores[k].append(s[k])

print(f"\n--- 5-Fold Grouped CV Summary ---")
for k, vals in cv_scores.items():
    vals = np.array(vals)
    print(f"{k:9s}: {vals.mean():.4f} ± {vals.std():.4f}")

# ------------------------------------------------------------------
# Final model: fit on full training set, evaluate on val + test
# ------------------------------------------------------------------
print("\n[Final model — fit on full training set]")
imputer = SimpleImputer(strategy="mean").fit(X_train_df.values)
scaler = StandardScaler().fit(imputer.transform(X_train_df.values))

X_train_final = scaler.transform(imputer.transform(X_train_df.values))
X_val_final = scaler.transform(imputer.transform(X_val_df.values))
X_test_final = scaler.transform(imputer.transform(X_test_df.values))

final_model = train_one_model(X_train_final, y_train, n_features=X_train_final.shape[1])

records = [{
    "split": "cv_train",
    "auroc": float(np.mean(cv_scores["auroc"])),
    "auroc_std": float(np.std(cv_scores["auroc"])),
    "f1": float(np.mean(cv_scores["f1"])),
    "precision": float(np.mean(cv_scores["precision"])),
    "recall": float(np.mean(cv_scores["recall"])),
}]

for label, X, y in [("val", X_val_final, y_val),
                    ("test", X_test_final, y_test)]:
    probs = predict_proba(final_model, X)
    s = score(y, probs)
    cm = confusion_matrix(y, s['preds'])
    print(f"\n--- {label} ---")
    print(f"AUROC    : {s['auroc']:.4f}")
    print(f"F1       : {s['f1']:.4f}")
    print(f"Precision: {s['precision']:.4f}")
    print(f"Recall   : {s['recall']:.4f}")
    print(f"Confusion matrix:\n{cm}")
    print(classification_report(y, s['preds'], target_names=['No Sepsis', 'Sepsis'], zero_division=0))
    records.append({
        "split": label,
        "auroc": s["auroc"],
        "f1": s["f1"],
        "precision": s["precision"],
        "recall": s["recall"],
    })
    # Save CM so baseline-vs-variant comparison figures are reproducible.
    pd.DataFrame(cm,
                 index=["true_no_sepsis", "true_sepsis"],
                 columns=["pred_no_sepsis", "pred_sepsis"]).to_csv(
        os.path.join(RESULTS_DIR, f"baseline_nn_confmat_{label}.csv")
    )

pd.DataFrame(records).to_csv(
    os.path.join(RESULTS_DIR, "baseline_nn_metrics.csv"), index=False
)
print(f"\nSaved: {os.path.join(RESULTS_DIR, 'baseline_nn_metrics.csv')}")
