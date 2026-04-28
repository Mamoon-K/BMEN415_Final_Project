"""
Consolidated baseline overview for the team-evaluated Baseline Models section.

Reads the three individual baseline metric CSVs (regression, DT, NN) that
were saved by the baseline scripts and produces:
    - results/baseline_summary.csv  (all models in one long-format table)
    - results/baseline_summary.png  (regression metrics + classification metrics)

Run after the three baseline scripts so their CSVs exist.
"""

import os
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_CSV = os.path.join(RESULTS_DIR, "baseline_summary.csv")
OUT_FIG = os.path.join(RESULTS_DIR, "baseline_summary.png")

BASELINES = {
    "regression_LR": ("regression",     "baseline_regression_metrics.csv"),
    "decision_tree": ("classification", "baseline_dt_metrics.csv"),
    "neural_net":    ("classification", "baseline_nn_metrics.csv"),
}

frames = []
for model, (task, fname) in BASELINES.items():
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        print(f"[skip] {path} not found — run {fname.replace('baseline_', '').replace('_metrics.csv', '')} baseline first")
        continue
    df = pd.read_csv(path)
    df.insert(0, "model", model)
    df.insert(1, "task", task)
    frames.append(df)

if not frames:
    raise SystemExit("No baseline CSVs found — run the baselines first.")

summary = pd.concat(frames, ignore_index=True)
summary.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}\n")
print(summary.to_string(index=False))

# Figure: regression panel on the left, classification panel on the right.
fig, (ax_reg, ax_clf) = plt.subplots(1, 2, figsize=(14, 5))

reg = summary[summary["task"] == "regression"]
if not reg.empty:
    metrics = [c for c in ["rmse", "mae", "r2"] if c in reg.columns]
    reg_plot = reg[reg["split"].isin(["cv_train", "val", "test"])].set_index("split")[metrics]
    reg_plot.plot.bar(ax=ax_reg)
    ax_reg.set_title("Regression baseline (LinearRegression) — RMSE / MAE / R²")
    ax_reg.set_ylabel("value")
    ax_reg.axhline(0, color="k", linewidth=0.5)

clf = summary[summary["task"] == "classification"]
if not clf.empty:
    metrics = [c for c in ["auroc", "auprc", "f1", "precision", "recall"] if c in clf.columns]
    test_rows = clf[clf["split"] == "test"].set_index("model")[metrics]
    test_rows.plot.bar(ax=ax_clf)
    ax_clf.set_title("Classification baselines — held-out test metrics")
    ax_clf.set_ylabel("value")
    ax_clf.set_ylim(0, 1)
    ax_clf.legend(loc="upper right", fontsize=9)

plt.tight_layout()
fig.savefig(OUT_FIG, dpi=150)
plt.close(fig)
print(f"\nSaved: {OUT_FIG}")
