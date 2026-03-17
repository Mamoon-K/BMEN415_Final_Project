import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────
# 1. Load and merge all .psv files into one CSV
# ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "combined_data.csv")

SET_A = "/Users/m.a.k/Desktop/training_setA"
SET_B = "/Users/m.a.k/Desktop/training_setB"


def load_psv_files(folder):
    """Load all .psv files from a folder, adding a patient_id column."""
    frames = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".psv"):
            continue
        patient_id = fname.replace(".psv", "")
        df = pd.read_csv(os.path.join(folder, fname), sep="|")
        df.insert(0, "patient_id", patient_id)
        frames.append(df)
    return frames


def build_combined_csv():
    """Merge training sets A & B into a single CSV."""
    print("Loading training_setA ...")
    frames_a = load_psv_files(SET_A)
    print(f"  {len(frames_a)} patients")

    print("Loading training_setB ...")
    frames_b = load_psv_files(SET_B)
    print(f"  {len(frames_b)} patients")

    df = pd.concat(frames_a + frames_b, ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved combined CSV -> {CSV_PATH}  ({len(df)} rows, {df['patient_id'].nunique()} patients)")
    return df


# ──────────────────────────────────────────
# 2. Exploratory Data Analysis
# ──────────────────────────────────────────
def run_eda(df):
    os.makedirs(os.path.join(DATA_DIR, "plots"), exist_ok=True)

    print("\n===== Dataset Overview =====")
    print(f"Total rows:     {len(df)}")
    print(f"Total patients: {df['patient_id'].nunique()}")
    print(f"Columns:        {df.shape[1]}")
    print(f"\nColumn types:\n{df.dtypes.value_counts()}")

    # --- 2a. Missing data percentage per feature ---
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    print("\n===== Missing Data (%) =====")
    print(missing_pct.to_string())

    fig, ax = plt.subplots(figsize=(12, 8))
    missing_pct.plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Data by Feature")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, "plots", "missing_data.png"), dpi=150)
    plt.close(fig)

    # --- 2b. SepsisLabel class distribution ---
    sepsis_counts = df["SepsisLabel"].value_counts()
    print(f"\n===== Sepsis Class Balance =====\n{sepsis_counts}")
    print(f"Sepsis prevalence: {sepsis_counts.get(1, 0) / len(df) * 100:.2f}%")

    fig, ax = plt.subplots(figsize=(5, 4))
    sepsis_counts.plot.bar(ax=ax, color=["#4c72b0", "#dd8452"])
    ax.set_xticklabels(["No Sepsis (0)", "Sepsis (1)"], rotation=0)
    ax.set_ylabel("Count")
    ax.set_title("SepsisLabel Distribution")
    for i, v in enumerate(sepsis_counts):
        ax.text(i, v + len(df) * 0.005, f"{v:,}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, "plots", "sepsis_distribution.png"), dpi=150)
    plt.close(fig)

    # --- 2c. Vital sign distributions (usable features) ---
    vitals = ["HR", "O2Sat", "Temp", "Resp", "EtCO2"]  # exclude MAP, SBP, DBP
    fig, axes = plt.subplots(1, len(vitals), figsize=(18, 4))
    for ax, col in zip(axes, vitals):
        df[col].dropna().hist(bins=50, ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel(col)
    fig.suptitle("Vital Sign Distributions (features)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(DATA_DIR, "plots", "vital_distributions.png"), dpi=150)
    plt.close(fig)

    # --- 2d. Basic descriptive stats ---
    print("\n===== Descriptive Statistics =====")
    print(df.describe().to_string())

    print(f"\nPlots saved to {os.path.join(DATA_DIR, 'plots')}/")


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        print(f"Loading existing CSV: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
    else:
        df = build_combined_csv()

    run_eda(df)
