import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

    # --- 2d. MAP distribution (regression target) ---
    print("\n===== MAP Distribution (Regression Target) =====")
    map_vals = df["MAP"].dropna()
    print(f"Count: {len(map_vals)}, Mean: {map_vals.mean():.1f}, Std: {map_vals.std():.1f}")
    print(f"Min: {map_vals.min():.1f}, Max: {map_vals.max():.1f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(map_vals, bins=80, color="steelblue", edgecolor="white")
    ax.axvline(map_vals.mean(), color="red", linestyle="--", label=f"Mean = {map_vals.mean():.1f}")
    ax.set_xlabel("MAP (mm Hg)")
    ax.set_ylabel("Count")
    ax.set_title("MAP Distribution (Regression Target)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, "plots", "map_distribution.png"), dpi=150)
    plt.close(fig)

    # --- 2e. Correlation heatmap (features vs. targets) ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "patient_id"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax,
                xticklabels=True, yticklabels=True, linewidths=0.3)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, "plots", "correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    # Print correlations with targets
    print("\n===== Correlations with MAP =====")
    map_corr = corr["MAP"].drop(["MAP", "SepsisLabel"], errors="ignore").sort_values(key=abs, ascending=False)
    print(map_corr.to_string())

    print("\n===== Correlations with SepsisLabel =====")
    sep_corr = corr["SepsisLabel"].drop(["MAP", "SepsisLabel"], errors="ignore").sort_values(key=abs, ascending=False)
    print(sep_corr.to_string())

    # Bar chart of correlations with targets
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    map_corr.sort_values().plot.barh(ax=axes[0], color="steelblue")
    axes[0].set_title("Feature Correlation with MAP")
    axes[0].set_xlabel("Pearson r")
    sep_corr.sort_values().plot.barh(ax=axes[1], color="#dd8452")
    axes[1].set_title("Feature Correlation with SepsisLabel")
    axes[1].set_xlabel("Pearson r")
    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, "plots", "target_correlations.png"), dpi=150)
    plt.close(fig)

    # --- 2f. Sepsis vs Non-Sepsis feature distributions ---
    sepsis_df = df[df["SepsisLabel"] == 1]
    no_sepsis_df = df[df["SepsisLabel"] == 0]

    # Vitals comparison
    compare_vitals = ["HR", "O2Sat", "Temp", "Resp"]
    fig, axes = plt.subplots(1, len(compare_vitals), figsize=(18, 4))
    for ax, col in zip(axes, compare_vitals):
        ax.hist(no_sepsis_df[col].dropna(), bins=50, alpha=0.6, color="#4c72b0", label="No Sepsis", density=True)
        ax.hist(sepsis_df[col].dropna(), bins=50, alpha=0.6, color="#dd8452", label="Sepsis", density=True)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.legend(fontsize=8)
    fig.suptitle("Vital Signs: Sepsis vs Non-Sepsis (Normalized)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(DATA_DIR, "plots", "vitals_sepsis_comparison.png"), dpi=150)
    plt.close(fig)

    # Key lab values comparison
    compare_labs = ["Creatinine", "BUN", "Lactate", "WBC", "Glucose", "Platelets"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, col in zip(axes.flat, compare_labs):
        vals_no = no_sepsis_df[col].dropna()
        vals_yes = sepsis_df[col].dropna()
        if len(vals_no) > 0 and len(vals_yes) > 0:
            ax.hist(vals_no, bins=50, alpha=0.6, color="#4c72b0", label="No Sepsis", density=True)
            ax.hist(vals_yes, bins=50, alpha=0.6, color="#dd8452", label="Sepsis", density=True)
            ax.legend(fontsize=8)
        ax.set_title(col)
        ax.set_xlabel(col)
    fig.suptitle("Lab Values: Sepsis vs Non-Sepsis (Normalized)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(DATA_DIR, "plots", "labs_sepsis_comparison.png"), dpi=150)
    plt.close(fig)

    # --- 2g. Patient-level sepsis prevalence ---
    patient_sepsis = df.groupby("patient_id")["SepsisLabel"].max()
    pat_counts = patient_sepsis.value_counts().sort_index()
    print(f"\n===== Patient-Level Sepsis Prevalence =====")
    print(f"Sepsis patients:     {pat_counts.get(1, 0)}")
    print(f"Non-sepsis patients: {pat_counts.get(0, 0)}")
    print(f"Prevalence: {pat_counts.get(1, 0) / len(patient_sepsis) * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(5, 4))
    pat_counts.plot.bar(ax=ax, color=["#4c72b0", "#dd8452"])
    ax.set_xticklabels(["No Sepsis (0)", "Sepsis (1)"], rotation=0)
    ax.set_ylabel("Number of Patients")
    ax.set_title("Patient-Level Sepsis Prevalence")
    for i, v in enumerate(pat_counts):
        ax.text(i, v + len(patient_sepsis) * 0.005, f"{v:,}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, "plots", "patient_sepsis_prevalence.png"), dpi=150)
    plt.close(fig)

    # --- 2h. Demographics ---
    # Age distribution by sepsis status
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(no_sepsis_df["Age"].dropna(), bins=40, alpha=0.6, color="#4c72b0", label="No Sepsis", density=True)
    axes[0].hist(sepsis_df["Age"].dropna(), bins=40, alpha=0.6, color="#dd8452", label="Sepsis", density=True)
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Age Distribution by Sepsis Status")
    axes[0].legend()

    # Gender split and sepsis rate
    gender_sepsis = df.groupby("Gender")["SepsisLabel"].mean() * 100
    gender_sepsis.index = ["Female", "Male"]
    gender_sepsis.plot.bar(ax=axes[1], color=["#c44e52", "#4c72b0"])
    axes[1].set_ylabel("Sepsis Prevalence (%)")
    axes[1].set_title("Sepsis Prevalence by Gender")
    axes[1].set_xticklabels(["Female", "Male"], rotation=0)
    for i, v in enumerate(gender_sepsis):
        axes[1].text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, "plots", "demographics.png"), dpi=150)
    plt.close(fig)

    # --- 2i. Basic descriptive stats ---
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
