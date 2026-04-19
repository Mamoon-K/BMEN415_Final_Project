"""
Unified data loader for the PhysioNet/CinC 2019 dataset.

Fast path: reads data/combined_data.csv if present.
Fallback: builds the combined CSV from data/training/training_setA and
training_setB raw .psv files, then caches it.

All model scripts import load_data() from here.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "combined_data.csv")
SET_A = os.path.join(DATA_DIR, "training", "training_setA")
SET_B = os.path.join(DATA_DIR, "training", "training_setB")

RANDOM_SEED = 42


def _load_psv_folder(folder):
    frames = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".psv"):
            continue
        df = pd.read_csv(os.path.join(folder, fname), sep="|")
        df.insert(0, "patient_id", fname.replace(".psv", ""))
        frames.append(df)
    return frames


def _build_combined_csv():
    if not (os.path.isdir(SET_A) and os.path.isdir(SET_B)):
        raise FileNotFoundError(
            f"Neither {CSV_PATH} nor raw .psv folders exist.\n"
            f"Expected either:\n"
            f"  - {CSV_PATH}, or\n"
            f"  - {SET_A}/ and {SET_B}/ with raw .psv files\n"
            f"Download the PhysioNet/CinC 2019 training sets into data/training/."
        )
    frames = _load_psv_folder(SET_A) + _load_psv_folder(SET_B)
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return df


def load_combined_df():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return _build_combined_csv()


def load_data():
    """Return (train, val, test) split by patient_id. 72/8/20 effective split."""
    data = load_combined_df()
    patient_ids = data["patient_id"].unique()
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.2, random_state=RANDOM_SEED)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.1, random_state=RANDOM_SEED)
    train = data[data["patient_id"].isin(train_ids)]
    val = data[data["patient_id"].isin(val_ids)]
    test = data[data["patient_id"].isin(test_ids)]
    return train, val, test
