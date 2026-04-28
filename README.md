# BMEN 415 Final Project

This repository contains:
- exploratory data analysis (`eda.py`)
- three team baseline models
- individual design-decision scripts
- a one-command runner (`run_all.py`)

## Environment

Use Python 3.10+ in a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Setup

The code supports two data layouts:

1. Preferred: place `combined_data.csv` at:

```text
data/combined_data.csv
```

2. Fallback: place the raw PhysioNet training folders at:

```text
data/training/training_setA/
data/training/training_setB/
```

If `combined_data.csv` is missing, `datasetup.py` will build it from the raw `.psv` files and cache it under `data/combined_data.csv`.

## One-Command Run

After activating the virtual environment, run:

```bash
python run_all.py
```
`run_all.py` automatically configures a headless Matplotlib backend and local cache directories, so no extra shell environment variables should be needed.

## Main Scripts

- `regression_baseline.py`: baseline linear regression for MAP
- `decision_tree_baseline.py`: baseline non-neural-network classifier for sepsis
- `neural_network_baseline.py`: baseline neural-network classifier for sepsis
- `mamoon_regression_ridge_lasso.py`: DD1
- `mamoon_decision_tree_missingness.py`: DD2
- `mamoon_nn_pos_weight.py`: DD3
- `baseline_summary.py`: consolidated baseline table and figure

## Outputs

All generated outputs are written to:

```text
results/
```

Key baseline outputs:
- `results/baseline_regression_metrics.csv`
- `results/baseline_dt_metrics.csv`
- `results/baseline_nn_metrics.csv`
- `results/baseline_nn_confmat_val.csv`
- `results/baseline_nn_confmat_test.csv`
- `results/baseline_summary.csv`
- `results/baseline_summary.png`

EDA figures are written to:
- `results/plots/`

Design-decision outputs are also written to `results/`.

## Reproducibility Notes

- Splits are done by `patient_id`, not by row.
- Shared random seed: `42`
- Feature exclusions are governed centrally in `feature_policy.py`.

## TA Run Checklist

1. Create and activate the virtual environment.
2. Install `requirements.txt`.
3. Ensure the dataset is present at `data/combined_data.csv` or under `data/training/`.
4. Run `python run_all.py`.
5. Inspect outputs in `results/`.

## Zip Submission Note

If this project is submitted as a `.zip`, the archive should include:
- the full codebase
- `data/combined_data.csv` or the raw `data/training/` folders
- the `results/` directory if you want generated outputs included directly
