"""
One-command orchestrator for the BMEN 415 Final Project.

Runs EDA, the three team baselines, and Mamoon's three design-decision variants.
All outputs land in results/ (metrics CSVs and figures).

Usage:
    python run_all.py
"""

import os
import subprocess
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "eda.py",
    "regression.py",
    "desicion_tree_classifier_and_random_forrest.py",
    "neural_network.py",
    "mamoon_regression_ridge_lasso.py",
    "mamoon_decision_tree_missingness.py",
    "mamoon_nn_pos_weight.py",
]


def run(script):
    path = os.path.join(THIS_DIR, script)
    print(f"\n{'=' * 70}\n▶ {script}\n{'=' * 70}", flush=True)
    t0 = time.time()
    result = subprocess.run([sys.executable, path], cwd=THIS_DIR)
    elapsed = time.time() - t0
    return result.returncode, elapsed


def main():
    os.makedirs(os.path.join(THIS_DIR, "results"), exist_ok=True)
    failures = []
    for script in SCRIPTS:
        code, elapsed = run(script)
        status = "OK" if code == 0 else f"FAIL ({code})"
        print(f"\n[{status}] {script}  —  {elapsed:.1f}s", flush=True)
        if code != 0:
            failures.append(script)

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    print(f"Ran {len(SCRIPTS)} scripts; {len(failures)} failed.")
    if failures:
        print("Failed:")
        for s in failures:
            print(f"  - {s}")
        sys.exit(1)
    print(f"All outputs in: {os.path.join(THIS_DIR, 'results')}")


if __name__ == "__main__":
    main()
