"""
Feature governance policy — single source of truth.

Based on the BMEN 415 Final Project Outline (updated 2026-02-26), p. 2:
- Features in YELLOW are outputs and cannot be used as INPUTS:
    MAP, SepsisLabel
- Features marked with a CROSS cannot be used AT ALL because they are too
  closely related to the outputs (direct components of MAP, which is an
  output): SBP, DBP.

patient_id is a row identifier, not a clinical feature.

Any script that builds X from the dataframe MUST import from this module.
Do not redefine EXCLUDED lists locally.
"""

# Globally disallowed as model inputs for EVERY task in this project.
# Includes both yellow outputs and crossed-out closely-related features.
GLOBALLY_DISALLOWED = ["MAP", "SepsisLabel", "SBP", "DBP", "patient_id"]

REGRESSION_TARGET = "MAP"
CLASSIFICATION_TARGET = "SepsisLabel"

# Both tasks draw inputs from the same allowed set; the task's target
# is already in GLOBALLY_DISALLOWED so it is never an input either.
REGRESSION_EXCLUDED = list(GLOBALLY_DISALLOWED)
CLASSIFICATION_EXCLUDED = list(GLOBALLY_DISALLOWED)


def regression_features(df):
    return [c for c in df.columns if c not in REGRESSION_EXCLUDED]


def classification_features(df):
    return [c for c in df.columns if c not in CLASSIFICATION_EXCLUDED]
