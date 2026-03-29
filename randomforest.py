from datasetup import load_data

train, val, test = load_data()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                              roc_auc_score, f1_score, 
                              recall_score, precision_score)
from sklearn.preprocessing import StandardScaler

#We can change it back to SVM if we would like but the performance is extremely slow, i think random forest will be our best bet for the base model

#Potential Design Decisions, change features, add more features, change scalling to better optimize  and prioritize sepsis cases

features = ['HR', 'O2Sat', 'Temp', 'Resp', 'Age', 'ICULOS'] #what should be the baseline features? Predicting SepsisLabel

train_clean = train[features + ['SepsisLabel']].dropna()
val_clean   = val[features + ['SepsisLabel']].dropna()
test_clean  = test[features + ['SepsisLabel']].dropna()

X_train = train_clean[features]
y_train = train_clean['SepsisLabel']
X_val   = val_clean[features]
y_val   = val_clean['SepsisLabel']
X_test  = test_clean[features]
y_test  = test_clean['SepsisLabel']

#scaling
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1, 
    verbose=1,
    class_weight='balanced' # important to handle sepsis cases better
)

model.fit(X_train, y_train)


# Validation metrics
preds = model.predict(X_val)
probs = model.predict_proba(X_val)[:, 1]

print("\n--- Random Forest Metrics (Validation) ---")
print(f"Accuracy:  {model.score(X_val, y_val):.4f}")
print(f"Precision: {precision_score(y_val, preds):.4f}")
print(f"Recall:    {recall_score(y_val, preds):.4f}")
print(f"F1 Score:  {f1_score(y_val, preds):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_val, probs):.4f}")
print("\nFull Report:")
print(classification_report(y_val, preds, target_names=['No Sepsis', 'Sepsis']))
print("Confusion Matrix:")
print(confusion_matrix(y_val, preds))

# Test metrics
test_preds = model.predict(X_test)
test_probs = model.predict_proba(X_test)[:, 1]

print("\n--- Random Forest Metrics (Test) ---")
print(f"Accuracy:  {model.score(X_test, y_test):.4f}")
print(f"Precision: {precision_score(y_test, test_preds):.4f}")
print(f"Recall:    {recall_score(y_test, test_preds):.4f}")
print(f"F1 Score:  {f1_score(y_test, test_preds):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, test_probs):.4f}")
print("\nFull Report:")
print(classification_report(y_test, test_preds, target_names=['No Sepsis', 'Sepsis']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_preds))