from datasetup import load_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

train, val, test = load_data()

target = 'MAP'

# Baseline: use all allowed input features (exclude outputs and banned features)
excluded = ['MAP', 'SBP', 'DBP', 'SepsisLabel', 'patient_id']
features = [col for col in train.columns if col not in excluded]

# Baseline missing data handling: drop rows with any NaN in features or target
train_clean = train[features + [target]].dropna()
val_clean   = val[features + [target]].dropna()
test_clean  = test[features + [target]].dropna()

X_train = train_clean[features]
y_train = train_clean[target]
X_val   = val_clean[features]
y_val   = val_clean[target]
X_test  = test_clean[features]
y_test  = test_clean[target]

print(f"Features ({len(features)}): {features}")
print(f"Train rows: {len(X_train)}, Val rows: {len(X_val)}, Test rows: {len(X_test)}")

model = LinearRegression()
model.fit(X_train, y_train)

# Validation metrics
val_preds = model.predict(X_val)
print("\n--- Validation Set ---")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, val_preds)):.2f} mmHg")
print(f"MAE:  {mean_absolute_error(y_val, val_preds):.2f} mmHg")
print(f"R²:   {r2_score(y_val, val_preds):.4f}")

# Test metrics
test_preds = model.predict(X_test)
print("\n--- Test Set ---")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, test_preds)):.2f} mmHg")
print(f"MAE:  {mean_absolute_error(y_test, test_preds):.2f} mmHg")
print(f"R²:   {r2_score(y_test, test_preds):.4f}")