from datasetup import load_data

train, val, test = load_data()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Basic feature selection, for regression we want to predict (MAP)
features = ['HR', 'O2Sat', 'Temp', 'Resp']

# Baseline, drop missing rows
train_clean = train[features + ['MAP']].dropna()
val_clean   = val[features + ['MAP']].dropna()

X_train = train_clean[features]
y_train = train_clean['MAP']
X_val   = val_clean[features]
y_val   = val_clean['MAP']

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
mae  = mean_absolute_error(y_val, preds)
r2   = r2_score(y_val, preds)

print(f"RMSE (Root Mean Squared Error): {rmse:.2f} mmHg")
print(f"MAE (Mean Absolute Error):     {mae:.2f} mmHg")
print(f"R-squared:               {r2:.4f}")

#R squared was extremely low, about 0.01 which is expected