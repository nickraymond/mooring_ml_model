# Mooring Load Prediction with Time Series Cross-Validation + Progress Feedback

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
df = pd.read_csv("synthetic_mooring_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Prepare features and target
features = ["wave_height", "current_speed", "sensor_tilt"]
target = "max_force"
X = df[features]
y = df[target]

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
results = []

print("Starting time series cross-validation...")

for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
    print(f"Processing fold {fold}/5")

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 1. Linear Regression
    lr = LinearRegression().fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # 2. Polynomial Regression (deg=2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    lr_poly = LinearRegression().fit(X_poly_train, y_train)
    y_pred_poly = lr_poly.predict(X_poly_test)

    # 3. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # 4. XGBoost
    xgb = XGBRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    # Evaluate
    models = {
        "Linear": y_pred_lr,
        "Polynomial (deg=2)": y_pred_poly,
        "Random Forest": y_pred_rf,
        "XGBoost": y_pred_xgb
    }

    for name, y_pred in models.items():
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R²": r2})

print("Cross-validation complete. Summary:")

# Summary
results_df = pd.DataFrame(results)
summary = results_df.groupby("Model").mean().reset_index()
print(summary)
