import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import time

# Simulated inputs and outputs for demo
wave_height = np.array([0.5, 0.8, 1.2, 1.5, 2.0, 2.5])
current_speed = np.array([0.2, 0.3, 0.4, 0.5, 0.7, 0.9])
sensor_tilt = np.array([5, 6, 8, 10, 15, 18])
load = np.array([50, 65, 90, 120, 160, 210])  # Target variable

# Feature matrix and target
X = np.vstack((wave_height, current_speed, sensor_tilt)).T
y = load

print("➡️  Starting model comparison...\n")

# ---- LINEAR REGRESSION ----
print("🔹 Running Linear Regression...")
lin_reg = LinearRegression()
cv_scores_lin = cross_val_score(lin_reg, X, y, scoring='neg_root_mean_squared_error', cv=3)
print(f"   RMSE (Linear): {-np.mean(cv_scores_lin):.2f}")
time.sleep(1)

# ---- POLYNOMIAL REGRESSION ----
print("🔹 Running Polynomial Regression (degree=2)...")
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
cv_scores_poly = cross_val_score(poly_reg, X_poly, y, scoring='neg_root_mean_squared_error', cv=3)
print(f"   RMSE (Polynomial): {-np.mean(cv_scores_poly):.2f}")
time.sleep(1)

# ---- RANDOM FOREST ----
print("🔹 Running Random Forest Regression...")
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores_rf = cross_val_score(rf_reg, X, y, scoring='neg_root_mean_squared_error', cv=3)
print(f"   RMSE (Random Forest): {-np.mean(cv_scores_rf):.2f}")
time.sleep(1)

print("\n✅ Model comparison complete.")
