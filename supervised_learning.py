"""
Week 4 Task: Supervised Learning Model Implementation
Dataset: Country Development Data (same dataset as Week 3)
Problem: REGRESSION — predict life_expec (life expectancy) from the other
socio-economic and health indicators.

Run this in Google Colab. Upload country_data_real.csv to the session first
(folder icon on the left -> upload), same as Week 3.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")

# ---------------------------------------------------------------
# STEP 1: LOAD THE DATA
# ---------------------------------------------------------------
df = pd.read_csv("country_data_real.csv")
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

# ---------------------------------------------------------------
# STEP 2: DEFINE THE PROBLEM
# ---------------------------------------------------------------
# Target: life_expec (continuous number) -> regression, not classification.
# Features: everything except country (identifier, not predictive) and
# life_expec itself (that's the target).
target = "life_expec"
feature_cols = [c for c in df.columns if c not in ["country", target]]

print("\nFeatures used:", feature_cols)
print("\nCorrelation of each feature with life_expec:")
print(df[feature_cols + [target]].corr()[target].sort_values(ascending=False))

# ---------------------------------------------------------------
# STEP 3: FEATURE ENGINEERING
# ---------------------------------------------------------------
# income and gdpp are heavily right-skewed (a small number of very rich
# countries stretch the scale). Log-transforming compresses that skew,
# which helps a linear model in particular treat "going from $1k to $2k
# income" as comparable in importance to "going from $50k to $100k",
# rather than the raw dollar gap dominating.
df["log_income"] = np.log1p(df["income"])
df["log_gdpp"] = np.log1p(df["gdpp"])

# health spending is reported as % of GDP, which doesn't say much on its
# own without knowing the size of the economy behind it. Multiplying by
# gdpp gives a rough absolute health-spending-per-person figure instead.
df["health_spend_per_capita"] = (df["health"] / 100) * df["gdpp"]

engineered_features = feature_cols + ["log_income", "log_gdpp", "health_spend_per_capita"]
# drop the raw income/gdpp now that we have log versions, to avoid feeding
# the model two versions of essentially the same signal
engineered_features = [f for f in engineered_features if f not in ["income", "gdpp"]]

print("\nFinal feature list after engineering:", engineered_features)

X = df[engineered_features]
y = df[target]

# ---------------------------------------------------------------
# STEP 4: TRAIN/TEST SPLIT
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining set: {X_train.shape[0]} countries")
print(f"Test set: {X_test.shape[0]} countries")

# ---------------------------------------------------------------
# STEP 5: SCALE FEATURES (needed for Linear Regression)
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# STEP 6: TRAIN TWO MODELS AND COMPARE
# ---------------------------------------------------------------
# Linear Regression: simple, interpretable baseline.
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# Random Forest: handles non-linear relationships and feature interactions
# without needing scaling — a stronger model to compare against the baseline.
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)  # tree models don't need scaled input
rf_pred = rf.predict(X_test)

def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name}: R2={r2:.3f}  MAE={mae:.2f} years  RMSE={rmse:.2f} years")
    return r2, mae, rmse

print("\nTest set performance:")
lr_r2, lr_mae, lr_rmse = evaluate("Linear Regression", y_test, lr_pred)
rf_r2, rf_mae, rf_rmse = evaluate("Random Forest    ", y_test, rf_pred)

# ---------------------------------------------------------------
# STEP 7: CROSS-VALIDATION (check the score isn't a lucky split)
# ---------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lr_cv_scores = cross_val_score(lr, scaler.fit_transform(X), y, cv=kf, scoring="r2")
rf_cv_scores = cross_val_score(rf, X, y, cv=kf, scoring="r2")

print(f"\nLinear Regression 5-fold CV R2: {lr_cv_scores.mean():.3f} (+/- {lr_cv_scores.std():.3f})")
print(f"Random Forest     5-fold CV R2: {rf_cv_scores.mean():.3f} (+/- {rf_cv_scores.std():.3f})")

# ---------------------------------------------------------------
# STEP 8: VISUALIZE PREDICTIONS
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, pred, name in zip(axes, [lr_pred, rf_pred], ["Linear Regression", "Random Forest"]):
    ax.scatter(y_test, pred, alpha=0.6, color="#2c7fb8")
    lims = [y_test.min() - 2, y_test.max() + 2]
    ax.plot(lims, lims, "--", color="gray")  # perfect-prediction line
    ax.set_xlabel("Actual Life Expectancy")
    ax.set_ylabel("Predicted Life Expectancy")
    ax.set_title(name)
plt.tight_layout()
plt.savefig("01_actual_vs_predicted.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# STEP 9: FEATURE IMPORTANCE (Random Forest)
# ---------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=engineered_features).sort_values()

plt.figure(figsize=(8, 6))
importances.plot(kind="barh", color="#d95f0e")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("02_feature_importance.png", dpi=150)
plt.show()

print("\nTop 5 most important features:")
print(importances.sort_values(ascending=False).head(5))

# ---------------------------------------------------------------
# STEP 10: RESIDUAL ANALYSIS (Random Forest)
# ---------------------------------------------------------------
residuals = y_test - rf_pred

plt.figure(figsize=(7, 5))
plt.scatter(rf_pred, residuals, alpha=0.6, color="#d95f0e")
plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("Predicted Life Expectancy")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot (Random Forest)")
plt.tight_layout()
plt.savefig("03_residuals.png", dpi=150)
plt.show()

print("\nLargest prediction errors:")
error_df = pd.DataFrame({
    "country": df.loc[X_test.index, "country"],
    "actual": y_test,
    "predicted": rf_pred.round(1),
    "error": (y_test - rf_pred).round(1)
}).sort_values("error", key=abs, ascending=False)
print(error_df.head(8))

print("\nDone.")
