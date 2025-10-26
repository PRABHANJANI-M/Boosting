import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv("Car.csv")
df = df.dropna()
df_encoded = pd.get_dummies(df, drop_first=True)

# Log-transform the target
df_encoded["selling_price"] = np.log1p(df_encoded["selling_price"])

# Split data
X = df_encoded.drop("selling_price", axis=1)
y = df_encoded["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define model
gbr = GradientBoostingRegressor(random_state=42)

# Parameter grid for tuning
params = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 4]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    gbr,
    param_grid=params,
    cv=kf,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

# Get best model
best_model = grid.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
y_test_exp = np.expm1(y_test)
y_pred_exp = np.expm1(y_pred)

rmse = mean_squared_error(y_test_exp, y_pred_exp)**0.5
r2 = r2_score(y_test_exp, y_pred_exp)

print("✅ Best Parameters:", grid.best_params_)
print("📉 RMSE:", round(rmse, 2))
print("📈 R² Score:", round(r2, 2))

# Save model and columns
joblib.dump(best_model, "car_price_model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")

print("\n💾 Model and column names saved successfully!")
