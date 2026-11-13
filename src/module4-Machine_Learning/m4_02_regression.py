import numpy as np
import pandas as pd
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Real dataset: 20,640 California districts. Target = median house value (in $100,000s)
data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.head())

X = df.drop(columns="MedHouseVal")
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"{name:18} MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
    return pred

lin_pred = evaluate("Linear Regression", LinearRegression())
rf_pred = evaluate("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))

# Interpreting linear regression: each coefficient = change in price per unit of the feature
coefs = pd.Series(LinearRegression().fit(X_train, y_train).coef_, index=X.columns).sort_values()
print("\nLinear regression coefficients:\n", coefs.round(3))

plt.figure(figsize=(6, 6))
plt.scatter(y_test, rf_pred, s=4, alpha=0.3)
plt.plot([0, 5], [0, 5], "r--", label="Perfect prediction")
plt.xlabel("Actual value"); plt.ylabel("Predicted value")
plt.title("Random Forest: actual vs predicted"); plt.legend()
plt.show()
