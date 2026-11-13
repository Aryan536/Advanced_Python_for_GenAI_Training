import numpy as np
import pandas as pd
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Real-world example: raw customer data with missing values and text categories
df = pd.DataFrame({
    "age":       [25, 41, np.nan, 19, 35, 52, 28, np.nan, 44, 31],
    "city_tier": ["Tier1", "Tier2", "Tier1", "Tier3", "Tier1",
                  "Tier2", "Tier3", "Tier1", "Tier2", "Tier1"],
    "device":    ["App", "Web", "App", np.nan, "Desktop", "App", "Web", "App", "Desktop", "App"],
    "orders":    [3, 12, 7, 1, 9, 15, 2, 5, 11, 6],
    "avg_spend": [1200, 3400, 2100, 600, 2800, 4100, 750, 1500, 3000, 1900],
    "churned":   [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
})
print("Missing values per column:\n", df.isna().sum())

X = df.drop(columns="churned")
y = df["churned"]

# 1) Split FIRST so the test set never influences preprocessing (avoids data leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 2) Different treatment for numeric and categorical columns
numeric_cols = ["age", "orders", "avg_spend"]
categorical_cols = ["city_tier", "device"]

numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
categorical_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                             ("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer([("num", numeric_pipe, numeric_cols),
                                  ("cat", categorical_pipe, categorical_cols)])

# 3) fit on train only, then transform both
X_train_ready = preprocessor.fit_transform(X_train)
X_test_ready = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()
print("\nFeatures after preprocessing:", list(feature_names))
print(pd.DataFrame(X_train_ready, columns=feature_names).round(2).to_string())
print("\nTrain shape:", X_train_ready.shape, "| Test shape:", X_test_ready.shape)
