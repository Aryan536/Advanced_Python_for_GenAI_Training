# STEP 12: Save the best pipeline and score new orders
import joblib
from sklearn.pipeline import Pipeline

final_model = Pipeline([("prep", preprocessor), ("model", models[best_ml])])
final_model.fit(X_train, y_train)                 # refit the whole pipeline on raw features
joblib.dump(final_model, "return_risk_model.joblib")
print("Model saved: return_risk_model.joblib")

def add_features(orders):
    o = orders.copy()
    o["Order_Date"] = pd.to_datetime(o["Order_Date"])
    o["Order_Month"] = o["Order_Date"].dt.month
    o["Order_Weekday"] = o["Order_Date"].dt.dayofweek
    o["Order_Hour"] = o["Order_Date"].dt.hour
    o["Is_Weekend"] = (o["Order_Weekday"] >= 5).astype(int)
    o["Is_Sale_Month"] = o["Order_Month"].isin([1, 8, 10, 11]).astype(int)
    o["High_Discount"] = (o["Discount_Pct"] > 40).astype(int)
    o["Slow_Delivery"] = (o["Delivery_Days"] > 5).astype(int)
    o["Is_New_Customer"] = (o["Previous_Orders"] == 0).astype(int)
    o["Order_Value"] = o["Unit_Price"] * o["Quantity"] * (1 - o["Discount_Pct"] / 100)
    o["Log_Order_Value"] = np.log1p(o["Order_Value"])
    o["Is_COD"] = (o["Payment_Method"] == "COD").astype(int)
    return o[numeric_features + categorical_features]

new_orders = pd.DataFrame([
    {"Order_Date": "2026-01-10 21:15", "Customer_Age": 22, "Gender": "Female", "City_Tier": 3,
     "Category": "Fashion", "Unit_Price": 1499, "Quantity": 2, "Discount_Pct": 45,
     "Payment_Method": "COD", "Device_Type": "Mobile App", "Is_Member": "No", "Previous_Orders": 0,
     "Session_Duration_Min": 6.5, "Delivery_Days": 8, "Customer_Rating": 2},
    {"Order_Date": "2026-01-12 10:05", "Customer_Age": 38, "Gender": "Male", "City_Tier": 1,
     "Category": "Books", "Unit_Price": 549, "Quantity": 1, "Discount_Pct": 10,
     "Payment_Method": "UPI", "Device_Type": "Desktop", "Is_Member": "Yes", "Previous_Orders": 14,
     "Session_Duration_Min": 18.0, "Delivery_Days": 2, "Customer_Rating": 5},
])
loaded = joblib.load("return_risk_model.joblib")
risk = loaded.predict_proba(add_features(new_orders))[:, 1]
for (_, o), r in zip(new_orders.iterrows(), risk):
    action = "CALL CUSTOMER" if r >= best_th else "no action"
    print(f"{o['Category']:8} | {o['Payment_Method']:4} | rating {o['Customer_Rating']} "
          f"-> return risk {r:.0%} -> {action}")
