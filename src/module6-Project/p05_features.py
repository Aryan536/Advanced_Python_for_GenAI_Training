# STEP 5: Feature engineering
# Business use case: score every order right after delivery + rating, so the support
# team can reach out (exchange, size help, fix) BEFORE the customer raises a return.
df["Order_Month"] = df["Order_Date"].dt.month
df["Order_Weekday"] = df["Order_Date"].dt.dayofweek
df["Order_Hour"] = df["Order_Date"].dt.hour
df["Is_Weekend"] = (df["Order_Weekday"] >= 5).astype(int)
df["Is_Sale_Month"] = df["Order_Month"].isin([1, 8, 10, 11]).astype(int)
df["High_Discount"] = (df["Discount_Pct"] > 40).astype(int)
df["Slow_Delivery"] = (df["Delivery_Days"] > 5).astype(int)
df["Is_New_Customer"] = (df["Previous_Orders"] == 0).astype(int)
df["Log_Order_Value"] = np.log1p(df["Order_Value"])
df["Is_COD"] = (df["Payment_Method"] == "COD").astype(int)

numeric_features = ["Customer_Age", "City_Tier", "Unit_Price", "Quantity", "Discount_Pct",
                    "Log_Order_Value", "Previous_Orders", "Session_Duration_Min", "Delivery_Days",
                    "Customer_Rating", "Order_Month", "Order_Weekday", "Order_Hour", "Is_Weekend",
                    "Is_Sale_Month", "High_Discount", "Slow_Delivery", "Is_New_Customer", "Is_COD"]
categorical_features = ["Gender", "Category", "Payment_Method", "Device_Type", "Is_Member"]
TARGET = "Returned_Flag"

X = df[numeric_features + categorical_features]
y = df[TARGET]
print("Feature matrix:", X.shape, "| positives:", y.sum())
