# STEP 3: Exploratory data analysis (EDA)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

rate = df.groupby("Category")["Returned_Flag"].mean().sort_values(ascending=False) * 100
rate.plot(kind="bar", ax=axes[0, 0], color="indianred")
axes[0, 0].set_title("Return rate by category (%)"); axes[0, 0].tick_params(axis="x", rotation=45)

pay = df.groupby("Payment_Method")["Returned_Flag"].mean().sort_values(ascending=False) * 100
pay.plot(kind="bar", ax=axes[0, 1], color="darkorange")
axes[0, 1].set_title("Return rate by payment method (%)"); axes[0, 1].tick_params(axis="x", rotation=45)

dd = df.groupby("Delivery_Days")["Returned_Flag"].mean() * 100
dd.plot(marker="o", ax=axes[0, 2], color="purple")
axes[0, 2].set_title("Return rate vs delivery days (%)")

rr = df.groupby("Customer_Rating")["Returned_Flag"].mean() * 100
rr.plot(kind="bar", ax=axes[1, 0], color="teal")
axes[1, 0].set_title("Return rate by customer rating (%)")

sns.histplot(np.log10(df["Order_Value"]), bins=50, ax=axes[1, 1], color="steelblue")
axes[1, 1].set_title("Order value (log10 scale)")

num_cols = ["Customer_Age", "Unit_Price", "Quantity", "Discount_Pct", "Order_Value",
            "Previous_Orders", "Session_Duration_Min", "Delivery_Days", "Customer_Rating",
            "Returned_Flag"]
sns.heatmap(df[num_cols].corr(), cmap="coolwarm", center=0, ax=axes[1, 2], annot=False)
axes[1, 2].set_title("Correlation heatmap")
plt.tight_layout()
plt.show()

# Monthly revenue trend
monthly = df.set_index("Order_Date").resample("MS")["Order_Value"].sum() / 1e5
monthly.plot(marker="o", figsize=(12, 4), title="Monthly revenue (Rs lakh)")
plt.show()
city_revenue = df.groupby("City")["Order_Value"].sum() / 1e5
print("Top 5 cities by revenue (Rs lakh):\n", city_revenue.nlargest(5).round(1))
