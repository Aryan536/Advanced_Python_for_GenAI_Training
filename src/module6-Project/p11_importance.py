# STEP 11: What drives returns? Feature importance + business insights
rf = models["Random Forest"]
importance = (pd.Series(rf.feature_importances_, index=feature_names)
                .sort_values(ascending=False).head(15))
importance.index = importance.index.str.replace("num__", "").str.replace("cat__", "")
importance.sort_values().plot(kind="barh", figsize=(9, 6), color="seagreen",
                              title="Top 15 drivers of returns (Random Forest)")
plt.show()

print("BUSINESS INSIGHTS")
fashion = df.loc[df.Category == "Fashion", "Returned_Flag"].mean()
grocery = df.loc[df.Category == "Grocery", "Returned_Flag"].mean()
print(f"1. Fashion return rate {fashion:.0%} vs Grocery {grocery:.0%} -> add size guides / fit tools.")
print(f"2. COD orders return {p_ret_cod:.0%} vs overall {p_ret:.0%} -> nudge UPI with prepaid offers.")
slow = df.loc[df.Slow_Delivery == 1, 'Returned_Flag'].mean()
fast = df.loc[df.Slow_Delivery == 0, 'Returned_Flag'].mean()
print(f"3. Slow deliveries (>5 days) return {slow:.0%} vs {fast:.0%} -> prioritise tier-3 logistics.")
low = df.loc[df.Customer_Rating <= 2, 'Returned_Flag'].mean()
print(f"4. Ratings of 1-2 stars return {low:.0%} -> trigger instant support call on low ratings.")
