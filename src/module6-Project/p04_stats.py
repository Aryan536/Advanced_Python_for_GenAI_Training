# STEP 4: Statistical tests - are the patterns real or just noise?
# 1) Chi-square: is Category related to Returned?
ct = pd.crosstab(df["Category"], df["Returned_Flag"])
chi2, p, dof, _ = stats.chi2_contingency(ct)
print(f"Category vs Returned       : chi2={chi2:.1f}, p={p:.2e}")

# 2) Chi-square: is Payment_Method related to Returned?
ct = pd.crosstab(df["Payment_Method"], df["Returned_Flag"])
chi2, p, dof, _ = stats.chi2_contingency(ct)
print(f"Payment method vs Returned : chi2={chi2:.1f}, p={p:.2e}")

# 3) Welch t-test: do returned orders have longer delivery times?
ret = df.loc[df["Returned_Flag"] == 1, "Delivery_Days"]
kept = df.loc[df["Returned_Flag"] == 0, "Delivery_Days"]
t, p = stats.ttest_ind(ret, kept, equal_var=False)
print(f"Delivery days (returned {ret.mean():.2f} vs kept {kept.mean():.2f}): t={t:.2f}, p={p:.2e}")

# 4) Bayes-style conditional probabilities
p_ret = df["Returned_Flag"].mean()
p_ret_cod = df.loc[df["Payment_Method"] == "COD", "Returned_Flag"].mean()
fashion_low = (df["Category"] == "Fashion") & (df["Customer_Rating"] <= 2)
p_ret_fashion_low = df.loc[fashion_low, "Returned_Flag"].mean()
print(f"\nP(return)                         = {p_ret:.3f}")
print(f"P(return | COD)                   = {p_ret_cod:.3f}")
print(f"P(return | Fashion & rating <= 2) = {p_ret_fashion_low:.3f}")

# 5) 95% confidence interval for the overall return rate
n = len(df)
se = np.sqrt(p_ret * (1 - p_ret) / n)
print(f"95% CI for return rate: {p_ret - 1.96 * se:.3f} to {p_ret + 1.96 * se:.3f}")
