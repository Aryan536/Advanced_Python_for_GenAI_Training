import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# PRACTICAL: A/B test of a new one-click checkout page
# Version A = old page, Version B = new page. Visitors were split randomly.
visitors_a, buyers_a = 12_000, 384      # 3.20% conversion
visitors_b, buyers_b = 12_000, 456      # 3.80% conversion

p_a, p_b = buyers_a / visitors_a, buyers_b / visitors_b
print(f"Conversion A = {p_a:.2%}, Conversion B = {p_b:.2%}, uplift = {(p_b - p_a) / p_a:.1%}")

# Step 1: hypotheses.  H0: p_A = p_B   H1: p_A != p_B   (alpha = 0.05)
# Step 2: two-proportion z-test
p_pool = (buyers_a + buyers_b) / (visitors_a + visitors_b)
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / visitors_a + 1 / visitors_b))
z = (p_b - p_a) / se_pool
p_value = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"z = {z:.2f}, p-value = {p_value:.4f}")

# Step 3: 95% confidence interval for the difference
se_diff = np.sqrt(p_a * (1 - p_a) / visitors_a + p_b * (1 - p_b) / visitors_b)
low, high = (p_b - p_a) - 1.96 * se_diff, (p_b - p_a) + 1.96 * se_diff
print(f"95% CI for (B - A): {low:.2%} to {high:.2%}")

# Step 4: the same answer with a chi-square test
table = [[buyers_a, visitors_a - buyers_a], [buyers_b, visitors_b - buyers_b]]
chi2, p_chi, _, _ = stats.chi2_contingency(table, correction=False)
print(f"Chi-square p-value = {p_chi:.4f} (matches the z-test)")

# Step 5: did the average basket value change? (Welch t-test on simulated baskets)
rng = np.random.default_rng(3)
basket_a = rng.normal(1480, 400, buyers_a)
basket_b = rng.normal(1495, 410, buyers_b)
t, p_t = stats.ttest_ind(basket_b, basket_a, equal_var=False)
print(f"Basket value t-test p-value = {p_t:.3f}")

# Step 6: business decision
if p_value < 0.05 and low > 0:
    extra = (p_b - p_a) * 100_000 * basket_b.mean()
    print("\nDECISION: roll out version B.")
    print(f"Expected extra revenue per 1 lakh visitors ~ Rs {extra:,.0f}")
else:
    print("\nDECISION: keep version A - the difference is not significant.")

plt.bar(["A (old)", "B (new)"], [p_a * 100, p_b * 100], color=["grey", "seagreen"],
        yerr=[1.96 * np.sqrt(p_a * (1 - p_a) / visitors_a) * 100,
              1.96 * np.sqrt(p_b * (1 - p_b) / visitors_b) * 100], capsize=8)
plt.ylabel("Conversion rate (%)")
plt.title("A/B test result with 95% error bars")
plt.show()
