import numpy as np
import pandas as pd
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
from scipy import stats

rng = np.random.default_rng(10)

# 1) 95% confidence interval for the average order value
sample = rng.normal(1450, 380, 200)           # 200 sampled orders
mean, se = sample.mean(), stats.sem(sample)
ci_low, ci_high = stats.t.interval(0.95, df=len(sample) - 1, loc=mean, scale=se)
print(f"Average order value = {mean:.0f}, 95% CI = ({ci_low:.0f}, {ci_high:.0f})")

# 2) One-sample t-test: is the average delivery time different from the promised 3 days?
delivery = rng.normal(3.3, 1.1, 120)
t, p = stats.ttest_1samp(delivery, popmean=3)
print(f"\nOne-sample t-test: t={t:.2f}, p={p:.4f} ->",
      "Reject H0: delivery differs from 3 days" if p < 0.05 else "Fail to reject H0")

# 3) Two-sample (Welch) t-test: do members spend more than non-members?
members = rng.normal(1600, 400, 150)
non_members = rng.normal(1450, 420, 180)
t, p = stats.ttest_ind(members, non_members, equal_var=False)
print(f"Two-sample t-test: t={t:.2f}, p={p:.4f} ->",
      "Members spend differently" if p < 0.05 else "No evidence of a difference")

# 4) Chi-square test of independence: is payment method related to returns?
table = pd.DataFrame({"Returned": [260, 310, 120], "Kept": [740, 1690, 880]},
                     index=["COD", "UPI", "Card"])
chi2, p, dof, expected = stats.chi2_contingency(table)
print(f"\nChi-square: chi2={chi2:.1f}, dof={dof}, p={p:.2e}")
print("Return rate by payment:\n", (table["Returned"] / table.sum(axis=1)).round(3))

# 5) One-way ANOVA: does the average rating differ across devices?
app = rng.normal(4.2, 0.6, 100)
web = rng.normal(4.0, 0.6, 100)
desktop = rng.normal(4.05, 0.6, 100)
f, p = stats.f_oneway(app, web, desktop)
print(f"\nANOVA: F={f:.2f}, p={p:.4f} ->",
      "At least one device differs" if p < 0.05 else "No significant difference")
