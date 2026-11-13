import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(1)
# Order values are typically right-skewed: many small orders, a few large ones
order_values = rng.lognormal(mean=7, sigma=0.6, size=2000)
# Time on site vs order value (positive relationship + noise)
time_on_site = rng.normal(15, 5, 2000).clip(1)
spend = 200 + 60 * time_on_site + rng.normal(0, 250, 2000)

print("Skewness :", round(stats.skew(order_values), 2), "(> 0 means right-skewed)")
print("Kurtosis :", round(stats.kurtosis(order_values), 2), "(> 0 means heavier tails than normal)")
print("Mean vs median:", round(order_values.mean()), "vs", round(np.median(order_values)))
print("Correlation (time on site vs spend):", round(np.corrcoef(time_on_site, spend)[0, 1], 2))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(order_values, bins=50, color="steelblue", edgecolor="white")
axes[0].axvline(order_values.mean(), color="red", label="Mean")
axes[0].axvline(np.median(order_values), color="green", label="Median")
axes[0].set_title("Histogram: order value (right-skewed)")
axes[0].legend()

axes[1].boxplot(order_values)
axes[1].set_title("Box plot: dots beyond whiskers are outliers")

axes[2].scatter(time_on_site, spend, s=5, alpha=0.4)
axes[2].set_xlabel("Minutes on site")
axes[2].set_ylabel("Spend (Rs)")
axes[2].set_title("Scatter: time on site vs spend")
plt.tight_layout()
plt.show()
