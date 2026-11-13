import numpy as np
import pandas as pd
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
from scipy import stats

# Real-world example: order values (Rs) from one day at a small online store.
# The last order is a bulk corporate purchase - an outlier.
order_values = np.array([450, 520, 610, 480, 700, 650, 530, 590, 610, 560,
                         620, 610, 480, 540, 25000])

print("Mean   :", round(order_values.mean(), 2))
print("Median :", np.median(order_values))
print("Mode   :", stats.mode(order_values, keepdims=False).mode)

# Spread
print("\nRange             :", order_values.max() - order_values.min())
print("Variance (sample) :", round(order_values.var(ddof=1), 2))
print("Std dev  (sample) :", round(order_values.std(ddof=1), 2))

q1, q3 = np.percentile(order_values, [25, 75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"Q1={q1}, Q3={q3}, IQR={iqr}, outlier fences=({lower}, {upper})")
print("Outliers (IQR rule):", order_values[(order_values < lower) | (order_values > upper)])

# Z-score: how many standard deviations a value is from the mean
z = (order_values - order_values.mean()) / order_values.std(ddof=1)
print("Z-score of the Rs 25,000 order:", round(z[-1], 2))

# Effect of removing the outlier
clean = order_values[order_values <= upper]
print(f"\nWithout outlier -> mean {clean.mean():.2f}, median {np.median(clean)}")

# Pandas gives everything in one line
print("\n", pd.Series(order_values, name="order_value").describe().round(2))
