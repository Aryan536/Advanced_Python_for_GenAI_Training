import time
import numpy as np
import pandas as pd
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

# 1) Vectorisation: NumPy is much faster than a Python loop
prices = np.random.default_rng(0).uniform(100, 5000, 1_000_000)

start = time.perf_counter()
loop_result = [p * 0.9 for p in prices]
loop_time = time.perf_counter() - start

start = time.perf_counter()
vector_result = prices * 0.9
vector_time = time.perf_counter() - start
print(f"Loop: {loop_time:.3f} s | NumPy: {vector_time:.4f} s | "
      f"NumPy is ~{loop_time / vector_time:.0f}x faster")

# 2) Pandas: a small order table
rng = np.random.default_rng(42)
n = 1000
orders = pd.DataFrame({
    "order_id": [f"ORD{i}" for i in range(1, n + 1)],
    "category": rng.choice(["Electronics", "Fashion", "Books", "Grocery"], n),
    "city": rng.choice(["Bengaluru", "Mumbai", "Delhi"], n),
    "price": rng.integers(200, 5000, n),
    "quantity": rng.integers(1, 4, n),
})
orders["revenue"] = orders["price"] * orders["quantity"]       # vectorised new column
print(orders.head())

# Filter, group, aggregate
high_value = orders[orders["revenue"] > 8000]
print("\nHigh-value orders:", len(high_value))

summary = (orders.groupby("category")
                 .agg(orders=("order_id", "count"),
                      total_revenue=("revenue", "sum"),
                      avg_revenue=("revenue", "mean"))
                 .sort_values("total_revenue", ascending=False)
                 .round(0))
print("\nRevenue by category:\n", summary)

# Pivot table: city x category revenue
pivot = orders.pivot_table(index="city", columns="category", values="revenue", aggfunc="sum")
print("\nCity x Category revenue:\n", pivot)

# Merge with a targets table (like a SQL JOIN)
targets = pd.DataFrame({"category": ["Electronics", "Fashion", "Books", "Grocery"],
                        "target": [2_000_000, 1_500_000, 900_000, 800_000]})
result = summary.reset_index().merge(targets, on="category")
result["achieved_pct"] = (result["total_revenue"] / result["target"] * 100).round(1)
print("\nTarget achievement:\n", result[["category", "total_revenue", "target", "achieved_pct"]])
