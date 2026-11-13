import numpy as np
import pandas as pd
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Real-world example: RFM customer segmentation
# Recency = days since last order, Frequency = orders per year, Monetary = yearly spend (Rs)
rng = np.random.default_rng(7)
groups = [  # (n, recency, frequency, monetary)
    (300, 10, 25, 60000),    # loyal high spenders
    (500, 40, 8, 15000),     # regular customers
    (400, 200, 2, 3000),     # lapsed / at-risk customers
    (200, 15, 4, 4000),      # new customers
]
parts = []
for n, r, f, m in groups:
    parts.append(pd.DataFrame({
        "recency": rng.normal(r, r * 0.2, n).clip(1),
        "frequency": rng.normal(f, f * 0.2, n).clip(1),
        "monetary": rng.normal(m, m * 0.2, n).clip(200),
    }))
rfm = pd.concat(parts, ignore_index=True)

X = StandardScaler().fit_transform(rfm)       # K-Means uses distances -> always scale

# Elbow + silhouette to choose K
inertias, sils = [], []
K = range(2, 9)
for k in K:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(K, inertias, "o-"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("K")
ax[1].plot(K, sils, "o-", color="green"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("K")
plt.show()
print("Best K by silhouette:", list(K)[int(np.argmax(sils))])

km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
rfm["segment"] = km.labels_
profile = rfm.groupby("segment").agg(customers=("recency", "size"),
                                     recency=("recency", "mean"),
                                     frequency=("frequency", "mean"),
                                     monetary=("monetary", "mean")).round(0)
print(profile.sort_values("monetary", ascending=False))
