# STEP 13 (bonus): Customer segmentation with K-Means (RFM)
from sklearn.cluster import KMeans

snapshot = df["Order_Date"].max() + pd.Timedelta(days=1)
rfm = df.groupby("Customer_ID").agg(
    Recency=("Order_Date", lambda d: (snapshot - d.max()).days),
    Frequency=("Order_ID", "count"),
    Monetary=("Order_Value", "sum"),
    Return_Rate=("Returned_Flag", "mean"))

rfm_scaled = StandardScaler().fit_transform(np.log1p(rfm[["Recency", "Frequency", "Monetary"]]))
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
rfm["Segment"] = kmeans.fit_predict(rfm_scaled)

profile = rfm.groupby("Segment").agg(Customers=("Recency", "size"), Recency=("Recency", "mean"),
                                     Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean"),
                                     Return_Rate=("Return_Rate", "mean")).round(2)
profile = profile.sort_values("Monetary", ascending=False)
profile["Label"] = ["Champions", "Loyal", "Potential", "At risk / one-time"]
print(profile)
