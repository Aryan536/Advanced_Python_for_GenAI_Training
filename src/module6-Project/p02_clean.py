# STEP 2: Clean the data
df = df.drop_duplicates(subset="Order_ID").copy()
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Fill missing values with sensible, explainable rules
df["Customer_Age"] = df["Customer_Age"].fillna(df["Customer_Age"].median())
df["Device_Type"] = df["Device_Type"].fillna("Unknown")
df["Customer_Rating"] = df["Customer_Rating"].fillna(
    df.groupby("Category")["Customer_Rating"].transform("median"))   # median of same category
df["Review_Text"] = df["Review_Text"].fillna("No review")

# Target to 0/1
df["Returned_Flag"] = (df["Returned"] == "Yes").astype(int)

# Sanity checks
assert df["Order_Value"].min() > 0
assert df[["Customer_Age", "Device_Type", "Customer_Rating"]].isna().sum().sum() == 0
print("Rows after cleaning:", len(df))
print("Remaining missing values:", df.isna().sum().sum())
print(f"Overall return rate: {df['Returned_Flag'].mean():.1%}")
