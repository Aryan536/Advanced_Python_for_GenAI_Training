# STEP 1: Load and understand the data
df = pd.read_excel(FILE, sheet_name="Orders")
print("Shape:", df.shape)
print(df.head())
print()
df.info()
print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])
print("\nTarget distribution:\n", df["Returned"].value_counts(normalize=True).round(3))
print("\nNumeric summary:\n", df.describe(include="number").round(1).T)
