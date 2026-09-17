import pandas as pd

df = pd.read_csv("../data/data_sorted.csv")
date_cols = df.columns[2:]

# Drop customers missing more than 50% of days
missing_pct = df[date_cols].isnull().sum(axis=1) / len(date_cols)
df = df[missing_pct <= 0.5].reset_index(drop=True)
print("Remaining customers after dropping high-missing:", len(df))
print("Label distribution after drop:")
print(df["FLAG"].value_counts())

# Forward-fill each customer's row (a meter reading carries forward
# if a day is missing), then back-fill any remaining leading gaps
df[date_cols] = df[date_cols].ffill(axis=1).bfill(axis=1)

# Confirm no missing values remain
print("\nRemaining missing values:", df[date_cols].isnull().sum().sum())

df.to_csv("../data/data_imputed.csv", index=False)
print("Saved: data_imputed.csv")