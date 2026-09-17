import pandas as pd

df = pd.read_csv("../data/data_sorted.csv")
date_cols = df.columns[2:]

missing_per_row = df[date_cols].isnull().sum(axis=1)
missing_pct = missing_per_row / len(date_cols)

df["missing_pct"] = missing_pct

print("Missing % by FLAG (mean):")
print(df.groupby("FLAG")["missing_pct"].mean())

print("\nCustomers with >50% missing, by FLAG:")
print(df[df["missing_pct"] > 0.5]["FLAG"].value_counts())

print("\nCustomers with <=50% missing, by FLAG:")
print(df[df["missing_pct"] <= 0.5]["FLAG"].value_counts())