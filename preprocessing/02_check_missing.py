import pandas as pd

df = pd.read_csv("../data/data_sorted.csv")
date_cols = df.columns[2:]

# How many missing values per row, distribution
missing_per_row = df[date_cols].isnull().sum(axis=1)
print("Missing values per customer - min/median/max:")
print(missing_per_row.min(), missing_per_row.median(), missing_per_row.max())

# Are missing values scattered or in big chunks (e.g. whole rows blank at start)?
print("\nCustomers missing >50% of days:", (missing_per_row > len(date_cols)*0.5).sum())
print("Customers missing <5% of days:", (missing_per_row < len(date_cols)*0.05).sum())

# Check the very first columns (2014/1/1 area) - your earlier output showed NaN there
print("\nMissing count for first 10 dates:")
print(df[date_cols[:10]].isnull().sum())