import pandas as pd

# Load the dataset
df = pd.read_csv("../data/data.csv")

# Basic info
print("Shape (rows, columns):", df.shape)
print("\nColumn names (first 5):", df.columns[:5].tolist())
print("Column names (last 5):", df.columns[-5:].tolist())
print("\nFirst 3 rows:")
print(df.head(3))

# Check the label column (should be named FLAG)
print("\nLabel distribution:")
print(df["FLAG"].value_counts())

# Check for missing values
print("\nTotal missing values:", df.isnull().sum().sum())
print("Rows with at least one missing value:", df.isnull().any(axis=1).sum())