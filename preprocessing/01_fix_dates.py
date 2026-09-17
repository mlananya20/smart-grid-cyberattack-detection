import pandas as pd

df = pd.read_csv("../data/data.csv")

# Separate ID/label columns from date columns
id_cols = ["CONS_NO", "FLAG"]
date_cols = [c for c in df.columns if c not in id_cols]

# Parse and sort date columns chronologically
sorted_dates = sorted(date_cols, key=lambda d: pd.to_datetime(d))

# Rebuild dataframe in correct order
df = df[id_cols + sorted_dates]

print("First 5 dates after sorting:", sorted_dates[:5])
print("Last 5 dates after sorting:", sorted_dates[-5:])

# Save the corrected version so we don't redo this every time
df.to_csv("../data/data_sorted.csv", index=False)
print("\nSaved: data_sorted.csv")