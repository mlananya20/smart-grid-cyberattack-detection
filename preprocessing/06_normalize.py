import pandas as pd
import numpy as np
import json

train_df = pd.read_csv("../data/train.csv")
val_df = pd.read_csv("../data/val.csv")
test_df = pd.read_csv("../data/test.csv")

date_cols = train_df.columns[2:]

# Compute mean/std from TRAINING data only (flatten all values across all customers/days)
train_values = train_df[date_cols].values.flatten()
mean = train_values.mean()
std = train_values.std()

print(f"Train mean: {mean:.4f}, Train std: {std:.4f}")

# Save these stats so they can be reused later (e.g. for a real deployment or re-run)
with open("../data/norm_stats.json", "w") as f:
    json.dump({"mean": float(mean), "std": float(std)}, f)

# Apply the SAME mean/std to all three splits
eps = 1e-8
for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    split_df[date_cols] = (split_df[date_cols] - mean) / (std + eps)
    split_df.to_csv(f"../data/{name}_normalized.csv", index=False)
    print(f"Saved {name}_normalized.csv")

print("\nSanity check - normalized train describe (first date col):")
print(train_df[date_cols[0]].describe())