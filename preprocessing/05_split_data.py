import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/data_imputed.csv")

# First split: 70% train, 30% temp (stratified so theft ratio stays consistent)
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["FLAG"], random_state=42
)

# Second split: split temp into 15% val, 15% test
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["FLAG"], random_state=42
)

print("Train:", len(train_df), "customers |", train_df["FLAG"].value_counts().to_dict())
print("Val:  ", len(val_df), "customers |", val_df["FLAG"].value_counts().to_dict())
print("Test: ", len(test_df), "customers |", test_df["FLAG"].value_counts().to_dict())

train_df.to_csv("../data/train.csv", index=False)
val_df.to_csv("../data/val.csv", index=False)
test_df.to_csv("../data/test.csv", index=False)
print("\nSaved train.csv, val.csv, test.csv")