import pandas as pd
import numpy as np

WINDOW_SIZE = 14
STEP_SIZE = 7  # 50% overlap between windows

def make_windows(df, split_name):
    date_cols = df.columns[2:]
    values = df[date_cols].values  # shape: (num_customers, num_days)
    flags = df["FLAG"].values
    cons_no = df["CONS_NO"].values

    windows = []
    labels = []
    customer_ids = []

    for i in range(len(df)):
        series = values[i]
        flag = flags[i]
        cid = cons_no[i]

        for start in range(0, len(series) - WINDOW_SIZE + 1, STEP_SIZE):
            window = series[start:start + WINDOW_SIZE]
            # Real label: 0 = Normal, 1 = Theft (from FLAG)
            windows.append(window)
            labels.append(flag)
            customer_ids.append(cid)

    windows = np.array(windows)
    labels = np.array(labels)
    customer_ids = np.array(customer_ids)

    print(f"{split_name}: {windows.shape[0]} windows created, shape {windows.shape}")
    print(f"  Label distribution: Normal={np.sum(labels==0)}, Theft={np.sum(labels==1)}")

    np.save(f"../data/{split_name}_windows.npy", windows)
    np.save(f"../data/{split_name}_labels.npy", labels)
    np.save(f"../data/{split_name}_customer_ids.npy", customer_ids)

train_df = pd.read_csv("../data/train_normalized.csv")
val_df = pd.read_csv("../data/val_normalized.csv")
test_df = pd.read_csv("../data/test_normalized.csv")

make_windows(train_df, "train")
make_windows(val_df, "val")
make_windows(test_df, "test")