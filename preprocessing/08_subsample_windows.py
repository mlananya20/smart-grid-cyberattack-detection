import numpy as np

MAX_WINDOWS_PER_SPLIT = {"train": 60000, "val": 12000, "test": 12000}

for split in ["train", "val", "test"]:
    windows = np.load(f"../data/{split}_windows.npy")
    labels = np.load(f"../data/{split}_labels.npy")
    customer_ids = np.load(f"../data/{split}_customer_ids.npy")

    n_total = len(windows)
    n_keep = min(MAX_WINDOWS_PER_SPLIT[split], n_total)

    # Stratified subsample: keep the same Normal/Theft ratio
    rng = np.random.default_rng(42)
    normal_idx = np.where(labels == 0)[0]
    theft_idx = np.where(labels == 1)[0]

    n_theft_keep = int(n_keep * (len(theft_idx) / n_total))
    n_normal_keep = n_keep - n_theft_keep

    keep_normal = rng.choice(normal_idx, size=min(n_normal_keep, len(normal_idx)), replace=False)
    keep_theft = rng.choice(theft_idx, size=min(n_theft_keep, len(theft_idx)), replace=False)

    keep_idx = np.concatenate([keep_normal, keep_theft])
    rng.shuffle(keep_idx)

    np.save(f"../data/{split}_windows_sub.npy", windows[keep_idx])
    np.save(f"../data/{split}_labels_sub.npy", labels[keep_idx])
    np.save(f"../data/{split}_customer_ids_sub.npy", customer_ids[keep_idx])

    print(f"{split}: kept {len(keep_idx)} windows "
          f"(Normal={np.sum(labels[keep_idx]==0)}, Theft={np.sum(labels[keep_idx]==1)})")