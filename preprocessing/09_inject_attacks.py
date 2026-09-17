import numpy as np

rng = np.random.default_rng(42)

def inject_fdi(window, intensity=0.5):
    """False Data Injection: scale down a contiguous chunk of values
    to simulate falsified low readings (a common FDI pattern)."""
    w = window.copy()
    length = rng.integers(3, 8)  # affected days
    start = rng.integers(0, len(w) - length + 1)
    w[start:start+length] = w[start:start+length] * (1 - intensity)
    return w

def inject_tampering(window, intensity=2.0):
    """Meter Tampering: abrupt, irregular spikes/drops at random
    scattered points (not contiguous - meant to look erratic)."""
    w = window.copy()
    n_points = rng.integers(2, 5)
    idxs = rng.choice(len(w), size=n_points, replace=False)
    signs = rng.choice([-1, 1], size=n_points)
    w[idxs] = w[idxs] + signs * intensity * np.abs(w[idxs]).mean()
    return w

def inject_replay(window, all_windows):
    """Replay: splice in a segment from a DIFFERENT time period
    (another random window) into part of this one."""
    w = window.copy()
    length = rng.integers(4, 9)
    start = rng.integers(0, len(w) - length + 1)
    # pick a random other window's segment to replay in
    donor = all_windows[rng.integers(0, len(all_windows))]
    donor_start = rng.integers(0, len(donor) - length + 1)
    w[start:start+length] = donor[donor_start:donor_start+length]
    return w

def build_attack_dataset(windows, labels, attack_fraction=0.15):
    """
    Takes Normal windows and converts a fraction into FDI/Tampering/Replay.
    Label scheme: 0=Normal, 1=Theft(real), 2=FDI, 3=Tampering, 4=Replay
    """
    windows = windows.copy()
    labels = labels.copy().astype(int)

    normal_idx = np.where(labels == 0)[0]
    n_per_attack = int(len(normal_idx) * attack_fraction / 3)

    rng.shuffle(normal_idx)
    fdi_idx = normal_idx[:n_per_attack]
    tamper_idx = normal_idx[n_per_attack:2*n_per_attack]
    replay_idx = normal_idx[2*n_per_attack:3*n_per_attack]

    for i in fdi_idx:
        windows[i] = inject_fdi(windows[i])
        labels[i] = 2
    for i in tamper_idx:
        windows[i] = inject_tampering(windows[i])
        labels[i] = 3
    for i in replay_idx:
        windows[i] = inject_replay(windows[i], windows)
        labels[i] = 4

    return windows, labels

for split in ["train", "val", "test"]:
    windows = np.load(f"../data/{split}_windows_sub.npy")
    labels = np.load(f"../data/{split}_labels_sub.npy")

    new_windows, new_labels = build_attack_dataset(windows, labels)

    np.save(f"../data/{split}_windows_final.npy", new_windows)
    np.save(f"../data/{split}_labels_final.npy", new_labels)

    unique, counts = np.unique(new_labels, return_counts=True)
    label_names = {0: "Normal", 1: "Theft", 2: "FDI", 3: "Tampering", 4: "Replay"}
    print(f"{split}:", {label_names[u]: c for u, c in zip(unique, counts)})