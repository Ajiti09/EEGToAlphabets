
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

ROOT = Path(r"D:\EEGToAlphabets")
PRE = ROOT / "preprocessed_data_version4"

with h5py.File(PRE / "epochs.mat", "r") as f:
    meta = pd.read_csv(
        PRE / "metadata.csv",
        usecols=["participant", "label", "trial", "trial_uid", "window_id"]
    )

    selected = meta[
        (meta["participant"] == "P01")
        & (meta["label"] == "A")
        & (meta["trial"] == 1)
    ]

    if len(selected) != 156:
        raise ValueError(f"Expected 156 windows, found {len(selected)}")

    window_ids = selected["window_id"].to_numpy()
    if not np.array_equal(window_ids, np.arange(1, 157)):
        raise ValueError("Selected windows are not ordered 1–156")

    # Find the contiguous window range in the source epochs tensor.
    first = selected.index.min()
    last = selected.index.max() + 1

    if last - first != 156:
        raise ValueError("Selected metadata rows are not contiguous")

    # epochs.mat X is stored as (samples, channels, windows).
    source = np.asarray(f["X"][:, :, first:last])
    source_trial = np.moveaxis(source, 2, 0).reshape(156 * 8, 15).T

saved = np.load(PRE / "trials_X.npy", mmap_mode="r")
trials_meta = pd.read_csv(PRE / "trials_meta.csv")
match = trials_meta[
    (trials_meta["participant"] == "P01")
    & (trials_meta["label"] == "A")
    & (trials_meta["trial"] == 1)
]

if len(match) != 1:
    raise ValueError(f"Expected one trial metadata row, found {len(match)}")

saved_trial = np.asarray(saved[match.index[0]])

print("Source epochs trial shape:", source_trial.shape)
print("Saved trial shape:", saved_trial.shape)
print("Maximum absolute difference:",
      float(np.max(np.abs(source_trial - saved_trial))))
print("\nChannel medians:")
for i, name in enumerate(
    pd.read_csv(PRE / "channel_names.csv", header=None)[0].astype(str)
):
    print(
        f"{name:4s} source={np.median(source_trial[i]):12.3f}"
        f" saved={np.median(saved_trial[i]):12.3f}"
    )
