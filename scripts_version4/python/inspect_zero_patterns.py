
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

ROOT = Path(r"D:\EEGToAlphabets")
EPOCHS_FILE = ROOT / "preprocessed_data_version4" / "epochs.mat"
OUT_DIR = ROOT / "results_version4" / "zero_value_investigation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRIALS = [405, 2405, 1331, 2056, 811]
WINDOW_SAMPLES = 8
SAMPLES_PER_TRIAL = 1248
WINDOWS_PER_TRIAL = 156
rows = []

def runs_of_true(mask):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    return [(int(g[0]), int(g[-1]), len(g)) for g in groups]

with h5py.File(EPOCHS_FILE, "r") as f:
    X = f["X"]
    print("Tensor shape:", X.shape)

    for trial in TRIALS:
        w0 = trial * WINDOWS_PER_TRIAL
        w1 = w0 + WINDOWS_PER_TRIAL

        # Convert this trial to (channels, time samples)
        block = np.asarray(X[:, :, w0:w1])
        trial_data = np.moveaxis(block, 2, 0).reshape(
            SAMPLES_PER_TRIAL, 15
        ).T

        zero = (trial_data == 0)
        any_zero = zero.any(axis=0)
        all_zero = zero.all(axis=0)

        # Report consecutive time positions containing any zeros
        any_runs = runs_of_true(any_zero)
        all_runs = runs_of_true(all_zero)

        print(f"\nTrial {trial}:")
        print("  Exact zero values:", int(zero.sum()))
        print("  Time positions with any zero:", int(any_zero.sum()))
        print("  Time positions zero in ALL 15 channels:", int(all_zero.sum()))
        print("  First/last affected sample:",
              int(np.flatnonzero(any_zero)[0]) if any_zero.any() else None,
              int(np.flatnonzero(any_zero)[-1]) if any_zero.any() else None)
        print("  Consecutive runs (any channel):", any_runs[:15])
        print("  Consecutive runs (all channels):", all_runs[:15])

        for s in np.flatnonzero(any_zero):
            rows.append({
                "trial_index_0_based": trial,
                "sample_index_0_based": int(s),
                "window_index_0_based": int(s // WINDOW_SAMPLES),
                "sample_within_8_sample_window": int(s % WINDOW_SAMPLES),
                "zero_channel_count": int(zero[:, s].sum()),
                "all_15_channels_zero": bool(all_zero[s]),
                "at_trial_start": bool(s < WINDOW_SAMPLES),
                "at_trial_end": bool(s >= SAMPLES_PER_TRIAL - WINDOW_SAMPLES),
            })

pd.DataFrame(rows).to_csv(OUT_DIR / "zero_positions_in_epochs.csv", index=False)
print("\nSaved:", OUT_DIR / "zero_positions_in_epochs.csv")
