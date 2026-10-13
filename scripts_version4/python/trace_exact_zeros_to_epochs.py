
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(r"D:\EEGToAlphabets")
DATA_DIR = ROOT / "preprocessed_data_version4"
RESULT_DIR = ROOT / "results_version4" / "zero_value_investigation"

EPOCHS_FILE = DATA_DIR / "epochs.mat"
TRIALS_FILE = DATA_DIR / "trials_X.npy"

# Trial 405 is zero-based index 405 (the 406th trial).
TRIAL_INDICES = [405, 2405, 1331, 2056, 811]

# Confirm source files exist.
for path in [EPOCHS_FILE, TRIALS_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

X = np.load(TRIALS_FILE, mmap_mode="r")

with h5py.File(EPOCHS_FILE, "r") as f:
    source_shape = f["X"].shape
    print("epochs.mat X shape:", source_shape)
    print("trials_X.npy shape:", X.shape)

    if source_shape != (8, 15, 390000):
        raise ValueError(
            "Unexpected source shape. Stop and verify before continuing."
        )

    rows = []

    for trial in TRIAL_INDICES:
        # Each trial contains 156 windows, each with 8 samples.
        window_start = trial * 156
        window_end = window_start + 156

        # Source: (8 samples, 15 channels, windows).
        source = f["X"][:, :, window_start:window_end]
        source_trial = np.moveaxis(source, 2, 0).reshape(
            1248, 15
        ).T

        # Reconstructed file: (trials, channels, samples).
        saved_trial = X[trial]

        # Compare zeros and numerical equality.
        source_zeros = source_trial == 0
        saved_zeros = saved_trial == 0

        exact_match = np.array_equal(
            source_trial, saved_trial
        )
        zero_mask_match = np.array_equal(
            source_zeros, saved_zeros
        )

        rows.append({
            "trial_index_zero_based": trial,
            "source_zero_count": int(source_zeros.sum()),
            "saved_zero_count": int(saved_zeros.sum()),
            "source_and_saved_zero_positions_match": zero_mask_match,
            "all_values_match_exactly": exact_match,
            "source_only_zero_count": int(
                (source_zeros & ~saved_zeros).sum()
            ),
            "saved_only_zero_count": int(
                (saved_zeros & ~source_zeros).sum()
            ),
            "max_absolute_difference": float(
                np.max(np.abs(
                    source_trial.astype(np.float64)
                    - saved_trial.astype(np.float64)
                ))
            ),
        })

        print(
            f"Trial {trial}: source zeros={source_zeros.sum()}, "
            f"saved zeros={saved_zeros.sum()}, "
            f"zero masks match={zero_mask_match}, "
            f"all values match={exact_match}"
        )

    pd.DataFrame(rows).to_csv(
        RESULT_DIR / "zero_source_trace.csv",
        index=False,
    )

print("\nSaved report:", RESULT_DIR / "zero_source_trace.csv")
