
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

ROOT = Path(r"D:\EEGToAlphabets")
DATA = ROOT / "preprocessed_data_version4"
OUT = ROOT / "results_version4" / "zero_value_investigation"
OUT.mkdir(parents=True, exist_ok=True)

EPOCHS_FILE = DATA / "epochs.mat"
META_FILE = DATA / "metadata.csv"

FS = 125
SAMPLES_PER_WINDOW = 8
WINDOWS_PER_TRIAL = 156
TRIALS_TO_CHECK = [405, 2405, 1331, 2056, 811]

meta = pd.read_csv(META_FILE)
required = ["task_start_used", "task_end_used"]
missing = [c for c in required if c not in meta.columns]
if missing:
    raise ValueError(f"Missing metadata columns: {missing}")

print("Metadata rows:", len(meta))
print("Metadata columns:", list(meta.columns))

records = []

with h5py.File(EPOCHS_FILE, "r") as f:
    X = f["X"]
    print("epochs.mat shape:", X.shape)

    if X.shape[0] != SAMPLES_PER_WINDOW:
        raise ValueError("Unexpected first dimension; expected 8 samples/window.")
    if len(meta) != X.shape[2]:
        raise ValueError(
            f"Metadata rows ({len(meta)}) do not match tensor windows ({X.shape[2]}). "
            "Stopping to avoid an incorrect alignment assumption."
        )

    for trial in TRIALS_TO_CHECK:
        first = trial * WINDOWS_PER_TRIAL
        last = first + WINDOWS_PER_TRIAL

        if last > X.shape[2]:
            raise ValueError(f"Trial {trial} is outside the tensor.")

        for w in range(first, last):
            # Source window has shape (8 samples, 15 channels)
            window = np.asarray(X[:, :, w])  # shape: (8 samples, 15 channels)
            zero_by_sample = np.all(window == 0, axis=1)
            last_sample_all_zero = bool(zero_by_sample[-1])
            zero_samples = np.flatnonzero(zero_by_sample).tolist()

            row = meta.iloc[w]
            start = float(row["task_start_used"])
            end = float(row["task_end_used"])

            # Diagnostic estimate only; does not reproduce MATLAB indexing
            estimated_count = round(end * FS) - round(start * FS)

            records.append({
                "trial_index_0_based": trial,
                "tensor_window_index_0_based": w,
                "window_in_trial_0_based": w - first,
                "task_start_sec": start,
                "task_end_sec": end,
                "duration_sec": end - start,
                "estimated_sample_count": estimated_count,
                "zero_samples_all_channels_0_based": ",".join(map(str, zero_samples)),
                "last_sample_all_channels_zero": last_sample_all_zero,
                "window_all_zero": bool(np.all(window == 0)),
            })

df = pd.DataFrame(records)
report = OUT / "zero_window_boundary_diagnostic.csv"
df.to_csv(report, index=False)

print("\nWindows with last sample zero across all channels:")
print(
    df[df["last_sample_all_channels_zero"]][[
        "trial_index_0_based",
        "window_in_trial_0_based",
        "task_start_sec",
        "task_end_sec",
        "duration_sec",
        "estimated_sample_count",
        "zero_samples_all_channels_0_based",
    ]].head(40).to_string(index=False)
)

print("\nCounts by trial:")
print(
    df.groupby("trial_index_0_based")[
        ["last_sample_all_channels_zero", "window_all_zero"]
    ].sum()
)

print("\nSaved report:", report)
