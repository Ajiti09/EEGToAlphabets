
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import numpy as np
import pandas as pd

ROOT = Path(r"D:\EEGToAlphabets")
PRE = ROOT / "preprocessed_data_version4"
ZIP_PATH = ROOT / "raw data_version4" / "TEST_SUBJECT_1.zip"

FS = 125
start_s, end_s = 6.9851, 16.9691

with ZipFile(ZIP_PATH) as zf:
    candidates = [
        n for n in zf.namelist()
        if Path(n).name.startswith("OpenBCI-RAW-")
        and "TEST_SUBJECT_1_A" in n
    ]
    if len(candidates) != 1:
        raise ValueError(f"Unexpected raw file candidates: {candidates}")

    with zf.open(candidates[0]) as f:
        raw = pd.read_csv(
            BytesIO(f.read()), comment="%", skipinitialspace=True
        )

raw.columns = raw.columns.astype(str).str.strip()

raw_names = [f"EXG Channel {i}" for i in range(16)]
raw_values = raw[raw_names].apply(
    pd.to_numeric, errors="coerce"
).to_numpy(dtype=float)

# This is the same timing assumption used in the earlier comparison.
raw_segment = raw_values[round(start_s * FS):round(end_s * FS)]

X = np.load(PRE / "trials_X.npy", mmap_mode="r")
trial_meta = pd.read_csv(PRE / "trials_meta.csv")
channels = pd.read_csv(
    PRE / "channel_names.csv", header=None
).iloc[:, 0].astype(str).str.strip().tolist()

selected = trial_meta[
    (trial_meta["participant"] == "P01")
    & (trial_meta["label"] == "A")
    & (trial_meta["trial"] == 1)
]
if len(selected) != 1:
    raise ValueError(f"Expected one trial row, found {len(selected)}")

processed = np.asarray(X[selected.index[0]], dtype=float)

# Match retained channels by the configured raw channel order.
configured = [
    "Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2",
    "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4"
]
raw_index = {name: i for i, name in enumerate(configured)}

rows = []
for i, name in enumerate(channels):
    rx = raw_segment[:, raw_index[name]]
    px = processed[i]

    rows.append({
        "channel": name,
        "raw_segment_median": np.nanmedian(rx),
        "processed_trial_median": np.nanmedian(px),
        "raw_segment_mean": np.nanmean(rx),
        "processed_trial_mean": np.nanmean(px),
        "raw_segment_std": np.nanstd(rx),
        "processed_trial_std": np.nanstd(px),
    })

report = pd.DataFrame(rows)
out = ROOT / "results_version4" / "processed_qc"
out.mkdir(parents=True, exist_ok=True)
report.to_csv(out / "raw_processed_median_comparison.csv", index=False)

print(report.to_string(index=False))
print("\nSaved:", out / "raw_processed_median_comparison.csv")
print("Note: raw and processed intervals still require exact alignment verification.")
