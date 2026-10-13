
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(r"D:\EEGToAlphabets")
pre = root / "preprocessed_data_version4"

X = np.load(pre / "trials_X.npy", mmap_mode="r")
meta = pd.read_csv(pre / "trials_meta.csv")
channels = pd.read_csv(
    pre / "channel_names.csv", header=None
).iloc[:, 0].astype(str).str.strip().tolist()

# Inspect P01, A, trial 1.
m = meta[
    (meta["participant"] == "P01")
    & (meta["label"] == "A")
    & (meta["trial"] == 1)
]
if len(m) != 1:
    raise ValueError(f"Expected one trial metadata row; found {len(m)}")

idx = m.index[0]
trial = np.asarray(X[idx], dtype=np.float64)

rows = []
for i, channel in enumerate(channels):
    x = trial[i]
    rows.append({
        "channel": channel,
        "median": np.median(x),
        "mean": np.mean(x),
        "minimum": np.min(x),
        "maximum": np.max(x),
        "std": np.std(x),
        "finite": bool(np.isfinite(x).all()),
    })

report = pd.DataFrame(rows)
out = root / "results_version4" / "processed_qc"
out.mkdir(parents=True, exist_ok=True)
report.to_csv(out / "trial1_channel_statistics.csv", index=False)

print("Trial index:", idx)
print(report.to_string(index=False))
print("\nSaved:", out / "trial1_channel_statistics.csv")
