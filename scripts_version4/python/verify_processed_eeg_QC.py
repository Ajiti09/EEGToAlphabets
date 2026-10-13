
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"D:\EEGToAlphabets")
PROCESSED_ROOT = ROOT / "preprocessed_data_version4"
OUT = ROOT / "results_version4" / "processed_qc"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED = [
    "Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2",
    "F7", "F8", "F3", "F4", "T7", "P3", "P4"
]

tensors = list(PROCESSED_ROOT.rglob("trials_X.npy"))
channel_files = list(PROCESSED_ROOT.rglob("channel_names.csv"))

print("Tensor candidates:")
for p in tensors:
    print(" ", p)

print("\nChannel-list candidates:")
for p in channel_files:
    print(" ", p)

if not tensors:
    raise FileNotFoundError(
        "trials_X.npy not found. Check the processed output folder."
    )
if not channel_files:
    raise FileNotFoundError(
        "channel_names.csv not found. Check the processed output folder."
    )

# Use the first discovered files for an initial inspection.
# If multiple output versions exist, verify the selected paths printed above.
tensor_path = tensors[0]
channel_path = channel_files[0]

X = np.load(tensor_path, mmap_mode="r")
channels_df = pd.read_csv(channel_path, header=None)
channels = channels_df.iloc[:, 0].astype(str).str.strip().tolist()

print("\nTensor:", tensor_path)
print("Tensor shape:", X.shape)
print("Tensor dtype:", X.dtype)
print("Channels:", channels)
print("Channel count:", len(channels))
print("T8 present:", "T8" in channels)
print("Expected channel order matches:", channels == EXPECTED)

if X.ndim != 3:
    raise ValueError(f"Expected a 3D tensor; got shape {X.shape}")

# Based on the known project layout: trials x channels x samples.
if X.shape[1] != len(channels):
    print(
        "WARNING: tensor channel dimension does not match channel_names.csv. "
        "Confirm the tensor layout before interpreting it."
    )

if "T8" in channels:
    print("FAIL: T8 appears in the processed channel list.")
elif len(channels) == 15 and channels == EXPECTED:
    print("PASS: channel list contains the expected 15 channels, without T8.")
else:
    print("REVIEW: channel list differs from the expected configuration.")

# Check finite values and basic statistics without loading the whole tensor.
finite_count = 0
total_count = int(np.prod(X.shape))
minimum = np.inf
maximum = -np.inf

# Scan one trial at a time to limit RAM usage.
for i in range(X.shape[0]):
    trial = np.asarray(X[i])
    finite = trial[np.isfinite(trial)]
    finite_count += len(finite)
    if len(finite):
        minimum = min(minimum, float(finite.min()))
        maximum = max(maximum, float(finite.max()))

print("Total tensor values:", total_count)
print("Finite values:", finite_count)
print("Non-finite values:", total_count - finite_count)
print("Global minimum:", minimum)
print("Global maximum:", maximum)

# Plot one representative trial, one panel per channel.
trial_index = 0
trial = np.asarray(X[trial_index])
if trial.shape[0] != len(channels):
    raise ValueError(
        "This plot assumes tensor layout is trials x channels x samples. "
        "Inspect the shape before plotting."
    )

fig, axes = plt.subplots(
    len(channels), 1, figsize=(12, 18), sharex=True
)
time_ms = np.arange(trial.shape[1]) / 125.0 * 1000

for i, ax in enumerate(axes):
    ax.plot(time_ms, trial[i], linewidth=0.7)
    ax.set_ylabel(channels[i])
    ax.grid(True, alpha=0.25)

axes[-1].set_xlabel("Time within trial (ms)")
fig.suptitle(f"Processed EEG — trial index {trial_index}")
fig.tight_layout()
fig.savefig(OUT / "representative_processed_trial.png", dpi=150)
plt.close(fig)

print("Processed trial plot saved to:", OUT)
