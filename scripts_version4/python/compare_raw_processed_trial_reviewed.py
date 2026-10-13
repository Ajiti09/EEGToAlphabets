
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(r"D:\EEGToAlphabets")
PRE = ROOT / "preprocessed_data_version4"
RAW_ZIP = ROOT / "raw data_version4" / "TEST_SUBJECT_1.zip"
OUT = ROOT / "results_version4" / "raw_processed_comparison"
OUT.mkdir(parents=True, exist_ok=True)

FS = 125
PARTICIPANT = "P01"
LABEL = "A"
TRIAL = 1

# Load trial metadata and select the target trial.
meta = pd.read_csv(PRE / "metadata.csv")
rows = meta[
    (meta["participant"] == PARTICIPANT)
    & (meta["label"] == LABEL)
    & (meta["trial"] == TRIAL)
].copy()

if rows.empty:
    raise ValueError("No metadata rows found for P01 / A / trial 1.")

rows = rows.sort_values("window_id")
print("Metadata rows:", len(rows))
print(rows[[
    "participant", "label", "trial", "window_id",
    "task_start_used", "task_end_used", "raw_path"
]].head(3).to_string(index=False))

if rows["raw_path"].nunique() != 1:
    raise ValueError("Selected windows refer to multiple raw files.")

# Processed tensor: trials x channels x samples.
X = np.load(PRE / "trials_X.npy", mmap_mode="r")
trials_meta = pd.read_csv(PRE / "trials_meta.csv")
match = trials_meta[
    (trials_meta["participant"] == PARTICIPANT)
    & (trials_meta["label"] == LABEL)
    & (trials_meta["trial"] == TRIAL)
]

if len(match) != 1:
    raise ValueError(
        f"Expected one trials_meta row; found {len(match)}."
    )

trial_index = match.index[0]
processed = np.asarray(X[trial_index])

channels = pd.read_csv(
    PRE / "channel_names.csv", header=None
).iloc[:, 0].astype(str).str.strip().tolist()

print("\nProcessed trial index:", trial_index)
print("Processed shape:", processed.shape)
print("Channel order:", channels)

# Raw recording member inside the ZIP.
with ZipFile(RAW_ZIP) as zf:
    candidates = [
        n for n in zf.namelist()
        if Path(n).name.startswith("OpenBCI-RAW-")
        and "TEST_SUBJECT_1_A" in n
    ]

    if len(candidates) != 1:
        raise ValueError(
            "Expected one raw A-session file; found: "
            + repr(candidates)
        )

    raw_member = candidates[0]
    with zf.open(raw_member) as f:
        raw = pd.read_csv(
            BytesIO(f.read()),
            comment="%",
            skipinitialspace=True,
            low_memory=False
        )

raw.columns = raw.columns.astype(str).str.strip()
print("\nRaw member:", raw_member)
print("Raw shape:", raw.shape)
print("Raw columns:", raw.columns.tolist())

raw_channels = [f"EXG Channel {i}" for i in range(16)]
missing = [c for c in raw_channels if c not in raw.columns]
if missing:
    raise ValueError(f"Raw channels missing: {missing}")

# Use the first and last metadata window boundaries.
start_s = float(rows["task_start_used"].min())
end_s = float(rows["task_end_used"].max())

print(f"\nMetadata interval: {start_s:.4f} to {end_s:.4f} seconds")
print("Interval duration:", end_s - start_s, "seconds")

# IMPORTANT:
# This assumes task_start_used/task_end_used are seconds from the beginning
# of the raw recording. Confirm these values against the session event log.
# The previously checked P01-A event log reported task_start=6.9851 and
# task_end=16.9888 seconds, which is consistent with this metadata interval.
# Keep this check explicit rather than silently treating the assumption as proven.
if not (0 <= start_s < end_s):
    raise ValueError("Invalid task interval in metadata.")

i0 = round(start_s * FS)
i1 = round(end_s * FS)

raw_numeric = raw[raw_channels].apply(
    pd.to_numeric, errors="coerce"
).to_numpy(dtype=float)

if i0 < 0 or i1 > len(raw_numeric) or i1 <= i0:
    raise ValueError(
        "The assumed raw interval is outside the recording. "
        "Do not treat the interval as matched."
    )

raw_segment = raw_numeric[i0:i1]
print("Assumed raw sample indices:", i0, "to", i1)
print("Raw segment shape:", raw_segment.shape)

# Map raw channels to the configured channel names, then retain exactly the
# channels present in the processed tensor (T8 is expected to be excluded).
configured_names = [
    "Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2",
    "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4"
]
if len(configured_names) != len(raw_channels):
    raise ValueError("Configured raw channel mapping does not match raw channel count.")

raw_by_name = {name: raw_segment[:, i] for i, name in enumerate(configured_names)}
missing_processed = [name for name in channels if name not in raw_by_name]
if missing_processed:
    raise ValueError(f"Cannot map processed channels to raw channels: {missing_processed}")

raw_retained = np.column_stack([raw_by_name[name] for name in channels])
pd.DataFrame(raw_retained, columns=channels).to_csv(
    OUT / "selected_raw_segment_retained_channels.csv", index=False
)
pd.DataFrame(processed.T, columns=channels).to_csv(
    OUT / "selected_processed_trial.csv", index=False
)

# Summary statistics help identify scale/baseline differences; they do not
# establish that the signals are physically calibrated to the same units.
summary_rows = []
for j, name in enumerate(channels):
    rv = raw_retained[:, j]
    pv = processed[j]
    summary_rows.append({
        "channel": name,
        "raw_finite_count": int(np.isfinite(rv).sum()),
        "processed_finite_count": int(np.isfinite(pv).sum()),
        "raw_median": float(np.nanmedian(rv)),
        "processed_median": float(np.nanmedian(pv)),
        "raw_min": float(np.nanmin(rv)),
        "raw_max": float(np.nanmax(rv)),
        "processed_min": float(np.nanmin(pv)),
        "processed_max": float(np.nanmax(pv)),
        "raw_std": float(np.nanstd(rv)),
        "processed_std": float(np.nanstd(pv)),
    })
pd.DataFrame(summary_rows).to_csv(OUT / "raw_processed_channel_summary.csv", index=False)

# Plot matched channel sets. Separate time axes are used because the raw
# event interval and processed trial can differ slightly in sample count.
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=False)
raw_time = np.arange(len(raw_retained)) / FS
for j, name in enumerate(channels):
    axes[0].plot(raw_time, raw_retained[:, j], label=name, linewidth=0.7)
axes[0].set_title(
    f"Raw EEG — {PARTICIPANT}-{LABEL}, event interval {start_s:.4f}–{end_s:.4f}s"
)
axes[0].set_xlabel("Seconds from selected segment start")
axes[0].set_ylabel("Raw numeric units (unit not verified)")
axes[0].legend(ncol=5, fontsize=8)
axes[0].grid(True, alpha=0.25)

processed_time = np.arange(processed.shape[-1]) / FS
for j, name in enumerate(channels):
    axes[1].plot(processed_time, processed[j], label=name, linewidth=0.7)
axes[1].set_title(
    f"Processed EEG — {PARTICIPANT}, {LABEL}, trial {TRIAL}"
)
axes[1].set_xlabel("Seconds from processed trial start")
axes[1].set_ylabel("Processed numeric units (unit not verified)")
axes[1].legend(ncol=5, fontsize=8)
axes[1].grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "raw_vs_processed_trial.png", dpi=150)
plt.close(fig)

print("\nSaved outputs in:", OUT)
print("- raw_vs_processed_trial.png")
print("- selected_raw_segment_retained_channels.csv")
print("- selected_processed_trial.csv")
print("- raw_processed_channel_summary.csv")
print("Confirm the event-log timing and raw channel mapping before drawing conclusions.")
