
from pathlib import Path
import csv
import pandas as pd
import h5py

ROOT = Path(r"D:\EEGToAlphabets")
DATA = ROOT / "preprocessed_data_version4"

print("\n=== 1. CHANNEL NAMES ===")
path = DATA / "channel_names.csv"

with path.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.reader(f))

channel_names = [row[0].strip() for row in rows if row]

print("Channel count:", len(channel_names))
print("Channel names:", channel_names)
print("Unique names:", len(set(channel_names)))

print("\n=== 2. MATLAB V7.3 VARIABLES ===")
path = DATA / "epochs.mat"

# with h5py.File(path, "r") as f:
#     def inspect(name, obj):
#         if isinstance(obj, h5py.Dataset):
#             print(
#                 name,
#                 "| shape:", obj.shape,
#                 "| dtype:", obj.dtype
#             )

    # f.visititems(inspect)

with h5py.File(path, "r") as f:
    print("Top-level entries:")

    for name, obj in f.items():
        if name == "#refs#":
            continue

        if isinstance(obj, h5py.Dataset):
            print(
                name,
                "| shape:", obj.shape,
                "| dtype:", obj.dtype
            )
        elif isinstance(obj, h5py.Group):
            print(name, "| Group")
            print("  Child entries:", list(obj.keys())[:20])


print("\n=== 3. TRIAL METADATA ===")
trials = pd.read_csv(DATA / "trials_meta.csv")

print("Rows:", len(trials))
print("Columns:", trials.columns.tolist())
print("Unique participants:", trials["participant"].nunique())
print("Class counts:")
print(trials["label"].value_counts(dropna=False).sort_index())

print("Duplicate trial UIDs:",
      trials["trial_uid"].duplicated().sum())

print("Trials per participant and class:")
print(
    trials.groupby(["participant", "label"])
    .size()
    .value_counts()
    .sort_index()
)

print("\n=== 4. WINDOW METADATA ===")
meta = pd.read_csv(DATA / "metadata.csv")

print("Rows:", len(meta))
print("Unique trial UIDs:", meta["trial_uid"].nunique())
print("Unique window UIDs:", meta["window_uid"].nunique())
print("Duplicate window UIDs:",
      meta["window_uid"].duplicated().sum())

windows_per_trial = meta.groupby("trial_uid").size()

print("Windows per trial distribution:")
print(windows_per_trial.value_counts().sort_index())

print("Trials with window count other than 156:",
      int((windows_per_trial != 156).sum()))

print("\n=== 5. WINDOW LABEL CONSISTENCY ===")
for col in ["participant", "label", "class_index", "trial"]:
    if col in meta.columns:
        distinct = meta.groupby("trial_uid")[col].nunique()
        print(
            f"Trials with multiple values for {col}:",
            int((distinct > 1).sum())
        )

print("\n=== 6. SESSION COVERAGE ===")
sessions = pd.read_csv(DATA / "fdf_session_index.csv")

print("Session rows:", len(sessions))
print("Unique participant-label pairs:",
      sessions[["participant", "label"]].drop_duplicates().shape[0])
print("Missing values per column:")
print(sessions.isna().sum())

print("\n=== 7. ICA AUDIT SUMMARY ===")
ica = pd.read_csv(DATA / "ica_log.csv")

print("ICA applied:")
print(ica["ica_applied"].value_counts(dropna=False))

print("ICA errors:", ica["ica_error"].notna().sum())
print("ICA error examples:")
print(ica.loc[ica["ica_error"].notna(), "ica_error"].head(10))

print("\n=== 8. BAD CHANNEL AUDIT SUMMARY ===")
bad = pd.read_csv(DATA / "bad_channel_audit.csv")

print("Rows:", len(bad))
print("Records with bad-channel values:")
print(bad["bad_channels"].value_counts(dropna=False).head(20))
