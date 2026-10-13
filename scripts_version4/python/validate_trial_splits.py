
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------
ROOT = Path(r"D:\EEGToAlphabets")
DATA = ROOT / "preprocessed_data_version4"
OUT = ROOT / "results_version4" / "split_audit"
OUT.mkdir(parents=True, exist_ok=True)

X_PATH = DATA / "trials_X.npy"
META_PATH = DATA / "trials_meta.csv"

# --------------------------------------------------
# 2. Load trial-level data
# --------------------------------------------------
X = np.load(X_PATH, mmap_mode="r")
meta = pd.read_csv(META_PATH)

required = {"participant", "label", "trial", "trial_uid"}
missing = required - set(meta.columns)
if missing:
    raise ValueError(f"Missing metadata columns: {sorted(missing)}")

if X.ndim != 3:
    raise ValueError(f"Expected 3D trial array; got {X.shape}")

if X.shape[0] != len(meta):
    raise ValueError(
        f"Trial count mismatch: X={X.shape[0]}, metadata={len(meta)}"
    )

if meta[list(required)].isna().any().any():
    raise ValueError("Missing values found in required metadata fields.")

if not meta["trial_uid"].is_unique:
    raise ValueError("trial_uid is not unique in trials_meta.csv.")

participants = set(meta["participant"].astype(str).unique())
required_duplicates = {"P04", "P10"}

if not required_duplicates.issubset(participants):
    raise ValueError(
        "Expected participant IDs P04 and P10 were not both found. "
        f"Found IDs: {sorted(participants)}"
    )

# --------------------------------------------------
# 3. Build duplicate-aware participant groups
# --------------------------------------------------
# P04 and P10 are held out together because their
# Z-session recordings were previously identified as duplicates.
meta["split_group"] = meta["participant"].astype(str)

meta.loc[
    meta["participant"].astype(str).isin(["P04", "P10"]),
    "split_group"
] = "P04_P10_DUPLICATE_GROUP"

print("Array shape:", X.shape)
print("Number of trials:", len(meta))
print("Number of participants:", len(participants))
print("Classes:", sorted(meta["label"].astype(str).unique()))
print("Number of split groups:", meta["split_group"].nunique())

# --------------------------------------------------
# 4. Create grouped LOSO folds
# --------------------------------------------------
logo = LeaveOneGroupOut()
groups = meta["split_group"].to_numpy()

fold_summaries = []
manifest_rows = []

for fold, (train_idx, test_idx) in enumerate(
    logo.split(np.zeros(len(meta)), meta["label"], groups),
    start=1
):
    train = meta.iloc[train_idx]
    test = meta.iloc[test_idx]

    train_trials = set(train["trial_uid"].astype(str))
    test_trials = set(test["trial_uid"].astype(str))

    train_participants = set(train["participant"].astype(str))
    test_participants = set(test["participant"].astype(str))

    train_groups = set(train["split_group"].astype(str))
    test_groups = set(test["split_group"].astype(str))

    # Mandatory leakage checks
    assert train_trials.isdisjoint(test_trials), (
        f"Trial leakage in fold {fold}"
    )
    assert train_participants.isdisjoint(test_participants), (
        f"Participant leakage in fold {fold}"
    )
    assert train_groups.isdisjoint(test_groups), (
        f"Split-group leakage in fold {fold}"
    )

    held_out_group = test["split_group"].iloc[0]

    fold_summaries.append({
        "fold": fold,
        "held_out_group": held_out_group,
        "held_out_participants": ",".join(
            sorted(test_participants)
        ),
        "train_trials": len(train_idx),
        "test_trials": len(test_idx),
        "train_participants": len(train_participants),
        "test_participants": len(test_participants),
        "trial_overlap": len(train_trials & test_trials),
        "participant_overlap": len(
            train_participants & test_participants
        ),
        "group_overlap": len(train_groups & test_groups),
        "status": "PASS"
    })

    for partition, indices in [
        ("train", train_idx),
        ("test", test_idx)
    ]:
        for idx in indices:
            row = meta.iloc[idx]
            manifest_rows.append({
                "fold": fold,
                "partition": partition,
                "trial_index": int(idx),
                "trial_uid": row["trial_uid"],
                "participant": row["participant"],
                "label": row["label"],
                "split_group": row["split_group"]
            })

# --------------------------------------------------
# 5. Save results
# --------------------------------------------------
summary = pd.DataFrame(fold_summaries)
manifest = pd.DataFrame(manifest_rows)

summary.to_csv(
    OUT / "duplicate_aware_loso_fold_summary.csv",
    index=False
)
manifest.to_csv(
    OUT / "duplicate_aware_loso_trial_manifest.csv",
    index=False
)

# Verify the expected number of folds and held-out group
assert len(summary) == 19, (
    f"Expected 19 folds, got {len(summary)}"
)
assert (summary["status"] == "PASS").all()
assert (summary["trial_overlap"] == 0).all()
assert (summary["participant_overlap"] == 0).all()
assert (summary["group_overlap"] == 0).all()

print("\nFOLD SUMMARY")
print(summary.to_string(index=False))

print("\nAll leakage checks passed.")
print("Fold summary:", OUT / "duplicate_aware_loso_fold_summary.csv")
print("Trial manifest:", OUT / "duplicate_aware_loso_trial_manifest.csv")
