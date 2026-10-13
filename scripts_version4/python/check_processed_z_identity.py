
from pathlib import Path
import csv
from collections import Counter

root = Path(r"D:\EEGToAlphabets")
processed = root / "preprocessed_data_version4"

def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

# 1. Check the processed session index.
index_path = processed / "fdf_session_index.csv"
index = read_csv(index_path)

print("\n" + "=" * 85)
print("PROCESSED SESSION INDEX — P04/P10, CLASS Z")
print("=" * 85)

for row in index:
    if row.get("participant", "").strip() in {"P04", "P10"} \
            and row.get("label", "").strip().upper() == "Z":
        for field in [
            "participant", "label", "folder",
            "raw_path", "segments_path", "event_path"
        ]:
            print(f"{field}: {row.get(field, '')}")
        print("-" * 85)

# 2. Check processed trial metadata.
trials_path = processed / "trials_meta.csv"
trials = read_csv(trials_path)

print("\n" + "=" * 85)
print("TRIAL METADATA — P04/P10, CLASS Z")
print("=" * 85)

selected = [
    row for row in trials
    if row.get("participant", "").strip() in {"P04", "P10"}
    and row.get("label", "").strip().upper() == "Z"
]

print("Matching trial rows:", len(selected))
print("Participant counts:", dict(Counter(
    row.get("participant", "") for row in selected
)))

for row in selected[:10]:
    print({
        key: row.get(key, "")
        for key in [
            "participant", "label", "class_index",
            "trial", "trial_uid"
        ]
    })

# 3. Check window metadata for the same participants/class.
metadata_path = processed / "metadata.csv"
metadata = read_csv(metadata_path)

print("\n" + "=" * 85)
print("WINDOW METADATA — P04/P10, CLASS Z")
print("=" * 85)

selected_windows = [
    row for row in metadata
    if row.get("participant", "").strip() in {"P04", "P10"}
    and row.get("label", "").strip().upper() == "Z"
]

print("Matching window rows:", len(selected_windows))
print("Participant counts:", dict(Counter(
    row.get("participant", "") for row in selected_windows
)))

for row in selected_windows[:3]:
    print({
        key: row.get(key, "")
        for key in [
            "participant", "session_label", "label",
            "trial", "trial_uid", "task_start_used",
            "task_end_used", "raw_path"
        ]
    })
