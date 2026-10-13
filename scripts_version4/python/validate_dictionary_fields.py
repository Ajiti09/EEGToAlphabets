
from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\EEGToAlphabets\preprocessed_data_version4")

trials = pd.read_csv(ROOT / "trials_meta.csv")
meta = pd.read_csv(ROOT / "metadata.csv")

print("=== LABEL TO CLASS INDEX ===")
print(
    trials[["label", "class_index"]]
    .drop_duplicates()
    .sort_values("class_index")
    .to_string(index=False)
)

print("\n=== SAMPLING FREQUENCY ===")
print(meta["fs"].value_counts(dropna=False).to_string())

print("\n=== PREPROCESSING REJECTION FLAG ===")
print(
    meta["sample_rejected_by_preproc"]
    .value_counts(dropna=False)
    .to_string()
)

print("\n=== WINDOW DURATION ===")
duration = meta["task_end_used"] - meta["task_start_used"]
print(duration.describe().to_string())

print("\n=== PARTICIPANT ORDER IN WINDOW METADATA ===")
print(meta["participant"].drop_duplicates().to_list())
