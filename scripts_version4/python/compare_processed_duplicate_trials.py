
from pathlib import Path
import csv
import numpy as np

root = Path(r"D:\EEGToAlphabets\preprocessed_data_version4")

meta_path = root / "trials_meta.csv"
data_path = root / "trials_X.npy"

with open(meta_path, "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

X = np.load(data_path, mmap_mode="r")

print("Trial array shape:", X.shape)
print("Trial metadata rows:", len(rows))

# Confirm that array rows and metadata rows have matching counts.
if len(rows) != X.shape[0]:
    raise ValueError(
        "Metadata row count does not match trials_X.npy. "
        "Do not compare array rows until their ordering is established."
    )

lookup = {}
for i, row in enumerate(rows):
    participant = row["participant"].strip()
    label = row["label"].strip().upper()
    trial = row["trial"].strip()

    if label == "Z" and participant in {"P04", "P10"}:
        lookup[(participant, trial)] = i

print("\nMatching trials:", len(lookup))

comparisons = []

for trial in range(1, 26):
    key4 = ("P04", str(trial))
    key10 = ("P10", str(trial))

    if key4 not in lookup or key10 not in lookup:
        print(f"Trial {trial}: missing counterpart")
        continue

    i4 = lookup[key4]
    i10 = lookup[key10]

    a = np.asarray(X[i4])
    b = np.asarray(X[i10])

    exact = np.array_equal(a, b)
    max_diff = float(np.max(np.abs(a - b)))

    comparisons.append((trial, exact, max_diff))

print("\nTRIAL-BY-TRIAL COMPARISON")
for trial, exact, max_diff in comparisons:
    print(
        f"Trial {trial:02d}: "
        f"exactly_equal={exact}, max_absolute_difference={max_diff:.8g}"
    )

print("\nSUMMARY")
print("Compared pairs:", len(comparisons))
print("Exactly equal pairs:", sum(x[1] for x in comparisons))
print("Different pairs:", sum(not x[1] for x in comparisons))
