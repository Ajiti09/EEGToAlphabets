
from pathlib import Path
import hashlib
import csv
import zipfile
import numpy as np
from scipy.io import whosmat

ROOT = Path(r"D:\EEGToAlphabets")
DATA = ROOT / "preprocessed_data_version4"

print("\n=== 1. NUMPY ARRAY DIMENSIONS ===")

for name in ["trials_X.npy", "epochs_slice.npz"]:
    path = DATA / name
    if not path.exists():
        continue

    try:
        if path.suffix == ".npy":
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            print(name, "shape:", arr.shape,
                  "dtype:", arr.dtype)
        else:
            with np.load(path, allow_pickle=False) as archive:
                for key in archive.files:
                    arr = archive[key]
                    print(name, "variable:", key,
                          "shape:", arr.shape,
                          "dtype:", arr.dtype)
    except Exception as exc:
        print(name, "ERROR:", exc)

print("\n=== 2. MATLAB FILE VARIABLES ===")

for name in ["epochs.mat", "epochs-W8Ehjb.mat"]:
    path = DATA / name
    if not path.exists():
        continue

    try:
        print("\n", name)
        for variable, shape, dtype in whosmat(path):
            print("Variable:", variable,
                  "| Shape:", shape,
                  "| Type:", dtype)
    except Exception as exc:
        print(name, "ERROR:", exc)

print("\n=== 3. CSV HEADERS AND ROW COUNTS ===")

csv_files = [
    "channel_names.csv",
    "fdf_epoch_manifest.csv",
    "fdf_session_index.csv",
    "trials_meta.csv",
    "bad_channel_audit.csv",
    "ica_log.csv",
]

for name in csv_files:
    path = DATA / name
    if not path.exists():
        continue

    try:
        with path.open("r", encoding="utf-8-sig",
                       newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            count = sum(1 for _ in reader)

        print(f"\n{name}: {count} data rows")
        print("Columns:", header)
    except Exception as exc:
        print(name, "ERROR:", exc)

print("\n=== 4. LARGE METADATA CSV ===")

path = DATA / "metadata.csv"
if path.exists():
    try:
        with path.open("r", encoding="utf-8-sig",
                       newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            count = sum(1 for _ in reader)

        print("metadata.csv rows:", count)
        print("Column count:", len(header))
        print("Columns:", header)
    except Exception as exc:
        print("metadata.csv ERROR:", exc)

print("\n=== 5. ZIP CONTENTS ===")

path = DATA / "epochs.zip"
if path.exists():
    try:
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                print(item.filename,
                      "| uncompressed bytes:", item.file_size)
    except Exception as exc:
        print("ZIP ERROR:", exc)

print("\n=== 6. HASH CHECK OF SAME-SIZED FILE PAIRS ===")

pairs = [
    ("epochs.mat", "epochs-W8Ehjb.mat"),
    ("metadata.csv", "metadata-7hsPK2.csv"),
    ("bad_channel_audit.csv", "bad_channel_audit-3IMfCJ.csv"),
    ("channel_names.csv", "channel_names-A5h6av.csv"),
    ("fdf_epoch_manifest.csv",
     "fdf_epoch_manifest-XOO7X7.csv"),
    ("fdf_session_index.csv",
     "fdf_session_index-8c1GJ4.csv"),
    ("ica_log.csv", "ica_log-mINhA9.csv"),
]

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

for first, second in pairs:
    p1, p2 = DATA / first, DATA / second
    if not p1.exists() or not p2.exists():
        continue

    try:
        same = sha256(p1) == sha256(p2)
        print(f"{first} vs {second}: "
              f"{'IDENTICAL' if same else 'DIFFERENT'}")
    except Exception as exc:
        print(first, second, "ERROR:", exc)
