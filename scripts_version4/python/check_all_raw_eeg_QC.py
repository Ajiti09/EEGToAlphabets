
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import numpy as np

ROOT = Path(r"D:\EEGToAlphabets")
RAW_ROOT = ROOT / "raw data_version4"
OUT = ROOT / "results_version4" / "raw_qc_all"
OUT.mkdir(parents=True, exist_ok=True)

CHANNELS = [f"EXG Channel {i}" for i in range(16)]
file_rows, channel_rows = [], []
files_checked = 0
errors = []

def longest_identical_run(s):
    """Longest consecutive run of identical finite readings."""
    a = s.to_numpy(dtype=float)
    if len(a) == 0:
        return 0
    best = current = 1
    for i in range(1, len(a)):
        if np.isfinite(a[i]) and np.isfinite(a[i-1]) and a[i] == a[i-1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best

zip_paths = sorted(RAW_ROOT.rglob("*.zip"))
print("ZIP files found:", len(zip_paths))

for zip_path in zip_paths:
    try:
        with ZipFile(zip_path) as zf:
            members = [
                n for n in zf.namelist()
                if Path(n).name.lower().startswith("openbci-raw-")
                and n.lower().endswith(".txt")
            ]

            for member in members:
                try:
                    with zf.open(member) as f:
                        df = pd.read_csv(
                            BytesIO(f.read()),
                            comment="%",
                            skipinitialspace=True,
                            low_memory=False
                        )
                    df.columns = df.columns.astype(str).str.strip()

                    required = ["Sample Index"] + CHANNELS
                    missing_cols = [c for c in required if c not in df.columns]
                    if missing_cols:
                        errors.append({
                            "zip": str(zip_path),
                            "file": member,
                            "error": f"Missing columns: {missing_cols}"
                        })
                        continue

                    files_checked += 1
                    idx = pd.to_numeric(df["Sample Index"], errors="coerce")
                    d = idx.diff().dropna()
                    d = d[np.isfinite(d)]

                    # OpenBCI counters can wrap from a high value back to zero.
                    wraps = int((d < 0).sum())
                    positive = d[d > 0]

                    file_rows.append({
                        "zip": str(zip_path),
                        "raw_file": member,
                        "rows": len(df),
                        "index_missing": int(idx.isna().sum()),
                        "index_wrap_transitions": wraps,
                        "positive_step_1": int((positive == 1).sum()),
                        "positive_step_2": int((positive == 2).sum()),
                        "positive_steps_gt_2": int((positive > 2).sum()),
                        "positive_step_values": str(
                            positive.value_counts().head(10).to_dict()
                        )
                    })

                    for ch in CHANNELS:
                        x = pd.to_numeric(df[ch], errors="coerce")
                        a = x.to_numpy(dtype=float)
                        finite = a[np.isfinite(a)]
                        repeated = (
                            (a[1:] == a[:-1])
                            & np.isfinite(a[1:])
                            & np.isfinite(a[:-1])
                        ) if len(a) > 1 else np.array([], dtype=bool)

                        channel_rows.append({
                            "zip": str(zip_path),
                            "raw_file": member,
                            "channel": ch,
                            "rows": len(a),
                            "missing_or_non_numeric": int(x.isna().sum()),
                            "infinite_values": int(np.isinf(a).sum()),
                            "minimum": np.min(finite) if len(finite) else np.nan,
                            "maximum": np.max(finite) if len(finite) else np.nan,
                            "median": np.median(finite) if len(finite) else np.nan,
                            "std_dev": np.std(finite) if len(finite) else np.nan,
                            "unique_finite_values": int(len(np.unique(finite))),
                            "repeated_consecutive_fraction": (
                                float(repeated.mean()) if len(repeated) else 0.0
                            ),
                            "longest_identical_run": longest_identical_run(x),
                            "possible_flat_channel": (
                                len(finite) > 0 and len(np.unique(finite)) <= 1
                            )
                        })

                except Exception as e:
                    errors.append({
                        "zip": str(zip_path),
                        "file": member,
                        "error": repr(e)
                    })
    except Exception as e:
        errors.append({
            "zip": str(zip_path),
            "file": "",
            "error": repr(e)
        })

pd.DataFrame(file_rows).to_csv(
    OUT / "all_sample_index_qc.csv", index=False
)
pd.DataFrame(channel_rows).to_csv(
    OUT / "all_channel_qc.csv", index=False
)
pd.DataFrame(errors).to_csv(
    OUT / "qc_errors.csv", index=False
)

print("\nQC finished")
print("ZIP files found:", len(zip_paths))
print("Raw EEG files checked:", files_checked)
print("Errors:", len(errors))
print("Reports saved to:", OUT)
