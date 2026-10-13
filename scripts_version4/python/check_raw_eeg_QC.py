from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

# Update this to the exact ZIP path on your computer.
zip_path = Path(
    r"D:\EEGToAlphabets\raw data_version4\TEST_SUBJECT_1.zip"
)

# Inspect every raw TXT file inside the ZIP.
out_dir = Path(r"D:\EEGToAlphabets\results_version4\raw_qc")
out_dir.mkdir(parents=True, exist_ok=True)

channel_names = [f"EXG Channel {i}" for i in range(16)]
channel_results = []
file_results = []

with zipfile.ZipFile(zip_path) as zf:
    txt_files = [
        name for name in zf.namelist()
        if name.lower().endswith(".txt")
        and "OpenBCI-RAW-" in Path(name).name
        and "TEST_SUBJECT_1_A" in name
    ]

    if not txt_files:
        raise FileNotFoundError(
            "No matching OpenBCI raw TXT file found in the ZIP. "
            "Check the session folder name."
        )

    for member in txt_files:
        with zf.open(member) as f:
            # OpenBCI header lines start with %, so pandas skips them.
            df = pd.read_csv(
                f,
                comment="%",
                skipinitialspace=True,
                low_memory=False
            )

        df.columns = df.columns.astype(str).str.strip()

        missing_cols = [
            c for c in ["Sample Index"] + channel_names
            if c not in df.columns
        ]
        if missing_cols:
            print(f"SKIPPED {member}: missing columns {missing_cols}")
            continue

        idx = pd.to_numeric(df["Sample Index"], errors="coerce")
        idx_diff = idx.diff().dropna()
        valid_diffs = idx_diff[np.isfinite(idx_diff)]

        file_results.append({
            "file": member,
            "rows": len(df),
            "sample_index_missing": int(idx.isna().sum()),
            "sample_index_min": idx.min(),
            "sample_index_max": idx.max(),
            "sample_index_step_1_count": int((valid_diffs == 1).sum()),
            "sample_index_step_2_count": int((valid_diffs == 2).sum()),
            "sample_index_nonpositive_steps": int((valid_diffs <= 0).sum()),
            "sample_index_gaps_gt_1": int((valid_diffs > 1).sum()),
            "sample_index_step_values": str(
                valid_diffs.value_counts().head(10).to_dict()
            ),
        })

        for ch in channel_names:
            x = pd.to_numeric(df[ch], errors="coerce")
            finite = x[np.isfinite(x)]
            # Longest run of identical consecutive finite readings.
            same = x.eq(x.shift()) & x.notna()
            groups = (~same).cumsum()
            longest_run = (
                same.groupby(groups).sum().max()
                if len(same) else 0
            )
            repeated_fraction = (
                float(same.sum() / max(len(x) - 1, 1))
            )

            channel_results.append({
                "file": member,
                "channel": ch,
                "rows": len(x),
                "missing_or_non_numeric": int(x.isna().sum()),
                "infinite_values": int(np.isinf(x).sum()),
                "minimum": finite.min() if len(finite) else np.nan,
                "maximum": finite.max() if len(finite) else np.nan,
                "median": finite.median() if len(finite) else np.nan,
                "standard_deviation": finite.std() if len(finite) else np.nan,
                "unique_finite_values": int(finite.nunique()),
                "repeated_consecutive_fraction": repeated_fraction,
                "longest_identical_run": int(longest_run),
                "possible_flat_channel": (
                    len(finite) > 0 and finite.nunique() <= 1
                ),
            })

pd.DataFrame(file_results).to_csv(
    out_dir / "sample_index_qc.csv", index=False
)
pd.DataFrame(channel_results).to_csv(
    out_dir / "channel_qc.csv", index=False
)

print("QC completed.")
print("Output folder:", out_dir)
print("Files checked:", len(file_results))
print("Reports: sample_index_qc.csv and channel_qc.csv")
