
from pathlib import Path
import zipfile
import io

import numpy as np
import pandas as pd


RAW_DIR = Path(
    r"D:\EEGToAlphabets\raw data_version4"
)

QC_FILE = (
    RAW_DIR
    / "raw_zip_audit_corrected"
    / "all_channel_qc_corrected.csv"
)

OUT_DIR = RAW_DIR / "raw_zip_audit_corrected"
FS = 125


def longest_identical_runs(values):
    """Return longest exact-value runs, splitting at non-finite values."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)

    if len(values) == 0:
        return 0, None, None

    same_as_previous = np.zeros(len(values), dtype=bool)
    same_as_previous[1:] = (
        finite[1:]
        & finite[:-1]
        & (values[1:] == values[:-1])
    )

    # A new run starts wherever the current value is not identical
    # to the preceding finite value.
    starts = np.ones(len(values), dtype=bool)
    starts[1:] = ~same_as_previous[1:]

    run_starts = np.flatnonzero(starts)
    run_ends = np.r_[run_starts[1:], len(values)]

    best_length = 0
    best_start = None
    best_value = None

    for start, end in zip(run_starts, run_ends):
        # Do not count missing/infinite readings as flat runs.
        if not finite[start]:
            continue

        length = int(end - start)

        if length > best_length:
            best_length = length
            best_start = int(start)
            best_value = float(values[start])

    return best_length, best_start, best_value


def main():
    qc = pd.read_csv(QC_FILE)

    flagged = qc[
        qc["repeated_consecutive_fraction"] >= 0.99
    ].copy()

    print("Flagged channel records:", len(flagged))
    print("Investigating original raw ZIP recordings...")

    results = []

    for archive_name, group in flagged.groupby("archive"):
        archive_path = RAW_DIR / archive_name

        if not archive_path.exists():
            print("WARNING: ZIP not found:", archive_path)
            continue

        with zipfile.ZipFile(archive_path) as zf:
            members = set(zf.namelist())

            for _, row in group.iterrows():
                member = row["raw_file"]
                channel = row["channel"]

                if member not in members:
                    print("WARNING: member not found:", member)
                    continue

                # Read only the actual recording referenced by the QC.
                with zf.open(member) as f:
                    df = pd.read_csv(
                        io.BytesIO(f.read()),
                        skiprows=4,
                        usecols=range(17),
                        low_memory=False,
                    )

                df.columns = [str(c).strip() for c in df.columns]

                if channel not in df.columns:
                    print("WARNING: channel not found:", channel)
                    continue

                values = pd.to_numeric(
                    df[channel], errors="coerce"
                ).to_numpy(dtype=float)

                finite = np.isfinite(values)
                valid_pairs = finite[1:] & finite[:-1]

                equal_pairs = (
                    (values[1:] == values[:-1])
                    & valid_pairs
                )

                valid_pair_count = int(valid_pairs.sum())
                equal_pair_count = int(equal_pairs.sum())

                exact_repeat_fraction = (
                    equal_pair_count / valid_pair_count
                    if valid_pair_count else np.nan
                )

                longest, start, value = longest_identical_runs(
                    values
                )

                # Longest run duration, in seconds.
                duration_seconds = longest / FS

                results.append({
                    "archive": archive_name,
                    "raw_file": member,
                    "channel": channel,
                    "sample_count": len(values),
                    "original_repeat_fraction": row[
                        "repeated_consecutive_fraction"
                    ],
                    "recalculated_repeat_fraction": (
                        exact_repeat_fraction
                    ),
                    "unique_finite_values": int(
                        np.unique(values[finite]).size
                    ),
                    "longest_identical_run_samples": longest,
                    "longest_identical_run_seconds": (
                        duration_seconds
                    ),
                    "run_start_row_zero_based": start,
                    "run_start_seconds_approx": (
                        start / FS if start is not None else np.nan
                    ),
                    "repeated_value": value,
                    "standard_deviation": (
                        float(np.std(values[finite], ddof=1))
                        if finite.sum() > 1 else np.nan
                    ),
                    "flat_for_at_least_1_second": (
                        duration_seconds >= 1.0
                    ),
                    "flat_for_at_least_5_seconds": (
                        duration_seconds >= 5.0
                    ),
                })

        print("Completed:", archive_name)

    out = pd.DataFrame(results)

    output_file = OUT_DIR / "high_repeat_channel_investigation.csv"
    out.to_csv(output_file, index=False)

    print("\nINVESTIGATION COMPLETE")
    print("Rows investigated:", len(out))
    print("Output:", output_file)

    if not out.empty:
        print("\nLongest repeated-value runs:")
        print(
            out.sort_values(
                "longest_identical_run_seconds",
                ascending=False
            )[
                [
                    "archive",
                    "channel",
                    "longest_identical_run_samples",
                    "longest_identical_run_seconds",
                    "run_start_seconds_approx",
                ]
            ].head(15).to_string(index=False)
        )


if __name__ == "__main__":
    main()
