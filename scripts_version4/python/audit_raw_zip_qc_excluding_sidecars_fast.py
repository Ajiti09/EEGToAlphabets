import argparse
import hashlib
import io
import zipfile
from pathlib import Path, PurePosixPath
from collections import Counter

import numpy as np
import pandas as pd


CHANNELS = [f"EXG Channel {i}" for i in range(16)]


def is_macos_sidecar(member):
    """Exclude macOS metadata folders and AppleDouble files."""
    path = PurePosixPath(member.replace("\\", "/"))
    return (
        "__MACOSX" in path.parts
        or any(part.startswith("._") for part in path.parts)
    )


def is_actual_raw(member):
    """Only process actual OpenBCI raw text recordings."""
    if is_macos_sidecar(member):
        return False

    name = PurePosixPath(member.replace("\\", "/")).name
    return name.startswith("OpenBCI-RAW") and name.lower().endswith(".txt")


def qc_one_recording(zf, archive_name, member):
    # Read the recording once; reuse the bytes for parsing and hashing.
    raw_bytes = zf.read(member)
    digest = hashlib.sha256(raw_bytes).hexdigest()

    # OpenBCI export: four metadata lines, followed by the CSV header.
    df = pd.read_csv(
        io.BytesIO(raw_bytes),
        skiprows=4,
        usecols=range(17),
        low_memory=False,
    )

    df.columns = [str(c).strip() for c in df.columns]
    expected = ["Sample Index"] + CHANNELS

    if list(df.columns) != expected:
        raise ValueError(
            f"Unexpected columns in {archive_name}:{member}\n"
            f"Found: {list(df.columns)}"
        )

    # Sample-index QC
    sample = pd.to_numeric(
        df["Sample Index"], errors="coerce"
    ).to_numpy(dtype=float)

    n = len(df)
    idx_missing = int(np.isnan(sample).sum())
    diffs = np.diff(sample)

    # OpenBCI index wraps from 254 to 0.
    wraps_mask = (
        (diffs < 0)
        & (sample[:-1] >= 250)
        & (sample[1:] <= 5)
    )
    wraps = int(wraps_mask.sum())
    positive = diffs[diffs > 0]
    step_counts = Counter(positive.tolist())

    finite_sample = sample[np.isfinite(sample)]

    index_row = {
        "archive": archive_name,
        "raw_file": member,
        "rows": n,
        "sample_index_missing": idx_missing,
        "sample_index_min": (
            float(finite_sample.min()) if len(finite_sample) else np.nan
        ),
        "sample_index_max": (
            float(finite_sample.max()) if len(finite_sample) else np.nan
        ),
        "index_wrap_transitions": wraps,
        "positive_step_1": int(np.sum(positive == 1)),
        "positive_step_2": int(np.sum(positive == 2)),
        "positive_steps_gt_2": int(np.sum(positive > 2)),
        "positive_step_values": str(dict(sorted(step_counts.items()))),
        "nonpositive_nonwrap_steps": int(
            np.sum((diffs <= 0) & ~wraps_mask)
        ),
    }

    channel_rows = []

    # Channel QC
    for ch in CHANNELS:
        vals = pd.to_numeric(
            df[ch], errors="coerce"
        ).to_numpy(dtype=float)

        finite_mask = np.isfinite(vals)
        finite = vals[finite_mask]

        missing_or_non_numeric = int((~finite_mask).sum())
        infinite = int(np.isinf(vals).sum())

        if len(vals) > 1:
            # Vectorized adjacent-equality check.
            equal_adjacent = (
                (vals[1:] == vals[:-1])
                & finite_mask[1:]
                & finite_mask[:-1]
            )

            repeated_fraction = float(
                equal_adjacent.sum() / (len(vals) - 1)
            )

            # Find the longest run of identical finite values.
            # Non-finite values break a run.
            run_start = np.empty(len(vals), dtype=bool)
            run_start[0] = True
            run_start[1:] = ~equal_adjacent

            run_ids = np.cumsum(run_start)
            run_lengths = np.bincount(
                run_ids[finite_mask]
            ) if finite_mask.any() else np.array([], dtype=int)

            longest_run = (
                int(run_lengths.max()) if len(run_lengths) else 0
            )
        else:
            repeated_fraction = np.nan
            longest_run = (
                1 if len(vals) == 1 and finite_mask[0] else 0
            )

        std = (
            float(np.std(finite, ddof=1))
            if len(finite) > 1
            else (0.0 if len(finite) == 1 else np.nan)
        )

        unique_count = int(len(np.unique(finite)))

        channel_rows.append({
            "archive": archive_name,
            "raw_file": member,
            "channel": ch,
            "rows": n,
            "missing_or_non_numeric": missing_or_non_numeric,
            "infinite_values": infinite,
            "minimum": float(finite.min()) if len(finite) else np.nan,
            "maximum": float(finite.max()) if len(finite) else np.nan,
            "median": float(np.median(finite)) if len(finite) else np.nan,
            "std_dev": std,
            "unique_finite_values": unique_count,
            "repeated_consecutive_fraction": repeated_fraction,
            "longest_identical_run": longest_run,
            "possible_flat_channel": bool(
                unique_count <= 1 and len(finite) > 0
            ),
        })

    return index_row, channel_rows, digest, len(raw_bytes)


def main():
    parser = argparse.ArgumentParser(
        description="Audit raw EEG ZIP archives, excluding macOS sidecars."
    )
    parser.add_argument(
        "--raw-dir",
        required=True,
        help="Directory containing participant ZIP archives",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (optional)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else raw_dir / "raw_zip_audit_corrected"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(
        p for p in raw_dir.glob("TEST_SUBJECT_*.zip") if p.is_file()
    )

    if not zips:
        raise SystemExit(
            f"No TEST_SUBJECT_*.zip archives found in {raw_dir}"
        )

    inventory_rows = []
    channel_rows = []
    index_rows = []
    digest_to_files = {}
    counts = Counter()

    for zip_number, zip_path in enumerate(zips, start=1):
        print(
            f"[{zip_number}/{len(zips)}] Processing {zip_path.name}",
            flush=True,
        )

        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                member = info.filename

                if info.is_dir():
                    continue

                sidecar = is_macos_sidecar(member)
                actual_raw = is_actual_raw(member)

                if sidecar:
                    counts["excluded_sidecars"] += 1
                    classification = "macOS sidecar"
                    status = "EXCLUDED: macOS sidecar"
                elif actual_raw:
                    counts["actual_raw_candidates"] += 1
                    classification = "Raw EEG recording"
                    status = "Included in raw EEG QC"
                else:
                    classification = "Other archive member"
                    status = "Inventory only"

                    lower = member.lower()
                    basename = PurePosixPath(member).name

                    if lower.endswith(".csv") and "event_log" in lower:
                        classification = "Event/marker log"
                    elif (
                        lower.endswith(".csv")
                        and "training_segments" in lower
                    ):
                        classification = "Training-segment CSV"
                    elif basename == ".DS_Store":
                        classification = "macOS metadata"

                inventory_rows.append({
                    "archive": zip_path.name,
                    "member_path": member,
                    "extension": Path(member).suffix.lower(),
                    "size_bytes": info.file_size,
                    "classification": classification,
                    "status": status,
                })

                if not actual_raw:
                    continue

                idx_row, rows, digest, byte_count = qc_one_recording(
                    zf, zip_path.name, member
                )

                index_rows.append(idx_row)
                channel_rows.extend(rows)

                digest_to_files.setdefault(digest, []).append(
                    (zip_path.name, member, byte_count)
                )

                counts["parsed_raw_recordings"] += 1

        print(
            f"  Completed {zip_path.name}; "
            f"raw recordings parsed so far: "
            f"{counts['parsed_raw_recordings']}",
            flush=True,
        )

    # Duplicate detection using SHA-256 of actual raw recordings only.
    duplicate_rows = []

    for digest, files in digest_to_files.items():
        if len(files) > 1:
            for archive, member, size in files:
                duplicate_rows.append({
                    "sha256": digest,
                    "duplicate_group_size": len(files),
                    "archive": archive,
                    "raw_file": member,
                    "size_bytes": size,
                })

    inventory = pd.DataFrame(inventory_rows)
    channels = pd.DataFrame(channel_rows)
    indices = pd.DataFrame(index_rows)

    duplicates = pd.DataFrame(
        duplicate_rows,
        columns=[
            "sha256",
            "duplicate_group_size",
            "archive",
            "raw_file",
            "size_bytes",
        ],
    )

    inventory.to_csv(
        out_dir / "archive_contents_inventory_corrected.csv",
        index=False,
    )
    channels.to_csv(
        out_dir / "all_channel_qc_corrected.csv",
        index=False,
    )
    indices.to_csv(
        out_dir / "all_sample_index_qc_corrected.csv",
        index=False,
    )
    duplicates.to_csv(
        out_dir / "raw_duplicate_hashes_corrected.csv",
        index=False,
    )

    summary = {
        "input_directory": str(raw_dir),
        "zip_archives_found": len(zips),
        "archive_file_members_in_inventory": len(inventory),
        "macOS_sidecar_files_excluded": int(
            counts["excluded_sidecars"]
        ),
        "actual_raw_recordings_parsed": int(
            counts["parsed_raw_recordings"]
        ),
        "channel_qc_rows": int(len(channels)),
        "sample_index_qc_rows": int(len(indices)),
        "raw_recordings_with_missing_or_nonnumeric_channel_values": (
            int(
                channels.groupby(["archive", "raw_file"])[
                    "missing_or_non_numeric"
                ].sum().gt(0).sum()
            )
            if len(channels) else 0
        ),
        "total_missing_or_nonnumeric_channel_values": (
            int(channels["missing_or_non_numeric"].sum())
            if len(channels) else 0
        ),
        "total_infinite_channel_values": (
            int(channels["infinite_values"].sum())
            if len(channels) else 0
        ),
        "flat_channel_records": (
            int(channels["possible_flat_channel"].sum())
            if len(channels) else 0
        ),
        "channel_records_with_repeated_consecutive_fraction_ge_0_99": (
            int(
                (channels["repeated_consecutive_fraction"] >= 0.99).sum()
            )
            if len(channels) else 0
        ),
        "raw_files_in_duplicate_hash_groups": len(duplicates),
        "duplicate_hash_groups": (
            int(duplicates["sha256"].nunique())
            if len(duplicates) else 0
        ),
        "note": (
            "Sidecars are excluded from raw parsing and hash comparisons. "
            "Inventory retains excluded entries for traceability."
        ),
    }

    pd.DataFrame([summary]).to_csv(
        out_dir / "raw_zip_audit_summary_corrected.csv",
        index=False,
    )

    print("\nRAW ZIP AUDIT COMPLETE")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nOutput directory: {out_dir}")
    print("Reports saved successfully.")


if __name__ == "__main__":
    main()