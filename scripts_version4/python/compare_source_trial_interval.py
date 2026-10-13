
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import re
import numpy as np
import pandas as pd

ROOT = Path(r"D:\EEGToAlphabets")
RAW_DIR = ROOT / "raw data_version4"
DATA_DIR = ROOT / "preprocessed_data_version4"
OUT_DIR = ROOT / "results_version4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS_DEFAULT = 125.0
WINDOW_SAMPLES = 8
WINDOW_SEC = WINDOW_SAMPLES / FS_DEFAULT
TOL = 1e-6

meta_path = DATA_DIR / "metadata.csv"
if not meta_path.exists():
    raise FileNotFoundError(
        f"metadata.csv not found at {meta_path}. "
        "Update DATA_DIR to the folder containing your processed metadata."
    )

meta = pd.read_csv(meta_path)

required_meta = {
    "participant", "label", "trial", "trial_uid", "window_id",
    "fs", "task_start_used", "task_end_used", "raw_path"
}
missing = required_meta - set(meta.columns)
if missing:
    raise ValueError(f"metadata.csv is missing columns: {sorted(missing)}")

for col in ["trial", "window_id", "fs", "task_start_used", "task_end_used"]:
    meta[col] = pd.to_numeric(meta[col], errors="coerce")

meta["participant"] = meta["participant"].astype(str).str.strip()
meta["label"] = meta["label"].astype(str).str.strip().str.upper()
meta["trial_uid"] = meta["trial_uid"].astype(str).str.strip()

report = []
seen_source_sessions = set()


def add_issue(row, category, detail):
    row["status"] = "REVIEW"
    row["issues"].append(f"{category}: {detail}")


def parse_participant(path):
    match = re.search(r"TEST[_ -]*SUBJECT[_ -]*(\d+)", path, re.I)
    if match:
        return f"P{int(match.group(1)):02d}"
    return None


def parse_class(path):
    parts = re.split(r"[/\\]", path)
    for part in reversed(parts):
        cleaned = re.sub(r"[^A-Za-z0-9]", " ", part).upper().split()
        for token in reversed(cleaned):
            if token in {"A", "K", "N", "P", "Z"}:
                return token
    return None


def read_csv_member(zf, name):
    with zf.open(name) as f:
        return pd.read_csv(BytesIO(f.read()))


for zip_path in sorted(RAW_DIR.glob("*.zip")):
    with ZipFile(zip_path) as zf:
        members = zf.namelist()

        raw_members = [
            m for m in members
            if Path(m).name.startswith("OpenBCI-RAW-")
            and m.lower().endswith(".txt")
            and "__MACOSX" not in m
        ]

        for raw_member in raw_members:
            folder = str(Path(raw_member).parent).replace("\\", "/")
            folder_members = [
                m for m in members
                if str(Path(m).parent).replace("\\", "/") == folder
            ]

            segment_files = [
                m for m in folder_members
                if "training_segments" in Path(m).name.lower()
                and m.lower().endswith(".csv")
            ]

            if len(segment_files) != 1:
                continue

            participant = parse_participant(raw_member)
            session_label = parse_class(raw_member)

            if not participant or not session_label:
                continue

            session_key = (zip_path.name, folder)
            seen_source_sessions.add(session_key)

            seg = read_csv_member(zf, segment_files[0])

            needed = {
                "participant_id", "trial", "label", "segment_type",
                "start_time_sec", "end_time_sec", "use_for_training"
            }
            if not needed.issubset(seg.columns):
                continue

            seg["trial_num"] = pd.to_numeric(seg["trial"], errors="coerce")
            seg["start"] = pd.to_numeric(seg["start_time_sec"], errors="coerce")
            seg["end"] = pd.to_numeric(seg["end_time_sec"], errors="coerce")
            seg["label_norm"] = seg["label"].astype(str).str.strip().str.upper()
            seg["type_norm"] = seg["segment_type"].astype(str).str.strip().str.lower()
            seg["use_num"] = pd.to_numeric(seg["use_for_training"], errors="coerce")

            tasks = seg[
                (seg["type_norm"] == "task")
                & (seg["use_num"] == 1)
                & (seg["label_norm"].isin(["A", "K", "N", "P", "Z"]))
            ]

            # Check source participant ID against participant inferred from path.
            source_pids = set(
                seg["participant_id"].astype(str).str.strip()
            ) - {""}

            if source_pids != {participant}:
                pid_note = (
                    f"Source participant_id={sorted(source_pids)}; "
                    f"recording path implies {participant}"
                )
            else:
                pid_note = None

            for _, source in tasks.iterrows():
                trial = source["trial_num"]
                label = source["label_norm"]
                start = source["start"]
                end = source["end"]

                row = {
                    "archive": zip_path.name,
                    "source_folder": folder,
                    "raw_file": raw_member,
                    "segments_file": segment_files[0],
                    "participant_from_path": participant,
                    "participant_in_source_csv": ";".join(sorted(source_pids)),
                    "session_class_from_path": session_label,
                    "source_label": label,
                    "source_trial": trial,
                    "source_start_sec": start,
                    "source_end_sec": end,
                    "source_duration_sec": end - start
                    if pd.notna(start) and pd.notna(end) else np.nan,
                    "expected_window_count": np.nan,
                    "observed_window_count": 0,
                    "observed_first_start_sec": np.nan,
                    "observed_last_end_sec": np.nan,
                    "start_difference_sec": np.nan,
                    "end_difference_sec": np.nan,
                    "sample_count_expected_per_window": WINDOW_SAMPLES,
                    "trial_uid_values": "",
                    "status": "PASS",
                    "issues": [],
                }

                if pid_note:
                    add_issue(row, "PARTICIPANT_ID", pid_note)

                if label != session_label:
                    add_issue(
                        row, "CLASS_PATH_MISMATCH",
                        f"Source label {label} differs from path class {session_label}"
                    )

                if pd.isna(trial) or pd.isna(start) or pd.isna(end) or end <= start:
                    add_issue(row, "INVALID_SOURCE_INTERVAL",
                              "Missing trial/time or end is not after start")
                    row["issues"] = " | ".join(row["issues"])
                    report.append(row)
                    continue

                # Mirror makeTaskWindows(): 8 samples, 8-sample step.
                fs = FS_DEFAULT
                duration = end - start
                step = WINDOW_SAMPLES / fs
                if duration < step:
                    expected_starts = np.array([])
                else:
                    expected_starts = np.arange(
                        start, end - step + 1e-9, step
                    )

                row["expected_window_count"] = len(expected_starts)

                # Match processed trial by path-inferred participant, class and trial.
                # The actual stored raw_path is also checked below.
                candidates = meta[
                    (meta["participant"] == participant)
                    & (meta["label"] == label)
                    & (meta["trial"] == trial)
                ].copy()

                if candidates.empty:
                    add_issue(row, "PROCESSED_TRIAL_MISSING",
                              "No matching participant/class/trial in metadata.csv")
                    row["issues"] = " | ".join(row["issues"])
                    report.append(row)
                    continue

                uids = sorted(candidates["trial_uid"].dropna().unique().tolist())
                row["trial_uid_values"] = ";".join(uids)

                if len(uids) != 1:
                    add_issue(row, "UID_AMBIGUITY",
                              f"Expected one trial_uid, found {uids}")

                # Verify that candidate processed rows refer to this source recording.
                paths = candidates["raw_path"].astype(str).unique().tolist()
                raw_name = Path(raw_member).name
                if not any(raw_name in p for p in paths):
                    add_issue(
                        row, "RAW_PATH_REFERENCE",
                        f"Processed raw_path does not contain source filename {raw_name}"
                    )

                if candidates["window_id"].isna().any():
                    add_issue(row, "MISSING_WINDOW_ID",
                              "One or more processed windows lack window_id")

                window_ids = sorted(
                    candidates["window_id"].dropna().astype(int).unique().tolist()
                )
                if window_ids != list(range(1, len(window_ids) + 1)):
                    add_issue(row, "WINDOW_ID_SEQUENCE",
                              f"Observed window IDs: {window_ids[:10]}...")

                row["observed_window_count"] = len(candidates)

                # Source interval expected to match task_start_used/task_end_used
                # as a complete interval, not each individual window.
                starts = candidates["task_start_used"].dropna().to_numpy(float)
                ends = candidates["task_end_used"].dropna().to_numpy(float)

                if len(starts) != len(candidates) or len(ends) != len(candidates):
                    add_issue(row, "MISSING_USED_TIMES",
                              "Processed task start/end values are missing")
                elif len(starts):
                    row["observed_first_start_sec"] = starts.min()
                    row["observed_last_end_sec"] = ends.max()

                    # The first window should start at the source task start.
                    row["start_difference_sec"] = starts.min() - start

                    # Last window ends at the end of the last complete window.
                    expected_last_end = (
                        expected_starts[-1] + step
                        if len(expected_starts) else np.nan
                    )
                    row["end_difference_sec"] = (
                        ends.max() - expected_last_end
                        if pd.notna(expected_last_end) else np.nan
                    )

                    if len(starts) != row["expected_window_count"]:
                        add_issue(
                            row, "WINDOW_COUNT_MISMATCH",
                            f"Expected {row['expected_window_count']}; "
                            f"metadata has {len(starts)} rows"
                        )

                    if len(expected_starts):
                        if abs(starts.min() - expected_starts[0]) > TOL:
                            add_issue(row, "FIRST_WINDOW_START",
                                      "First processed start differs from expected start")

                        if abs(ends.max() - expected_last_end) > TOL:
                            add_issue(row, "LAST_WINDOW_END",
                                      "Last processed end differs from last full window end")

                        # Compare every processed window to its expected interval.
                        ordered = candidates.sort_values("window_id")
                        if len(ordered) == len(expected_starts):
                            observed_starts = ordered["task_start_used"].to_numpy(float)
                            observed_ends = ordered["task_end_used"].to_numpy(float)

                            if not np.allclose(
                                observed_starts, expected_starts,
                                atol=TOL, rtol=0
                            ):
                                add_issue(row, "WINDOW_START_SEQUENCE",
                                          "Processed window starts differ from expected sequence")

                            expected_ends = expected_starts + step
                            if not np.allclose(
                                observed_ends, expected_ends,
                                atol=TOL, rtol=0
                            ):
                                add_issue(row, "WINDOW_END_SEQUENCE",
                                          "Processed window ends differ from expected sequence")

                if len(candidates) != row["expected_window_count"]:
                    add_issue(
                        row, "PROCESSED_ROW_COUNT",
                        f"Expected {row['expected_window_count']} rows; "
                        f"found {len(candidates)}"
                    )

                row["issues"] = " | ".join(row["issues"])
                report.append(row)

report_df = pd.DataFrame(report)
out_path = OUT_DIR / "task_1_2_source_processed_interval_comparison.csv"
report_df.to_csv(out_path, index=False)

print("Source sessions discovered:", len(seen_source_sessions))
print("Task segments compared:", len(report_df))
print("PASS:", int((report_df["status"] == "PASS").sum()) if len(report_df) else 0)
print("REVIEW:", int((report_df["status"] == "REVIEW").sum()) if len(report_df) else 0)
print("Report:", out_path)

if len(report_df):
    print("\nIssues by category:")
    issue_counts = (
        report_df.loc[report_df["issues"].ne(""), "issues"]
        .str.split(" | ", regex=False)
        .explode()
        .value_counts()
    )
    print(issue_counts.to_string() if len(issue_counts) else "No issues detected.")

    print("\nFirst 25 review rows:")
    cols = [
        "participant_from_path", "session_class_from_path",
        "source_trial", "source_label", "expected_window_count",
        "observed_window_count", "status", "issues"
    ]
    print(
        report_df.loc[report_df["status"] == "REVIEW", cols]
        .head(25).to_string(index=False)
    )
