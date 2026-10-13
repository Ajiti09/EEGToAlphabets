
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from io import BytesIO
import re
import pandas as pd

ROOT = Path(r"D:\EEGToAlphabets\raw data_version4")
OUT = Path(r"D:\EEGToAlphabets\results_version4")
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = {"A", "K", "N", "P", "Z"}
EXPECTED_TRIALS = 25
TOLERANCE_SEC = 0.02

issues = []
summaries = []
sessions = {}


def issue(session, category, detail, severity="REVIEW"):
    issues.append({
        "session": session,
        "category": category,
        "severity": severity,
        "detail": str(detail),
    })


def norm(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def trial_num(value):
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return None


def label_value(value):
    return norm(value).upper()


def read_csv_from_zip(zf, member):
    with zf.open(member) as f:
        return pd.read_csv(BytesIO(f.read()))


def add_session_file(zf, archive_name, member, kind):
    # The parent session directory is the primary pairing key.
    folder = str(Path(member).parent).replace("\\", "/")
    key = (archive_name, folder)
    sessions.setdefault(key, {})
    sessions[key][kind] = member


def get_task_rows(df):
    required = {
        "participant_id", "trial", "label", "segment_type",
        "start_time_sec", "end_time_sec", "duration_sec",
        "use_for_training"
    }
    missing = required - set(df.columns)
    if missing:
        return None, f"Missing columns: {sorted(missing)}"

    segment = df["segment_type"].astype(str).str.strip().str.lower()
    use = pd.to_numeric(df["use_for_training"], errors="coerce")
    task_mask = segment.eq("task") & use.eq(1)

    return df.loc[task_mask].copy(), None


# Discover source event logs and training-segment CSVs.
for zip_path in sorted(ROOT.glob("*.zip")):
    try:
        with ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if "__MACOSX" in member or not member.lower().endswith(".csv"):
                    continue

                name = Path(member).name.lower()

                if "event_log" in name:
                    add_session_file(
                        zf, zip_path.name, member, "event"
                    )
                elif "training_segments" in name:
                    add_session_file(
                        zf, zip_path.name, member, "segments"
                    )

    except BadZipFile:
        issue(zip_path.name, "INVALID_ARCHIVE", "Could not read ZIP")


print("Session folders discovered:", len(sessions))

for (archive_name, folder), files in sorted(sessions.items()):
    session = f"{archive_name} :: {folder}"

    if "event" not in files:
        issue(session, "MISSING_EVENT_LOG", "No event log found")
        continue

    if "segments" not in files:
        issue(session, "MISSING_SEGMENTS", "No training-segment file found")
        continue

    try:
        with ZipFile(ROOT / archive_name) as zf:
            ev = read_csv_from_zip(zf, files["event"])
            seg = read_csv_from_zip(zf, files["segments"])
    except Exception as exc:
        issue(session, "READ_ERROR", repr(exc))
        continue

    # Verify required event-log fields.
    event_required = {
        "participant_id", "trial", "label",
        "event", "time_from_start_sec"
    }
    missing = event_required - set(ev.columns)

    if missing:
        issue(session, "EVENT_COLUMNS", f"Missing columns: {sorted(missing)}")
        continue

    task, error = get_task_rows(seg)
    if error:
        issue(session, "SEGMENT_COLUMNS", error)
        continue

    # Normalize IDs, labels and trial numbers.
    ev["_pid"] = ev["participant_id"].map(norm)
    ev["_label"] = ev["label"].map(label_value)
    ev["_trial"] = ev["trial"].map(trial_num)
    ev["_event"] = ev["event"].map(norm).str.lower()
    ev["_time"] = pd.to_numeric(
        ev["time_from_start_sec"], errors="coerce"
    )

    seg["_pid"] = seg["participant_id"].map(norm)
    seg["_label"] = seg["label"].map(label_value)
    seg["_trial"] = seg["trial"].map(trial_num)
    seg["_start"] = pd.to_numeric(
        seg["start_time_sec"], errors="coerce"
    )
    seg["_end"] = pd.to_numeric(
        seg["end_time_sec"], errors="coerce"
    )
    seg["_duration"] = pd.to_numeric(
        seg["duration_sec"], errors="coerce"
    )

    task["_pid"] = task["participant_id"].map(norm)
    task["_label"] = task["label"].map(label_value)
    task["_trial"] = task["trial"].map(trial_num)
    task["_start"] = pd.to_numeric(
        task["start_time_sec"], errors="coerce"
    )
    task["_end"] = pd.to_numeric(
        task["end_time_sec"], errors="coerce"
    )
    task["_duration"] = pd.to_numeric(
        task["duration_sec"], errors="coerce"
    )

    # Participant identity consistency within the source files.
    event_pids = sorted(set(ev["_pid"]) - {""})
    segment_pids = sorted(set(seg["_pid"]) - {""})

    if len(event_pids) != 1:
        issue(session, "EVENT_PARTICIPANT",
              f"Event-log participant IDs: {event_pids}")

    if len(segment_pids) != 1:
        issue(session, "SEGMENT_PARTICIPANT",
              f"Segment participant IDs: {segment_pids}")

    if set(event_pids) != set(segment_pids):
        issue(session, "PARTICIPANT_MISMATCH",
              f"Event IDs {event_pids}; segment IDs {segment_pids}")

    # Every task label must be an expected class.
    for source, df in [("event", ev), ("segments", task)]:
        invalid = sorted(
            set(df["_label"]) - CLASSES - {""}
        )
        blank_count = int(df["_label"].eq("").sum())

        if invalid or blank_count:
            issue(session, "INVALID_LABEL",
                  f"{source}: invalid={invalid}, blank_rows={blank_count}")

    # Detect mixed labels for the same trial in event logs.
    for trial, group in ev.dropna(subset=["_trial"]).groupby("_trial"):
        labels = sorted(set(group["_label"]) - {""})
        if trial != 0 and len(labels) > 1:
            issue(session, "EVENT_MIXED_LABELS",
                  f"Trial {trial}: labels {labels}")

    # Expected number of task trials.
    task_trials = task["_trial"].dropna().astype(int)
    unique_trials = sorted(set(task_trials))
    if len(task) != EXPECTED_TRIALS:
        issue(session, "TASK_ROW_COUNT",
              f"Expected {EXPECTED_TRIALS} task rows; found {len(task)}")

    if len(unique_trials) != EXPECTED_TRIALS:
        issue(session, "UNIQUE_TASK_TRIAL_COUNT",
              f"Expected {EXPECTED_TRIALS} unique task trials; "
              f"found {len(unique_trials)}")

    if len(task_trials) != len(set(task_trials)):
        issue(session, "DUPLICATE_TASK_TRIAL",
              "A task trial occurs more than once")

    if unique_trials != list(range(1, EXPECTED_TRIALS + 1)):
        issue(session, "TRIAL_SEQUENCE",
              f"Observed task trial IDs: {unique_trials}")

    # Check boundaries, durations and overlap.
    for _, row in task.iterrows():
        t = row["_trial"]
        start, end, duration = row["_start"], row["_end"], row["_duration"]

        if pd.isna(start) or pd.isna(end) or pd.isna(duration):
            issue(session, "MISSING_BOUNDARY",
                  f"Trial {t}: missing/non-numeric start, end or duration")
            continue

        if end <= start:
            issue(session, "INVALID_BOUNDARY",
                  f"Trial {t}: start={start}, end={end}")

        if abs((end - start) - duration) > TOLERANCE_SEC:
            issue(session, "DURATION_MISMATCH",
                  f"Trial {t}: end-start={end-start:.4f}, "
                  f"recorded duration={duration:.4f}")

    ordered = task.dropna(subset=["_start", "_end"]).sort_values("_start")
    previous_end = None
    previous_trial = None

    for _, row in ordered.iterrows():
        if previous_end is not None and row["_start"] < previous_end - TOLERANCE_SEC:
            issue(session, "OVERLAPPING_TASKS",
                  f"Trials {previous_trial} and {row['_trial']} overlap")
        previous_end = row["_end"]
        previous_trial = row["_trial"]

    # Compare each task trial's label to its event-log records.
    event_boundary_checks = 0

    for _, row in task.iterrows():
        t = row["_trial"]
        if pd.isna(t):
            continue

        t = int(t)
        matching = ev[ev["_trial"] == t]

        if matching.empty:
            issue(session, "MISSING_TRIAL_EVENTS",
                  f"Task trial {t} has no event-log rows")
            continue

        event_labels = set(matching["_label"]) - {""}
        segment_label = row["_label"]

        if event_labels and event_labels != {segment_label}:
            issue(session, "LABEL_MISMATCH",
                  f"Trial {t}: segment label={segment_label}; "
                  f"event labels={sorted(event_labels)}")

        # Compare only explicit task-start/task-end events.
        start_events = matching[
            matching["_event"].str.contains("task", na=False)
            & matching["_event"].str.contains("start", na=False)
        ]
        end_events = matching[
            matching["_event"].str.contains("task", na=False)
            & matching["_event"].str.contains("end", na=False)
        ]

        if len(start_events) == 1 and len(end_events) == 1:
            event_start = start_events["_time"].iloc[0]
            event_end = end_events["_time"].iloc[0]

            if pd.notna(event_start) and pd.notna(event_end):
                event_boundary_checks += 1

                if abs(event_start - row["_start"]) > TOLERANCE_SEC:
                    issue(session, "START_BOUNDARY_MISMATCH",
                          f"Trial {t}: segment={row['_start']}; "
                          f"event={event_start}")

                if abs(event_end - row["_end"]) > TOLERANCE_SEC:
                    issue(session, "END_BOUNDARY_MISMATCH",
                          f"Trial {t}: segment={row['_end']}; "
                          f"event={event_end}")
        elif len(start_events) or len(end_events):
            issue(session, "AMBIGUOUS_BOUNDARY_EVENTS",
                  f"Trial {t}: task-start events={len(start_events)}, "
                  f"task-end events={len(end_events)}")

    # Summarize each paired session.
    summaries.append({
        "archive": archive_name,
        "session_folder": folder,
        "event_file": files["event"],
        "segment_file": files["segments"],
        "event_rows": len(ev),
        "training_segment_rows": len(seg),
        "task_rows": len(task),
        "unique_task_trials": len(unique_trials),
        "participant_ids_event": ";".join(event_pids),
        "participant_ids_segments": ";".join(segment_pids),
        "event_labels": ";".join(sorted(set(ev["_label"]) - {""})),
        "task_labels": ";".join(sorted(set(task["_label"]) - {""})),
        "explicit_boundary_pairs_checked": event_boundary_checks,
    })

summary_df = pd.DataFrame(summaries)
issues_df = pd.DataFrame(issues)

summary_path = OUT / "task_1_2_source_session_summary.csv"
issues_path = OUT / "task_1_2_source_discrepancies.csv"

summary_df.to_csv(summary_path, index=False)
issues_df.to_csv(issues_path, index=False)

print("\nSessions with both source files:", len(summary_df))
print("Discrepancies/issues recorded:", len(issues_df))
print("Summary:", summary_path)
print("Issues:", issues_path)

if not issues_df.empty:
    print("\nIssue counts by category:")
    print(issues_df.groupby(["category", "severity"]).size().to_string())
    print("\nFirst 30 issues:")
    print(issues_df.head(30).to_string(index=False))
else:
    print("No discrepancies were detected by these checks.")

print("\nIMPORTANT:")
print("- No source files were changed.")
print("- A boundary check is performed only where explicit task-start/task-end")
print("  events exist. Missing explicit events do not prove boundaries are correct.")
print("- Compare the resulting reports with the processed metadata before closing Task 1.2.")
