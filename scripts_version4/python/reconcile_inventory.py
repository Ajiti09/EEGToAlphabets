
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
import csv
import io
import re

ROOT = Path(r"D:\EEGToAlphabets")
RAW_DIR = ROOT / "raw data_version4"

DETAILS_OUT = ROOT / "session_file_details.csv"
SUMMARY_OUT = ROOT / "session_inventory_reconciled.csv"

CLASSES = ["A", "K", "N", "P", "Z"]
counts = defaultdict(lambda: {
    "raw_eeg": 0,
    "event_log": 0,
    "training_segments": 0,
    "csv_participant_ids": set(),
    "issues": []
})
details = []

# Handles ordinary and inconsistent names such as:
# TEST_SUBJECT_3_Z
# TEST_SUBJECT _ 3_Z
session_pattern = re.compile(
    r"TEST_SUBJECT\s*_\s*(\d+)_([AKNPZ])(?:/|\\|$)",
    re.IGNORECASE
)

for archive_path in sorted(RAW_DIR.glob("*.zip")):
    archive_match = re.fullmatch(
        r"TEST_SUBJECT_(\d+)\.zip",
        archive_path.name,
        re.IGNORECASE
    )
    if not archive_match:
        continue

    archive_subject = int(archive_match.group(1))

    with ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            path = member.filename
            base = Path(path).name

            # Ignore macOS metadata/sidecar files.
            if "__MACOSX" in path or base.startswith("._"):
                continue

            match = session_pattern.search(path)
            if not match:
                continue

            folder_subject = int(match.group(1))
            label = match.group(2).upper()
            key = (folder_subject, label)

            lower = base.lower()
            if "openbci-raw" in lower and lower.endswith(".txt"):
                kind = "raw_eeg"
            elif "event_log" in lower and lower.endswith(".csv"):
                kind = "event_log"
            elif "training_segments" in lower and lower.endswith(".csv"):
                kind = "training_segments"
            else:
                continue

            participant_ids = set()
            read_status = "Not applicable"

            # Read participant_id values from CSV content.
            if kind in {"event_log", "training_segments"}:
                try:
                    with archive.open(member, "r") as binary_file:
                        text_file = io.TextIOWrapper(
                            binary_file,
                            encoding="utf-8-sig",
                            errors="replace"
                        )
                        reader = csv.DictReader(text_file)
                        if reader.fieldnames and "participant_id" in reader.fieldnames:
                            for row in reader:
                                value = (row.get("participant_id") or "").strip()
                                if value:
                                    participant_ids.add(value.upper())
                            read_status = "OK"
                        else:
                            read_status = "participant_id column missing"
                except Exception as exc:
                    read_status = f"CSV read error: {exc}"

                counts[key]["csv_participant_ids"].update(participant_ids)

            counts[key][kind] += 1

            issues = []
            if archive_subject != folder_subject:
                issues.append("Archive/folder participant mismatch")

            if participant_ids:
                expected_id = f"P{folder_subject:02d}"
                if participant_ids != {expected_id}:
                    issues.append(
                        "CSV participant_id differs from session folder: "
                        + ",".join(sorted(participant_ids))
                    )

            counts[key]["issues"].extend(issues)

            details.append({
                "archive": archive_path.name,
                "archive_subject": f"P{archive_subject:02d}",
                "folder_subject": f"P{folder_subject:02d}",
                "class": label,
                "file_type": kind,
                "member_path": path,
                "csv_participant_ids": ";".join(sorted(participant_ids)),
                "read_status": read_status,
                "issues": "; ".join(issues)
            })

with DETAILS_OUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "archive", "archive_subject", "folder_subject", "class",
        "file_type", "member_path", "csv_participant_ids",
        "read_status", "issues"
    ])
    writer.writeheader()
    writer.writerows(details)

with SUMMARY_OUT.open("w", newline="", encoding="utf-8-sig") as f:
    fields = [
        "participant", "class", "raw_eeg_count",
        "event_log_count", "training_segments_count",
        "csv_participant_ids", "issues"
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for subject in range(1, 21):
        for label in CLASSES:
            item = counts[(subject, label)]
            writer.writerow({
                "participant": f"P{subject:02d}",
                "class": label,
                "raw_eeg_count": item["raw_eeg"],
                "event_log_count": item["event_log"],
                "training_segments_count": item["training_segments"],
                "csv_participant_ids": ";".join(
                    sorted(item["csv_participant_ids"])
                ),
                "issues": "; ".join(sorted(set(item["issues"])))
            })

print("File details:", DETAILS_OUT)
print("Reconciled session summary:", SUMMARY_OUT)
print("Files classified:", len(details))
print("Expected participant-class combinations:", 20 * len(CLASSES))
