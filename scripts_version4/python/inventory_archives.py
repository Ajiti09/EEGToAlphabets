from pathlib import Path
from zipfile import ZipFile, BadZipFile
import csv

ROOT = Path(r"D:\EEGToAlphabets")
RAW_DIR = ROOT / "raw data_version4"
OUTPUT = ROOT / "archive_contents_inventory.csv"

def classify_file(name):
    lower = name.lower()
    if "openbci" in lower and "raw" in lower and lower.endswith(".txt"):
        return "Raw EEG recording"
    if "training" in lower and "segment" in lower and lower.endswith(".csv"):
        return "Training-segment CSV"
    if "segment" in lower and lower.endswith(".csv"):
        return "Possible segment CSV"
    if any(word in lower for word in ("event", "marker", "trigger")):
        return "Possible event/marker log"
    return "Other / needs review"

rows = []

for archive_path in sorted(RAW_DIR.glob("*.zip")):
    try:
        with ZipFile(archive_path, "r") as archive:
            members = archive.infolist()

            if not members:
                rows.append({
                    "archive": archive_path.name,
                    "member_path": "",
                    "extension": "",
                    "size_bytes": 0,
                    "classification": "EMPTY ARCHIVE",
                    "status": "Review"
                })
                continue

            for member in members:
                if member.is_dir():
                    continue

                member_path = member.filename
                extension = Path(member_path).suffix.lower()

                rows.append({
                    "archive": archive_path.name,
                    "member_path": member_path,
                    "extension": extension,
                    "size_bytes": member.file_size,
                    "classification": classify_file(member_path),
                    "status": "Review" if classify_file(member_path).startswith("Possible")
                              or classify_file(member_path) == "Other / needs review"
                              else "Classified by filename"
                })

    except BadZipFile:
        rows.append({
            "archive": archive_path.name,
            "member_path": "",
            "extension": "",
            "size_bytes": "",
            "classification": "INVALID ZIP",
            "status": "Investigate"
        })

fieldnames = [
    "archive", "member_path", "extension",
    "size_bytes", "classification", "status"
]

with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Archives found: {len(list(RAW_DIR.glob('*.zip')))}")
print(f"Files indexed: {sum(1 for r in rows if r['member_path'])}")
print(f"Inventory saved to: {OUTPUT}")



import re
from collections import Counter

summary_counts = Counter()

for row in rows:
    path = row["member_path"]
    category = row["classification"]
    archive_name = row["archive"]

    if not path:
        continue

    # Exclude macOS metadata files and directories.
    if "__MACOSX" in path or "/._" in path or path.startswith("._"):
        continue

    # Participant from archive: TEST_SUBJECT_1.zip -> P01
    archive_match = re.search(
        r"TEST_SUBJECT_(\d+)\.zip",
        archive_name,
        re.IGNORECASE
    )

    # Class from session folder name.
    # Example: ...TEST_SUBJECT_1_A/OpenBCI-RAW-....txt
    session_match = re.search(
        r"TEST_SUBJECT_\d+_([AKNPZ])(?:/|\\|$)",
        path,
        re.IGNORECASE
    )

    # Participant/class from event or training CSV filename.
    # Example: P01_A_event_log_....csv
    file_match = re.search(
        r"P(\d+)_([AKNPZ])_(?:event_log|training_segments)",
        path,
        re.IGNORECASE
    )

    if category == "Raw EEG recording":
        if not archive_match or not session_match:
            continue

        participant = f"P{int(archive_match.group(1)):02d}"
        letter = session_match.group(1).upper()
        kind = "raw_eeg"

    elif category in (
        "Possible event/marker log",
        "Training-segment CSV"
    ):
        if not file_match:
            continue

        participant = f"P{int(file_match.group(1)):02d}"
        letter = file_match.group(2).upper()

        if category == "Possible event/marker log":
            kind = "event_log"
        else:
            kind = "training_segments"

    else:
        continue

    summary_counts[(participant, letter, kind)] += 1

summary_path = ROOT / "participant_class_summary.csv"

with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "participant", "class",
        "raw_eeg_count",
        "event_log_count",
        "training_segments_count"
    ])

    for participant_num in range(1, 21):
        participant = f"P{participant_num:02d}"

        for letter in ["A", "K", "N", "P", "Z"]:
            writer.writerow([
                participant,
                letter,
                summary_counts[(participant, letter, "raw_eeg")],
                summary_counts[(participant, letter, "event_log")],
                summary_counts[(participant, letter, "training_segments")]
            ])

print(f"Participant/class summary saved to: {summary_path}")
