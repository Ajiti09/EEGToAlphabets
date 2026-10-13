
from pathlib import Path
from zipfile import ZipFile
import csv
import io
import re

ROOT = Path(r"D:\EEGToAlphabets")
RAW_DIR = ROOT / "raw data_version4"

PATH_REPORT = ROOT / "z_session_path_audit.csv"
CSV_REPORT = ROOT / "z_session_csv_content_audit.csv"

targets = {3, 4, 8}
path_rows = []
csv_rows = []

for archive_path in sorted(RAW_DIR.glob("*.zip")):
    match = re.fullmatch(
        r"TEST_SUBJECT_(\d+)\.zip",
        archive_path.name,
        re.IGNORECASE
    )

    if not match or int(match.group(1)) not in targets:
        continue

    archive_subject = int(match.group(1))

    with ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            name = member.filename
            base = Path(name).name

            # Skip macOS sidecar files and directories.
            if "__MACOSX" in name or base.startswith("._"):
                continue

            # Focus on Z session folders and Z-related files.
            if not re.search(r"_Z(?:/|\\|_|\.)", name, re.IGNORECASE):
                continue

            if not (
                "OpenBCI-RAW" in base
                or "event_log" in base.lower()
                or "training_segments" in base.lower()
            ):
                continue

            path_rows.append({
                "archive": archive_path.name,
                "archive_subject": f"P{archive_subject:02d}",
                "member_path": name,
                "filename": base,
                "size_bytes": member.file_size,
            })

            if not base.lower().endswith(".csv"):
                continue

            try:
                with archive.open(member, "r") as f:
                    text = io.TextIOWrapper(
                        f, encoding="utf-8-sig", errors="replace"
                    )
                    reader = csv.reader(text)
                    rows = list(reader)

                header = rows[0] if rows else []
                data_rows = rows[1:] if rows else []

                # Capture a small sample for inspection.
                sample = " | ".join(
                    ",".join(row) for row in data_rows[:3]
                )

                csv_rows.append({
                    "archive": archive_path.name,
                    "archive_subject": f"P{archive_subject:02d}",
                    "member_path": name,
                    "filename": base,
                    "header": " | ".join(header),
                    "data_row_count": len(data_rows),
                    "first_3_data_rows": sample,
                    "read_status": "OK",
                })

            except Exception as exc:
                csv_rows.append({
                    "archive": archive_path.name,
                    "archive_subject": f"P{archive_subject:02d}",
                    "member_path": name,
                    "filename": base,
                    "header": "",
                    "data_row_count": "",
                    "first_3_data_rows": "",
                    "read_status": f"ERROR: {exc}",
                })

with PATH_REPORT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "archive", "archive_subject", "member_path",
        "filename", "size_bytes"
    ])
    writer.writeheader()
    writer.writerows(path_rows)

with CSV_REPORT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "archive", "archive_subject", "member_path", "filename",
        "header", "data_row_count", "first_3_data_rows",
        "read_status"
    ])
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"Z-session paths inspected: {len(path_rows)}")
print(f"CSV files inspected: {len(csv_rows)}")
print(f"Path report: {PATH_REPORT}")
print(f"CSV content report: {CSV_REPORT}")
