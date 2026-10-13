
# from pathlib import Path
# from zipfile import ZipFile
# import csv
# import io

# zip_path = Path(
#     r"D:\EEGToAlphabets\raw data_version4\TEST_SUBJECT_8.zip"
# )

# search_text = "20260511_115558"

# if not zip_path.exists():
#     raise FileNotFoundError(f"ZIP archive not found: {zip_path}")

# with ZipFile(zip_path, "r") as archive:
#     # Find the files using the timestamp, without assuming
#     # a particular folder separator or participant prefix.
#     matches = [
#         name for name in archive.namelist()
#         if search_text in name and name.lower().endswith(".csv")
#         and "__MACOSX" not in name
#     ]

#     print(f"Archive: {zip_path.name}")
#     print(f"Matching CSV files: {len(matches)}")

#     if not matches:
#         print("\nCSV files containing 'Z' or the target date:")
#         for name in archive.namelist():
#             if name.lower().endswith(".csv") and (
#                 "20260511" in name or "_z_" in name.lower()
#             ):
#                 print(name)

#     for name in matches:
#         print("\n" + "=" * 90)
#         print(f"FILE: {name}")
#         print("=" * 90)

#         # Read the CSV directly from the ZIP; no extraction.
#         raw_bytes = archive.read(name)
#         text = raw_bytes.decode("utf-8-sig", errors="replace")

#         reader = csv.reader(io.StringIO(text))
#         rows = list(reader)

#         if not rows:
#             print("File is empty.")
#             continue

#         print("HEADERS:")
#         print(rows[0])

#         print("\nFIRST 5 DATA ROWS:")
#         for row in rows[1:6]:
#             print(row)

#         print(f"\nTotal data rows: {max(0, len(rows) - 1)}")

# from pathlib import Path
# from zipfile import ZipFile

# zip_path = Path(
#     r"D:\EEGToAlphabets\raw data_version4\TEST_SUBJECT_8.zip"
# )

# with ZipFile(zip_path, "r") as archive:
#     names = archive.namelist()

#     print("ZIP:", zip_path.name)
#     print("Total entries:", len(names))

#     print("\nAll entries containing TEST_SUBJECT_4:")
#     for name in names:
#         if "TEST_SUBJECT_4" in name and "__MACOSX" not in name:
#             print(name)

#     print("\nAll CSV entries containing 20260511:")
#     for name in names:
#         if name.lower().endswith(".csv") and "20260511" in name:
#             print(name)


# from pathlib import Path
# from zipfile import ZipFile, BadZipFile

# root = Path(r"D:\EEGToAlphabets\raw data_version4")

# terms = [
#     "20260511_115558",
#     "OpenBCI-RAW-2026-05-11_11-56-02",
#     "P08_Z_event_log",
#     "P08_Z_training_segments",
# ]

# found_any = False

# for zip_path in sorted(root.glob("*.zip")):
#     try:
#         with ZipFile(zip_path, "r") as archive:
#             matches = [
#                 name for name in archive.namelist()
#                 if any(term.lower() in name.lower() for term in terms)
#                 and "__MACOSX" not in name
#             ]

#             if matches:
#                 found_any = True
#                 print(f"\n=== ARCHIVE: {zip_path.name} ===")
#                 for name in matches:
#                     print(name)

#     except BadZipFile:
#         print(f"WARNING: Invalid ZIP: {zip_path.name}")

# if not found_any:
#     print("No matching files found in any participant ZIP.")


# from pathlib import Path
# from zipfile import ZipFile
# import hashlib
# import csv
# import io

# root = Path(r"D:\EEGToAlphabets\raw data_version4")

# candidates = {
#     "TEST_SUBJECT_10.zip": "20260511_115558",
#     "TEST_SUBJECT_4.zip": "20260511_115558",
# }

# def sha256_from_zip(archive, member):
#     digest = hashlib.sha256()
#     with archive.open(member) as f:
#         for chunk in iter(lambda: f.read(1024 * 1024), b""):
#             digest.update(chunk)
#     return digest.hexdigest()

# for archive_name, timestamp in candidates.items():
#     archive_path = root / archive_name
#     print(f"\n{'=' * 100}")
#     print(f"ARCHIVE: {archive_name}")
#     print("=" * 100)

#     with ZipFile(archive_path, "r") as archive:
#         members = [
#             name for name in archive.namelist()
#             if timestamp in name
#             and "__MACOSX" not in name
#             and not name.endswith("/")
#         ]

#         for member in members:
#             if member.lower().endswith(".csv"):
#                 print(f"\n--- CSV: {member} ---")
#                 content = archive.read(member).decode(
#                     "utf-8-sig", errors="replace"
#                 )
#                 rows = list(csv.reader(io.StringIO(content)))

#                 if rows:
#                     print("Headers:", rows[0])
#                     print("First 3 data rows:")
#                     for row in rows[1:4]:
#                         print(row)
#                     print("Total data rows:", len(rows) - 1)

#             elif "OpenBCI-RAW-" in member:
#                 print(f"\n--- RAW EEG: {member} ---")
#                 print("Size:", archive.getinfo(member).file_size)
#                 print("SHA-256:", sha256_from_zip(archive, member))


# from pathlib import Path
# from zipfile import ZipFile
# import hashlib

# root = Path(r"D:\EEGToAlphabets\raw data_version4")

# sessions = {
#     "P10": {
#         "zip": "TEST_SUBJECT_10.zip",
#         "folder": "TEST_SUBJECT_10/OpenBCISession_INNER_SPEECH_TEST_SUBJECT_10_Z/",
#         "csv_prefix": "P10_Z_",
#     },
#     "P04": {
#         "zip": "TEST_SUBJECT_4.zip",
#         "folder": "TEST_SUBJECT_4/OpenBCISession_INNER_SPEECH_TEST_SUBJECT_4_Z/",
#         "csv_prefix": "P08_Z_",
#     },
# }

# def sha256_bytes(data):
#     return hashlib.sha256(data).hexdigest()

# results = {}

# for key, info in sessions.items():
#     archive_path = root / info["zip"]
#     print(f"\n{'=' * 80}\nSESSION CANDIDATE: {key} | {archive_path.name}")

#     with ZipFile(archive_path, "r") as z:
#         # Locate entries by their session folder.
#         entries = [
#             n for n in z.namelist()
#             if n.startswith(info["folder"])
#             and "__MACOSX" not in n
#             and not n.endswith("/")
#         ]

#         results[key] = {}

#         for name in entries:
#             filename = name.rsplit("/", 1)[-1]

#             if filename.endswith(".csv") or filename.startswith("OpenBCI-RAW-"):
#                 data = z.read(name)
#                 digest = sha256_bytes(data)
#                 results[key][filename] = digest

#                 print(f"\nFile: {filename}")
#                 print(f"Bytes: {len(data):,}")
#                 print(f"SHA-256: {digest}")

# print("\n" + "=" * 80)
# print("CROSS-ARCHIVE COMPARISON")

# for key_a, key_b in [("P10", "P04")]:
#     files_a = results[key_a]
#     files_b = results[key_b]

#     for name_a, hash_a in files_a.items():
#         if name_a.startswith("OpenBCI-RAW-"):
#             for name_b, hash_b in files_b.items():
#                 if name_b.startswith("OpenBCI-RAW-"):
#                     print(
#                         f"\nRaw EEG comparison:\n"
#                         f"  {key_a}: {name_a}\n"
#                         f"  {key_b}: {name_b}\n"
#                         f"  Byte-identical: {hash_a == hash_b}"
#                     )

#     print("\nMatching CSV hashes:")
#     for name_a, hash_a in files_a.items():
#         if name_a.endswith(".csv"):
#             for name_b, hash_b in files_b.items():
#                 if name_b.endswith(".csv") and hash_a == hash_b:
#                     print(f"  IDENTICAL: {name_a} == {name_b}")

#     print("\nDifferent or unmatched CSV hashes:")
#     for name_a, hash_a in files_a.items():
#         if name_a.endswith(".csv"):
#             matches = [
#                 name_b for name_b, hash_b in files_b.items()
#                 if name_b.endswith(".csv") and hash_a == hash_b
#             ]
#             if not matches:
#                 print(f"  No identical counterpart: {name_a}")


from pathlib import Path
from zipfile import ZipFile
import csv
import io

root = Path(r"D:\EEGToAlphabets\raw data_version4")

candidates = {
    "P10": {
        "zip": "TEST_SUBJECT_10.zip",
        "prefix": "TEST_SUBJECT_10/OpenBCISession_INNER_SPEECH_TEST_SUBJECT_10_Z/",
        "event": "P10_Z_event_log_20260511_115558.csv",
        "segments": "P10_Z_training_segments_20260511_115558.csv",
    },
    "P04": {
        "zip": "TEST_SUBJECT_4.zip",
        "prefix": "TEST_SUBJECT_4/OpenBCISession_INNER_SPEECH_TEST_SUBJECT_4_Z/",
        "event": "P08_Z_event_log_20260511_115558.csv",
        "segments": "P08_Z_training_segments_20260511_115558.csv",
    },
}

def load_csv(zip_name, member):
    with ZipFile(root / zip_name, "r") as archive:
        content = archive.read(member).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(content)))

def compare_csv(kind):
    info_a = candidates["P10"]
    info_b = candidates["P04"]

    rows_a = load_csv(
        info_a["zip"], info_a["prefix"] + info_a[kind]
    )
    rows_b = load_csv(
        info_b["zip"], info_b["prefix"] + info_b[kind]
    )

    print(f"\n{'=' * 80}\nCOMPARE: {kind}")
    print(f"P10 rows: {len(rows_a)} | P04 rows: {len(rows_b)}")

    if len(rows_a) != len(rows_b):
        print("WARNING: Row counts differ.")

    differences = []

    for i, (a, b) in enumerate(zip(rows_a, rows_b), start=1):
        keys = sorted((set(a) | set(b)) - {"participant_id"})
        for key in keys:
            va, vb = a.get(key), b.get(key)
            if va != vb:
                differences.append((i, key, va, vb))

    print("Differences excluding participant_id:", len(differences))

    for row_num, column, va, vb in differences[:20]:
        print(
            f"Row {row_num}, column {column}: "
            f"P10={va!r} | P04={vb!r}"
        )

    if not differences and len(rows_a) == len(rows_b):
        print("All compared fields match when participant_id is ignored.")

compare_csv("event")
compare_csv("segments")
