
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from collections import defaultdict
import hashlib

root = Path(r"D:\EEGToAlphabets\raw data_version4")

hash_groups = defaultdict(list)
raw_count = 0

def sha256_from_zip(archive, member):
    digest = hashlib.sha256()
    with archive.open(member) as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for zip_path in sorted(root.glob("*.zip")):
    try:
        with ZipFile(zip_path, "r") as archive:
            raw_files = [
                name for name in archive.namelist()
                if Path(name).name.startswith("OpenBCI-RAW-")
                and name.lower().endswith(".txt")
                and "__MACOSX" not in name
            ]

            for member in raw_files:
                raw_count += 1
                digest = sha256_from_zip(archive, member)
                hash_groups[digest].append({
                    "archive": zip_path.name,
                    "member": member,
                    "size": archive.getinfo(member).file_size,
                })

    except BadZipFile:
        print(f"WARNING: Invalid archive: {zip_path.name}")

duplicates = {
    digest: files
    for digest, files in hash_groups.items()
    if len(files) > 1
}

print("Raw EEG files scanned:", raw_count)
print("Unique raw EEG hashes:", len(hash_groups))
print("Duplicate hash groups:", len(duplicates))

for number, (digest, files) in enumerate(duplicates.items(), start=1):
    print(f"\n{'=' * 80}")
    print(f"DUPLICATE GROUP {number}")
    print("SHA-256:", digest)
    for item in files:
        print(f"Archive: {item['archive']}")
        print(f"Path:    {item['member']}")
        print(f"Bytes:   {item['size']:,}")
