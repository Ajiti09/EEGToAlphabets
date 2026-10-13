
from pathlib import Path
import pandas as pd
import csv

ROOT = Path(r"D:\EEGToAlphabets")
META_DIR = ROOT / "metadata_version4"
OUTPUT = ROOT / "metadata_inventory.csv"

rows = []

for path in sorted(META_DIR.rglob("*")):
    if not path.is_file():
        continue

    record = {
        "file": str(path.relative_to(ROOT)),
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "rows": "",
        "columns_count": "",
        "column_names": "",
        "status": ""
    }

    if path.suffix.lower() == ".csv":
        try:
            # Read the header first, then count data rows.
            with path.open(
                "r", newline="", encoding="utf-8-sig"
            ) as f:
                reader = csv.reader(f)
                header = next(reader, [])
                row_count = sum(1 for _ in reader)

            record["rows"] = row_count
            record["columns_count"] = len(header)
            record["column_names"] = " | ".join(header)
            record["status"] = "Read successfully"

        except Exception as exc:
            record["status"] = f"Error: {exc}"
    else:
        record["status"] = "Non-CSV file; inspect separately"

    rows.append(record)

pd.DataFrame(rows).to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print(f"Metadata files inspected: {len(rows)}")
print(f"Inventory saved to: {OUTPUT}")
