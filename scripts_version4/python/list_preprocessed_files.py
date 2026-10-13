from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\EEGToAlphabets")
DATA_DIR = ROOT / "preprocessed_data_version4"
OUTPUT = ROOT / "preprocessed_file_listing.csv"

rows = []

for path in sorted(DATA_DIR.rglob("*")):
    if path.is_file():
        rows.append({
            "relative_path": str(path.relative_to(ROOT)),
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size
        })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print(f"Files found: {len(df)}")
print(df.to_string(index=False))
print(f"\nSaved to: {OUTPUT}")