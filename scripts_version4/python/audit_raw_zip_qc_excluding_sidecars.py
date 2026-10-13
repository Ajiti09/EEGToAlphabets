#!/usr/bin/env python3
"""Audit OpenBCI raw EEG files inside participant ZIP archives, excluding macOS sidecars.

Examples:
  python audit_raw_zip_qc_excluding_sidecars.py --raw-dir "D:\\EEGToAlphabets\\raw data_version4"
  python audit_raw_zip_qc_excluding_sidecars.py --raw-dir /mnt/data --output-dir /mnt/data/raw_zip_audit_corrected

Only actual OpenBCI-RAW*.txt recordings are parsed. ZIP entries are excluded if any path
component is __MACOSX or the basename starts with '._'. Original ZIPs are not modified.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import os
import zipfile
from pathlib import Path, PurePosixPath
from collections import Counter
import pandas as pd
import numpy as np

CHANNELS = [f"EXG Channel {i}" for i in range(16)]


def is_macos_sidecar(member: str) -> bool:
    p = PurePosixPath(member.replace('\\', '/'))
    return '__MACOSX' in p.parts or any(part.startswith('._') for part in p.parts)


def is_actual_raw(member: str) -> bool:
    if is_macos_sidecar(member):
        return False
    name = PurePosixPath(member.replace('\\', '/')).name
    return name.startswith('OpenBCI-RAW') and name.lower().endswith('.txt')


def sha256_member(zf: zipfile.ZipFile, member: str) -> str:
    h = hashlib.sha256()
    with zf.open(member) as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def qc_one_recording(zf: zipfile.ZipFile, archive_name: str, member: str):
    # Read each ZIP member once: reuse bytes for parsing and SHA-256 to avoid a second decompression pass.
    raw_bytes = zf.read(member)
    digest = hashlib.sha256(raw_bytes).hexdigest()
    # OpenBCI export has four metadata lines, then a CSV header.
    df = pd.read_csv(io.BytesIO(raw_bytes), skiprows=4, usecols=range(17), low_memory=False)
    expected = ['Sample Index'] + CHANNELS
    if list(df.columns) != expected:
        # Normalize harmless whitespace in headers, then enforce the expected schema.
        df.columns = [str(c).strip() for c in df.columns]
        if list(df.columns) != expected:
            raise ValueError(f"Unexpected first 17 columns in {archive_name}:{member}: {list(df.columns)}")

    sample = pd.to_numeric(df['Sample Index'], errors='coerce').to_numpy(dtype=float)
    n = len(df)
    idx_missing = int(np.isnan(sample).sum())
    diffs = np.diff(sample)
    # OpenBCI sample index wraps from 254 to 0; count these separately, not as data gaps.
    wraps = int(np.sum((diffs < 0) & (sample[:-1] >= 250) & (sample[1:] <= 5)))
    positive = diffs[diffs > 0]
    step_counts = Counter(positive.tolist())
    index_row = {
        'archive': archive_name, 'raw_file': member, 'rows': n,
        'sample_index_missing': idx_missing,
        'sample_index_min': float(np.nanmin(sample)) if np.isfinite(sample).any() else np.nan,
        'sample_index_max': float(np.nanmax(sample)) if np.isfinite(sample).any() else np.nan,
        'index_wrap_transitions': wraps,
        'positive_step_1': int(np.sum(positive == 1)),
        'positive_step_2': int(np.sum(positive == 2)),
        'positive_steps_gt_2': int(np.sum(positive > 2)),
        'positive_step_values': str(dict(sorted(step_counts.items()))),
        'nonpositive_nonwrap_steps': int(np.sum((diffs <= 0) & ~((diffs < 0) & (sample[:-1] >= 250) & (sample[1:] <= 5)))),
    }

    channel_rows = []
    for ch in CHANNELS:
        vals = pd.to_numeric(df[ch], errors='coerce').to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        nonnumeric_or_missing = int((~np.isfinite(vals)).sum())
        infinite = int(np.isinf(vals).sum())
        if len(vals) > 1:
            equal_adjacent = (vals[1:] == vals[:-1]) & np.isfinite(vals[1:]) & np.isfinite(vals[:-1])
            repeated_fraction = float(equal_adjacent.sum() / (len(vals) - 1))
            longest_run = 1
            run = 1
            for a, b in zip(vals[:-1], vals[1:]):
                if np.isfinite(a) and np.isfinite(b) and a == b:
                    run += 1
                    longest_run = max(longest_run, run)
                else:
                    run = 1
        else:
            repeated_fraction = np.nan
            longest_run = len(vals)
        std = float(np.std(finite, ddof=1)) if len(finite) > 1 else (0.0 if len(finite) == 1 else np.nan)
        unique_count = int(len(np.unique(finite)))
        channel_rows.append({
            'archive': archive_name, 'raw_file': member, 'channel': ch, 'rows': n,
            'missing_or_non_numeric': nonnumeric_or_missing, 'infinite_values': infinite,
            'minimum': float(np.min(finite)) if len(finite) else np.nan,
            'maximum': float(np.max(finite)) if len(finite) else np.nan,
            'median': float(np.median(finite)) if len(finite) else np.nan,
            'std_dev': std, 'unique_finite_values': unique_count,
            'repeated_consecutive_fraction': repeated_fraction,
            'longest_identical_run': int(longest_run),
            'possible_flat_channel': bool(unique_count <= 1 and len(finite) > 0),
        })
    return index_row, channel_rows, digest, len(raw_bytes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', required=True, help='Directory containing participant ZIP archives')
    parser.add_argument('--output-dir', default=None, help='Output directory; defaults to <raw-dir>/raw_zip_audit_corrected')
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else raw_dir / 'raw_zip_audit_corrected'
    out_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(p for p in raw_dir.glob('TEST_SUBJECT_*.zip') if p.is_file())
    if not zips:
        raise SystemExit(f'No TEST_SUBJECT_*.zip archives found in {raw_dir}')

    inventory_rows = []
    channel_rows = []
    index_rows = []
    duplicate_rows = []
    digest_to_files = {}
    counts = Counter()

    for zip_path in zips:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                member = info.filename
                if info.is_dir():
                    continue
                sidecar = is_macos_sidecar(member)
                actual_raw = is_actual_raw(member)
                if sidecar:
                    counts['excluded_sidecars'] += 1
                    status = 'EXCLUDED: macOS sidecar (__MACOSX or ._* path component)'
                    classification = 'macOS sidecar'
                elif actual_raw:
                    counts['actual_raw_candidates'] += 1
                    status = 'Included in raw EEG QC' if not info.file_size == 0 else 'Included candidate; empty file (check parser)'
                    classification = 'Raw EEG recording'
                else:
                    status = 'Not a raw EEG recording; retained in inventory only'
                    lower = member.lower()
                    if lower.endswith('.csv') and 'event_log' in lower:
                        classification = 'Event/marker log'
                    elif lower.endswith('.csv') and 'training_segments' in lower:
                        classification = 'Training-segment CSV'
                    elif PurePosixPath(member).name == '.DS_Store':
                        classification = 'macOS metadata'
                    else:
                        classification = 'Other archive member'
                inventory_rows.append({
                    'archive': zip_path.name, 'member_path': member,
                    'extension': Path(member).suffix.lower(), 'size_bytes': info.file_size,
                    'classification': classification, 'status': status,
                })
                if not actual_raw:
                    continue
                idx_row, rows, digest, byte_count = qc_one_recording(zf, zip_path.name, member)
                index_rows.append(idx_row)
                channel_rows.extend(rows)
                digest_to_files.setdefault(digest, []).append((zip_path.name, member, byte_count))
                counts['parsed_raw_recordings'] += 1
                counts['parsed_channel_records'] += len(rows)

    for digest, files in digest_to_files.items():
        if len(files) > 1:
            for archive, member, size in files:
                duplicate_rows.append({
                    'sha256': digest, 'duplicate_group_size': len(files),
                    'archive': archive, 'raw_file': member, 'size_bytes': size,
                })

    inventory = pd.DataFrame(inventory_rows)
    channels = pd.DataFrame(channel_rows)
    indices = pd.DataFrame(index_rows)
    duplicates = pd.DataFrame(duplicate_rows, columns=['sha256','duplicate_group_size','archive','raw_file','size_bytes'])
    inventory.to_csv(out_dir / 'archive_contents_inventory_corrected.csv', index=False)
    channels.to_csv(out_dir / 'all_channel_qc_corrected.csv', index=False)
    indices.to_csv(out_dir / 'all_sample_index_qc_corrected.csv', index=False)
    duplicates.to_csv(out_dir / 'raw_duplicate_hashes_corrected.csv', index=False)

    summary = {
        'input_directory': str(raw_dir), 'zip_archives_found': len(zips),
        'archive_file_members_in_inventory': len(inventory),
        'macOS_sidecar_files_excluded': int(counts['excluded_sidecars']),
        'actual_raw_recordings_parsed': int(counts['parsed_raw_recordings']),
        'channel_qc_rows': int(len(channels)),
        'sample_index_qc_rows': int(len(indices)),
        'raw_recordings_with_missing_or_nonnumeric_channel_values': int(channels.groupby(['archive','raw_file'])['missing_or_non_numeric'].sum().gt(0).sum()) if len(channels) else 0,
        'total_missing_or_nonnumeric_channel_values': int(channels['missing_or_non_numeric'].sum()) if len(channels) else 0,
        'total_infinite_channel_values': int(channels['infinite_values'].sum()) if len(channels) else 0,
        'flat_channel_records': int(channels['possible_flat_channel'].sum()) if len(channels) else 0,
        'channel_records_with_repeated_consecutive_fraction_ge_0_99': int((channels['repeated_consecutive_fraction'] >= .99).sum()) if len(channels) else 0,
        'raw_files_in_duplicate_hash_groups': int(len(duplicates)),
        'duplicate_hash_groups': int(duplicates['sha256'].nunique()) if len(duplicates) else 0,
        'note': 'Sidecars are excluded from all raw parsing and hash comparisons. Inventory retains excluded entries for traceability.'
    }
    pd.DataFrame([summary]).to_csv(out_dir / 'raw_zip_audit_summary_corrected.csv', index=False)
    print('RAW ZIP AUDIT COMPLETE')
    for k, v in summary.items():
        print(f'{k}: {v}')
    print(f'Output directory: {out_dir}')
    print('Files written: archive_contents_inventory_corrected.csv, all_channel_qc_corrected.csv, all_sample_index_qc_corrected.csv, raw_duplicate_hashes_corrected.csv, raw_zip_audit_summary_corrected.csv')

if __name__ == '__main__':
    main()
