
from pathlib import Path
from zipfile import ZipFile
import csv
import hashlib
import io
import re
import numpy as np
import pandas as pd
import h5py

ROOT = Path(r"D:\EEGToAlphabets")
DATA = ROOT / "preprocessed_data_version4"
RAW_ROOT = ROOT / "raw data_version4"
OUT = ROOT / "results_version4" / "complete_dataset_qc"
OUT.mkdir(parents=True, exist_ok=True)

CHUNK = 100
EPS = 1e-12


def locate(name, roots=(DATA, ROOT)):
    for base in roots:
        if base.exists():
            found = list(base.rglob(name))
            if found:
                return found[0]
    return None


def save_df(rows, name):
    pd.DataFrame(rows).to_csv(OUT / name, index=False)



def signal_stats(x):
    """Lightweight signal QC without expensive run-length calculations."""
    x = np.asarray(x)
    flat = x.reshape(-1)

    if flat.size == 0:
        return {
            "n_values": 0,
            "n_nonfinite": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "zero_fraction": np.nan,
            "max_abs": np.nan,
        }

    finite = np.isfinite(flat)
    n_finite = int(finite.sum())

    if n_finite == 0:
        return {
            "n_values": int(flat.size),
            "n_nonfinite": int(flat.size),
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "zero_fraction": float(np.mean(flat == 0)),
            "max_abs": np.nan,
        }

    values = flat[finite]

    return {
        "n_values": int(flat.size),
        "n_nonfinite": int((~finite).sum()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "zero_fraction": float(np.mean(values == 0)),
        "max_abs": float(np.max(np.abs(values))),
    }


# ------------------------------------------------------------
# 1. Locate the supplied files
# ------------------------------------------------------------
epochs_path = locate("epochs.mat")
metadata_path = locate("metadata.csv")
manifest_path = locate("fdf_epoch_manifest.csv")
channels_path = locate("channel_names.csv")
trials_path = locate("trials_X.npy")
trial_meta_path = locate("trials_meta.csv")

print("=== INPUT FILES ===")
for name, path in [
    ("epochs.mat", epochs_path),
    ("metadata.csv", metadata_path),
    ("fdf_epoch_manifest.csv", manifest_path),
    ("channel_names.csv", channels_path),
    ("trials_X.npy", trials_path),
    ("trials_meta.csv", trial_meta_path),
]:
    print(f"{name}: {path if path else 'NOT FOUND'}")

# Prefer the project data folder for files with duplicate names.
if epochs_path is None or metadata_path is None:
    raise FileNotFoundError(
        "Required epochs.mat or metadata.csv not found. "
        "Check DATA and ROOT paths."
    )

channels = None
if channels_path:
    channels = pd.read_csv(channels_path, header=None)[0].astype(str).tolist()
    print("\nChannel count:", len(channels))
    print("Channel names:", channels)

# ------------------------------------------------------------
# 2. Audit supplied metadata
# ------------------------------------------------------------
meta = pd.read_csv(metadata_path, low_memory=False)

print("\n=== SUPPLIED METADATA ===")
print("Rows:", len(meta))
print("Columns:", list(meta.columns))

meta_rows = []
for col in [
    "participant", "label", "class_index", "trial",
    "trial_uid", "window_uid", "window_id", "fs",
    "task_start_used", "task_end_used", "bad_channels", "raw_path"
]:
    if col in meta:
        meta_rows.append({
            "field": col,
            "missing_values": int(meta[col].isna().sum()),
            "unique_nonmissing": int(meta[col].nunique(dropna=True)),
        })

save_df(meta_rows, "metadata_field_quality.csv")

if "label" in meta:
    save_df(
        [{"label": str(k), "rows": int(v)}
         for k, v in meta["label"].value_counts(dropna=False).items()],
        "metadata_label_counts.csv",
    )

if "participant" in meta and "label" in meta:
    counts = (
        meta.groupby(["participant", "label"], dropna=False)
        .size().reset_index(name="window_count")
    )
    save_df(counts.to_dict("records"), "metadata_participant_class_counts.csv")

# Check duplicate IDs without assuming duplicate rows are invalid.
for col in ["window_uid", "trial_uid"]:
    if col in meta:
        print(
            f"{col}: rows={len(meta)}, "
            f"unique={meta[col].nunique(dropna=True)}, "
            f"missing={meta[col].isna().sum()}"
        )


# ------------------------------------------------------------
# 3. Audit supplied epochs.mat efficiently
# ------------------------------------------------------------
epoch_rows = []
epoch_channel_rows = []

with h5py.File(epochs_path, "r") as f:
    print("\n=== SUPPLIED epochs.mat ===")
    print("Top-level keys:", list(f.keys()))

    if "X" not in f:
        raise KeyError("Variable X was not found in epochs.mat")

    ds = f["X"]
    print("X shape in HDF5:", ds.shape)
    print("X dtype:", ds.dtype)

    if len(ds.shape) != 3:
        raise ValueError(f"Expected 3D X, got {ds.shape}")

    n_time, n_channels, n_windows = ds.shape
    print(
        "Interpreted dimensions:",
        n_windows, "windows x",
        n_channels, "channels x",
        n_time, "samples"
    )

    # Check every window, using vectorized operations.
    nonfinite_total = 0
    zero_total = 0
    value_total = 0
    global_min = np.inf
    global_max = -np.inf
    global_max_abs = 0.0

    # Only detailed-statistics sample: every 100th window.
    detailed_indices = set(range(0, n_windows, 100))

    for start in range(0, n_windows, 2000):
        end = min(start + 2000, n_windows)

        # (samples, channels, windows)
        block = np.asarray(ds[:, :, start:end])

        nonfinite_total += int((~np.isfinite(block)).sum())
        zero_total += int((block == 0).sum())
        value_total += int(block.size)

        finite_values = block[np.isfinite(block)]
        if finite_values.size:
            global_min = min(global_min, float(finite_values.min()))
            global_max = max(global_max, float(finite_values.max()))
            global_max_abs = max(
                global_max_abs,
                float(np.max(np.abs(finite_values)))
            )

        # Detailed checks only for sampled windows.
        for idx in range(start, end):
            if idx not in detailed_indices:
                continue

            window = np.asarray(ds[:, :, idx], dtype=np.float32).T

            epoch_rows.append({
                "window_index": idx,
                **signal_stats(window),
            })

            for c, sig in enumerate(window):
                epoch_channel_rows.append({
                    "window_index": idx,
                    "channel_index": c,
                    "channel": (
                        channels[c]
                        if channels and c < len(channels)
                        else f"channel_{c+1}"
                    ),
                    **signal_stats(sig),
                })

    save_df(epoch_rows, "epochs_mat_sampled_window_quality.csv")
    save_df(
        epoch_channel_rows,
        "epochs_mat_sampled_window_channel_quality.csv"
    )

    save_df([{
        "windows": n_windows,
        "channels": n_channels,
        "samples_per_window": n_time,
        "total_values": value_total,
        "nonfinite_values": nonfinite_total,
        "zero_values": zero_total,
        "global_min": global_min if global_min != np.inf else np.nan,
        "global_max": global_max if global_max != -np.inf else np.nan,
        "global_max_abs": global_max_abs,
    }], "epochs_mat_full_tensor_summary.csv")

    print("Full-tensor values checked:", value_total)
    print("Nonfinite values:", nonfinite_total)
    print("Zero values:", zero_total)
    print("Global minimum:", global_min)
    print("Global maximum:", global_max)
    print("Detailed windows sampled:", len(epoch_rows))

# ------------------------------------------------------------
# 4. Audit trials_X.npy across every trial
# ------------------------------------------------------------
if trials_path:
    X = np.load(trials_path, mmap_mode="r")
    print("\n=== trials_X.npy ===")
    print("Shape:", X.shape, "dtype:", X.dtype)

    if X.ndim != 3:
        raise ValueError("trials_X.npy must be a 3D array")

    n_trials, n_channels, n_time = X.shape
    trial_rows = []
    trial_channel_rows = []

    for start in range(0, n_trials, CHUNK):
        end = min(start + CHUNK, n_trials)
        block = np.asarray(X[start:end], dtype=np.float64)

        for offset, trial in enumerate(block):
            ti = start + offset
            trial_rows.append({
                "trial_index": ti,
                **signal_stats(trial),
            })

            for c, sig in enumerate(trial):
                trial_channel_rows.append({
                    "trial_index": ti,
                    "channel_index": c,
                    "channel": channels[c] if channels and c < len(channels)
                               else f"channel_{c+1}",
                    **signal_stats(sig),
                })

    save_df(trial_rows, "trials_X_trial_quality.csv")
    save_df(trial_channel_rows, "trials_X_trial_channel_quality.csv")

    if trial_meta_path:
        tm = pd.read_csv(trial_meta_path, low_memory=False)
        print("Trial metadata rows:", len(tm))
        print("Tensor trial count:", n_trials)

        alignment = [{
            "check": "trial_count_vs_metadata_rows",
            "tensor_count": n_trials,
            "metadata_count": len(tm),
            "pass": n_trials == len(tm),
        }]

        if "trial_uid" in tm:
            alignment.append({
                "check": "unique_trial_uid",
                "tensor_count": n_trials,
                "metadata_count": int(tm["trial_uid"].nunique(dropna=True)),
                "pass": n_trials == tm["trial_uid"].nunique(dropna=True),
            })

        save_df(alignment, "trial_tensor_metadata_checks.csv")

# ------------------------------------------------------------
# 5. Cross-check metadata and manifest
# ------------------------------------------------------------
if manifest_path:
    manifest = pd.read_csv(manifest_path)
    manifest.to_csv(OUT / "manifest_copy_for_review.csv", index=False)
    print("\nManifest rows:", len(manifest))
    print("Manifest columns:", list(manifest.columns))


# ------------------------------------------------------------
# 6. Audit raw EEG files inside ZIP archives (streaming)
# ------------------------------------------------------------
zip_candidates = set()

if RAW_ROOT.exists():
    zip_candidates.update(RAW_ROOT.rglob("*.zip"))
zip_candidates.update(ROOT.glob("*.zip"))

raw_rows = []
seen_hashes = {}

for zip_path in sorted(zip_candidates):
    print(f"\nScanning ZIP: {zip_path.name}", flush=True)

    try:
        with ZipFile(zip_path) as zf:
            members = [
                n for n in zf.namelist()
                if not n.endswith("/")
                and "OpenBCI-RAW" in n
                and n.lower().endswith(".txt")
            ]

            for member in members:
                print(f"  Reading: {member}", flush=True)

                try:
                    # Hash source bytes in chunks.
                    digest = hashlib.sha256()
                    with zf.open(member) as src:
                        for chunk in iter(
                            lambda: src.read(1024 * 1024), b""
                        ):
                            digest.update(chunk)
                    sha = digest.hexdigest()

                    seen_hashes.setdefault(sha, []).append(
                        f"{zip_path.name}::{member}"
                    )

                    # Streaming statistics; no full-recording list.
                    fs_header = None
                    n_rows = 0
                    n_nonfinite = np.zeros(16, dtype=np.int64)
                    mins = np.full(16, np.inf)
                    maxs = np.full(16, -np.inf)
                    sums = np.zeros(16, dtype=np.float64)
                    sums_sq = np.zeros(16, dtype=np.float64)
                    repeat_count = np.zeros(16, dtype=np.int64)
                    comparable_count = np.zeros(16, dtype=np.int64)
                    constant = np.ones(16, dtype=bool)
                    first_values = np.full(16, np.nan)
                    previous = np.full(16, np.nan)
                    have_previous = np.zeros(16, dtype=bool)

                    with zf.open(member) as src:
                        text_stream = io.TextIOWrapper(
                            src, encoding="utf-8", errors="replace"
                        )

                        for line_no, line in enumerate(
                            text_stream, start=1
                        ):
                            if line_no <= 30:
                                match = re.search(
                                    r"Sample\s*Rate\s*=\s*([0-9.]+)",
                                    line, re.I
                                )
                                if match:
                                    fs_header = float(match.group(1))

                            line = line.strip()
                            if not line or line.startswith("%"):
                                continue

                            parts = next(csv.reader([line]))
                            if len(parts) < 17:
                                continue

                            eeg = pd.to_numeric(
                                pd.Series(parts[1:17]),
                                errors="coerce"
                            ).to_numpy(dtype=np.float64)

                            # Preserve the original inclusion rule.
                            if np.isfinite(eeg).sum() < 12:
                                continue

                            n_rows += 1
                            finite = np.isfinite(eeg)
                            n_nonfinite += (~finite).astype(np.int64)

                            for c in range(16):
                                if not finite[c]:
                                    continue

                                value = eeg[c]
                                mins[c] = min(mins[c], value)
                                maxs[c] = max(maxs[c], value)
                                sums[c] += value
                                sums_sq[c] += value * value

                                if np.isnan(first_values[c]):
                                    first_values[c] = value
                                elif value != first_values[c]:
                                    constant[c] = False

                                if have_previous[c]:
                                    comparable_count[c] += 1
                                    if value == previous[c]:
                                        repeat_count[c] += 1

                                previous[c] = value
                                have_previous[c] = True

                    for c in range(16):
                        count = n_rows - int(n_nonfinite[c])
                        if count > 0:
                            mean = sums[c] / count
                            variance = max(
                                0.0, sums_sq[c] / count - mean * mean
                            )
                            std = float(np.sqrt(variance))
                        else:
                            mean = np.nan
                            std = np.nan

                        raw_rows.append({
                            "zip": zip_path.name,
                            "member": member,
                            "sha256": sha,
                            "rows": n_rows,
                            "sampling_rate_header": fs_header,
                            "channel_index_1based": c + 1,
                            "nonfinite_count": int(n_nonfinite[c]),
                            "min": (
                                float(mins[c])
                                if count > 0 else np.nan
                            ),
                            "max": (
                                float(maxs[c])
                                if count > 0 else np.nan
                            ),
                            "mean": float(mean),
                            "std": float(std),
                            "max_abs": (
                                float(max(abs(mins[c]), abs(maxs[c])))
                                if count > 0 else np.nan
                            ),
                            "adjacent_repeat_fraction": (
                                float(repeat_count[c] /
                                      comparable_count[c])
                                if comparable_count[c] else np.nan
                            ),
                            "constant_channel": (
                                bool(constant[c]) if count > 0 else False
                            ),
                            "status": (
                                "CHECKED" if n_rows else
                                "NO_NUMERIC_EEG_ROWS"
                            ),
                        })

                except Exception as exc:
                    raw_rows.append({
                        "zip": zip_path.name,
                        "member": member,
                        "status": f"ERROR: {exc}",
                    })

    except Exception as exc:
        raw_rows.append({
            "zip": zip_path.name,
            "status": f"ZIP_ERROR: {exc}",
        })

    # Save progress after each archive, so partial results survive.
    save_df(raw_rows, "raw_zip_channel_quality.csv")

    duplicate_rows = [
        {
            "sha256": sha,
            "duplicate_count": len(paths),
            "locations": " | ".join(paths),
        }
        for sha, paths in seen_hashes.items()
        if len(paths) > 1
    ]
    save_df(duplicate_rows, "raw_zip_duplicate_hashes.csv")

print("\n=== COMPLETE DATASET AUDIT FINISHED ===")
print("Reports saved in:", OUT)
print("Raw EEG channel records:", len(raw_rows))
print(
    "Exact duplicate raw-file groups:",
    len(duplicate_rows) if zip_candidates else 0
)
print("Review the CSV reports before classifying anomalies.")
