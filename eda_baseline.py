from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "preprocessed_data_version4"
META_DIR = ROOT / "metadata_version4"
OUT = ROOT / "results_version4" / "eda"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

X_PATH = DATA / "trials_X.npy"
TRIAL_META_PATH = DATA / "trials_meta.csv"
CHANNELS_PATH = META_DIR / "channels.csv"
TASK_PATH = META_DIR / "task_description.csv"

FALLBACK_CHANNELS = [
    "Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2",
    "F7", "F8", "F3", "F4", "T7", "P3", "P4"
]
CLASS_ORDER = ["A", "K", "N", "P", "Z"]


def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close()


def read_sampling_rate():
    if TASK_PATH.exists():
        try:
            task = pd.read_csv(TASK_PATH)
            if {"section", "field", "value"}.issubset(task.columns):
                row = task[(task["section"] == "acquisition") &
                           (task["field"] == "sampling_rate")]
                if not row.empty:
                    return float(row.iloc[0]["value"])
        except Exception as exc:
            print(f"Warning reading task_description.csv: {exc}")
    return 125.0


def read_channels(n_channels):
    if CHANNELS_PATH.exists():
        try:
            df = pd.read_csv(CHANNELS_PATH)
            candidates = [c for c in df.columns if str(c).lower() in
                          {"channel", "channel_name", "name", "label"}]
            if candidates:
                names = df[candidates[0]].dropna().astype(str).tolist()
                if len(names) == n_channels:
                    return names
            if df.shape[1] == 1:
                names = df.iloc[:, 0].dropna().astype(str).tolist()
                if len(names) == n_channels:
                    return names
        except Exception as exc:
            print(f"Warning reading channels.csv: {exc}")
    if len(FALLBACK_CHANNELS) == n_channels:
        return FALLBACK_CHANNELS
    return [f"Ch{i+1}" for i in range(n_channels)]


def main():
    if not X_PATH.exists():
        raise FileNotFoundError(f"Missing {X_PATH}")
    if not TRIAL_META_PATH.exists():
        raise FileNotFoundError(f"Missing {TRIAL_META_PATH}")

    print("Loading trial metadata...")
    meta = pd.read_csv(TRIAL_META_PATH)
    print("Loading EEG tensor read-only...")
    X = np.load(X_PATH, mmap_mode="r", allow_pickle=False)

    if X.ndim != 3:
        raise ValueError(f"Expected 3D tensor (trials, channels, time), got {X.shape}")
    n_trials, n_channels, n_time = X.shape
    fs = read_sampling_rate()
    channels = read_channels(n_channels)
    duration_s = n_time / fs

    if not {"participant", "label"}.issubset(meta.columns):
        raise ValueError("trials_meta.csv must contain participant and label columns")

    labels = meta["label"].astype(str).str.strip().str.upper()
    participants = meta["participant"].astype(str).str.strip()

    checks = [
        {"check": "tensor_is_3d", "result": True, "details": str(X.shape)},
        {"check": "tensor_trial_count_matches_metadata_rows",
         "result": n_trials == len(meta), "details": f"tensor={n_trials}; metadata={len(meta)}"},
        {"check": "sampling_rate_hz", "result": fs, "details": "From task_description.csv or 125 Hz fallback"},
        {"check": "samples_per_trial", "result": n_time, "details": f"duration_seconds={duration_s:.6f}"},
    ]
    if "trial_uid" in meta:
        checks.append({
            "check": "trial_uid_unique_and_complete",
            "result": bool(meta["trial_uid"].notna().all() and meta["trial_uid"].nunique() == n_trials),
            "details": f"unique={meta['trial_uid'].nunique(dropna=True)}; rows={n_trials}; missing={meta['trial_uid'].isna().sum()}"
        })
    pd.DataFrame(checks).to_csv(OUT / "alignment_and_schema_checks.csv", index=False)

    field_rows = [{
        "field": c,
        "missing_count": int(meta[c].isna().sum()),
        "missing_fraction": float(meta[c].isna().mean()),
        "unique_nonmissing": int(meta[c].nunique(dropna=True)),
    } for c in meta.columns]
    pd.DataFrame(field_rows).to_csv(OUT / "metadata_quality.csv", index=False)

    id_rows = []
    for col in ["trial_uid", "participant", "trial"]:
        if col in meta:
            id_rows.append({
                "field": col,
                "rows": len(meta),
                "unique_nonmissing": int(meta[col].nunique(dropna=True)),
                "missing": int(meta[col].isna().sum()),
                "duplicate_nonmissing_count": int(meta[col].notna().sum() - meta[col].nunique(dropna=True)),
            })
    pd.DataFrame(id_rows).to_csv(OUT / "metadata_id_summary.csv", index=False)

    label_counts = labels.value_counts().reindex(CLASS_ORDER, fill_value=0)
    label_counts.rename_axis("label").reset_index(name="trial_count").to_csv(
        OUT / "class_distribution.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.bar(label_counts.index, label_counts.values)
    plt.title("Trial count by imagined alphabet class")
    plt.xlabel("Class")
    plt.ylabel("Number of trials")
    save_fig("01_class_distribution.png")

    participant_counts = pd.crosstab(participants, labels).reindex(columns=CLASS_ORDER, fill_value=0)
    participant_counts.to_csv(OUT / "participant_class_counts.csv")
    plt.figure(figsize=(11, 6))
    plt.imshow(participant_counts.values, aspect="auto")
    plt.colorbar(label="Trial count")
    plt.xticks(np.arange(len(participant_counts.columns)), participant_counts.columns)
    plt.yticks(np.arange(len(participant_counts.index)), participant_counts.index)
    plt.xlabel("Class")
    plt.ylabel("Participant")
    plt.title("Participant × class trial counts")
    save_fig("02_participant_class_heatmap.png")

    trial_rows, channel_rows = [], []
    finite_total = nonfinite_total = zero_total = 0
    total_values = int(n_trials * n_channels * n_time)
    global_min, global_max, global_max_abs = np.inf, -np.inf, 0.0

    for start in range(0, n_trials, 100):
        stop = min(start + 100, n_trials)
        block = np.asarray(X[start:stop], dtype=np.float32)
        finite = np.isfinite(block)
        nonfinite_total += int((~finite).sum())
        zero_total += int((block == 0).sum())
        finite_total += int(finite.sum())
        vals = block[finite]
        if vals.size:
            global_min = min(global_min, float(vals.min()))
            global_max = max(global_max, float(vals.max()))
            global_max_abs = max(global_max_abs, float(np.max(np.abs(vals))))

        for local_i, trial in enumerate(block):
            idx = start + local_i
            ftrial = np.isfinite(trial)
            values = trial[ftrial]
            row = {
                "trial_index": idx,
                "participant": participants.iloc[idx] if idx < len(meta) else None,
                "label": labels.iloc[idx] if idx < len(meta) else None,
                "nonfinite_count": int((~ftrial).sum()),
                "zero_fraction": float(np.mean(trial == 0)),
                "min": float(values.min()) if values.size else np.nan,
                "max": float(values.max()) if values.size else np.nan,
                "mean": float(values.mean()) if values.size else np.nan,
                "std": float(values.std()) if values.size else np.nan,
                "rms": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))) if values.size else np.nan,
                "max_abs": float(np.max(np.abs(values))) if values.size else np.nan,
            }
            if "trial_uid" in meta:
                row["trial_uid"] = meta["trial_uid"].iloc[idx]
            trial_rows.append(row)

            for ci, sig in enumerate(trial):
                v = sig[np.isfinite(sig)]
                channel_rows.append({
                    "trial_index": idx,
                    "participant": row["participant"],
                    "label": row["label"],
                    "channel_index": ci,
                    "channel": channels[ci],
                    "nonfinite_count": int((~np.isfinite(sig)).sum()),
                    "zero_fraction": float(np.mean(sig == 0)),
                    "mean": float(v.mean()) if v.size else np.nan,
                    "std": float(v.std()) if v.size else np.nan,
                    "rms": float(np.sqrt(np.mean(v.astype(np.float64) ** 2))) if v.size else np.nan,
                    "min": float(v.min()) if v.size else np.nan,
                    "max": float(v.max()) if v.size else np.nan,
                    "max_abs": float(np.max(np.abs(v))) if v.size else np.nan,
                })

    trial_df = pd.DataFrame(trial_rows)
    channel_df = pd.DataFrame(channel_rows)
    trial_df.to_csv(OUT / "trial_signal_quality.csv", index=False)
    channel_df.to_csv(OUT / "trial_channel_signal_quality.csv", index=False)
    pd.DataFrame([{
        "n_trials": n_trials, "n_channels": n_channels, "n_samples_per_trial": n_time,
        "sampling_rate_hz": fs, "duration_seconds": duration_s, "total_values": total_values,
        "finite_values": finite_total, "nonfinite_values": nonfinite_total,
        "zero_values": zero_total, "zero_fraction": zero_total / total_values if total_values else np.nan,
        "global_min": global_min if np.isfinite(global_min) else np.nan,
        "global_max": global_max if np.isfinite(global_max) else np.nan,
        "global_max_abs": global_max_abs, "dtype": str(X.dtype),
    }]).to_csv(OUT / "tensor_summary.csv", index=False)

    channel_summary = channel_df.groupby("channel", sort=False).agg(
        mean_of_trial_means=("mean", "mean"),
        median_trial_std=("std", "median"),
        median_trial_rms=("rms", "median"),
        median_abs_max=("max_abs", "median"),
        total_nonfinite=("nonfinite_count", "sum"),
        mean_zero_fraction=("zero_fraction", "mean"),
    ).reset_index()
    channel_summary.to_csv(OUT / "channel_summary.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.bar(channel_summary["channel"], channel_summary["median_trial_rms"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Median trial RMS (stored units)")
    plt.title("Channel-wise signal amplitude (median trial RMS)")
    save_fig("03_channel_median_rms.png")

    plt.figure(figsize=(9, 5))
    plt.hist(trial_df["rms"].replace([np.inf, -np.inf], np.nan).dropna(), bins=40)
    plt.xlabel("Trial RMS across channels and time (stored units)")
    plt.ylabel("Number of trials")
    plt.title("Distribution of trial signal amplitude")
    save_fig("04_trial_rms_distribution.png")

    plt.figure(figsize=(9, 5))
    data = [trial_df.loc[trial_df["label"] == c, "rms"].dropna().values for c in CLASS_ORDER]
    plt.boxplot(data, labels=CLASS_ORDER, showfliers=False)
    plt.xlabel("Class")
    plt.ylabel("Trial RMS (stored units)")
    plt.title("Trial amplitude distribution by class")
    save_fig("05_trial_rms_by_class.png")

    participant_rms = trial_df.groupby("participant", dropna=False)["rms"].agg(
        trial_count="count", median_rms="median", mean_rms="mean", std_rms="std"
    ).reset_index()
    participant_rms.to_csv(OUT / "participant_signal_variability.csv", index=False)
    participant_rms = participant_rms.sort_values("participant")
    plt.figure(figsize=(12, 5))
    plt.bar(participant_rms["participant"].astype(str), participant_rms["median_rms"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Participant")
    plt.ylabel("Median trial RMS (stored units)")
    plt.title("Between-participant signal amplitude variability")
    save_fig("06_participant_variability.png")

    chosen = []
    for c in CLASS_ORDER:
        ids = np.flatnonzero(labels.to_numpy() == c)
        if len(ids):
            chosen.append((c, int(ids[0])))
    if chosen:
        fig, axes = plt.subplots(len(chosen), 1, figsize=(12, 2.5 * len(chosen)), sharex=True)
        if len(chosen) == 1:
            axes = [axes]
        for ax, (c, idx) in zip(axes, chosen):
            trial = np.asarray(X[idx], dtype=np.float32)
            t = np.arange(n_time) / fs
            for ci in [2, 3, 6]:
                if ci < n_channels:
                    ax.plot(t, trial[ci], linewidth=0.7, label=channels[ci])
            ax.set_title(f"Representative trial: class {c}, participant {participants.iloc[idx]}")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right", ncol=3)
        axes[-1].set_xlabel("Time (seconds)")
        fig.suptitle("Representative preprocessed EEG traces", y=1.005)
        save_fig("07_representative_eeg_traces.png")

    try:
        from scipy.signal import welch
        psd_rows = []
        plt.figure(figsize=(10, 6))
        plotted = 0
        for c, idx in chosen:
            trial = np.asarray(X[idx], dtype=np.float32)
            for ci in [2, 3, 6]:
                if ci >= n_channels:
                    continue
                sig = trial[ci]
                if not np.isfinite(sig).all():
                    sig = sig[np.isfinite(sig)]
                if len(sig) < 32:
                    continue
                freqs, power = welch(sig, fs=fs, nperseg=min(256, len(sig)))
                keep = freqs <= min(40, fs / 2)
                plt.plot(freqs[keep], 10 * np.log10(power[keep] + 1e-20),
                         linewidth=0.8, label=f"{c}-{channels[ci]}")
                for f, p in zip(freqs[keep], power[keep]):
                    psd_rows.append({"class": c, "trial_index": idx, "channel": channels[ci],
                                     "frequency_hz": f, "psd_power": p})
                plotted += 1
        if plotted:
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD (dB; stored-unit²/Hz)")
            plt.title(f"Welch PSD on representative full trials (fs={fs:g} Hz)")
            plt.legend(fontsize=7, ncol=3)
            save_fig("08_representative_trial_psd.png")
            pd.DataFrame(psd_rows).to_csv(OUT / "representative_trial_psd.csv", index=False)
        else:
            plt.close()
            warnings.warn("No PSD plot produced; insufficient finite data.")
    except Exception as exc:
        plt.close("all")
        warnings.warn(f"PSD generation skipped: {exc}")

    checks_df = pd.DataFrame(checks)
    failed = checks_df[checks_df["result"].astype(str).str.lower().isin(["false", "0"])]
    report = [
        "# EEG EDA report", "",
        "## Dataset",
        f"- Tensor shape: `{X.shape}`, dtype `{X.dtype}`.",
        f"- Metadata rows: {len(meta)}.",
        f"- Sampling rate used: {fs:g} Hz.",
        f"- Reconstructed trial duration: {duration_s:.6f} seconds.",
        f"- Channels: {', '.join(channels)}.",
        f"- Class counts: {label_counts.to_dict()}.", "",
        "## Integrity",
        f"- Total tensor values: {total_values:,}.",
        f"- Non-finite values: {nonfinite_total:,}.",
        f"- Exact zero values: {zero_total:,} ({zero_total / total_values:.6%}).",
        f"- Global finite minimum / maximum: {global_min} / {global_max}.",
        f"- Checks marked false: {len(failed)}.", "",
        "## Interpretation notes",
        "- Descriptive EDA only; no model training or evaluation.",
        "- Input trials_X.npy is opened read-only and is not modified.",
        "- Amplitude units are reported as stored; verify units before applying absolute thresholds.",
        "- Exact zeros are reported as diagnostics, not automatically called bad data.",
        "- PSD uses reconstructed full trials, not isolated 8-sample windows.",
        "- No data-dependent normalization, feature selection, or tuning is performed.",
        "- P04/P10 duplicate grouping is a model-splitting requirement; this EDA does not assign partitions.", "",
        "## Outputs",
        "- alignment_and_schema_checks.csv; metadata_quality.csv; metadata_id_summary.csv",
        "- class_distribution.csv; participant_class_counts.csv; tensor_summary.csv",
        "- trial_signal_quality.csv; trial_channel_signal_quality.csv; channel_summary.csv",
        "- participant_signal_variability.csv; representative_trial_psd.csv (if generated)",
        "- All plots are in `figures/`.",
    ]
    (OUT / "EDA_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("\nEDA completed.")
    print(f"Tensor: {X.shape} | fs={fs:g} Hz | trial duration={duration_s:.3f}s")
    print(f"Class counts: {label_counts.to_dict()}")
    print(f"Non-finite values: {nonfinite_total:,}; exact zeros: {zero_total:,}")
    print(f"Reports: {OUT}")
    print(f"Figures: {FIG}")


if __name__ == "__main__":
    main()
