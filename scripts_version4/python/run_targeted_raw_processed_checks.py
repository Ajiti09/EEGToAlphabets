from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"D:\EEGToAlphabets")
PRE = ROOT / "preprocessed_data_version4"
RAW_ROOT = ROOT / "raw data_version4"
OUT = ROOT / "results_version4" / "targeted_raw_processed_checks"
OUT.mkdir(parents=True, exist_ok=True)

FS = 125
CHANNELS = pd.read_csv(PRE / "channel_names.csv", header=None)[0].astype(str).str.strip().tolist()
ALL_CONFIGURED = [
    "Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2",
    "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4"
]
RAW_COLS = [f"EXG Channel {i}" for i in range(16)]

def raw_zip_for(participant):
    # P01 -> TEST_SUBJECT_1.zip; P19 -> TEST_SUBJECT_19.zip
    n = int(participant[1:])
    p = RAW_ROOT / f"TEST_SUBJECT_{n}.zip"
    if not p.exists():
        raise FileNotFoundError(f"Raw ZIP not found: {p}")
    return p

def read_raw_session(participant, label):
    zpath = raw_zip_for(participant)
    session_tag = f"TEST_SUBJECT_{int(participant[1:])}_{label}"
    with ZipFile(zpath) as zf:
        members = [
            n for n in zf.namelist()
            if Path(n).name.startswith("OpenBCI-RAW-")
            and "/._" not in n
            and "__MACOSX" not in n
        ]

        # Match the participant/session using the containing folder names.
        # Some ZIPs have slightly different nesting or duplicated session-name
        # prefixes, so don't require the exact session_tag substring in the
        # entire path.
        participant_token = f"TEST_SUBJECT_{int(participant[1:])}"
        candidates = []
        for n in members:
            parts = [part for part in n.split("/") if part]
            path_text = "/".join(parts)
            if participant_token not in path_text:
                continue
            # Prefer paths whose session folder or any path component ends
            # with the class label, while tolerating inconsistent prefixes.
            session_match = any(
                part.endswith(f"_{label}") or part.endswith(f"_{participant_token}_{label}")
                for part in parts[:-1]
            )
            if session_match or session_tag in path_text:
                candidates.append(n)

        # If the folder naming is still inconsistent, report the available
        # raw members for this participant so the actual path can be selected.
        if len(candidates) != 1:
            available = [n for n in members if participant_token in n]
            raise ValueError(
                f"Expected one raw file for {participant}-{label}; got {candidates}. "
                f"Raw files available for this participant: {available}"
            )
        member = candidates[0]
        with zf.open(member) as f:
            raw = pd.read_csv(BytesIO(f.read()), comment="%", skipinitialspace=True, low_memory=False)
    raw.columns = raw.columns.astype(str).str.strip()
    missing = [c for c in RAW_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")
    arr = raw[RAW_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return arr, member

def load_trial(participant, label, trial):
    tm = pd.read_csv(PRE / "trials_meta.csv")
    matches = tm[
        (tm.participant.astype(str) == participant)
        & (tm.label.astype(str) == label)
        & (tm.trial.astype(int) == int(trial))
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one trial metadata row for {participant}-{label}-{trial}, found {len(matches)}")
    trial_index = int(matches.index[0])
    X = np.load(PRE / "trials_X.npy", mmap_mode="r")
    processed = np.asarray(X[trial_index], dtype=np.float64)
    meta = pd.read_csv(PRE / "metadata.csv")
    rows = meta[
        (meta.participant.astype(str) == participant)
        & (meta.label.astype(str) == label)
        & (meta.trial.astype(int) == int(trial))
    ].sort_values("window_id")
    if rows.empty:
        raise ValueError("No window metadata found for requested trial")
    return processed, rows, trial_index

def raw_segment_for(participant, label, rows):
    raw, member = read_raw_session(participant, label)
    start_s = float(rows["task_start_used"].min())
    end_s = float(rows["task_end_used"].max())
    # Assumption: metadata times are seconds from recording start. Check against
    # the corresponding event/training-segment CSV before treating as definitive.
    i0, i1 = round(start_s * FS), round(end_s * FS)
    if i0 < 0 or i1 > len(raw) or i1 <= i0:
        raise ValueError(f"Metadata interval {start_s}-{end_s} is outside raw recording")
    seg = raw[i0:i1]
    raw_by_name = {name: seg[:, i] for i, name in enumerate(ALL_CONFIGURED)}
    raw_retained = np.column_stack([raw_by_name[name] for name in CHANNELS])
    return raw_retained, member, start_s, end_s, raw.shape[0]

def window_metrics(x, channels, prefix):
    rows = []
    for j, ch in enumerate(channels):
        v = np.asarray(x[:, j], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            continue
        d = np.diff(v)
        rows.append({
            "signal": prefix, "channel": ch, "n_finite": len(v),
            "min": float(np.min(v)), "max": float(np.max(v)),
            "mean": float(np.mean(v)), "std": float(np.std(v)),
            "max_abs": float(np.max(np.abs(v))),
            "max_abs_step": float(np.max(np.abs(d))),
            "step_p99_abs": float(np.quantile(np.abs(d), 0.99)),
            "n_steps_gt_10x_median": int(np.sum(np.abs(d) > 10 * max(np.median(np.abs(d)), 1e-12))),
        })
    return rows

def save_case(participant, label, trial, step_seconds=None, peak_channel=None, peak_index=None):
    processed, rows, trial_index = load_trial(participant, label, trial)
    raw, member, start_s, end_s, raw_n = raw_segment_for(participant, label, rows)

    if len(raw) != processed.shape[1]:
        print(f"WARNING {participant}-{label}-{trial}: raw samples={len(raw)}, processed samples={processed.shape[1]}")
    n = min(len(raw), processed.shape[1])
    raw = raw[:n]
    proc = processed[:, :n].T

    # Export whole matched arrays for reproducible review.
    pd.DataFrame(raw, columns=CHANNELS).to_csv(OUT / f"{participant}_{label}_T{trial:02d}_raw.csv", index=False)
    pd.DataFrame(proc, columns=CHANNELS).to_csv(OUT / f"{participant}_{label}_T{trial:02d}_processed.csv", index=False)

    # Whole-trial stats and first-difference statistics.
    stats = window_metrics(raw, CHANNELS, "raw") + window_metrics(proc, CHANNELS, "processed")
    pd.DataFrame(stats).to_csv(OUT / f"{participant}_{label}_T{trial:02d}_channel_metrics.csv", index=False)

    # Focus around requested relative time or peak sample. For a step use ±0.8 sec.
    if step_seconds is not None:
        center = int(round(step_seconds * FS))
        half = int(round(0.8 * FS))
        lo, hi = max(0, center-half), min(n, center+half)
        tag = f"step{step_seconds:g}s"
    elif peak_index is not None:
        center = int(peak_index)
        half = int(round(0.8 * FS))
        lo, hi = max(0, center-half), min(n, center+half)
        tag = f"peakidx{center}"
    else:
        lo, hi = 0, n
        tag = "full"

    # Use local time on both traces; these are not sample-wise validated.
    t = np.arange(lo, hi) / FS
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    for j, ch in enumerate(CHANNELS):
        axes[0].plot(t, raw[lo:hi, j], linewidth=0.75, label=ch)
        axes[1].plot(t, proc[lo:hi, j], linewidth=0.75, label=ch)
    axes[0].set_title(f"RAW {participant}-{label} trial {trial}: {member}")
    axes[0].set_ylabel("Raw numeric units (unverified)")
    axes[1].set_title("PROCESSED trial — matched channel names; sample alignment still to verify")
    axes[1].set_ylabel("Processed numeric units (unverified)")
    axes[1].set_xlabel("Seconds from trial start (relative)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=5, fontsize=7)
    fig.tight_layout()
    plot_path = OUT / f"{participant}_{label}_T{trial:02d}_{tag}.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(f"\nCASE {participant}-{label} trial {trial}")
    print("Raw member:", member)
    print(f"Metadata interval: {start_s:.4f} to {end_s:.4f} s; raw rows in recording: {raw_n}")
    print(f"Raw selected samples: {len(raw)}; processed samples: {processed.shape[1]}")
    print("Processed trial tensor index:", trial_index)
    print("Saved plot:", plot_path)
    print("Saved metrics:", OUT / f"{participant}_{label}_T{trial:02d}_channel_metrics.csv")
    return raw, proc

# Case A: the reported ~4 s step in P01-A trial 1.
save_case("P01", "A", 1, step_seconds=4.0)

# Extreme cases previously flagged in the trial/channel QC report.
# These plots focus on the full trial plus local metrics; adjust if the QC
# report indicates a different channel/trial indexing convention.
save_case("P19", "P", 10, peak_channel="P3")
save_case("P03", "Z", 17, peak_channel="P8")
save_case("P03", "Z", 18, peak_channel="P8")
save_case("P14", "P", 11, peak_channel="F4")

print("\nFinished. Review all CSVs and plots under:", OUT)
print("IMPORTANT: Verify raw task times against each session event log before drawing causal conclusions.")
