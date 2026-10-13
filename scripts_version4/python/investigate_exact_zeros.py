
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths
ROOT = Path(r"D:\EEGToAlphabets")
DATA_FILE = ROOT / "preprocessed_data_version4" / "trials_X.npy"
OUT_DIR = ROOT / "results_version4" / "zero_value_investigation"

OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SHAPE = (2500, 15, 1248)


def longest_true_run(mask):
    """Return longest consecutive run of True values."""
    mask = np.asarray(mask, dtype=bool)

    if not mask.any():
        return 0, None, None

    changes = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    lengths = ends - starts
    i = int(np.argmax(lengths))

    return int(lengths[i]), int(starts[i]), int(ends[i] - 1)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find trials_X.npy at:\n{DATA_FILE}\n"
            "Check the actual folder location before proceeding."
        )

    # Read-only loading; do not edit or resave the original tensor.
    X = np.load(DATA_FILE, mmap_mode="r")

    print("File:", DATA_FILE)
    print("Shape:", X.shape)
    print("Data type:", X.dtype)

    if X.shape != EXPECTED_SHAPE:
        print("WARNING: Shape differs from the expected shape.")

    # Count exact zeros.
    zero_mask = (X == 0)
    total_values = int(X.size)
    total_zeros = int(np.count_nonzero(zero_mask))

    print("\nTOTAL ZERO SUMMARY")
    print("Total tensor values:", total_values)
    print("Exact zeros:", total_zeros)
    print("Percentage of all values:", 100 * total_zeros / total_values)

    # Report 1: zero count by trial.
    trial_rows = []

    for t in range(X.shape[0]):
        z = zero_mask[t]
        positions = np.flatnonzero(z.any(axis=0))

        trial_rows.append({
            "trial_index_zero_based": t,
            "zero_count": int(z.sum()),
            "zero_fraction": float(z.mean()),
            "channels_with_zeros": int(z.any(axis=1).sum()),
            "sample_positions_with_any_zero": int(len(positions)),
            "first_sample_with_zero": (
                int(positions[0]) if len(positions) else None
            ),
            "last_sample_with_zero": (
                int(positions[-1]) if len(positions) else None
            ),
            "has_any_zero": bool(z.any()),
        })

    trial_df = pd.DataFrame(trial_rows)
    trial_df.to_csv(OUT_DIR / "zeros_by_trial.csv", index=False)

    # Report 2: zero count by retained channel.
    channel_rows = []

    for c in range(X.shape[1]):
        channel = X[:, c, :]
        mask = (channel == 0)

        channel_rows.append({
            "channel_index_zero_based": c,
            "zero_count": int(mask.sum()),
            "total_values": int(mask.size),
            "zero_fraction": float(mask.mean()),
            "trials_with_zeros": int(mask.any(axis=1).sum()),
            "trials_entirely_zero": int(
                np.all(mask, axis=1).sum()
            ),
        })

    channel_df = pd.DataFrame(channel_rows)
    channel_df.to_csv(OUT_DIR / "zeros_by_channel.csv", index=False)

    # Report 3: consecutive zero runs in each trial/channel.
    run_rows = []

    for t in range(X.shape[0]):
        for c in range(X.shape[1]):
            mask = zero_mask[t, c, :]
            longest, start, end = longest_true_run(mask)

            if mask.any():
                positions = np.flatnonzero(mask)

                run_rows.append({
                    "trial_index_zero_based": t,
                    "channel_index_zero_based": c,
                    "zero_count": int(mask.sum()),
                    "zero_fraction": float(mask.mean()),
                    "zero_runs_count": int(
                        np.sum(
                            np.diff(
                                np.r_[False, mask, False].astype(int)
                            ) == 1
                        )
                    ),
                    "longest_zero_run_samples": longest,
                    "longest_zero_run_seconds": longest / 125,
                    "longest_run_start_sample": start,
                    "longest_run_end_sample": end,
                    "first_zero_sample": int(positions[0]),
                    "last_zero_sample": int(positions[-1]),
                    "zero_at_first_sample": bool(mask[0]),
                    "zero_at_last_sample": bool(mask[-1]),
                })

    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(OUT_DIR / "zero_runs_by_trial_channel.csv", index=False)

    # Report 4: sample positions where multiple channels are zero together.
    # This helps identify shared padding-like regions.
    shared_rows = []

    for t in range(X.shape[0]):
        counts_per_sample = zero_mask[t].sum(axis=0)

        for s in np.flatnonzero(counts_per_sample >= 3):
            shared_rows.append({
                "trial_index_zero_based": t,
                "sample_index_zero_based": int(s),
                "time_seconds_approx": float(s / 125),
                "channels_zero_at_sample": int(counts_per_sample[s]),
                "fraction_of_channels_zero": float(
                    counts_per_sample[s] / X.shape[1]
                ),
            })

    shared_df = pd.DataFrame(shared_rows)
    shared_df.to_csv(
        OUT_DIR / "samples_with_3_or_more_zero_channels.csv",
        index=False,
    )

    # Report 5: overall tensor edge summary.
    edge_rows = []

    for edge_name, sample in [
        ("first_sample", 0),
        ("last_sample", X.shape[2] - 1),
    ]:
        edge_mask = zero_mask[:, :, sample]

        edge_rows.append({
            "edge": edge_name,
            "sample_index_zero_based": sample,
            "zero_values": int(edge_mask.sum()),
            "total_values_at_edge": int(edge_mask.size),
            "zero_fraction": float(edge_mask.mean()),
            "trials_with_any_zero_channel": int(
                edge_mask.any(axis=1).sum()
            ),
            "trials_all_channels_zero": int(
                edge_mask.all(axis=1).sum()
            ),
        })

    pd.DataFrame(edge_rows).to_csv(
        OUT_DIR / "tensor_edge_zero_summary.csv",
        index=False,
    )

    print("\nREPORTS CREATED")
    print("Output directory:", OUT_DIR)
    print("zeros_by_trial.csv")
    print("zeros_by_channel.csv")
    print("zero_runs_by_trial_channel.csv")
    print("samples_with_3_or_more_zero_channels.csv")
    print("tensor_edge_zero_summary.csv")

    print("\nTOP TRIALS BY ZERO COUNT")
    print(
        trial_df.sort_values("zero_count", ascending=False)
        .head(15).to_string(index=False)
    )

    print("\nZERO COUNTS BY CHANNEL")
    print(
        channel_df.sort_values("zero_count", ascending=False)
        .to_string(index=False)
    )

    if not runs_df.empty:
        print("\nLONGEST ZERO RUNS")
        print(
            runs_df.sort_values(
                "longest_zero_run_samples", ascending=False
            ).head(15).to_string(index=False)
        )

    print("\nInvestigation complete. Original data was not modified.")


if __name__ == "__main__":
    main()
