# EEG EDA report

## Dataset
- Tensor shape: `(2500, 15, 1248)`, dtype `float32`.
- Metadata rows: 2500.
- Sampling rate used: 125 Hz.
- Reconstructed trial duration: 9.984000 seconds.
- Channels: Fp1, Fp2, C3, C4, P7, P8, O1, O2, F7, F8, F3, F4, T7, P3, P4.
- Class counts: {'A': 500, 'K': 500, 'N': 500, 'P': 500, 'Z': 500}.

## Integrity
- Total tensor values: 46,800,000.
- Non-finite values: 0.
- Exact zero values: 7,515 (0.016058%).
- Global finite minimum / maximum: -249052.890625 / 111618.484375.
- Checks marked false: 0.

## Interpretation notes
- Descriptive EDA only; no model training or evaluation.
- Input trials_X.npy is opened read-only and is not modified.
- Amplitude units are reported as stored; verify units before applying absolute thresholds.
- Exact zeros are reported as diagnostics, not automatically called bad data.
- PSD uses reconstructed full trials, not isolated 8-sample windows.
- No data-dependent normalization, feature selection, or tuning is performed.
- P04/P10 duplicate grouping is a model-splitting requirement; this EDA does not assign partitions.

## Outputs
- alignment_and_schema_checks.csv; metadata_quality.csv; metadata_id_summary.csv
- class_distribution.csv; participant_class_counts.csv; tensor_summary.csv
- trial_signal_quality.csv; trial_channel_signal_quality.csv; channel_summary.csv
- participant_signal_variability.csv; representative_trial_psd.csv (if generated)
- All plots are in `figures/`.