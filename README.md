# EEGToAlphabets
### EEG-Based Alphabet Classification Using Deep Learning

## 1. Project Overview

**EEGToAlphabets** is a research project that investigates deep-learning approaches for classifying alphabet-related patterns from Electroencephalography (EEG) recordings.

The project uses preprocessed EEG windows and evaluates CNN-LSTM and LSTM-based neural-network configurations using PyTorch. The current experimental workflow uses Leave-One-Subject-Out (LOSO) evaluation, with optional subject-specific calibration.

**Current status:** EEG preprocessing and dataset validation are completed; deep-learning model development and evaluation are ongoing.

## 2. Objectives

- Organize and validate the raw EEG recordings.
- Preprocess EEG signals and assess data quality.
- Prepare EEG windows and corresponding metadata for model development.
- Develop and compare five neural-network configurations.
- Evaluate generalization to held-out participants using LOSO evaluation.
- Study the effect of calibration using a small number of trials from the held-out participant.

## 3. Dataset

The repository contains raw subject archives and preprocessed EEG data. The raw-data directory contains 20 subject ZIP archives.

| Parameter | Value |
|---|---|
| Dataset | https://data.mendeley.com/datasets/cp586sfzrt/4 |
| Raw subject archives | 20 |
| Target alphabet classes | A, K, N, P, Z |
| Sampling frequency | 125 Hz |
| Original channels | 16 |
| Retained channels | 15 |
| Window length | 8 samples |
| Window duration | 64 ms |
| Expected window tensor shape | `(N, 15, 8)` |

The exact number of usable trials and windows should be taken from the current dataset manifest and audit outputs rather than assumed from a fixed count.

## 4. EEG Preprocessing Pipeline

The EEG preprocessing and epoch-export pipeline is implemented in MATLAB using two configuration files:

- `scripts_version4/matlab/export_c3_picard_50HzNotch_FDF_epochs_PC.m`
- `scripts_version4/matlab/fdf_config_pc.m`

The exporter reads the raw OpenBCI recordings and their task-segment CSV files, processes the recordings, creates fixed-length EEG windows, and exports the EEG tensor together with labels, participant identifiers, trial/window identifiers, metadata, bad-channel audits, and ICA logs.

### 4.1 Dataset and channel configuration

- Sampling frequency: **125 Hz** (the configuration default; the loader also checks the raw-file header for a sample-rate value).
- Original EEG channels: **16**.
- Globally excluded channel: **T8**.
- Expected retained channels: **15**.
- Target classes: **A, K, N, P, Z**.
- Task trimming at the beginning/end: **0 seconds**.

### 4.2 Signal preprocessing sequence

For each recording, the MATLAB pipeline performs the following operations in this order:

1. **Median-centering:** subtracts the median of each channel from that channel's signal.
2. **50 Hz notch filtering:** applies a zero-phase notch filter at 50 Hz with quality factor **Q = 30** (`filtfilt` is used).
3. **Picard ICA:** attempts ICA on the good channels. Picard is preferred and required by the supplied configuration. The pipeline identifies candidate components by their absolute correlation with available frontal references (`Fp1` and/or `Fp2`), uses a correlation threshold of **0.35**, and removes no more than **2 components**.
4. **Frequency-domain filtering (FDF):** applies FFT-based suppression of the **4–15 Hz** band, with **1 Hz cosine transition bands** (3–4 Hz and 15–16 Hz), followed by the inverse FFT.
5. **Referencing:** the configured reference is `none`; no CAR or median reference is applied.
6. **Bad-channel imputation:** the configured strategy is `mean_good`, which replaces each identified bad channel with the per-sample mean of the good channels.

The order above reflects the implementation in `preprocessRaw`.

### 4.3 Bad-channel detection

The configuration identifies channels for exclusion from ICA/reference calculations and subsequent imputation using these checks:

- Rail threshold: **180,000 µV**.
- Maximum permitted rail-sample ratio: **5%**.
- Minimum channel standard deviation: **0.01**.
- Robust z-score threshold for channel standard deviation: **8**.
- The globally excluded channel **T8** is also marked for dropping.

The pipeline writes a `bad_channel_audit.csv` file for the recording-level channel-quality audit.

### 4.4 Epoch and window generation

The exporter reads task segments from the segment CSV files and selects rows whose segment type is `task` and whose label belongs to the configured classes.

The configured feature mode is `win8_0`:

| Parameter | Configuration |
|---|---:|
| Window length | 8 samples |
| Window step | 8 samples |
| Overlap | None |
| Sampling frequency | 125 Hz |
| Window duration | 64 ms |
| Task trim at start/end | 0 seconds |

Each exported window is stored as **15 channels × 8 time samples**. The resulting tensor is arranged as:

`X.shape = (number_of_windows, 15, 8)`

The exporter also saves labels (`y`), class names, participant IDs, trial and window IDs, sampling frequency, channel names, preprocessing name, feature-mode name, and a metadata CSV.

### 4.5 Epoch rejection and ICA logging

- `APPLY_BAD_EPOCH_REJECTION` is set to **false** in the supplied configuration. Therefore, the optional bad-epoch rejection routine is not enabled by this configuration.
- The exporter records whether ICA was applied, which algorithm was used, the removed component indices, and any ICA error in `ica_log.csv`.
- `REQUIRE_PICARD` is set to **true**. If Picard is unavailable or fails, the supplied configuration does not permit the runica fallback to be used as a substitute. Check the ICA log and run output to verify successful ICA processing.

### 4.6 Preprocessing output files

The MATLAB exporter writes the following principal outputs for each preprocessing/feature-mode combination:

- `epochs.mat` – EEG tensor and associated variables.
- `metadata.csv` – window/trial-level metadata.
- `bad_channel_audit.csv` – bad-channel audit.
- `ica_log.csv` – ICA status, algorithm, removed components, and errors.
- `channel_names.csv` – retained channel names.
- `fdf_session_index.csv` – discovered recording-session index.
- `fdf_epoch_manifest.csv` – exported epoch/feature-mode manifest.

The output directory and input-data directory are configured in `fdf_config_pc.m`; update those paths for the machine on which preprocessing is run.

## 5. Deep-Learning Models

The PyTorch notebook defines a shared model class, `CNNLSTMFamily`, and a model factory that supports the following five model names:

| Model name | CNN feature extractor | LSTM configuration |
|---|---|---|
| `CNN-1-LSTM` | One Conv1D block | 1-layer unidirectional LSTM |
| `CNN-2-LSTM` | One Conv1D block | 2-layer unidirectional LSTM |
| `CNN-2-Bi-LSTM` | One Conv1D block | 2-layer bidirectional LSTM |
| `CNN-3-LSTM` | One Conv1D block | 3-layer unidirectional LSTM |
| `3-LSTM` | No CNN block | 3-layer unidirectional LSTM |

**Implementation note:** In the supplied notebook, the `CNN-1`, `CNN-2`, and `CNN-3` names do not indicate one, two, or three stacked convolutional blocks. The shared implementation contains one Conv1D block when `use_cnn=True`; the model variants primarily differ in the number/directionality of LSTM layers. The `3-LSTM` configuration bypasses the CNN block.

### Shared CNN block

For CNN-enabled configurations, the model applies:

- Conv1D with 64 output filters.
- Batch normalization.
- ELU activation.
- Max pooling when the input time dimension permits it.
- Dropout (0.35).

The convolution kernel is selected from the input time dimension by the notebook code.

### LSTM and classification head

The LSTM hidden size is 64. The classification head applies dropout followed by a fully connected layer that produces class logits. Bidirectional mode doubles the LSTM output dimension before the classification head.

## 6. Experimental Protocol

The supplied notebook performs experiments using the FDF variant and `win8` mode selected from the epoch manifest.

### Data split

- Evaluation strategy: Leave-One-Subject-Out (LOSO).
- In each fold, 16 participants are used for training, 3 for validation (used for early stopping and checkpoint selection) and 1 is held out for testing.
- Training and validation data are prepared from the remaining participants by the helper functions in `deep_utils.py`.
- The notebook records split-audit information.
- Adaptive z-score normalization is fitted on the training windows and then applied to the training, validation, calibration, and test sets.

### Calibration settings

The final experimental setup evaluates Pure LOSO and three calibration conditions:

| Condition | Calibration trials per class | Total calibration trials | Share of the subject's trials | Test trials |
|---|---:|---:|---:|---:|
| Pure LOSO | 0 | 0 | 0% | 125 |
| 1-trial calibration | 1 | 5 | ~4% | 120 |
| 5-trial calibration | 5 | 25 | 20% | 100 |
| 10-trial calibration | 10 | 50 | 40% | 75 |

The calibration step fine-tunes the trained model using the selected calibration windows from the held-out participant. The test set is then evaluated at trial level.

These 1-, 5-, and 10-trial conditions represent the final calibration settings for the project. Earlier experimental configurations should not be treated as the final setup.

### Training configuration

The supplied notebook uses the following settings:

| Parameter | Value |
|---|---:|
| Random seed | 42 |
| Optimizer / loss | Adam / cross-entropy (gradient clipping at 5.0) |
| Maximum source-training epochs | 500 |
| Early-stopping patience | 30 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Calibration learning rate | 0.001 |
| Calibration patience | 30 |
| Maximum calibration epochs (1-trial condition) | 180 |
| Maximum calibration epochs (5- and 10-trial conditions) | 260 |
| Calibration batch size | 256 |
| Prediction batch size | 512 |

The actual device is selected by the helper function `get_device`; the notebook requests XPU preference through `prefer_xpu=True`. The available device depends on the runtime environment.

## 7. Evaluation Metrics and Outputs

The notebook evaluates predictions at trial level and records:

- Accuracy
- Balanced accuracy
- Macro F1-score

It also saves fold-level results, training history, split-leakage audit records, trial-level predictions, summary tables, and best-by-calibration summaries.

The results table below is intentionally left as a placeholder until the final model runs have been completed.

| Model | Accuracy | Balanced accuracy | Macro F1-score |
|---|---:|---:|---:|
| CNN-1-LSTM | TBD | TBD | TBD |
| CNN-2-LSTM | TBD | TBD | TBD |
| CNN-2-Bi-LSTM | TBD | TBD | TBD |
| CNN-3-LSTM | TBD | TBD | TBD |
| 3-LSTM | TBD | TBD | TBD |

Do not interpret existing classical-baseline or earlier experiment outputs as final results for all five deep-learning models.

## 8. Repository Structure

The following is a guide to the actual repository layout. The repository also contains detailed CSV audit outputs, plots, logs, and supporting evidence files.

```text
EEGToAlphabets/
├── .gitattributes
├── .gitignore
├── requirements.txt
├── classical_baseline.ipynb
├── eda_baseline.py
├── eegtoalphabets-cnn-1-lstm.ipynb
├── eegtoalphabets-cnn-2-lstm.ipynb
├── data_docs/
│   ├── baseline_handoff.md
│   └── dataset dictionaries and audit workbooks
├── metadata_version4/
│   ├── channels.csv
│   ├── participants.csv
│   └── task_description.csv
├── raw data_version4/
│   ├── TEST_SUBJECT_1.zip ... TEST_SUBJECT_20.zip
│   └── raw_zip_audit_corrected/
├── preprocessed_data_version4/
│   ├── epochs.mat
│   ├── epochs.zip
│   ├── trials_X.npy
│   ├── trials_meta.csv
│   ├── metadata.csv
│   ├── channel_names.csv
│   ├── bad_channel_audit.csv
│   ├── ica_log.csv
│   ├── fdf_epoch_manifest.csv
├── results_version4/
│   ├── classical_baseline/
│   ├── complete_dataset_qc/
│   ├── eda/
│   ├── figures/
│   ├── processed_qc/
│   ├── raw_processed_comparison/
│   ├── raw_qc/
│   ├── raw_qc_all/
│   ├── resultsDL/
│   ├── split_audit/
│   ├── targeted_raw_processed_checks/
│   ├── task_1_2_evidence/
│   └── zero_value_investigation/
└── scripts_version4/
    ├── matlab/
    │   ├── export_c3_picard_50HzNotch_FDF_epochs_PC.m
    │   └── fdf_config_pc.m
    └── python/
        ├── deep_utils.py
        ├── C3_Picard_50HzNotch_FDF_Baseline_LOSO.ipynb
        └── dataset audit, validation, and inspection scripts
```

## 9. Environment Setup

The supplied `requirements.txt` contains the following pinned packages:

```text
numpy==2.4.6
pandas==3.0.6
scipy==1.17.1
scikit-learn==1.9.1
h5py==3.16.0
```

The model notebook additionally imports PyTorch, Matplotlib, and IPython/Jupyter components. Install PyTorch using the official installation selector for the target CPU/GPU environment, and ensure Matplotlib and Jupyter are available in the notebook environment.

### Clone the repository

```bash
git clone https://github.com/Ajiti09/EEGToAlphabets.git
cd EEGToAlphabets
```

### Pull Git LFS data

```bash
git lfs install
git lfs pull
```

### Create and activate a virtual environment (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Install the pinned requirements

```bash
pip install -r requirements.txt
```

Install the compatible PyTorch build and notebook dependencies separately if they are not already present.

## 10. Running the Deep-Learning Experiment

The supplied notebook is configured for a Kaggle environment. Its setup cell searches `/kaggle/input` for the manifest, epoch data, metadata, audit files, channel names, and `deep_utils.py`, then stages the required files under `/kaggle/working/epochs_root`.

The notebook also expects `deep_utils.py` to be available from `/kaggle/working`. In this repository, the helper is located at:

`scripts_version4/python/deep_utils.py`

Make that helper available at the path expected by the notebook, or update the notebook's setup cell and import path to match your environment.

### Selecting the models

The uploaded `eegtoalphabets-cnn-1-lstm.ipynb` contains the five-model factory in its model-definition cell. The later `MODEL_NAMES` assignment controls which models are actually scheduled. For a run covering all five configurations, set that cell to:

```python
MODEL_NAMES = [
    'CNN-1-LSTM',
    'CNN-2-LSTM',
    'CNN-2-Bi-LSTM',
    'CNN-3-LSTM',
    '3-LSTM',
]
```

If using separate copies of the notebook for individual model experiments, change only `MODEL_NAMES` in the model-selection cell, as applicable; keep the remaining shared workflow unchanged unless the experiment design is intentionally modified.

Run the notebook cells in order. The notebook filters the manifest to `win8` mode and FDF variants, creates the experiment plan for each selected model and calibration setting, and runs the LOSO folds.

### Output location

The supplied notebook writes experiment outputs to:

`/kaggle/working/results`

The outputs include fold results, training history, split audit, trial-level predictions, summary files, and comparison figures.

Copy or download the results directory after the Kaggle session if the results need to be retained.

## 11. Current Project Status

| Component | Status |
|---|---|
| Raw dataset organization | Completed |
| Dataset inventory and audit | Completed |
| EEG preprocessing | Completed |
| EDA and signal-quality review | Completed |
| Metadata and window alignment checks | Completed |
| Deep-learning model experiments | Ongoing |
| Final comparison of all five models | Pending |
| Final performance analysis | Pending |

## 12. Future Work

- Complete training and evaluation for all five model configurations.
- Compare model performance across LOSO folds.
- Analyze the effect of 1-, 5- and 10-trial calibration.
- Review class-wise performance and confusion patterns.
- Assess generalization across participants.
- Document final experimental findings.

## 13. Contributor

**GitHub:** [Ajiti09](https://github.com/Ajiti09)
**GitHub:** [akashs62](https://github.com/akashs62)
**GitHub:** [Ananya-Priya21](https://github.com/Ananya-Priya21)
**GitHub:** [Khushi310105](https://github.com/Khushi310105)
**GitHub:** [ShivaniAgarwal04](https://github.com/ShivaniAgarwal04)

---

*EEGToAlphabets — Exploring deep-learning approaches for EEG-based alphabet classification.*
