# Classical Baseline and EDA 

## Dataset and Signal Findings 
- Trials: 2,500
- Participants: 20
- Classes: A, K, N, P, Z
- EEG channels: 15
- Tensor shape: `(2500, 15, 1248)`
- Sampling rate: 125 Hz
- Trial duration: 9.984 seconds
- Data type: float32
Four time-domain features were extracted per channel: mean, standard deviation, RMS, and peak-to-peak amplitude. This gives 60 features per trial.

Some unusually high signal amplitudes were observed during EDA. Signal units have not been independently verified, and these values were not automatically removed or clipped.

## Classical Baseline 
- Model: Logistic Regression
- Preprocessing: StandardScaler in a scikit-learn Pipeline
- Evaluation: Existing 19-fold participant-wise manifest
- Features: 60 time-domain features per trial
- Maximum iterations: 2,000
- Random state: 42
The scaler is fitted on training data within each fold to prevent test-fold leakage.

## Initial Results
- Mean fold accuracy: 22.40%
- Mean fold balanced accuracy: 22.40%
- Mean fold macro F1: 0.2015
- Mean fold weighted F1: 0.2015
Most-confused alphabet pairs, measured by mutual confusions:
1. K–P: 261
2. A–K: 240
3. A–N: 235
4. K–Z: 212
5. P–Z: 206
The five-class chance-level reference is 20%. These results provide an initial baseline and indicate limited class discrimination using these features alone.

## Reproducibility
Run `classical_baseline.ipynb` in Google Colab after making the tensor, metadata, and split manifest available in the configured data directory. Execute notebook cells in order. The notebook saves predictions, fold scores, class-wise metrics, model settings, and confusion-matrix outputs.

