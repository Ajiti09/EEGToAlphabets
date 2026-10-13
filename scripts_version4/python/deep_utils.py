"""Fully paper-style FD-F deep-learning utilities.

This module is for the MATLAB export produced by
run_01_export_paperstyle_fdf_epochs_PC.m

Pipeline philosophy:
- Picard/runica ICA, FD-F 4-15 Hz suppression, and 8-sample non-overlap segmentation already done in MATLAB
- adaptive z-score normalization is fitted inside each LOSO fold using source-training data only
- calibration/test split is by trial_uid, never by window index
- predictions are aggregated to trial level for reported metrics
"""
from __future__ import annotations

import copy
import csv
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import h5py
except Exception:
    h5py = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


# ----------------------------- Basic helpers -----------------------------

def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu"):
        try:
            if torch.xpu.is_available():
                torch.xpu.manual_seed_all(seed)
        except Exception:
            pass


def get_device(prefer_xpu: bool = True) -> torch.device:
    if prefer_xpu and hasattr(torch, "xpu"):
        try:
            if torch.xpu.is_available():
                return torch.device("xpu")
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps"):
        try:
            if torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------- Data loading -----------------------------

@dataclass
class EEGDataset:
    X: np.ndarray                 # N,C,T
    y: np.ndarray                 # zero-based class index
    labels: np.ndarray
    participants: np.ndarray
    trial_uid: np.ndarray
    window_uid: np.ndarray
    metadata: pd.DataFrame
    class_names: List[str]
    channel_names: List[str]
    fs: float
    variant: str
    mode: str
    window_sec: float


def read_channel_names(channel_path: str | Path) -> List[str]:
    p = Path(channel_path)
    if not p.exists():
        return []
    vals = []
    with open(p, "r", newline="") as f:
        for row in csv.reader(f):
            if row:
                vals.append(str(row[0]).strip())
    return vals


def _load_x_from_mat(mat_path: Path, expected_n: int) -> np.ndarray:
    if h5py is None:
        raise ImportError("h5py is required to load MATLAB -v7.3 .mat files")
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(mat_path)
    with h5py.File(mat_path, "r") as f:
        if "X" not in f:
            raise KeyError(f"X not found in {mat_path}")
        raw = np.array(f["X"], dtype=np.float32)
    if raw.ndim != 3:
        raise ValueError(f"Expected 3D X, got {raw.shape}")

    # MATLAB HDF5 often presents dimensions reversed. Use metadata length.
    if raw.shape[0] == expected_n:
        X = raw
    elif raw.shape[-1] == expected_n:
        X = np.moveaxis(raw, -1, 0)
    elif raw.shape[1] == expected_n:
        X = np.moveaxis(raw, 1, 0)
    else:
        # Fallback to closest dimension to expected_n.
        axis = int(np.argmin([abs(s - expected_n) for s in raw.shape]))
        X = np.moveaxis(raw, axis, 0)
    # After sample axis, ensure channels second and time last.
    # Channels expected <=64, time likely 250 or 500.
    if X.shape[1] > 64 and X.shape[2] <= 64:
        X = np.transpose(X, (0, 2, 1))
    if X.shape[0] != expected_n:
        raise ValueError(f"Loaded X has {X.shape[0]} rows but metadata has {expected_n}; raw={raw.shape}, X={X.shape}")
    return np.ascontiguousarray(X, dtype=np.float32)


def load_manifest(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    mf_path = root / "paperstyle_fdf_epoch_manifest.csv"
    if not mf_path.exists():
        raise FileNotFoundError(f"Manifest not found: {mf_path}")
    mf = pd.read_csv(mf_path)
    # Normalize columns to easier names.
    mf = mf.rename(columns={
        "preprocessing": "variant",
        "feature_mode": "mode",
        "epoch_mat_path": "mat_path",
    })
    for col in ["mat_path", "metadata_path", "audit_path", "ica_log_path", "channel_names_path"]:
        if col in mf.columns:
            fixed = []
            for i, p in enumerate(mf[col].astype(str).tolist()):
                pp = Path(p)
                if pp.exists():
                    fixed.append(str(pp))
                else:
                    fixed.append(str(root / str(mf.loc[i, "variant"]) / str(mf.loc[i, "mode"]) / pp.name))
            mf[col] = fixed
    return mf


def load_epoch_dataset(row: pd.Series) -> EEGDataset:
    meta_path = Path(str(row["metadata_path"]))
    meta = pd.read_csv(meta_path)
    if len(meta) == 0:
        raise ValueError(f"No metadata rows in {meta_path}")
    X = _load_x_from_mat(Path(str(row["mat_path"])), expected_n=len(meta))
    if "class_index" not in meta.columns:
        raise KeyError("metadata.csv must contain class_index")
    y = meta["class_index"].astype(int).to_numpy() - 1
    labels = meta["label"].astype(str).to_numpy() if "label" in meta.columns else y.astype(str)
    participants = meta["participant"].astype(str).to_numpy()
    trial_uid = meta["trial_uid"].astype(str).to_numpy()
    window_uid = meta["window_uid"].astype(str).to_numpy() if "window_uid" in meta.columns else trial_uid.copy()
    class_names = meta[["class_index", "label"]].drop_duplicates().sort_values("class_index")["label"].astype(str).tolist() if "label" in meta.columns else [str(i) for i in sorted(np.unique(y))]
    ch_names = read_channel_names(row.get("channel_names_path", ""))
    if not ch_names:
        ch_names = [f"ch{i+1}" for i in range(X.shape[1])]
    mode = str(row.get("mode", "unknown"))
    if "win2" in mode:
        window_sec = 2.0
    elif "win4" in mode:
        window_sec = 4.0
    else:
        window_sec = X.shape[2] / 125.0
    return EEGDataset(
        X=X, y=y.astype(np.int64), labels=labels, participants=participants,
        trial_uid=trial_uid, window_uid=window_uid, metadata=meta,
        class_names=class_names, channel_names=ch_names, fs=125.0,
        variant=str(row.get("variant", "C2_paperstyle_picard_fdf")),
        mode=mode, window_sec=window_sec,
    )


# ----------------------------- LOSO / calibration split -----------------------------

def deterministic_val_subjects(train_subjects: Sequence[str], heldout: str, val_fraction: float = 0.15) -> Tuple[List[str], List[str]]:
    subs = sorted([str(s) for s in train_subjects])
    rng = random.Random(abs(hash((heldout, "val_subjects"))) % (2**32))
    rng.shuffle(subs)
    n_val = max(1, int(round(len(subs) * val_fraction))) if len(subs) > 4 else 1
    val_subs = sorted(subs[:n_val])
    train_subs = sorted(subs[n_val:])
    return train_subs, val_subs


def select_calibration_trial_uids(y: np.ndarray, participants: np.ndarray, trial_uid: np.ndarray,
                                  heldout: str, trials_per_class: int, seed: int = 42) -> List[str]:
    if trials_per_class <= 0:
        return []
    rng = random.Random(abs(hash((heldout, trials_per_class, seed))) % (2**32))
    held_mask = participants == heldout
    calib_trials: List[str] = []
    for cls in sorted(np.unique(y)):
        class_trials = sorted(np.unique(trial_uid[held_mask & (y == cls)]).tolist())
        rng.shuffle(class_trials)
        chosen = class_trials[:min(trials_per_class, len(class_trials))]
        calib_trials.extend(chosen)
    return sorted(calib_trials)


def make_fold_indices(ds: EEGDataset, heldout: str, calib_trials_per_class: int, seed: int = 42) -> Dict[str, np.ndarray | List[str]]:
    subjects = sorted(np.unique(ds.participants).tolist())
    source_subjects = [s for s in subjects if s != heldout]
    tr_subs, val_subs = deterministic_val_subjects(source_subjects, heldout)
    train_idx = np.where(np.isin(ds.participants, tr_subs))[0]
    val_idx = np.where(np.isin(ds.participants, val_subs))[0]

    held_idx = np.where(ds.participants == heldout)[0]
    calib_trials = select_calibration_trial_uids(ds.y, ds.participants, ds.trial_uid, heldout, calib_trials_per_class, seed=seed)
    if calib_trials:
        calib_idx = held_idx[np.isin(ds.trial_uid[held_idx], calib_trials)]
        test_idx = held_idx[~np.isin(ds.trial_uid[held_idx], calib_trials)]
    else:
        calib_idx = np.array([], dtype=int)
        test_idx = held_idx
    return dict(train_idx=train_idx, val_idx=val_idx, calib_idx=calib_idx, test_idx=test_idx,
                train_subjects=tr_subs, val_subjects=val_subs, calib_trials=calib_trials)


def audit_split(ds: EEGDataset, heldout: str, split: Dict, calib_name: str, model_name: str) -> Dict:
    calib_idx = np.asarray(split["calib_idx"], dtype=int)
    test_idx = np.asarray(split["test_idx"], dtype=int)
    train_idx = np.asarray(split["train_idx"], dtype=int)
    val_idx = np.asarray(split["val_idx"], dtype=int)
    calib_trials = set(ds.trial_uid[calib_idx].tolist())
    test_trials = set(ds.trial_uid[test_idx].tolist())
    source_trials = set(ds.trial_uid[np.concatenate([train_idx, val_idx])].tolist()) if (len(train_idx)+len(val_idx)) else set()
    held_trials = set(ds.trial_uid[ds.participants == heldout].tolist())
    return dict(
        variant=ds.variant, mode=ds.mode, model=model_name, calibration_name=calib_name, subject=heldout,
        n_train_subjects=len(split["train_subjects"]), n_val_subjects=len(split["val_subjects"]),
        n_train_windows=int(len(train_idx)), n_val_windows=int(len(val_idx)),
        n_calib_windows=int(len(calib_idx)), n_test_windows=int(len(test_idx)),
        n_calib_trials=int(len(calib_trials)), n_test_trials=int(len(test_trials)),
        n_source_heldout_trial_overlap=int(len(source_trials & held_trials)),
        n_calib_test_trial_overlap=int(len(calib_trials & test_trials)),
        pass_no_source_heldout_overlap=(len(source_trials & held_trials) == 0),
        pass_no_calib_test_overlap=(len(calib_trials & test_trials) == 0),
    )


# ----------------------------- Adaptive z-score -----------------------------

def fit_adaptive_zscore(X_train: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel adaptive z-score. Fit only on source training data.

    X: [N, C, T]. Statistics are over samples and time for each channel.
    """
    mean = X_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = X_train.std(axis=(0, 2), keepdims=True).astype(np.float32)
    std = np.maximum(std, eps)
    return mean, std


def apply_adaptive_zscore(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((X - mean) / std, dtype=np.float32)


# ----------------------------- Models -----------------------------

class EEGNetLite(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int, F1: int = 8, D: int = 2, dropout: float = 0.5):
        super().__init__()
        F2 = F1 * D
        k = min(64, max(8, n_times // 4))
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, k), padding=(0, k//2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F2, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F2, F2, kernel_size=(1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            feat = self.block2(self.block1(dummy)).flatten(1).shape[1]
        self.classifier = nn.Linear(feat, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # B,1,C,T
        x = self.block2(self.block1(x))
        return self.classifier(x.flatten(1))


class ShallowConvNetLite(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int, n_filters: int = 40, dropout: float = 0.5):
        super().__init__()
        k = min(32, max(8, n_times // 8))
        self.temporal = nn.Conv2d(1, n_filters, kernel_size=(1, k), padding=(0, k//2), bias=False)
        self.spatial = nn.Conv2d(n_filters, n_filters, kernel_size=(n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.pool = nn.AvgPool2d(kernel_size=(1, max(2, n_times // 32)), stride=(1, max(2, n_times // 64)))
        self.drop = nn.Dropout(dropout)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            feat = self._features(dummy).flatten(1).shape[1]
        self.classifier = nn.Linear(feat, n_classes)

    def _features(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = x * x
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.drop(x)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.classifier(self._features(x).flatten(1))


class CNNLSTMFamily(nn.Module):
    """CNN-LSTM family inspired by the Scientific Reports paper.

    Input X: B,C,T. A temporal Conv1D extracts local features, then LSTM/BiLSTM layers
    model temporal dynamics. Variants are controlled by n_lstm_layers and bidirectional.
    """
    def __init__(self, n_channels: int, n_times: int, n_classes: int,
                 n_lstm_layers: int = 1, bidirectional: bool = False,
                 use_cnn: bool = True, conv_filters: int = 64, hidden: int = 64,
                 dropout: float = 0.35):
        super().__init__()
        self.use_cnn = use_cnn
        self.bidirectional = bidirectional
        self.n_lstm_layers = n_lstm_layers
        if use_cnn:
            self.conv = nn.Sequential(
                nn.Conv1d(n_channels, conv_filters, kernel_size=min(7, max(3, n_times if n_times % 2 == 1 else n_times-1)), padding=(min(7, max(3, n_times if n_times % 2 == 1 else n_times-1))//2), bias=False),
                nn.BatchNorm1d(conv_filters),
                nn.ELU(),
                nn.MaxPool1d(kernel_size=2) if n_times >= 4 else nn.Identity(),
                nn.Dropout(dropout),
            )
            lstm_input = conv_filters
        else:
            self.conv = None
            lstm_input = n_channels
        self.lstm = nn.LSTM(
            input_size=lstm_input, hidden_size=hidden, num_layers=n_lstm_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, n_classes))

    def forward(self, x):
        # x: B,C,T
        if self.use_cnn:
            z = self.conv(x)       # B,F,T'
            z = z.transpose(1, 2)  # B,T',F
        else:
            z = x.transpose(1, 2)  # B,T,C
        out, _ = self.lstm(z)
        last = out[:, -1, :]
        return self.head(last)


def build_model(name: str, n_channels: int, n_times: int, n_classes: int) -> nn.Module:
    name = str(name)
    if name == "EEGNet":
        return EEGNetLite(n_channels, n_times, n_classes)
    if name == "ShallowConvNet":
        return ShallowConvNetLite(n_channels, n_times, n_classes)
    if name in {"CNN_LSTM", "CNN-1-LSTM"}:
        return CNNLSTMFamily(n_channels, n_times, n_classes, n_lstm_layers=1, bidirectional=False, use_cnn=True)
    if name == "CNN_BiLSTM":
        return CNNLSTMFamily(n_channels, n_times, n_classes, n_lstm_layers=1, bidirectional=True, use_cnn=True)
    if name == "CNN-2-LSTM":
        return CNNLSTMFamily(n_channels, n_times, n_classes, n_lstm_layers=2, bidirectional=False, use_cnn=True)
    if name == "CNN-2-Bi-LSTM":
        return CNNLSTMFamily(n_channels, n_times, n_classes, n_lstm_layers=2, bidirectional=True, use_cnn=True)
    if name == "CNN-3-LSTM":
        return CNNLSTMFamily(n_channels, n_times, n_classes, n_lstm_layers=3, bidirectional=False, use_cnn=True)
    if name == "3-LSTM":
        return CNNLSTMFamily(n_channels, n_times, n_classes, n_lstm_layers=3, bidirectional=False, use_cnn=False)
    raise ValueError(f"Unknown model name: {name}")


# ----------------------------- Training and evaluation -----------------------------

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tx = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    ty = torch.from_numpy(y.astype(np.int64))
    return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
                device: torch.device, max_epochs: int = 220, patience: int = 28,
                batch_size: int = 128, lr: float = 1e-3, weight_decay: float = 1e-4,
                verbose: bool = False) -> Tuple[nn.Module, Dict]:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = None
    if X_val is not None and len(X_val) > 0:
        val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)
    best_state = copy.deepcopy(model.state_dict())
    best_score = -np.inf
    best_epoch = 0
    no_improve = 0
    history = []
    start = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.detach().cpu()) * len(yb)
            n_seen += len(yb)
        train_loss = total_loss / max(1, n_seen)

        if val_loader is not None:
            val_metrics = evaluate_windows(model, X_val, y_val, device, batch_size=batch_size)
            score = val_metrics["balanced_accuracy"]
        else:
            val_metrics = {"balanced_accuracy": -train_loss, "accuracy": np.nan, "macro_f1": np.nan}
            score = -train_loss
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k,v in val_metrics.items()}})
        if score > best_score:
            best_score = float(score)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if verbose and (epoch == 1 or epoch % 20 == 0):
            print(f"epoch={epoch:03d} loss={train_loss:.4f} val_bal={score:.4f}")
        if no_improve >= patience:
            break
    model.load_state_dict(best_state)
    info = dict(best_epoch=best_epoch, best_val_score=best_score, epochs_run=epoch, runtime_sec=time.time()-start, history=history)
    return model, info


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    loader = make_loader(X, np.zeros(len(X), dtype=np.int64), batch_size=batch_size, shuffle=False)
    probs = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probs.append(p)
    return np.vstack(probs) if probs else np.zeros((0, 0), dtype=np.float32)


def evaluate_windows(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int = 256) -> Dict[str, float]:
    if len(X) == 0:
        return {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan}
    p = predict_proba(model, X, device, batch_size=batch_size)
    pred = p.argmax(axis=1)
    return dict(
        accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
    )


def aggregate_trial_predictions(probs: np.ndarray, y: np.ndarray, trial_uid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    rows = []
    true = []
    trial_ids = []
    for tu in sorted(np.unique(trial_uid).tolist()):
        idx = np.where(trial_uid == tu)[0]
        if len(idx) == 0:
            continue
        mean_prob = probs[idx].mean(axis=0)
        rows.append(mean_prob)
        vals, counts = np.unique(y[idx], return_counts=True)
        true.append(vals[np.argmax(counts)])
        trial_ids.append(tu)
    if not rows:
        return np.zeros((0, probs.shape[1]), dtype=np.float32), np.array([], dtype=int), [], np.array([], dtype=int)
    arr = np.vstack(rows)
    pred = arr.argmax(axis=1)
    return arr, np.array(true, dtype=int), trial_ids, pred


def evaluate_trial_level(model: nn.Module, X: np.ndarray, y: np.ndarray, trial_uid: np.ndarray,
                         device: torch.device, batch_size: int = 256) -> Tuple[Dict[str, float], pd.DataFrame]:
    probs = predict_proba(model, X, device, batch_size=batch_size)
    if probs.size == 0:
        return {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan}, pd.DataFrame()
    _, y_true, trial_ids, y_pred = aggregate_trial_predictions(probs, y, trial_uid)
    metrics = dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )
    pred_df = pd.DataFrame({"trial_uid": trial_ids, "y_true": y_true, "y_pred": y_pred})
    return metrics, pred_df


# ----------------------------- Result IO -----------------------------

RESULT_COLUMNS = [
    "pipeline", "variant", "mode", "window_sec", "model", "calibration_name", "calib_trials_per_class",
    "subject", "n_train_subjects", "n_val_subjects", "n_train_windows", "n_val_windows", "n_calib_trials",
    "n_calib_windows", "n_test_trials", "n_test_windows", "accuracy", "balanced_accuracy", "macro_f1",
    "best_epoch", "epochs_run", "train_runtime_sec", "calib_runtime_sec", "total_runtime_sec", "device", "seed"
]

AUDIT_COLUMNS = [
    "variant", "mode", "model", "calibration_name", "subject", "n_train_subjects", "n_val_subjects",
    "n_train_windows", "n_val_windows", "n_calib_windows", "n_test_windows", "n_calib_trials", "n_test_trials",
    "n_source_heldout_trial_overlap", "n_calib_test_trial_overlap", "pass_no_source_heldout_overlap", "pass_no_calib_test_overlap"
]


def append_row_csv(path: str | Path, row: Dict, columns: List[str] = RESULT_COLUMNS) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    clean = {c: row.get(c, "") for c in columns}
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(clean)


def read_results(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.read_csv(path)


def result_exists(result_csv: str | Path, mode: str, model: str, calibration_name: str, subject: str) -> bool:
    df = read_results(result_csv)
    if df.empty:
        return False
    m = (df["mode"].astype(str) == str(mode)) & (df["model"].astype(str) == str(model)) & \
        (df["calibration_name"].astype(str) == str(calibration_name)) & (df["subject"].astype(str) == str(subject))
    return bool(m.any())


def summarize_results(result_csv: str | Path, out_csv: Optional[str | Path] = None) -> pd.DataFrame:
    df = read_results(result_csv)
    if df.empty:
        return df
    group_cols = ["pipeline", "variant", "mode", "window_sec", "model", "calibration_name", "calib_trials_per_class"]
    summary = df.groupby(group_cols, dropna=False).agg(
        n_subjects=("subject", "nunique"),
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        std_balanced_accuracy=("balanced_accuracy", "std"),
        mean_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std"),
        mean_train_runtime_sec=("train_runtime_sec", "mean"),
        mean_total_runtime_sec=("total_runtime_sec", "mean"),
    ).reset_index()
    summary = summary.sort_values("mean_balanced_accuracy", ascending=False)
    if out_csv is not None:
        ensure_dir(Path(out_csv).parent)
        summary.to_csv(out_csv, index=False)
    return summary


def save_best_by_calibration(summary: pd.DataFrame, out_csv: str | Path) -> pd.DataFrame:
    if summary.empty:
        best = summary
    else:
        best = summary.sort_values(["calibration_name", "mean_balanced_accuracy"], ascending=[True, False]).groupby("calibration_name", as_index=False).head(1)
    ensure_dir(Path(out_csv).parent)
    best.to_csv(out_csv, index=False)
    return best


def cleanup_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu" and hasattr(torch, "xpu"):
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass

