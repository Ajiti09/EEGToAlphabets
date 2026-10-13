import h5py
import numpy as np
import pandas as pd
data_dir = r"D:\EEGToAlphabets\preprocessed_data_version4"
with h5py.File(data_dir + r"\epochs.mat", "r") as f:
    meta = pd.read_csv(data_dir + r"\metadata.csv")

    def read_strings(name):
        refs = f[name][0]
        return np.array([
            "".join(chr(c) for c in f[ref][:].ravel())
            for ref in refs
        ])

    checks = {
        "participant": read_strings("participants") == meta["participant"].astype(str).to_numpy(),
        "label": read_strings("labels") == meta["label"].astype(str).to_numpy(),
        "trial_uid": read_strings("trial_uid") == meta["trial_uid"].astype(str).to_numpy(),
        "window_uid": read_strings("window_uid") == meta["window_uid"].astype(str).to_numpy(),
        "class_index": f["y"][0].astype(int) == meta["class_index"].to_numpy(dtype=int),
    }

    for name, result in checks.items():
        print(f"{name}: {result.sum()}/{len(result)} match")
