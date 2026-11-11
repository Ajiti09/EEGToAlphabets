import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Reset matplotlib parameters to clean defaults
plt.rcdefaults()

# 2. Data setup for all 4 calibration levels
data = {
    "calib_40pct_10trials (CNN-2-Bi-LSTM)": [92.13, 92.13, 92.08],
    "calib_20pct_5trials (CNN-2-Bi-LSTM)": [79.50, 79.50, 79.21],
    "calib_approx5pct_1trial (CNN-2-LSTM)": [39.63, 39.63, 38.98],
    "pure_loso (CNN-2-Bi-LSTM)": [26.32, 26.32, 23.77],
}

metrics = ["Accuracy", "Balanced accuracy", "Macro F1"]
df = pd.DataFrame(data, index=metrics)

# 3. Setup Plot Dimensions
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

x = np.arange(len(df.index))
width = 0.16  # Symmetrical bar width

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# 4. Plot Grouped Bars & Bar Labels
models = list(df.columns)
n_models = len(models)

for i, model in enumerate(models):
    offset = (i - n_models / 2 + 0.5) * width
    bars = ax.bar(
        x + offset,
        df[model],
        width,
        label=model,
        color=colors[i],
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

# 5. Place Main Title High Above Legend (using fig.suptitle)
fig.suptitle(
    "Best model metrics by calibration setting",
    fontsize=12,
    fontweight="bold",
    y=0.96,
)

# 6. Axis Labels & Formatting
ax.set_ylabel("Score (%)", fontsize=8.5, labelpad=8)
ax.set_xticks(x)
ax.set_xticklabels(df.index, fontsize=8.5)
ax.set_ylim(0, 105)
ax.set_yticks(np.arange(0, 101, 20))
ax.set_yticklabels([f"{y}%" for y in np.arange(0, 101, 20)], fontsize=7.5)

# Grid and Spine Styling
ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)

# 7. Legend Placed Below Main Title (no overlap)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.14),  # Positions legend neatly between title & top axis
    ncol=2,
    frameon=False,
    fontsize=7.5,
    handletextpad=0.4,
    columnspacing=1.5,
)

# 8. Set Margins to Leave Room for Legend & Title
plt.subplots_adjust(top=0.82, bottom=0.10, left=0.08, right=0.96)
plt.show()