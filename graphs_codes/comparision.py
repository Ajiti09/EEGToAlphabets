import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reset matplotlib parameters to clean defaults
plt.rcdefaults()

# 1. Data setup
calibrations = [
    "0 trials\n(pure_loso)",
    "1 trial\n(approx 5%)",
    "5 trials\n(20%)",
    "10 trials\n(40%)",
]

data = {
    "CNN-1-LSTM": [23.76, 38.58, 76.25, 90.40],
    "CNN-2-LSTM": [24.24, 39.63, 79.05, 91.60],
    "CNN-2-Bi-LSTM": [26.32, 37.17, 79.50, 92.13],
    "CNN-3-LSTM": [23.04, 39.21, 76.00, 91.33],
    "3-LSTM": [24.62, 33.64, 72.25, 87.20],
}

df = pd.DataFrame(data, index=calibrations)

# 2. Setup Plot Dimensions
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

x = np.arange(len(df.index))
width = 0.14  # Symmetrical bar spacing

# Color palette matching reference
colors = ["#4C72B0", "#F58518", "#54A24B", "#B07AA1", "#E45756"]

# 3. Draw Chance Level Line (20%)
ax.axhline(
    y=20,
    color="gray",
    linestyle="--",
    linewidth=1.0,
    zorder=1,
    label="Chance level (20%)",
)

# 4. Plot Grouped Bars & Compact Bar Labels
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

    # Small numbers directly above bars (6pt font, upright/horizontal)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,  # Reduced font size to avoid overlap
            fontweight="bold",
        )

# 5. Title & Label Formatting with strict font controls
ax.set_title(
    "Deep-learning model accuracy comparison",
    fontsize=11,
    fontweight="bold",
    pad=28,  # Extra space below title for legend
)
ax.set_ylabel("Mean trial-level accuracy (%)", fontsize=8.5, labelpad=8)
ax.set_xlabel("Evaluation setting", fontsize=8.5, labelpad=8)

# Set ticks and limits
ax.set_xticks(x)
ax.set_xticklabels(df.index, fontsize=7.5)
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

# 6. Compact Horizontal Legend
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=6,  # Single horizontal row for legend items
    frameon=False,
    fontsize=7.5,
    handletextpad=0.3,
    columnspacing=1.0,
)

# 7. Explicit Layout Adjustments (Leaves ample margin at top and bottom)
plt.subplots_adjust(top=0.88, bottom=0.15, left=0.08, right=0.98)
plt.show()