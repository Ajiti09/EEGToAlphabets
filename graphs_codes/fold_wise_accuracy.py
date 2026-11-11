import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# ---------------------------------------------------------
# 1. Load & Filter Data
# ---------------------------------------------------------
df_folds = pd.read_csv('C:\\Users\\91629\\Desktop\\results\\all_fold_results.csv')

model_name = 'CNN-2-Bi-LSTM'
calib_name = 'calib_40pct_10trials'

df_filtered = df_folds[
    (df_folds['model'] == model_name) & 
    (df_folds['calibration_name'] == calib_name)
].sort_values('subject').copy()

subjects = df_filtered['subject'].tolist()
accuracies = (df_filtered['accuracy'] * 100).tolist()
mean_accuracy = np.mean(accuracies)

# ---------------------------------------------------------
# 2. Setup Figure & Color Mapping
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)

norm = Normalize(vmin=min(accuracies), vmax=max(accuracies))
colors = [cm.viridis(norm(acc)) for acc in accuracies]
bars = ax.bar(subjects, accuracies, color=colors, width=0.65)

# ---------------------------------------------------------
# 3. Add Reference Lines
# ---------------------------------------------------------
ax.axhline(
    y=mean_accuracy, 
    color='red', 
    linestyle='--', 
    linewidth=1.0, 
    label=f'Mean accuracy = {mean_accuracy:.1f}%'
)

ax.axhline(
    y=20, 
    color='gray', 
    linestyle=':', 
    linewidth=1.0, 
    label='Chance level (20%)'
)

# ---------------------------------------------------------
# 4. Annotations & Axis Formatting (Reduced Fonts & Margins)
# ---------------------------------------------------------
# Bar labels reduced to 6.5pt
for bar, acc in zip(bars, accuracies):
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0, 
        yval + 1.2, 
        f"{acc:.1f}%", 
        ha='center', 
        va='bottom', 
        fontsize=6.5, 
        fontweight='medium'
    )

# Axis ticks and labels
ax.set_ylim(0, 105)
ax.set_yticks(np.arange(0, 101, 20))
ax.set_yticklabels([f"{y}%" for y in np.arange(0, 101, 20)], fontsize=8)

ax.set_xticks(np.arange(len(subjects)))
ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=8)
ax.set_xlabel('Held-out participant', fontsize=9.5, labelpad=6)
ax.set_ylabel('Accuracy (%)', fontsize=9.5, labelpad=6)

# Grid styling
ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='lightgray')
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------------------------------------------------------
# 5. Title & Legend
# ---------------------------------------------------------
ax.set_title(
    f'Fold-wise LOSO accuracy: {model_name} | {calib_name}', 
    fontsize=10, 
    fontweight='bold', 
    pad=10
)

ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.22), 
    ncol=2, 
    frameon=False, 
    fontsize=8
)

# Adjust layout margins so y-axis text never gets clipped off
plt.tight_layout(rect=[0.03, 0.0, 1.0, 1.0])
plt.savefig('loso_accuracy_compact.png', bbox_inches='tight', pad_inches=0.2)
plt.show()