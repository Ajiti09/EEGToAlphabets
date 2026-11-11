import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------
# 1. Load Data from extracted results
# ---------------------------------------------------------
df = pd.read_csv('C:\\Users\\91629\\Desktop\\results\\fold_trial_predictions.csv')

# Filter for the CNN-2-Bi-LSTM model and 10-trials calibration
df_filtered = df[(df['model'] == 'CNN-2-Bi-LSTM') & (df['calibration_name'] == 'calib_40pct_10trials')]

# Extract unique labels in sorted order
labels = sorted(df_filtered['label_true'].unique())

# Compute raw confusion matrix counts
cm_counts = confusion_matrix(
    df_filtered['label_true'], 
    df_filtered['label_pred'], 
    labels=labels
)

# ---------------------------------------------------------
# 2. Calculate Row-Normalized Proportions (%)
# ---------------------------------------------------------
row_sums = cm_counts.sum(axis=1, keepdims=True)
cm_normalized = (cm_counts / row_sums) * 100

# ---------------------------------------------------------
# 3. Setup Custom Colormap & Compact Figure Size
# ---------------------------------------------------------
colors = ["#ffffcc", "#c7e9b4", "#41b6c4", "#1d91c0", "#225ea8", "#0c2c84"]
cmap = LinearSegmentedColormap.from_list("custom_blue_yellow", colors)

# Reduced canvas size to scale down matrix area
fig, ax = plt.subplots(figsize=(5.0, 4.2), dpi=150)

# Heatmap plot
sns.heatmap(
    cm_normalized,
    annot=False,
    cmap=cmap,
    vmin=0,
    vmax=100,
    cbar_kws={'label': 'Row-normalized proportion', 'ticks': [0, 20, 40, 60, 80, 100]},
    linewidths=0.6,
    linecolor='white',
    ax=ax
)

# Colorbar font formatting
cbar = ax.collections[0].colorbar
cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
cbar.ax.tick_params(labelsize=6.5)
cbar.ax.yaxis.label.set_size(8)

# ---------------------------------------------------------
# 4. Annotate Cells & Highlight Main Diagonal
# ---------------------------------------------------------
n_classes = len(labels)

for i in range(n_classes):
    for j in range(n_classes):
        pct = cm_normalized[i, j]
        count = cm_counts[i, j]
        
        is_diagonal = (i == j)
        
        text = f"{pct:.1f}%\n(n={count})"
        weight = 'bold' if is_diagonal else 'normal'
        color = 'white' if pct > 50 else 'black'
        fontsize = 7.0 if is_diagonal else 6.0
            
        ax.text(
            j + 0.5, i + 0.5, text,
            ha='center', va='center',
            color=color, fontweight=weight, fontsize=fontsize
        )

        # Highlight diagonal cells
        if is_diagonal:
            rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='#4a4a4a', lw=1.0)
            ax.add_patch(rect)

# ---------------------------------------------------------
# 5. Labels & Centered Large Title Styling
# ---------------------------------------------------------
ax.set_xticks(np.arange(n_classes) + 0.5)
ax.set_yticks(np.arange(n_classes) + 0.5)
ax.set_xticklabels(labels, fontweight='bold', fontsize=7.5)
ax.set_yticklabels(labels, fontweight='bold', fontsize=7.5, rotation=0)

ax.set_xlabel('Predicted label', fontsize=8, fontweight='bold', labelpad=5)
ax.set_ylabel('True label', fontsize=8, fontweight='bold', labelpad=5)

# Centered heading with increased font size (11.5pt)
ax.set_title('Confusion matrix: CNN-2-Bi-LSTM | calib_40pct_10trials', 
             fontsize=11.5, fontweight='bold', pad=14, loc='center')

# Save plot with outer margin padding so no label touches the edge
plt.savefig('confusion_matrix_centered_heading.png', bbox_inches='tight', pad_inches=0.3)
plt.show()