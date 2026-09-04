import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FIGURE_DIR = REPO_ROOT / "test" / "result" / "figure"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Data
data = {
    "Participant": [
        'A', 'B', 'C', 'D', 'E',
        'F', 'G', 'H', 'I'
    ],
    "V1": [
        0.227, 0.966, 0.005, 0.010, 0.964,
        0.990, 0.993, 0.008, 0.058
    ],
    "V2": [
        0.234, 0.997, np.nan, 0.970, 0.582,
        0.974, 0.328, 0.919, 0.770
    ],
    "V3": [
        np.nan, 0.574, np.nan, 0.602, np.nan,
        np.nan, np.nan, np.nan, np.nan
    ],
    "V4": [
        np.nan, 0.851, np.nan, 0.773, np.nan,
        np.nan, np.nan, 0.718, np.nan
    ],
}

df = pd.DataFrame(data).set_index("Participant")

# Visit labels
visit_labels = [
    "Third trimester",
    "4-6 weeks\npostpartum",
    "4 months\npostpartum",
    "8 months\npostpartum"
]

values = df.to_numpy(dtype=float)
masked_values = np.ma.masked_invalid(values)

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Low probability = light; high probability = dark
cmap = plt.cm.Blues.copy()

# Missing values shown in light gray
cmap.set_bad(color="lightgray")

heatmap = ax.imshow(
    masked_values,
    aspect="auto",
    cmap=cmap,
    vmin=0,
    vmax=1
)

# Axis labels
ax.set_xticks(range(len(visit_labels)))
ax.set_xticklabels(visit_labels)

ax.set_yticks(range(len(df.index)))
ax.set_yticklabels(df.index.astype(str))

ax.set_xlabel("Visit")
ax.set_ylabel("Participant")
ax.set_title("Predicted Probability by Participant and Visit")

# Add probability values to each cell
for row_idx in range(values.shape[0]):
    for col_idx in range(values.shape[1]):
        value = values[row_idx, col_idx]

        if np.isnan(value):
            ax.text(
                col_idx,
                row_idx,
                "-",
                ha="center",
                va="center",
                color="black"
            )
        else:
            # White text on dark cells, black text on light cells
            text_color = "white" if value > 0.60 else "black"

            ax.text(
                col_idx,
                row_idx,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9
            )

# Color bar
colorbar = fig.colorbar(heatmap, ax=ax)
colorbar.set_label("Predicted probability")

plt.tight_layout()
plt.savefig(FIGURE_DIR / "participant_probability_heatmap.png", dpi=600, bbox_inches="tight")
plt.show()