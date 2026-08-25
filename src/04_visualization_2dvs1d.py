import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Load the files ----------
df_2d = pd.read_csv("../test/result/data/meta+2dnmr/model_results_all_visits_selected_features_meta+selected2dnmr.csv")
df_1d = pd.read_csv("../test/result/data/meta+1dnmr/model_results_all_visits_selected_features_meta+selected1dnmr.csv")
visit_label_map = {
    "V1": "Third trimester",
    "V2": "4-6 weeks\npostpartum",
    "V3": "4 months\npostpartum",
    "V4": "8 months\npostpartum",
    "V5": "12 months\npostpartum"
}

visit_order = ["V1", "V2", "V3", "V4", "V5"]

# ---------- 2. Get the best CV ROC AUC for each Visit-Model ----------
def pick_best(df, data_type):
    best = df.loc[df.groupby(["Visit", "Model"])["CV ROC AUC"].idxmax()].copy()
    best["Data Type"] = data_type
    # Fill NaN for missing columns
    if "SE CV ROC AUC" not in best.columns:
        best["SE CV ROC AUC"] = np.nan
    return best[["Visit", "Model", "CV ROC AUC", "SE CV ROC AUC", "Data Type"]]

best_all = pd.concat([
    pick_best(df_2d, "2D"),
    pick_best(df_1d, "1D")
], ignore_index=True)

# Sort the data
visit_order = ["V1", "V2", "V3", "V4", "V5"]
model_order = sorted(best_all["Model"].unique())
best_all["Visit"] = pd.Categorical(best_all["Visit"], categories=visit_order, ordered=True)
best_all["Model"] = pd.Categorical(best_all["Model"], categories=model_order, ordered=True)

# ---------- 3. Plotting ----------
fig, axes = plt.subplots(2, 3, figsize=(12, 9), sharey=True)
axes = axes.flatten()

bar_width = 0.35
colors = {"2D": "#4C72B0", "1D": "#55A868"}

for idx, visit in enumerate(visit_order):
    ax = axes[idx]
    sub = best_all[best_all["Visit"] == visit].sort_values("Model")
    for i, model in enumerate(model_order):
        for dtype, offset in zip(["2D", "1D"], [-bar_width/2, bar_width/2]):
            row = sub[(sub["Model"] == model) & (sub["Data Type"] == dtype)]
            if row.empty:
                continue
            auc = row["CV ROC AUC"].values[0]
            se  = row["SE CV ROC AUC"].values[0]
            xpos = i + offset
            ax.bar(xpos, auc, width=bar_width, color=colors[dtype],
                   label=dtype if i == 0 else "")
            if not np.isnan(se):
                ax.errorbar(xpos, auc, yerr=se, fmt='none',
                            ecolor='black', capsize=4, lw=1)
            ax.text(xpos, auc + 0.02, f"{auc:.2f}", 
                    ha='center', va='bottom', fontsize=6)
    ax.set_title(visit_label_map[visit])
    ax.set_xticks(np.arange(len(model_order)))
    ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("CV ROC AUC")
    ax.grid(False)

# Remove the 6th empty subplot
fig.delaxes(axes[-1])

# Legend
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[d]) for d in colors]
labels  = list(colors.keys())
fig.legend(handles, labels, title="Data Type", loc="upper right")

fig.suptitle("Best CV ROC AUC per Model (1D vs 2D) across Visits",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("../test/result/figure/best_model_comparison_1d_vs_2d.png", dpi=600, bbox_inches='tight')
plt.show()