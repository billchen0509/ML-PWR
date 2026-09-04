import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULT_DATA_DIR = REPO_ROOT / "test" / "result" / "data"
FIGURE_DIR = REPO_ROOT / "test" / "result" / "figure"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 1. Load the files ----------
df_2d = pd.read_csv(RESULT_DATA_DIR / "meta+2dnmr" / "model_results_all_visits_selected_features_meta+selected2dnmr.csv")
df_1d = pd.read_csv(RESULT_DATA_DIR / "meta+1dnmr" / "model_results_all_visits_selected_features_meta+selected1dnmr.csv")

visit_label_map = {
    "V1": "Third trimester",
    "V2": "4-6 weeks\npostpartum",
    "V3": "4 months\npostpartum",
    "V4": "8 months\npostpartum",
    "V5": "12 months\npostpartum"
}

visit_order = ["V1", "V2", "V3", "V4", "V5"]

# ---------- 2. Get the best CV ROC AUC for each Visit ----------
def pick_best_by_visit(df, data_type):
    df = df.copy()
    
    # Make sure CV ROC AUC is numeric
    df["CV ROC AUC"] = pd.to_numeric(df["CV ROC AUC"], errors="coerce")
    
    # Remove rows without CV ROC AUC
    df = df.dropna(subset=["CV ROC AUC"])
    
    # Pick the best model within each visit
    best = df.loc[df.groupby("Visit")["CV ROC AUC"].idxmax()].copy()
    best["Data Type"] = data_type
    
    # Fill NaN for missing SE column
    if "SE CV ROC AUC" not in best.columns:
        best["SE CV ROC AUC"] = np.nan
    
    return best[["Visit", "Model", "CV ROC AUC", "SE CV ROC AUC", "Data Type"]]

best_all = pd.concat([
    pick_best_by_visit(df_2d, "2D"),
    pick_best_by_visit(df_1d, "1D")
], ignore_index=True)

# Sort visits
best_all["Visit"] = pd.Categorical(best_all["Visit"], categories=visit_order, ordered=True)
best_all = best_all.sort_values(["Visit", "Data Type"])

# ---------- 3. Plotting ----------
fig, ax = plt.subplots(figsize=(9, 5.5))

bar_width = 0.34
x = np.arange(len(visit_order))

colors = {"2D": "#4C72B0", "1D": "#55A868"}
offsets = {"2D": -bar_width / 2, "1D": bar_width / 2}

for dtype in ["2D", "1D"]:
    sub = best_all[best_all["Data Type"] == dtype].copy()
    
    auc_values = []
    se_values = []
    model_labels = []
    
    for visit in visit_order:
        row = sub[sub["Visit"] == visit]
        if row.empty:
            auc_values.append(np.nan)
            se_values.append(np.nan)
            model_labels.append("")
        else:
            auc_values.append(row["CV ROC AUC"].values[0])
            se_values.append(row["SE CV ROC AUC"].values[0])
            model_labels.append(row["Model"].values[0])
    
    xpos = x + offsets[dtype]
    
    bars = ax.bar(
        xpos,
        auc_values,
        width=bar_width,
        color=colors[dtype],
        label=dtype
    )
    
    # Add error bars if SE is available
    for i, (auc, se) in enumerate(zip(auc_values, se_values)):
        if not np.isnan(auc) and not np.isnan(se):
            ax.errorbar(
                xpos[i],
                auc,
                yerr=se,
                fmt="none",
                ecolor="black",
                capsize=4,
                lw=1
            )
    
    # Abbreviate model names for cleaner figure annotation
    model_abbrev = {
    "MultiLayer Perceptron": "MLP",
    "Logistic Regression": "LR",
    "Random Forest": "RF"
    }

    # Add AUC values and abbreviated model names above each bar
    for i, (auc, se, model) in enumerate(zip(auc_values, se_values, model_labels)):
        if not np.isnan(auc):
            short_model = model_abbrev.get(model, model)
            
            # Put label above the error bar if SE exists
            if not np.isnan(se):
                label_y = auc + se + 0.025
            else:
                label_y = auc + 0.025
            
            ax.text(
                xpos[i],
                label_y,
                f"{auc:.2f}\n({short_model})",
                ha="center",
                va="bottom",
                fontsize=8,
                linespacing=1.25
            )

# X-axis
ax.set_xticks(x)
ax.set_xticklabels([visit_label_map[v] for v in visit_order], fontsize=9)

# Y-axis
ax.set_ylabel("CV ROC AUC")
ax.set_ylim(0, 1.15)

# Legend and title
ax.legend(title="Data Type", fontsize=9)
ax.set_title("Best-performing 1D and 2D Models Across Visits", fontsize=13,pad=18)

ax.grid(False)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "best_1d_2d_model_by_visit.png", dpi=600, bbox_inches="tight")
plt.show()