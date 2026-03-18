import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.container import BarContainer

# =========================
# 1. Load and combine data
# =========================
file_paths = {
    "Meta + Selected 2D NMR": "../result/data/meta+nmr/selection/model_results_all_visits_selection_leak.xlsx",
    "Meta + All 2D NMR": "../result/data/meta+nmr/model_results_all_visits_hyper_filtered.xlsx",
    "Meta Only": "../result/data/meta_only/model_results_all_visits_hyper_meta_filtered.xlsx",
    "All 2D NMR Only": "../result/data/nmr_only/model_results_all_visits_hyper_nmr.xlsx",
}

dataframes = []

for source_name, path in file_paths.items():
    df = pd.read_excel(path)
    df["Source"] = source_name
    df["Model"] = df["Model"].str.strip()
    dataframes.append(df)

df_all = pd.concat(dataframes, ignore_index=True)

# Standardize Selected 2D metric column names
df_all.loc[
    df_all["Source"] == "Meta + Selected 2D NMR", "Average CV Accuracy"
] = df_all.loc[
    df_all["Source"] == "Meta + Selected 2D NMR", "CV Accuracy"
]

df_all.loc[
    df_all["Source"] == "Meta + Selected 2D NMR", "Average CV ROC AUC"
] = df_all.loc[
    df_all["Source"] == "Meta + Selected 2D NMR", "CV ROC AUC"
]

# Keep only highest AUC per Visit + Model for Selected 2D NMR
df_all_selected = df_all[df_all["Source"] == "Meta + Selected 2D NMR"]
df_all_selected = (
    df_all_selected.sort_values(by="Average CV ROC AUC", ascending=False)
    .groupby(["Visit", "Model"], as_index=False)
    .first()
)

df_all_others = df_all[df_all["Source"] != "Meta + Selected 2D NMR"]
df_all = pd.concat([df_all_others, df_all_selected], ignore_index=True)


# =========================================
# 2. Plot performance per visit
# =========================================
def plot_model_per_visit(df, metric="Average CV Accuracy"):
    df = df.copy()
    df["Model"] = df["Model"].str.strip()
    df["Visit"] = df["Visit"].astype(str)

    # Choose corresponding SE column
    if "Accuracy" in metric:
        se_col = "SE CV Accuracy"
    elif "ROC AUC" in metric:
        se_col = "SE CV ROC AUC"
    else:
        raise ValueError(f"Unrecognized metric: {metric}")

    visits = sorted(df["Visit"].dropna().unique())
    sources = sorted(df["Source"].dropna().unique())
    palette = sns.color_palette("Set2", n_colors=len(sources))

    for visit in visits:
        data = df[df["Visit"] == visit]
        if data.empty:
            continue

        plt.figure(figsize=(10, 8))
        ax = sns.barplot(
            data=data,
            x="Model",
            y=metric,
            hue="Source",
            palette=palette,
            errorbar=None,
            dodge=True,
        )

        ax.set_title(f"Model Performance {metric} - Visit {visit}", fontsize=12, pad=10)
        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=60, labelsize=9)
        ax.tick_params(axis="y", labelsize=10)

        # Add error bars using direct data mapping
        for i, bar in enumerate(ax.patches):
            model_idx = i % len(data["Model"].unique())
            source_idx = i // len(data["Model"].unique())

            models = data["Model"].unique()
            sources = data["Source"].unique()

            if source_idx >= len(sources) or model_idx >= len(models):
                continue

            model = models[model_idx]
            source = sources[source_idx]

            row = data[
                (data["Model"] == model) &
                (data["Source"] == source)
            ]

            if not row.empty:
                se = row[se_col].values[0]
                ax.errorbar(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    yerr=se,
                    fmt="none",
                    color="black",
                    capsize=5,
                    elinewidth=2,
                    alpha=0.6,
                )

        # Add value labels
        for container in ax.containers:
            if isinstance(container, BarContainer):
                labels = [f"{val:.3f}" for val in container.datavalues]
                ax.bar_label(
                    container,
                    labels=labels,
                    label_type="edge",
                    fontsize=6,
                    padding=3
                )

        plt.legend(
            title="Source",
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=10,
            title_fontsize=11
        )
        plt.tight_layout()
        # plt.savefig(f"../result/figure/model_performance_{metric.lower().replace(' ', '_')}_visit_{visit}.png", dpi=600, bbox_inches='tight')
        plt.show()


plot_model_per_visit(df_all, metric="Average CV Accuracy")
plot_model_per_visit(df_all, metric="Average CV ROC AUC")


# =========================================
# 3. Plot comparison per model across visits
# =========================================
def plot_model_comparison_per_model(df_all, metrics="Average CV Accuracy", error="SE CV Accuracy"):
    df_all = df_all.copy()
    df_all["Model"] = df_all["Model"].str.strip()

    models = sorted(df_all["Model"].unique())
    visits = sorted(df_all["Visit"].unique())
    metrics = metrics.strip()
    error = error.strip() if error else None
    palette = sns.color_palette("Set2")

    # Determine subplot grid size
    n_models = len(models)
    ncols = 3
    nrows = int(np.ceil(n_models / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows), sharey=True)
    axes = axes.flatten()

    legend_handles = None

    for i, model in enumerate(models):
        ax = axes[i]
        data = df_all[df_all["Model"] == model]

        sns.barplot(
            data=data,
            x="Visit",
            y=metrics,
            hue="Source",
            ax=ax,
            palette=palette[:4],
            errorbar=None
        )

        ax.set_title(model)

        # Save legend handles from the first plot only
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        # Remove legends from individual subplots
        ax.get_legend().remove()

        if i % ncols == 0:
            ax.set_ylabel(metrics)
        else:
            ax.set_ylabel("")

        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Visit")
        else:
            ax.set_xlabel("")

        ax.set_xticks(range(len(visits)))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        # Add error bars using direct data mapping
        data_sorted = data.sort_values(["Visit", "Source"]).reset_index(drop=True)

        for j, bar in enumerate(ax.patches):
            if j < len(data_sorted):
                err_val = data_sorted.loc[j, error]
                if pd.notna(err_val) and err_val > 0:
                    x = bar.get_x() + bar.get_width() / 2
                    height = bar.get_height()

                    ax.errorbar(
                        x,
                        height,
                        yerr=err_val,
                        fmt="none",
                        color="black",
                        capsize=5,
                        elinewidth=1.5,
                        alpha=0.8
                    )

        # Add value labels
        for container in ax.containers:
            if isinstance(container, BarContainer):
                labels = [f"{val:.3f}" for val in container.datavalues]
                ax.bar_label(
                    container,
                    labels=labels,
                    label_type="edge",
                    fontsize=6,
                    padding=3
                )

    # Hide unused subplots
    for j in range(len(models), len(axes)):
        axes[j].set_visible(False)

    # Add global legend to the right of the whole figure
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            bbox_to_anchor=(1.02, 1),
            fontsize=12,
            title="Source"
        )

    plt.suptitle(f"Model Comparison Across Visits: {metrics}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 0.90, 0.96])
    plt.savefig(
        f"../result/figure/model_comparison_{metrics.lower().replace(' ', '_')}.png",
        dpi=750,
        bbox_inches="tight"
    )
    plt.show()


plot_model_comparison_per_model(df_all, metrics="Average CV Accuracy", error="SE CV Accuracy")
plot_model_comparison_per_model(df_all, metrics="Average CV ROC AUC", error="SE CV ROC AUC")


# =========================================
# 4. Delta bar plot across visits
# =========================================
def plot_model_delta(df, metrics="Average CV Accuracy"):
    delta_df = df.pivot_table(
        index=["Visit", "Model"],
        columns="Source",
        values=metrics
    ).reset_index()

    delta_df["Meta + Selected 2D NMR - Meta + All 2D NMR"] = (
        delta_df["Meta + Selected 2D NMR"] - delta_df["Meta + All 2D NMR"]
    )
    delta_df["Meta + Selected 2D NMR - All 2D NMR Only"] = (
        delta_df["Meta + Selected 2D NMR"] - delta_df["All 2D NMR Only"]
    )
    delta_df["Meta + Selected 2D NMR - Meta Only"] = (
        delta_df["Meta + Selected 2D NMR"] - delta_df["Meta Only"]
    )

    delta_long = pd.melt(
        delta_df,
        id_vars=["Visit", "Model"],
        value_vars=[
            "Meta + Selected 2D NMR - Meta + All 2D NMR",
            "Meta + Selected 2D NMR - All 2D NMR Only",
            "Meta + Selected 2D NMR - Meta Only"
        ],
        var_name="Delta Type",
        value_name="Delta"
    )

    visits = sorted(delta_long["Visit"].unique())
    delta_types = delta_long["Delta Type"].unique()
    color_map = {dt: palette[i] for i, dt in enumerate(delta_types)}

    n_visits = len(visits)
    fig, axes = plt.subplots(1, n_visits, figsize=(15, 8), sharey=True)

    for i, visit in enumerate(visits):
        ax = axes[i]
        visit_data = delta_long[delta_long["Visit"] == visit]

        sns.barplot(
            data=visit_data,
            x="Model",
            y="Delta",
            hue="Delta Type",
            hue_order=delta_types,
            palette=[color_map[dt] for dt in delta_types],
            errorbar=None,
            ax=ax
        )

        ax.set_title(f"Visit {visit}")
        ax.set_xlabel("Model")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel("Performance Gain" if i == 0 else "")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=6)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[dt], label=dt)
        for dt in delta_types
    ]

    fig.legend(
        handles=legend_elements,
        loc="center right",
        bbox_to_anchor=(1.1, 1.0),
        title="Delta Type",
        frameon=True,
        framealpha=0.9
    )

    plt.suptitle(
        f"Performance Gain of Meta + Selected 2D NMR Compared with Others - {metrics}",
        fontsize=14
    )
    plt.savefig(
        f"../result/figure/model_delta_{metrics.lower().replace(' ', '_')}.png",
        dpi=600,
        bbox_inches="tight"
    )
    plt.show()


plot_model_delta(df_all, metrics="Average CV Accuracy")
plot_model_delta(df_all, metrics="Average CV ROC AUC")