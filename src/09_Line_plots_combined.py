# Combined line plots of selected 1D and 2D NMR metabolites
# using the strongest signal per metabolite
# with Welch t-tests comparing PWR and Non-PWR at each visit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math

from pathlib import Path
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


# File paths
files_2d = [
    "../../result/data/nmr_only/DF1_nmr_only_annotated_2d.csv",
    "../../result/data/nmr_only/DF2_nmr_only_annotated_2d.csv",
    "../../result/data/nmr_only/DF3_nmr_only_annotated_2d.csv",
    "../../result/data/nmr_only/DF4_nmr_only_annotated_2d.csv",
    "../../result/data/nmr_only/DF5_nmr_only_annotated_2d.csv"
]

files_1d = [
    "../../result/1D_data/nmr_only/DF1_nmr_only_annotated_1d.csv",
    "../../result/1D_data/nmr_only/DF2_nmr_only_annotated_1d.csv",
    "../../result/1D_data/nmr_only/DF3_nmr_only_annotated_1d.csv",
    "../../result/1D_data/nmr_only/DF4_nmr_only_annotated_1d.csv",
    "../../result/1D_data/nmr_only/DF5_nmr_only_annotated_1d.csv"
]

#Visit information

visit_order = ["V1", "V2", "V3", "V4", "V5"]

visit_time_months = {
    "V1": -2.0,
    "V2": 1.25,
    "V3": 4.0,
    "V4": 8.0,
    "V5": 12.0
}

short_visit_labels = {
    "V1": "Third tri",
    "V2": "4-6 wk",
    "V3": "4 mo",
    "V4": "8 mo",
    "V5": "12 mo"
}

# 7 selected 2D metabolites
target_2d = [
    "Alanine",
    "Cysteine",
    "Glucose",
    "Glutathione",
    "Glycine",
    "Lactic acid",
    "Lysine"
]


# 7 selected 1D metabolites
target_1d = [
    "4-Hydroxyproline",
    "Acetic acid",
    "Acetone",
    "Creatinine",   
    "Galactose",
    "Pyruvic acid",
    "Xanthurenic acid"
]

metabolite_order = target_2d + target_1d

# Output directory

output_dir = Path("../test/result/figure")
output_dir.mkdir(parents=True, exist_ok=True)

def clean_metabolite_name(column_name):
    """
    Remove suffixes such as .1, .2, etc. added to duplicated
    metabolite columns.

    Example:
    Glucose.1 -> Glucose
    """

    return re.sub(r"\.\d+$", "", column_name)


def get_metabolite_cols(df, metabolite):
    """
    Find all signal columns corresponding to the same
    annotated metabolite.
    """

    return [
        column
        for column in df.columns
        if clean_metabolite_name(column) == metabolite
    ]

# Load 2D and 1D datasets

dfs_2d = []

for visit, file_path in zip(visit_order, files_2d):

    df = pd.read_csv(file_path)

    if "pwr_current" not in df.columns:
        raise KeyError(
            f"'pwr_current' was not found in {file_path}"
        )

    df["Visit"] = visit
    dfs_2d.append(df)


dfs_1d = []

for visit, file_path in zip(visit_order, files_1d):

    df = pd.read_csv(file_path)

    if "pwr_current" not in df.columns:
        raise KeyError(
            f"'pwr_current' was not found in {file_path}"
        )

    df["Visit"] = visit
    dfs_1d.append(df)


#Function to select strongest signal
def select_strongest_signals(
    dfs,
    target_metabolites,
    nmr_type
):

    """
    For each metabolite:
    1. Find all annotated signal columns.
    2. Calculate mean signal intensity at each visit.
    3. Average those visit-specific means across all visits.
    4. Select the signal with the highest average intensity.
    """

    signal_records = []

    for visit, df in zip(visit_order, dfs):

        for metabolite in target_metabolites:

            columns = get_metabolite_cols(
                df,
                metabolite
            )

            for column in columns:

                values = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

                signal_records.append({
                    "metabolite": metabolite,
                    "signal_col": column,
                    "visit": visit,
                    "mean_intensity": values.mean(
                        skipna=True
                    )
                })


    signal_mean_df = pd.DataFrame(signal_records)


    if signal_mean_df.empty:

        raise ValueError(
            f"No matching {nmr_type} metabolite columns "
            f"were found."
        )


    selected_signal_df = (
        signal_mean_df
        .groupby(
            ["metabolite", "signal_col"],
            as_index=False
        )["mean_intensity"]
        .mean()
        .sort_values(
            ["metabolite", "mean_intensity"],
            ascending=[True, False]
        )
        .groupby(
            "metabolite",
            as_index=False
        )
        .first()
        .rename(columns={
            "signal_col": "selected_signal",
            "mean_intensity":
                "mean_intensity_across_visits"
        })
    )


    selected_signal_df["NMR_type"] = nmr_type

    return selected_signal_df


# Select strongest signals separately for 2D and 1D
selected_signal_2d = select_strongest_signals(
    dfs=dfs_2d,
    target_metabolites=target_2d,
    nmr_type="2D"
)


selected_signal_1d = select_strongest_signals(
    dfs=dfs_1d,
    target_metabolites=target_1d,
    nmr_type="1D"
)


# Combine selected signal information
selected_signal_df = pd.concat(
    [
        selected_signal_2d,
        selected_signal_1d
    ],
    ignore_index=True
)

#Function to build participant-level data
def build_plot_data(
    dfs,
    target_metabolites,
    selected_signal_df,     
    nmr_type
):

    plot_records = []

    selected_map = dict(
        zip(
            selected_signal_df["metabolite"],
            selected_signal_df["selected_signal"]
        )
    )


    for visit, df in zip(visit_order, dfs):

        for metabolite in target_metabolites:

            selected_column = selected_map.get(
                metabolite
            )

            if selected_column is None:
                continue

            if selected_column not in df.columns:
                continue


            values = pd.to_numeric(
                df[selected_column],
                errors="coerce"
            )


            temp = pd.DataFrame({
                "visit": visit,
                "time_months":
                    visit_time_months[visit],

                "Groups":
                    df["pwr_current"].map({
                        0: "Non-PWR",
                        1: "PWR"
                    }),

                "metabolite": metabolite,
                "NMR_type": nmr_type,
                "selected_signal":
                    selected_column,
                "value": values
            })


            temp = temp.dropna(
                subset=[
                    "Groups",
                    "value"
                ]
            )


            if not temp.empty:
                plot_records.append(temp)


    if not plot_records:

        raise ValueError(
            f"No valid plotting data were created "
            f"for {nmr_type}."
        )


    return pd.concat(
        plot_records,
        ignore_index=True
    )

# Build 2D and 1D plotting data
plot_df_2d = build_plot_data(
    dfs=dfs_2d,
    target_metabolites=target_2d,
    selected_signal_df=selected_signal_2d,
    nmr_type="2D"
)


plot_df_1d = build_plot_data(
    dfs=dfs_1d,
    target_metabolites=target_1d,
    selected_signal_df=selected_signal_1d,
    nmr_type="1D"
)


# Combine 1D and 2D participant-level data
plot_df = pd.concat(
    [
        plot_df_2d,
        plot_df_1d
    ],
    ignore_index=True
)


plot_df["visit"] = pd.Categorical(
    plot_df["visit"],
    categories=visit_order,
    ordered=True
)


plot_df["metabolite"] = pd.Categorical(
    plot_df["metabolite"],
    categories=metabolite_order,
    ordered=True
)


plot_df = (
    plot_df
    .sort_values(
        [
            "metabolite",
            "visit",
            "Groups"
        ]
    )
    .reset_index(drop=True)
)

# 11. Welch t-test: PWR vs Non-PWR at each metabolite × visit
test_records = []


for (
    metabolite,
    nmr_type,
    visit,
    time_months
), sub_df in plot_df.groupby(
    [
        "metabolite",
        "NMR_type",
        "visit",
        "time_months"
    ],
    observed=True
):


    non_pwr_values = (
        pd.to_numeric(
            sub_df.loc[
                sub_df["Groups"] == "Non-PWR",
                "value"
            ],
            errors="coerce"
        )
        .dropna()
    )


    pwr_values = (
        pd.to_numeric(
            sub_df.loc[
                sub_df["Groups"] == "PWR",
                "value"
            ],
            errors="coerce"
        )
        .dropna()
    )


    non_pwr_n = len(non_pwr_values)
    pwr_n = len(pwr_values)


    non_pwr_mean = (
        non_pwr_values.mean()
        if non_pwr_n > 0
        else np.nan
    )


    pwr_mean = (
        pwr_values.mean()
        if pwr_n > 0
        else np.nan
    )


    # Welch independent-samples t-test
    if non_pwr_n >= 2 and pwr_n >= 2:

        t_statistic, p_value = ttest_ind(
            non_pwr_values,
            pwr_values,
            equal_var=False,
            nan_policy="omit"
        )

    else:

        t_statistic = np.nan
        p_value = np.nan


    test_records.append({

        "NMR_type": nmr_type,

        "metabolite": metabolite,

        "visit": visit,

        "time_months":
            time_months,

        "non_pwr_n":
            non_pwr_n,

        "pwr_n":
            pwr_n,

        "non_pwr_mean":
            non_pwr_mean,

        "pwr_mean":
            pwr_mean,

        "mean_difference_pwr_minus_nonpwr":
            pwr_mean - non_pwr_mean,

        "welch_t_statistic":
            t_statistic,

        "p_value":
            p_value
    })


test_results_df = pd.DataFrame(
    test_records
)


test_results_df["visit"] = pd.Categorical(
    test_results_df["visit"],
    categories=visit_order,
    ordered=True
)


test_results_df["metabolite"] = pd.Categorical(
    test_results_df["metabolite"],
    categories=metabolite_order,
    ordered=True
)


test_results_df = (
    test_results_df
    .sort_values(
        [
            "metabolite",
            "visit"
        ]
    )
    .reset_index(drop=True)
)

# Benjamini-Hochberg FDR correction

test_results_df["p_fdr"] = np.nan


valid_p_mask = (
    test_results_df["p_value"]
    .notna()
)


if valid_p_mask.sum() > 0:

    adjusted_p_values = multipletests(
        test_results_df.loc[
            valid_p_mask,
            "p_value"
        ],
        alpha=0.05,
        method="fdr_bh"
    )[1]


    test_results_df.loc[
        valid_p_mask,
        "p_fdr"
    ] = adjusted_p_values


test_results_df["significant_raw"] = (
    test_results_df["p_value"] <= 0.05
)


test_results_df["significant_fdr"] = (
    test_results_df["p_fdr"] <= 0.05
)

p_column_for_plot = "p_value"

# p_column_for_plot = "p_fdr"

significance_threshold = 0.05

# 14. Calculate mean, SD, n, and SE

summary_df = (
    plot_df
    .groupby(
        [
            "metabolite",
            "NMR_type",
            "Groups",
            "visit",
            "time_months"
        ],
        as_index=False,
        observed=True
    )
    .agg(
        mean_value=("value", "mean"),
        sd_value=("value", "std"),
        n=("value", "count")
    )
)


summary_df["se_value"] = (
    summary_df["sd_value"] /
    np.sqrt(summary_df["n"])
)


summary_df["visit"] = pd.Categorical(
    summary_df["visit"],
    categories=visit_order,
    ordered=True
)


summary_df["metabolite"] = pd.Categorical(
    summary_df["metabolite"],
    categories=metabolite_order,
    ordered=True
)


summary_df = (
    summary_df
    .sort_values(
        [
            "metabolite",
            "visit",
            "Groups"
        ]
    )
    .reset_index(drop=True)
)

# 15. Plot combined line plots
metabolites = [
    metabolite
    for metabolite in metabolite_order
    if metabolite in summary_df[
        "metabolite"
    ].astype(str).unique()
]


ncols = 4

nrows = math.ceil(
    len(metabolites) / ncols
)


fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(
        ncols * 3.3,
        nrows * 2.8
    ),
    sharex=True
)


axes = np.atleast_1d(
    axes
).flatten()


colors = {
    "Non-PWR": "#4C72B0",
    "PWR": "#C44E52"
}

# Draw each metabolite

for ax, metabolite in zip(
    axes,
    metabolites
):


    sub_metabolite = summary_df[
        summary_df["metabolite"].astype(str)
        == metabolite
    ].copy()


    # Identify whether this is 1D or 2D
    nmr_type = (
        sub_metabolite["NMR_type"]
        .iloc[0]
    )

    # Plot Non-PWR and PWR
    
    for group in [
        "Non-PWR",
        "PWR"
    ]:


        sub_group = (
            sub_metabolite[
                sub_metabolite["Groups"]
                == group
            ]
            .sort_values(
                "time_months"
            )
        )


        if sub_group.empty:
            continue


        plotting_se = (
            sub_group["se_value"]
            .fillna(0)
        )


        ax.errorbar(

            sub_group["time_months"],

            sub_group["mean_value"],

            yerr=plotting_se,

            marker="o",

            linestyle="-",

            capsize=3,

            linewidth=1.3,

            markersize=4,

            color=colors[group],

            label=group
        )

    # Add significance stars
    
    significant_visits = test_results_df[
        (
            test_results_df[
                "metabolite"
            ].astype(str)
            == metabolite
        )
        &
        (
            test_results_df[
                p_column_for_plot
            ]
            <= significance_threshold
        )
    ].copy()


    # Determine plotted range including SE
    se_for_position = (
        sub_metabolite["se_value"]
        .fillna(0)
    )


    lower_values = (
        sub_metabolite["mean_value"]
        -
        se_for_position
    )


    upper_values = (
        sub_metabolite["mean_value"]
        +
        se_for_position
    )


    data_lower = lower_values.min()
    data_upper = upper_values.max()

    data_range = (
        data_upper -
        data_lower
    )


    # Avoid zero range
    if (
        not np.isfinite(data_range)
        or data_range == 0
    ):

        reference_value = abs(
            data_upper
        )

        if (
            not np.isfinite(
                reference_value
            )
            or reference_value == 0
        ):

            reference_value = 1.0


        data_range = (
            reference_value
            * 0.10
        )


    star_positions = []


    for _, significant_row in (
        significant_visits.iterrows()
    ):


        visit_summary = sub_metabolite[
            sub_metabolite["visit"]
            ==
            significant_row["visit"]
        ].copy()


        if visit_summary.empty:
            continue


        # Position star above the higher error bar
        visit_upper = (
            visit_summary[
                "mean_value"
            ]
            +
            visit_summary[
                "se_value"
            ].fillna(0)
        ).max()


        star_y = (
            visit_upper
            +
            0.07 * data_range
        )


        ax.text(
            significant_row[
                "time_months"
            ],
            star_y,
            "*",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color="black"
        )


        star_positions.append(
            star_y
        )


    # Expand y-axis if stars are present
    if star_positions:


        new_lower_limit = (
            data_lower
            -
            0.05 * data_range
        )


        new_upper_limit = (
            max(
                max(star_positions),
                data_upper
            )
            +
            0.10 * data_range
        )


        ax.set_ylim(
            new_lower_limit,
            new_upper_limit
        )


    # Subplot formatting
    
    # Metabolite name only
    ax.set_title(
        metabolite,
        fontsize=9
    )


    ax.grid(False)


    ax.set_xticks([
        visit_time_months[visit]
        for visit in visit_order
    ])


    ax.set_xticklabels(
        [
            short_visit_labels[visit]
            for visit in visit_order
        ],
        rotation=45,
        ha="right",
        fontsize=8
    )


for ax in axes[
    len(metabolites):
]:

    fig.delaxes(ax)

fig.supxlabel(
    "Time relative to delivery",
    fontsize=11
)


fig.supylabel(
    "NMR signal intensity",
    fontsize=11
)

# legend

legend_handles = []
legend_labels = []


for ax in axes[
    :len(metabolites)
]:

    handles, labels = (
        ax.get_legend_handles_labels()
    )


    if handles:

        legend_handles = handles
        legend_labels = labels

        break


if legend_handles:

    fig.legend(
        legend_handles,
        legend_labels,
        title="Group",
        loc="upper right",
        bbox_to_anchor=(
            0.95,
            0.98
        ),
        ncol=2,
        frameon=False
    )

fig.suptitle(
    "PWR vs Non-PWR Across Visits for Selected NMR Metabolites",
    fontsize=14,
    y=0.98
)


plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.96
    ]
)

figure_path = (
    output_dir /
    "lineplots_selected_1d_2d_metabolites_with_significance.png"
)


plt.savefig(
    figure_path,
    dpi=600,
    bbox_inches="tight"
)

plt.show()
