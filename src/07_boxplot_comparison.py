import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re


def build_plot_df(files, target_metabolites, group_col="pwr_current"):
    """
    Build tidy dataframe for plotting metabolite boxplots across visits.
    Replicated signals for the same metabolite are averaged within each sample.
    """
    dfs = [pd.read_csv(f) for f in files]
    plot_data = []

    for i, df in enumerate(dfs, start=1):
        for met in target_metabolites:
            cols = [c for c in df.columns if re.sub(r"\.\d+$", "", c) == met]
            if not cols:
                continue

            avg_val = df[cols].mean(axis=1)

            temp = pd.DataFrame({
                "visit": i,
                "Groups": df[group_col].map({0: "Non-PWR", 1: "PWR"}),
                "metabolite": met,
                "value": avg_val
            })
            plot_data.append(temp)

    if not plot_data:
        return pd.DataFrame(columns=["visit", "Groups", "metabolite", "value"])

    return pd.concat(plot_data, ignore_index=True)


def plot_metabolite_boxplots(plot_df, title, output_path, col_wrap=4, height=3, aspect=1):
    """
    Plot boxplots for metabolites across visits.
    """
    if plot_df.empty:
        print(f"No data available for plotting: {title}")
        return

    g = sns.catplot(
        data=plot_df,
        x="visit",
        y="value",
        hue="Groups",
        col="metabolite",
        col_wrap=col_wrap,
        kind="box",
        sharey=False,
        height=height,
        aspect=aspect
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Visit", "Average NMR Signal")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(title)
    g.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.show()


# =========================
# 2D NMR settings
# =========================
files_2d = [
    "../../result/data/nmr_only/DF1_nmr_only_annotated_2d.csv",
    "../../result/data/nmr_only/DF2_nmr_only_annotated_2d.csv",
    "../../result/data/nmr_only/DF3_nmr_only_annotated_2d.csv",
    "../../result/data/nmr_only/DF4_nmr_only_annotated_2d.csv",
    "../../result/data/nmr_only/DF5_nmr_only_annotated_2d.csv"
]

target_2d_only = {
    "Alanine", "Arabitol", "Creatine", "Cysteine", "Deoxyguanosine", "Glucose",
    "Glutamic acid", "Glutathione", "Glycerophosphocholine",
    "Glycine", "Guanosine diphosphate", "Isoleucine", "Lactic acid",
    "Leucine", "Lysine", "Maltose", "Ornithine", "Pinitol",
    "Proline", "Sulforaphane", "Syringic acid", "Triethylene Glycol",
    "Valine", "Xylitol", "Xylose", "o-Cresol"
}

plot_df_2d = build_plot_df(files_2d, target_2d_only)
plot_metabolite_boxplots(
    plot_df_2d,
    title="PWR vs Non-PWR Across Visits For 2D NMR Metabolites",
    output_path="../test/result/figure/boxplots_metabolites_2d.png"
)


# =========================
# 1D NMR settings
# =========================
files_1d = [
    "../../result/1D_data/nmr_only/DF1_nmr_only_annotated_1d.csv",
    "../../result/1D_data/nmr_only/DF2_nmr_only_annotated_1d.csv",
    "../../result/1D_data/nmr_only/DF3_nmr_only_annotated_1d.csv",
    "../../result/1D_data/nmr_only/DF4_nmr_only_annotated_1d.csv",
    "../../result/1D_data/nmr_only/DF5_nmr_only_annotated_1d.csv"
]

target_1d_only = {
    "1,1-Dimethylbiguanide",
    "1,5-Anhydrosorbitol",
    "2-Aminoisobutyric acid",
    "2-Chlorobenzoic acid",
    "3-Hydroxybenzoic acid",
    "4-Hydroxyproline",
    "Acetic acid",
    "Acetone",
    "Citric acid",
    "Creatinine",
    "Dimethylsulfide",
    "Galactose",
    "Glycolic acid",
    "Homocysteic acid",
    "Isovalerylglycine",
    "Mannose",
    "Methanol",
    "Paracetamol sulfate",
    "Pipecolic acid",
    "Putrescine",
    "Pyruvic acid",
    "Urea",
    "Uridine",
    "Xanthurenic acid",
    "Xylobiose",
    "alpha-Ketoisovaleric acid"
}

plot_df_1d = build_plot_df(files_1d, target_1d_only)
plot_metabolite_boxplots(
    plot_df_1d,
    title="PWR vs Non-PWR Across Visits For 1D NMR Metabolites",
    output_path="../test/result/figure/boxplots_metabolites_1d.png"
)