import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

# Input files: full filtered data
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULT_DIR = REPO_ROOT / "test" / "result"

files = [
    RESULT_DIR / "data" / "meta+2dnmr" / "DF1_filtered_2d.csv",
    RESULT_DIR / "data" / "meta+2dnmr" / "DF2_filtered_2d.csv",
    RESULT_DIR / "data" / "meta+2dnmr" / "DF3_filtered_2d.csv",
    RESULT_DIR / "data" / "meta+2dnmr" / "DF4_filtered_2d.csv",
    RESULT_DIR / "data" / "meta+2dnmr" / "DF5_filtered_2d.csv",
]

# Outcome
outcome = "pwr_current"

# Variables by type
continuous_cols = ['wt_gain', 'weight_change', 'age_delivery_range2', 'num_alc_weekly']
categorical_cols = ['vig_activity', 'race_1', 'race_2', 'caffeine', 'smoke_current', 'ppg_BMI', 'first_BMI']

# Required columns to keep from DFx_filtered.csv
meta_cols = ["visit", outcome] + continuous_cols + categorical_cols

# Output paths
out_dir = RESULT_DIR / "meta"
out_dir.mkdir(parents=True, exist_ok=True)

out_global = out_dir / "pwr_meta_global_tests.csv"
out_visit_all = out_dir / "pwr_meta_by_visit_tests.csv"
out_visit_sig = out_dir / "pwr_meta_by_visit_significant_only.csv"


# Load full data and keep only metadata part
dfs = []
for f in files:
    if not f.exists():
        raise FileNotFoundError(f"Missing input file: {f}")
    df = pd.read_csv(f)
    df = df[meta_cols].copy()
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# Ensure outcome is numeric
df_all[outcome] = pd.to_numeric(df_all[outcome], errors="coerce")


def test_continuous_vs_binary(series, y):
    """Mann-Whitney U test between two groups defined by binary y."""
    a = series[y == 0].dropna()
    b = series[y == 1].dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    stat, p = mannwhitneyu(a, b, alternative='two-sided')
    return stat, p


def test_categorical_vs_binary(series, y):
    """
    Chi-square for general tables.
    Fisher's exact only when table is 2x2.
    Returns (test_name, statistic_OR, p).
    For Fisher: 'statistic_OR' is the odds ratio.
    For Chi-square: 'statistic_OR' is the chi2 statistic.
    """
    ct = pd.crosstab(series, y)

    # Drop empty rows/cols to avoid errors
    ct = ct.loc[(ct.sum(axis=1) > 0), (ct.sum(axis=0) > 0)]

    if ct.empty or ct.shape[1] < 2:
        return np.nan, np.nan, np.nan

    if ct.shape == (2, 2):
        try:
            odds, p = fisher_exact(ct.values)
            return "Fisher's exact", odds, p
        except Exception:
            chi2, p, dof, exp = chi2_contingency(ct)
            return "Chi-square", chi2, p
    else:
        chi2, p, dof, exp = chi2_contingency(ct)
        return "Chi-square", chi2, p


# Global tests pooled across visits
global_results = []

# Continuous variables
for col in continuous_cols:
    stat, p = test_continuous_vs_binary(df_all[col], df_all[outcome])
    global_results.append([col, "Mann-Whitney U", stat, p])

# Categorical variables
for col in categorical_cols:
    test_name, stat_or, p = test_categorical_vs_binary(df_all[col], df_all[outcome])
    global_results.append([col, test_name, stat_or, p])

global_df = pd.DataFrame(
    global_results,
    columns=["Variable", "Test", "Statistic/OR", "p_value"]
)

# FDR correction
global_df["p_fdr_bh"] = multipletests(global_df["p_value"], method="fdr_bh")[1]

# Sort by adjusted p-value
global_df = global_df.sort_values("p_fdr_bh").reset_index(drop=True)

# Save global results
global_df.to_csv(out_global, index=False)


# Per-visit tests + per-visit FDR
visit_rows = []

for visit, dfv in df_all.groupby("visit"):
    yv = pd.to_numeric(dfv[outcome], errors="coerce")
    tmp_rows = []

    # Continuous variables
    for col in continuous_cols:
        stat, p = test_continuous_vs_binary(dfv[col], yv)
        tmp_rows.append([visit, col, "Mann-Whitney U", stat, p])

    # Categorical variables
    for col in categorical_cols:
        test_name, stat_or, p = test_categorical_vs_binary(dfv[col], yv)
        tmp_rows.append([visit, col, test_name, stat_or, p])

    tmp_df = pd.DataFrame(
        tmp_rows,
        columns=["Visit", "Variable", "Test", "Statistic/OR", "p_value"]
    )

    # FDR within visit
    tmp_df["p_fdr_bh"] = multipletests(tmp_df["p_value"], method="fdr_bh")[1]
    visit_rows.append(tmp_df)

visit_df = pd.concat(visit_rows, ignore_index=True)

# Save all per-visit results
visit_df.to_csv(out_visit_all, index=False)


# Significant-only summary per visit
visit_sig_df = (
    visit_df.loc[visit_df["p_fdr_bh"] < 0.05, ["Visit", "Variable", "p_fdr_bh"]]
    .sort_values(["Visit", "p_fdr_bh"])
    .reset_index(drop=True)
)

visit_sig_df.to_csv(out_visit_sig, index=False)