import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path

# Input
INPUT = Path("../../Matthias Klein's files - GWG/OB70 SHIPP3 SELECTED_marked.xlsx")
BMI_INPUT = Path("../../Matthias Klein's files - GWG/OB70 SHIPP3 SMALL DATASET-BMI.xlsx")
# Output
OUT_XLSX = Path('../test/result/data/Table1_PWR_nonPWR_recalculated.xlsx')
OUT_CSV_1A = Path('../test/result/data/Table1a_overall_PWR_nonPWR.csv')
OUT_CSV_1B = Path('../test/result/data/Table1b_timepoints_PWR_nonPWR.csv')

# -------------------------
# Load and clean data
# -------------------------
raw_df = pd.read_excel(INPUT)
meta_df = raw_df.dropna(subset=['pwr_current']).copy()
meta_df.drop(columns=[c for c in ['pwr_any','pwr_first','pp_weight_loss1','pp_weight_loss2','pp_weight_loss3'] if c in meta_df.columns], inplace=True)
meta_df.replace(to_replace=['.', 'NA', ''], value=np.nan, inplace=True)
for col in meta_df.columns:
    if meta_df[col].dtype == 'object':
        meta_df[col] = pd.to_numeric(meta_df[col], errors='coerce')

# Impute missing wt_gain using previous project method
mask = meta_df['weight_change_v1'].notna() & meta_df['wt_gain'].notna()
delta = (meta_df.loc[mask, 'wt_gain'] - meta_df.loc[mask, 'weight_change_v1']).mean()
missing_mask = meta_df['wt_gain'].isna() & meta_df['weight_change_v1'].notna()
n_imputed_wt_gain = int(missing_mask.sum())
meta_df.loc[missing_mask, 'wt_gain'] = meta_df.loc[missing_mask, 'weight_change_v1'] + delta

# Define groups: pwr_current 1 = PWR, 0 = non-PWR
meta_df['group'] = np.where(meta_df['pwr_current'].astype(float) == 1, 'PWR', 'non-PWR')

# Add BMI values 
bmi_df = pd.read_excel(BMI_INPUT)
bmi_df = bmi_df[['participant_id', 'ppg_BMI_value']].copy()

bmi_df['participant_id'] = pd.to_numeric(bmi_df['participant_id'], errors='coerce')
bmi_df['ppg_BMI_value'] = pd.to_numeric(bmi_df['ppg_BMI_value'], errors='coerce')

meta_df['participant_id'] = pd.to_numeric(meta_df['participant_id'], errors='coerce')

meta_df = meta_df.merge(
    bmi_df,
    on = 'participant_id',
    how = 'left'
)

vig_activity_map = {
    1: 0,
    2: 1,
    3: 2.5,
    4: 4,
    5: 8
}

meta_df['vig_activity_numeric'] = meta_df['vig_activity'].map(vig_activity_map)
# -------------------------
# Helper functions
# -------------------------
def fmt_mean_sd(s):
    x = pd.to_numeric(s, errors='coerce').dropna()
    if len(x) == 0:
        return ''
    if len(x) == 1:
        return f'{x.mean():.2f} ± NA'
    return f'{x.mean():.2f} ± {x.std(ddof=1):.2f}'

def fmt_count_pct(mask, denom):
    n = int(mask.sum())
    return f'{n} ({n/denom*100:.1f}%)' if denom else ''

def continuous_test(series):
    data = pd.DataFrame({'x': pd.to_numeric(series, errors='coerce'), 'g': meta_df['group']}).dropna()
    x0 = data.loc[data['g'] == 'non-PWR', 'x']
    x1 = data.loc[data['g'] == 'PWR', 'x']
    if len(x0) == 0 or len(x1) == 0:
        return np.nan, 'Mann-Whitney U'
    p = stats.mannwhitneyu(x0, x1, alternative='two-sided', method='auto').pvalue
    return float(p), 'Mann-Whitney U'

def regular_ttest(series):
    data = pd.DataFrame({
        'x': pd.to_numeric(series, errors='coerce'),
        'g': meta_df['group']
    }).dropna()
    x0 = data.loc[data['g'] == 'non-PWR', 'x']
    x1 = data.loc[data['g'] == 'PWR', 'x']

    if len(x0) < 2 or len(x1) < 2:
        return np.nan, 'Student\'s t-test'
    p = stats.ttest_ind(
        x0, x1, equal_var=True
    ).pvalue
    return float(p), 'Student\'s t-test'
def categorical_test(cat_series):
    data = pd.DataFrame({'cat': cat_series, 'g': meta_df['group']}).dropna()
    if data.empty or data['cat'].nunique() < 2 or data['g'].nunique() < 2:
        return np.nan, ''
    ct = pd.crosstab(data['cat'], data['g'])
    for g in ['non-PWR', 'PWR']:
        if g not in ct.columns:
            ct[g] = 0
    ct = ct[['non-PWR', 'PWR']]
    ct = ct.loc[ct.sum(axis=1) > 0]

    if ct.shape == (2, 2):
        chi2, p_chi, dof, expected = stats.chi2_contingency(ct, correction=False)
        if (expected < 5).any():
            _, p = stats.fisher_exact(ct.to_numpy())
            return float(p), 'Fisher exact'
        return float(p_chi), 'Chi-square'
    else:
        chi2, p, dof, expected = stats.chi2_contingency(ct, correction=False)
        return float(p), 'Chi-square'

def fmt_p(p):
    if pd.isna(p):
        return ''
    return '<0.001' if p < 0.001 else f'{p:.3f}'

def add_fdr_across_tables(overall_df, time_df):
    all_p = pd.concat([overall_df['p_raw'], time_df['p_raw']], ignore_index=True)
    valid = all_p.notna()
    q_all = pd.Series(np.nan, index=all_p.index)
    reject_all = pd.Series('', index=all_p.index, dtype='object')
    if valid.sum() > 0:
        reject, qvals, _, _ = multipletests(all_p[valid], alpha=0.05, method='fdr_bh')
        q_all.loc[valid] = qvals
        reject_all.loc[valid] = np.where(reject, 'Yes', 'No')
    n_overall = len(overall_df)
    overall_df = overall_df.copy()
    time_df = time_df.copy()
    overall_df['p_FDR'] = q_all.iloc[:n_overall].values
    overall_df['FDR<0.05'] = reject_all.iloc[:n_overall].values
    time_df['p_FDR'] = q_all.iloc[n_overall:].values
    time_df['FDR<0.05'] = reject_all.iloc[n_overall:].values
    return overall_df, time_df

def make_display(df):
    out = df.copy()
    out['p-value'] = out['p_raw'].apply(fmt_p)
    out['FDR-adjusted p'] = out['p_FDR'].apply(fmt_p)
    return out[['Variable', 'Overall sample', 'non-PWR', 'PWR', 'p-value', 'FDR-adjusted p', 'FDR<0.05', 'test']]

# -------------------------
# Create any-time variables
# -------------------------
visits = ['v1', 'v2', 'v3', 'v4', 'v5']
behavior_labels = {
    'smoke_current': 'Smoked at any time from third trimester to 12 months postpartum, n (%)',
    'caffeine': 'Consumed caffeine at any time from third trimester to 12 months postpartum, n (%)',
    'num_alc_weekly': 'Consumed alcohol at any time from third trimester to 12 months postpartum, n (%)'
}
for prefix in behavior_labels:
    cols = [f'{prefix}_{v}' for v in visits if f'{prefix}_{v}' in meta_df.columns]
    vals = meta_df[cols].apply(pd.to_numeric, errors='coerce')
    meta_df[f'{prefix}_anytime'] = np.where(vals.notna().any(axis=1), vals.gt(0).any(axis=1).astype(float), np.nan)

# -------------------------
# Table 1a: Overall data
# -------------------------
overall_rows = []

def add_cont_row(label, col):
    p, test = continuous_test(meta_df[col])
    overall_rows.append({'Variable': label,
                         'Overall sample': fmt_mean_sd(meta_df[col]),
                         'non-PWR': fmt_mean_sd(meta_df.loc[meta_df['group'] == 'non-PWR', col]),
                         'PWR': fmt_mean_sd(meta_df.loc[meta_df['group'] == 'PWR', col]),
                         'p_raw': p,
                         'test': test})
def add_ttest_cont_row(label, col):
    p, test = regular_ttest(meta_df[col])

    overall_rows.append({
        'Variable': label,
        'Overall sample': fmt_mean_sd(meta_df[col]),
        'non-PWR': fmt_mean_sd(
            meta_df.loc[meta_df['group'] == 'non-PWR', col]
        ),
        'PWR': fmt_mean_sd(
            meta_df.loc[meta_df['group'] == 'PWR', col]
        ),
        'p_raw': p,
        'test': test
    })
    
def add_cat_block(label, cat_series, categories):
    p, test = categorical_test(cat_series)
    overall_rows.append({'Variable': label, 'Overall sample': '', 'non-PWR': '', 'PWR': '', 'p_raw': p, 'test': test})
    valid = cat_series.notna()
    for value, lab in categories:
        row_mask = (cat_series == value) & valid
        overall_rows.append({'Variable': f'  {lab}',
                             'Overall sample': fmt_count_pct(row_mask, int(valid.sum())),
                             'non-PWR': fmt_count_pct(row_mask[meta_df['group'] == 'non-PWR'], int(valid[meta_df['group'] == 'non-PWR'].sum())),
                             'PWR': fmt_count_pct(row_mask[meta_df['group'] == 'PWR'], int(valid[meta_df['group'] == 'PWR'].sum())),
                             'p_raw': np.nan,
                             'test': ''})

def add_binary_row(label, col):
    series = meta_df[col]
    p, test = categorical_test(series)
    valid = series.notna()
    yes_mask = series.eq(1) & valid
    overall_rows.append({'Variable': label,
                         'Overall sample': fmt_count_pct(yes_mask, int(valid.sum())),
                         'non-PWR': fmt_count_pct(yes_mask[meta_df['group'] == 'non-PWR'], int(valid[meta_df['group'] == 'non-PWR'].sum())),
                         'PWR': fmt_count_pct(yes_mask[meta_df['group'] == 'PWR'], int(valid[meta_df['group'] == 'PWR'].sum())),
                         'p_raw': p,
                         'test': test})

add_cont_row('Total gestational weight gain, mean ± SD', 'wt_gain')
add_cont_row('Age at delivery, years, mean ± SD', 'age_delivery_range2')
add_cont_row('Pre-pregnancy BMI, mean ± SD', 'ppg_BMI_value')

race = pd.Series(np.nan, index=meta_df.index, dtype='object')
race[meta_df['race_1'].eq(1)] = 'White'
race[(meta_df['race_1'].ne(1)) & meta_df['race_2'].eq(1)] = 'Black'
race[(meta_df['race_1'].eq(0)) & (meta_df['race_2'].eq(0))] = 'Other'
add_cat_block('Race, n (%)', race, [('White', 'White'), ('Black', 'Black'), ('Other', 'Other')])
add_cat_block('Pre-pregnancy BMI category, n (%)', meta_df['ppg_BMI'], [(0, 'Underweight (<18.5)'), (1, 'Normal weight (18.5 to <25)'), (2, 'Overweight (25 to <30)'), (3, 'Obese (≥30)')])
add_ttest_cont_row(
    'Vigorous activity frequency, estimated times/month, mean ± SD',
    'vig_activity_numeric'
)
for prefix, label in behavior_labels.items():
    add_binary_row(label, f'{prefix}_anytime')
overall_df = pd.DataFrame(overall_rows)

# -------------------------
# Table 1b: Time point data
# -------------------------
visit_labels = {'v1': 'Third trimester', 'v2': '4-6 weeks postpartum', 'v3': '4 months postpartum', 'v4': '8 months postpartum', 'v5': '12 months postpartum'}
time_rows = []

def add_visit_header(v):
    time_rows.append({'Variable': visit_labels[v], 'Overall sample': '', 'non-PWR': '', 'PWR': '', 'p_raw': np.nan, 'test': ''})

def add_binary_visit(label, col):
    series = pd.to_numeric(meta_df[col], errors='coerce')
    binary_series = series.gt(0).where(series.notna(), np.nan)
    p, test = categorical_test(binary_series)
    valid = series.notna()
    yes_mask = series.gt(0) & valid
    time_rows.append({'Variable': f'  {label}',
                      'Overall sample': fmt_count_pct(yes_mask, int(valid.sum())),
                      'non-PWR': fmt_count_pct(yes_mask[meta_df['group'] == 'non-PWR'], int(valid[meta_df['group'] == 'non-PWR'].sum())),
                      'PWR': fmt_count_pct(yes_mask[meta_df['group'] == 'PWR'], int(valid[meta_df['group'] == 'PWR'].sum())),
                      'p_raw': p,
                      'test': test})

def add_cont_visit(label, col):
    p, test = continuous_test(meta_df[col])
    time_rows.append({'Variable': f'  {label}',
                      'Overall sample': fmt_mean_sd(meta_df[col]),
                      'non-PWR': fmt_mean_sd(meta_df.loc[meta_df['group'] == 'non-PWR', col]),
                      'PWR': fmt_mean_sd(meta_df.loc[meta_df['group'] == 'PWR', col]),
                      'p_raw': p,
                      'test': test})

for v in visits:
    add_visit_header(v)
    add_binary_visit('Smokers, n (%)', f'smoke_current_{v}')
    add_binary_visit('Caffeine consumers, n (%)', f'caffeine_{v}')
    add_binary_visit('Alcohol consumers, n (%)', f'num_alc_weekly_{v}')
    add_cont_visit('Weight change from pre-pregnancy, mean ± SD', f'weight_change_{v}')
time_df = pd.DataFrame(time_rows)

# -------------------------
# FDR and display tables
# -------------------------
overall_df, time_df = add_fdr_across_tables(overall_df, time_df)
overall_display = make_display(overall_df)
time_display = make_display(time_df)

overall_display.to_csv(OUT_CSV_1A, index=False)
time_display.to_csv(OUT_CSV_1B, index=False)

notes = pd.DataFrame({'Item': ['Input file', 'Analysis sample', 'PWR coding', 'PWR n', 'non-PWR n', 'Excluded due to missing pwr_current', 'wt_gain imputation', 'Number of wt_gain values imputed', 'Continuous variable test', 'Binary categorical variable test', 'Multi-category variable test', 'FDR method', 'FDR scope', 'FDR significance threshold'],
                      'Details': [INPUT.name, f'n = {len(meta_df)}', 'pwr_current: 1 = PWR, 0 = non-PWR', int((meta_df['group'] == 'PWR').sum()), int((meta_df['group'] == 'non-PWR').sum()), int(raw_df['pwr_current'].isna().sum()), 'Missing wt_gain imputed as weight_change_v1 + mean(wt_gain - weight_change_v1)', n_imputed_wt_gain, 'Mann–Whitney U test', 'Chi-square if expected counts adequate; Fisher exact test for sparse 2x2 tables', 'Chi-square test', 'Benjamini–Hochberg', 'Across all valid p-values from Table 1a and Table 1b combined', 'FDR-adjusted p < 0.05']})

with pd.ExcelWriter(OUT_XLSX, engine='xlsxwriter') as writer:
    overall_display.to_excel(writer, sheet_name='Table 1a Overall', index=False, startrow=2)
    time_display.to_excel(writer, sheet_name='Table 1b Timepoints', index=False, startrow=2)
    notes.to_excel(writer, sheet_name='Analysis notes', index=False)
    workbook = writer.book
    title_fmt = workbook.add_format({'bold': True, 'font_size': 14, 'align': 'left', 'valign': 'vcenter'})
    subtitle_fmt = workbook.add_format({'italic': True, 'font_size': 10, 'text_wrap': True})
    header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9EAF7', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
    body_fmt = workbook.add_format({'border': 1, 'valign': 'top', 'text_wrap': True})
    group_fmt = workbook.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1, 'valign': 'top'})
    sig_fmt = workbook.add_format({'bg_color': '#FFF2CC', 'border': 1, 'valign': 'top', 'text_wrap': True})
    sheets_info = [('Table 1a Overall', overall_display, 'Table 1a. Overall participant characteristics and behavioral variables by postpartum weight retention status.'), ('Table 1b Timepoints', time_display, 'Table 1b. Behavioral variables and weight change across time points by postpartum weight retention status.')]
    for sheet_name, table_df, title in sheets_info:
        ws = writer.sheets[sheet_name]
        ws.write(0, 0, title, title_fmt)
        ws.write(1, 0, 'Values are presented as mean ± standard deviation or n (%). P-values compare PWR and non-PWR groups. FDR-adjusted p-values were calculated using Benjamini–Hochberg across all valid tests in Table 1a and Table 1b combined.', subtitle_fmt)
        for col_num, col_name in enumerate(table_df.columns):
            ws.write(2, col_num, col_name, header_fmt)
        for row_num in range(len(table_df)):
            excel_row = row_num + 3
            var_text = str(table_df.iloc[row_num]['Variable'])
            is_group_header = var_text in [
    'Race, n (%)',
    'Pre-pregnancy BMI category, n (%)',
    'Third trimester',
    '4-6 weeks postpartum',
    '4 months postpartum',
    '8 months postpartum',
    '12 months postpartum'
]
            is_sig = table_df.iloc[row_num]['FDR<0.05'] == 'Yes'
            row_fmt = sig_fmt if is_sig else (group_fmt if is_group_header else body_fmt)
            for col_num, value in enumerate(table_df.iloc[row_num]):
                ws.write(excel_row, col_num, value, row_fmt)
        ws.freeze_panes(3, 1)
        ws.set_column(0, 0, 58)
        ws.set_column(1, 3, 20)
        ws.set_column(4, 6, 15)
        ws.set_column(7, 7, 18)
        ws.set_row(0, 24)
        ws.set_row(1, 48)
    ws_notes = writer.sheets['Analysis notes']
    ws_notes.set_column(0, 0, 35)
    ws_notes.set_column(1, 1, 90)
    for col_num, col_name in enumerate(notes.columns):
        ws_notes.write(0, col_num, col_name, header_fmt)

print(f'Wrote {OUT_XLSX}')
print(f'Wrote {OUT_CSV_1A}')
print(f'Wrote {OUT_CSV_1B}')
