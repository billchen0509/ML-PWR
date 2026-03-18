from pathlib import Path
import pandas as pd
import numpy as np
import sys


def safe_drop(df: pd.DataFrame, cols, axis=1):
    """Drop columns/rows safely if they exist."""
    return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore") if axis == 1 else df.drop(index=cols, errors="ignore")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def prepare_meta(meta_xlsx: Path) -> dict:
    """Load and preprocess metadata, then split into visit-specific DataFrames."""
    try:
        meta_df = pd.read_excel(meta_xlsx)
    except Exception as e:
        print(f"[ERROR] Failed to read metadata Excel: {meta_xlsx}\n{e}")
        sys.exit(1)

    print(f"Loaded meta_df shape: {meta_df.shape}")

    # Drop rows where pwr_current is missing
    if "pwr_current" not in meta_df.columns:
        print("[WARN] Column 'pwr_current' not found in metadata; skipping row drop on pwr_current.")
    else:
        before = meta_df.shape[0]
        meta_df = meta_df.dropna(subset=["pwr_current"])
        after = meta_df.shape[0]
        print(f"Dropped {before - after} rows with NaN pwr_current (remaining: {after})")

    # Drop unnecessary columns if present
    cols_to_drop = ["pwr_any", "pwr_first", "pp_weight_loss1", "pp_weight_loss2", "pp_weight_loss3"]
    present_drop = [c for c in cols_to_drop if c in meta_df.columns]
    meta_df = safe_drop(meta_df, cols_to_drop, axis=1)
    print(f"Dropped columns (if present): {present_drop}")

    # Replace invalid placeholders with NaN
    placeholders = ['.', 'NA', '']
    meta_df.replace(to_replace=placeholders, value=np.nan, inplace=True)

    # Convert object columns to numeric where possible
    obj_cols = meta_df.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        print(f"Attempting numeric conversion for object columns (n={len(obj_cols)}):")
    for col in obj_cols:
        before_na = meta_df[col].isna().sum()
        meta_df[col] = pd.to_numeric(meta_df[col], errors='coerce')
        after_na = meta_df[col].isna().sum()
        print(f"  Converted column: {col} → {meta_df[col].dtype} (NaNs: {before_na} → {after_na})")

    # Impute wt_gain using delta from rows with both wt_gain and weight_change_v1
    if ("wt_gain" in meta_df.columns) and ("weight_change_v1" in meta_df.columns):
        mask = meta_df['weight_change_v1'].notna() & meta_df['wt_gain'].notna()
        n_calc = int(mask.sum())
        if n_calc > 0:
            print(f"Calculating delta for {n_calc} rows where both wt_gain and weight_change_v1 are not NaN")
            delta = (meta_df.loc[mask, 'wt_gain'] - meta_df.loc[mask, 'weight_change_v1']).mean()
            missing_mask = meta_df['wt_gain'].isna() & meta_df['weight_change_v1'].notna()
            n_impute = int(missing_mask.sum())
            meta_df.loc[missing_mask, 'wt_gain'] = meta_df.loc[missing_mask, 'weight_change_v1'] + delta
            print(f"Filled {n_impute} missing wt_gain values using estimated delta = {delta:.2f}")
        else:
            print("[WARN] No rows with both wt_gain and weight_change_v1 present; skipping wt_gain imputation.")
    else:
        print("[WARN] Missing column(s) for wt_gain imputation; require 'wt_gain' and 'weight_change_v1'.")

    prefixes = ['weight_change', 'caffeine', 'num_alc_weekly', 'smoke_current']
    visits = [f"V{i}" for i in range(1, 6)]

    def is_visit_suffixed(col: str) -> bool:
        return any(col.endswith(f"_v{i}") for i in range(1, 6))

    meta_columns = [c for c in meta_df.columns if not is_visit_suffixed(c)]

    if 'participant_id' not in meta_columns:
        if 'participant_id' in meta_df.columns:
            meta_columns.append('participant_id')
        else:
            print("[WARN] 'participant_id' not found in metadata. Merge will fail unless 'participant_id' exists in meta_df.")

    visit_dfs = {}
    for v in range(1, 6):
        visit = f"V{v}"
        visit_columns = [f"{prefix}_v{v}" for prefix in prefixes if f"{prefix}_v{v}" in meta_df.columns]
        missing = [f"{p}_v{v}" for p in prefixes if f"{p}_v{v}" not in meta_df.columns]
        if missing:
            print(f"[WARN] For {visit}, missing meta columns: {missing}")

        selected_columns = list(dict.fromkeys(meta_columns + visit_columns))
        visit_df = meta_df[selected_columns].copy()
        rename_dict = {f"{prefix}_v{v}": prefix for prefix in prefixes}
        visit_df.rename(columns=rename_dict, inplace=True)
        visit_df['visit'] = visit
        visit_dfs[visit] = visit_df

    print("Built visit-specific meta DataFrames.")
    return visit_dfs


def process_nmr_dataset(nmr_csv: Path, out_dir: Path, suffix: str):
    """Load one NMR dataset, merge with visit-specific metadata, and save outputs."""
    meta_xlsx = Path("../../Matthias Klein's files - GWG/OB70 SHIPP3 SELECTED_marked.xlsx")
    visit_dfs = prepare_meta(meta_xlsx)
    visits = [f"V{i}" for i in range(1, 6)]

    ensure_dir(out_dir)

    try:
        nmr_data = pd.read_csv(nmr_csv, index_col=0)
    except Exception as e:
        print(f"[ERROR] Failed to read NMR CSV: {nmr_csv}\n{e}")
        return

    print(f"Loaded {suffix.upper()} nmr_data shape: {nmr_data.shape}")

    nmr_data['participant_id'] = nmr_data.index.to_series().astype(str).str.extract(r'GWG-(\d+)', expand=False)
    nmr_data['visit'] = nmr_data.index.to_series().astype(str).str.extract(r'GWG-\d+-(V\d)', expand=False)
    before_drop = nmr_data.shape[0]
    nmr_data = nmr_data.dropna(subset=['visit'])
    print(f"Dropped {before_drop - nmr_data.shape[0]} {suffix.upper()} NMR rows without a parsed visit. Remaining: {nmr_data.shape[0]}")

    def get_visit_df(visit: str) -> pd.DataFrame:
        if visit not in visit_dfs:
            print(f"[WARN] No meta visit_df for {visit}; returning empty DataFrame.")
            return pd.DataFrame()

        nmr_visit_df = nmr_data[nmr_data['visit'] == visit].copy()
        meta_visit_df = visit_dfs[visit].copy()

        if 'participant_id' not in meta_visit_df.columns:
            print(f"[ERROR] 'participant_id' missing in meta for {visit}. Cannot merge.")
            return pd.DataFrame()

        nmr_visit_df['participant_id'] = nmr_visit_df['participant_id'].astype(str)
        meta_visit_df['participant_id'] = meta_visit_df['participant_id'].astype(str)

        merged = pd.merge(meta_visit_df, nmr_visit_df, on=['participant_id', 'visit'], how='inner')
        print(
            f"{suffix.upper()} {visit}: merged rows = {merged.shape[0]} "
            f"(meta: {meta_visit_df.shape[0]}, nmr: {nmr_visit_df.shape[0]})"
        )
        return merged

    for i, visit in enumerate(visits, start=1):
        df = get_visit_df(visit)
        raw_file = out_dir / f"DF{i}_{suffix}.csv"
        df.to_csv(raw_file, index=False)
        print(f"Saved {visit} → {raw_file} (rows={df.shape[0]}, cols={df.shape[1]})")

        if 'weight_change' not in df.columns:
            print(f"[WARN] {suffix.upper()} DF{i} has no 'weight_change' column; skipping filtering.")
            continue

        df['weight_change'] = pd.to_numeric(df['weight_change'], errors='coerce')
        before_rows = df.shape[0]
        df = df[df['weight_change'] > 0]
        after_rows = df.shape[0]

        filtered_file = out_dir / f"DF{i}_filtered_{suffix}.csv"
        df.to_csv(filtered_file, index=False)
        print(f"Saved filtered → {filtered_file} (kept {after_rows}/{before_rows})")


def main():
    print("Loading files...")

    datasets = [
        {
            "nmr_csv": Path("../../mrbin_result/1D data/75_19_2dregions_13/mrbin_2025-07-19_16-00-21bins.csv"),
            "out_dir": Path("../test/result/data/meta+1dnmr"),
            "suffix": "1d",
        },
        {
            "nmr_csv": Path("../../mrbin_result/Result_final/updated/mrbin_final.csv"),
            "out_dir": Path("../test/result/data/meta+2dnmr"),
            "suffix": "2d",
        },
    ]

    for dataset in datasets:
        print("\n" + "=" * 80)
        print(f"Processing {dataset['suffix'].upper()} dataset")
        print("=" * 80)
        process_nmr_dataset(
            nmr_csv=dataset["nmr_csv"],
            out_dir=dataset["out_dir"],
            suffix=dataset["suffix"],
        )

    print("All done.")


if __name__ == "__main__":
    main()
