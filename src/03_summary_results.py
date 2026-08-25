import os
import pandas as pd

# Results files of 5 configurations
result_files = {
    "Meta + Selected 2D NMR": "../test/result/data/meta+2dnmr/model_results_all_visits_selected_features_meta+selected2dnmr.csv",
    "Meta + Selected 1D NMR": "../test/result/data/meta+1dnmr/model_results_all_visits_selected_features_meta+selected1dnmr.csv",
    "Meta + All 2D NMR": "../test/result/data/meta+all2dnmr/model_results_all_visits_selected_features_meta+all2dnmr.csv",
    "All 2D NMR Only": "../test/result/data/all2dnmronly/model_results_all_visits_selected_features_all2dnmronly.csv",
    "Meta Only": "../test/result/data/metaonly/model_results_all_visits_selected_features_metaonly.csv",
}

# Output
output_dir = "../test/result/data"
os.makedirs(output_dir, exist_ok=True)

def standardize_result_columns(df, input_name, file_path):
    """
    Standardize metric column names across different result files.
    """

    df = df.copy()
    df["Input Type"] = input_name

    rename_map = {}

    # Accuracy
    if "CV Accuracy" in df.columns and "Average CV Accuracy" not in df.columns:
        rename_map["CV Accuracy"] = "Average CV Accuracy"

    # F1
    if "f1" in df.columns and "Average CV F1 Score" not in df.columns:
        rename_map["f1"] = "Average CV F1 Score"

    if "CV F1" in df.columns and "Average CV F1 Score" not in df.columns:
        rename_map["CV F1"] = "Average CV F1 Score"

    # Precision
    if "precision" in df.columns and "Average CV Precision" not in df.columns:
        rename_map["precision"] = "Average CV Precision"

    if "CV Precision" in df.columns and "Average CV Precision" not in df.columns:
        rename_map["CV Precision"] = "Average CV Precision"

    # Recall
    if "recall" in df.columns and "Average CV Recall" not in df.columns:
        rename_map["recall"] = "Average CV Recall"

    if "CV Recall" in df.columns and "Average CV Recall" not in df.columns:
        rename_map["CV Recall"] = "Average CV Recall"

    # ROC AUC
    if "CV ROC AUC" in df.columns and "Average CV ROC AUC" not in df.columns:
        rename_map["CV ROC AUC"] = "Average CV ROC AUC"

    # Standard error F1
    if "SE CV F1" in df.columns and "SE CV F1 Score" not in df.columns:
        rename_map["SE CV F1"] = "SE CV F1 Score"


    # Apply renaming
    df = df.rename(columns=rename_map)

    duplicate_pairs = {
        "CV Accuracy": "Average CV Accuracy",
        "f1": "Average CV F1 Score",
        "CV F1": "Average CV F1 Score",
        "precision": "Average CV Precision",
        "CV Precision": "Average CV Precision",
        "recall": "Average CV Recall",
        "CV Recall": "Average CV Recall",
        "CV ROC AUC": "Average CV ROC AUC",
        "SE CV F1": "SE CV F1 Score",
    }

    for old_col, new_col in duplicate_pairs.items():

        if old_col in df.columns and new_col in df.columns:

            df[new_col] = df[new_col].fillna(df[old_col])

            df = df.drop(columns=old_col)
    optional_columns = [
        "Top NMR Features",
        "Selected Features",
    ]

    for col in optional_columns:
        if col not in df.columns:
            df[col] = pd.NA

    required_columns = [
        "Visit",
        "Average CV ROC AUC",
    ]

    missing_required = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_required:
        raise ValueError(
            f"\nMissing required columns in:\n"
            f"{file_path}\n"
            f"Missing: {missing_required}\n"
            f"Available columns: {list(df.columns)}"
        )
    df["Average CV ROC AUC"] = pd.to_numeric(
        df["Average CV ROC AUC"],
        errors="coerce"
    )


    return df


all_best_tables = []
all_full_tables = []

for input_name, file_path in result_files.items():

    print("\n" + "=" * 70)
    print(f"Reading: {input_name}")
    print(f"File: {file_path}")

    # --------------------------------------------------------
    # Check file existence
    # --------------------------------------------------------

    if not os.path.exists(file_path):
        print(f"[WARNING] File not found: {file_path}")
        continue


    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(file_path)


    # --------------------------------------------------------
    # Show original shape
    # --------------------------------------------------------

    print(f"Original shape: {df.shape}")


    # --------------------------------------------------------
    # Standardize columns
    # --------------------------------------------------------

    df = standardize_result_columns(
        df=df,
        input_name=input_name,
        file_path=file_path
    )


    print(f"Standardized shape: {df.shape}")
    print(f"Visits: {df['Visit'].unique().tolist()}")


    # --------------------------------------------------------
    # Remove rows without ROC AUC
    # --------------------------------------------------------

    valid_df = df.dropna(
        subset=["Visit", "Average CV ROC AUC"]
    ).copy()


    # --------------------------------------------------------
    # Select best model for each visit
    # within this input configuration
    # --------------------------------------------------------

    best_indices = (
        valid_df
        .groupby("Visit")["Average CV ROC AUC"]
        .idxmax()
    )

    best_by_visit = (
        valid_df
        .loc[best_indices]
        .reset_index(drop=True)
        .copy()
    )


    all_best_tables.append(best_by_visit)
    all_full_tables.append(df)


# 3. Combine best-by-visit results across all five input types
summary_best_by_input = pd.concat(
    all_best_tables,
    ignore_index=True,
    sort=False
)
visit_order = ["V1", "V2", "V3", "V4", "V5"]

summary_best_by_input["Visit"] = pd.Categorical(
    summary_best_by_input["Visit"],
    categories=visit_order,
    ordered=True
)
summary_best_by_input = (
    summary_best_by_input
    .sort_values(
        ["Visit", "Input Type"]
    )
    .reset_index(drop=True)
)

preferred_columns = [
    "Visit",
    "Input Type",
    "Model",
    "Average CV Accuracy",
    "Average CV F1 Score",
    "Average CV Precision",
    "Average CV Recall",
    "Average CV ROC AUC",
    "SE CV Accuracy",
    "SE CV F1 Score",
    "SE CV Precision",
    "SE CV Recall",
    "SE CV ROC AUC",
    "Selected Features",
    "Top NMR Features",
]


def reorder_columns(df, preferred_columns):

    first_columns = [
        col for col in preferred_columns
        if col in df.columns
    ]

    remaining_columns = [
        col for col in df.columns
        if col not in first_columns
    ]

    return df[first_columns + remaining_columns]

# Save best model by input type and visit
summary_best_by_input = reorder_columns(
    summary_best_by_input,
    preferred_columns
)

best_by_input_path = os.path.join(
    output_dir,
    "summary_best_model_by_input_and_visit.csv"
)

summary_best_by_input.to_csv(
    best_by_input_path,
    index=False
)

# Select overall best configuration for each visit
overall_valid = summary_best_by_input.dropna(
    subset=["Average CV ROC AUC"]
).copy()


overall_best_indices = (
    overall_valid
    .groupby("Visit", observed=True)["Average CV ROC AUC"]
    .idxmax()
)


overall_best_by_visit = (
    overall_valid
    .loc[overall_best_indices]
    .sort_values("Visit")
    .reset_index(drop=True)
)


overall_best_by_visit = reorder_columns(
    overall_best_by_visit,
    preferred_columns
)
overall_best_path = os.path.join(
    output_dir,
    "summary_overall_best_configuration_by_visit.csv"
)

overall_best_by_visit.to_csv(
    overall_best_path,
    index=False
)

# Combine ALL model results from all input configurations
all_results_combined = pd.concat(
    all_full_tables,
    ignore_index=True,
    sort=False
)

all_results_combined["Visit"] = pd.Categorical(
    all_results_combined["Visit"],
    categories=visit_order,
    ordered=True
)
sort_columns = [
    col
    for col in ["Visit", "Input Type", "Model"]
    if col in all_results_combined.columns
]

all_results_combined = (
    all_results_combined
    .sort_values(sort_columns)
    .reset_index(drop=True)
)
all_results_combined = reorder_columns(
    all_results_combined,
    preferred_columns
)
combined_path = os.path.join(
    output_dir,
    "summary_all_results_combined.csv"
)

all_results_combined.to_csv(
    combined_path,
    index=False
)