import os
import pandas as pd

# 1. Result files for the five input configurations
result_files = {
    "Meta + Selected 2D NMR": "../test/result/model/meta+selected2dnmr/model_results_all_visits_meta+selected2dnmr.csv",
    "Meta + Selected 1D NMR": "../test/result/model/meta+selected1dnmr/model_results_all_visits_meta+selected1dnmr.csv",
    "Meta + All 2D NMR": "../test/result/model/meta+all2dnmr/model_results_all_visits_meta+all2dnmr.csv",
    "All 2D NMR Only": "../test/result/model/all2dnmronly/model_results_all_visits_all2dnmronly.csv",
    "Meta Only": "../test/result/model/metaonly/model_results_all_visits_metaonly.csv",
}

# 2. Read and standardize result tables
all_best_tables = []
all_full_tables = []

for input_name, file_path in result_files.items():
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}")
        continue

    df = pd.read_csv(file_path)
    df["Input Type"] = input_name

    # Standardize ROC AUC column
    if "CV ROC AUC" in df.columns:
        df["Unified CV ROC AUC"] = df["CV ROC AUC"]
    elif "Average CV ROC AUC" in df.columns:
        df["Unified CV ROC AUC"] = df["Average CV ROC AUC"]
    else:
        raise ValueError(f"No ROC AUC column found in {file_path}")

    # Standardize accuracy column
    if "CV Accuracy" in df.columns:
        df["Unified CV Accuracy"] = df["CV Accuracy"]
    elif "Average CV Accuracy" in df.columns:
        df["Unified CV Accuracy"] = df["Average CV Accuracy"]

    # Add empty feature column if not available
    if "Top NMR Features" not in df.columns:
        df["Top NMR Features"] = pd.NA

    # Select the best model for each visit within each input type
    best_by_visit = (
        df.loc[df.groupby("Visit")["Unified CV ROC AUC"].idxmax()]
        .reset_index(drop=True)
        .copy()
    )

    all_best_tables.append(best_by_visit)
    all_full_tables.append(df)

    print(f"\nBest model using {input_name} by visit based on CV ROC AUC:")
    print(
        best_by_visit[
            ["Visit", "Model", "Unified CV ROC AUC", "Top NMR Features", "Std CV ROC AUC"]
        ]
    )

# 3. Combine best-by-visit results across all five input types
summary_best_by_input = pd.concat(all_best_tables, ignore_index=True)

visit_order = ["V1", "V2", "V3", "V4", "V5"]
summary_best_by_input["Visit"] = pd.Categorical(
    summary_best_by_input["Visit"], categories=visit_order, ordered=True
)
summary_best_by_input = summary_best_by_input.sort_values(
    ["Visit", "Input Type"]
).reset_index(drop=True)

print("\n===== Combined best model for each input type at each visit =====")
print(
    summary_best_by_input[
        ["Input Type", "Visit", "Model", "Unified CV ROC AUC", "Top NMR Features", "Std CV ROC AUC"]
    ]
)

summary_best_by_input.to_csv(
    "../test/result/model/summary_best_model_by_input_and_visit.csv",
    index=False
)

# 4. Select the overall best configuration across the five input types for each visit
overall_best_by_visit = (
    summary_best_by_input.loc[
        summary_best_by_input.groupby("Visit")["Unified CV ROC AUC"].idxmax()
    ]
    .reset_index(drop=True)
)

print("\n===== Overall best input configuration by visit =====")
print(
    overall_best_by_visit[
        ["Visit", "Input Type", "Model", "Unified CV ROC AUC", "Top NMR Features", "Std CV ROC AUC"]
    ]
)

overall_best_by_visit.to_csv(
    "../test/result/model/summary_overall_best_configuration_by_visit.csv",
    index=False
)

# 5. Save the fully combined result table
all_results_combined = pd.concat(all_full_tables, ignore_index=True)
all_results_combined.to_csv(
    "../test/result/model/summary_all_results_combined.csv",
    index=False
)

print("\nSaved files:")
print("- ../test/result/model/summary_best_model_by_input_and_visit.csv")
print("- ../test/result/model/summary_overall_best_configuration_by_visit.csv")
print("- ../test/result/model/summary_all_results_combined.csv")