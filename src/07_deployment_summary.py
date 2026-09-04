import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PREDICTION_DIR = REPO_ROOT / "test" / "result" / "deploy" / "prediction"

files = [
    PREDICTION_DIR / "V1_MultiLayerPerceptron_N20_predictions.csv",
    PREDICTION_DIR / "V2_MultiLayerPerceptron_N10_predictions.csv",
    PREDICTION_DIR / "V3_LogisticRegression_N5_predictions.csv",
    PREDICTION_DIR / "V4_RandomForest_N15_predictions.csv",
]

visit_columns = ["V1", "V2", "V3", "V4"]

# def a function to load the files
def load_visit(file, visit_name):
    df = pd.read_csv(file)
    df = df.drop_duplicates(subset=["participant_id"], keep="last")
    df['participant_id'] = df['participant_id'].astype(str)
    df['pred_proba'] = pd.to_numeric(df["pred_proba"], errors="coerce")
    df["pred_label"] = df["pred_label"].astype("Int64")
    df[visit_name] = df.apply(
        lambda r:(
            f"{int(r['pred_label'])} ({r['pred_proba']:.4f})"
            if pd.notna(r['pred_proba']) and pd.notna(r['pred_label'])
            else "NA"
        ),
        axis=1
    )
    return df[["participant_id", visit_name]]

# Build the combined DataFrame
wide = None
for file, visit in zip(files, visit_columns):
    if not file.exists():
        raise FileNotFoundError(f"Missing prediction file: {file}")
    visit_df = load_visit(file, visit)
    wide = visit_df if wide is None else pd.merge(wide, visit_df, on="participant_id", how="outer")

wide.to_csv(PREDICTION_DIR / "combined_visit_predictions.csv", index=False)