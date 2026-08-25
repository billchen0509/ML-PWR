import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve

# Configuration
INPUT_FILE = "../test/result/data/meta+2dnmr/model_results_all_visits_selected_features_meta+selected2dnmr.csv"
FIGURE_DIR = "../test/result/figure"

PALETTE = [
    "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
    "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6"
]

BEST_MODELS = {
    "V1": "MultiLayer Perceptron",
    "V2": "MultiLayer Perceptron",
    "V3": "Logistic Regression",
    "V4": "Random Forest",
    "V5": "Logistic Regression"
}

SUPPLEMENTAL_TARGETS = [
    {"Visit": "V2", "Model": "MultiLayer Perceptron", "Features": 10, "Output": f"{FIGURE_DIR}/Supplemental_Figure_S1_V2.png"},
    {"Visit": "V3", "Model": "Logistic Regression",   "Features": 5,  "Output": f"{FIGURE_DIR}/Supplemental_Figure_S2_V3.png"},
    {"Visit": "V4", "Model": "Random Forest",         "Features": 15, "Output": f"{FIGURE_DIR}/Supplemental_Figure_S3_V4.png"},
    {"Visit": "V5", "Model": "Logistic Regression",   "Features": 8,  "Output": f"{FIGURE_DIR}/Supplemental_Figure_S4_V5.png"}
]

VISIT_ORDER = ["V1", "V2", "V3", "V4", "V5"]
VISIT_LABEL_MAP = {
    "V1": "Third trimester",
    "V2": "4-6 weeks\npostpartum",
    "V3": "4 months\npostpartum",
    "V4": "8 months\npostpartum",
    "V5": "12 months\npostpartum"
}

def load_results(filename):
    return pd.read_csv(filename)


def parse_literal(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def count_selected_features(selected_features_value):
    parsed = parse_literal(selected_features_value)
    if isinstance(parsed, list) and len(parsed) > 0 and "selected_nmr" in parsed[0]:
        return len(parsed[0]["selected_nmr"])
    return np.nan


def get_best_model_per_visit(df, metric="CV ROC AUC"):
    best_rows = df.loc[df.groupby("Visit")[metric].idxmax()].copy()
    best_rows["Visit"] = pd.Categorical(best_rows["Visit"], categories=VISIT_ORDER, ordered=True)
    best_rows = best_rows.sort_values("Visit").reset_index(drop=True)
    return best_rows


def get_best_model_per_visit_per_model(df, metric="CV ROC AUC"):
    best_rows = df.loc[df.groupby(["Visit", "Model"])[metric].idxmax()].copy()
    best_rows["Visit"] = pd.Categorical(best_rows["Visit"], categories=VISIT_ORDER, ordered=True)
    best_rows = best_rows.sort_values(["Visit", "Model"]).reset_index(drop=True)
    return best_rows


def get_model_data_by_feature_count(df, visit, model_name, target_n_features):
    subset = df[(df["Visit"] == visit) & (df["Model"] == model_name)].copy()
    if subset.empty:
        return None, None, None

    subset["n_features"] = subset["Selected Features"].apply(count_selected_features)
    matched = subset[subset["n_features"] == target_n_features]

    if matched.empty:
        return None, None, None

    best_row = matched.sort_values("CV ROC AUC", ascending=False).iloc[0]
    y_true = np.array(parse_literal(best_row["All y True"]))
    y_prob = np.array(parse_literal(best_row["All y Proba"]))
    auc_value = best_row["CV ROC AUC"]

    return y_true, y_prob, auc_value


def get_best_row_for_visit_model(df, visit, model_name):
    subset = df[(df["Visit"] == visit) & (df["Model"] == model_name)].copy()
    if subset.empty:
        return None

    return subset.loc[subset["CV ROC AUC"].idxmax()]


# Figure: Best CV ROC AUC per model per visit
def plot_model_performance_over_visits(df, metric="CV ROC AUC"):
    plot_df = get_best_model_per_visit_per_model(df, metric=metric)

    plt.figure(figsize=(12, 9))
    ax = sns.barplot(
        data=plot_df,
        x="Visit",
        y=metric,
        hue="Model",
        order=VISIT_ORDER,
        palette=PALETTE,
        errorbar=None)
    ax.set_xticks(range(len(VISIT_ORDER)))
    ax.set_xticklabels([VISIT_LABEL_MAP[v] for v in VISIT_ORDER],rotation=30,ha="right",fontsize=10)
    # Add error bars safely in bar order
    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        se = row["SE CV ROC AUC"] if "SE CV ROC AUC" in row and pd.notna(row["SE CV ROC AUC"]) else None
        if se is not None:
            ax.errorbar(
                x=x,
                y=y,
                yerr=se,
                fmt="none",
                color="black",
                capsize=4,
                elinewidth=1.5,
                alpha=0.8
            )
        ax.text(
            x,
            y + 0.01,
            f"{y:.3f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    ax.set_title("Best CV ROC AUC per Model per Visit")
    ax.set_xlabel("Visit")
    ax.set_ylabel(metric)
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/model_performance_over_visits.png", dpi=600, bbox_inches="tight")
    plt.show()


# Figure: ROC curves of best models across visits
def plot_roc_curves_best_models(df, best_models):
    plt.figure(figsize=(8, 7))

    for visit, model_name in best_models.items():
        row = get_best_row_for_visit_model(df, visit, model_name)
        if row is None:
            continue

        y_true = parse_literal(row["All y True"])
        y_prob = parse_literal(row["All y Proba"])

        if y_true is None or y_prob is None:
            continue
        if len(set(y_prob)) <= 1:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        visit_label = VISIT_LABEL_MAP.get(visit, visit)
        plt.plot(fpr,tpr,label=f"{visit_label.replace(chr(10), ' ')} - {model_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves of Best Models per Visit")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/roc_curve_best_models_per_visit.png", dpi=600, bbox_inches="tight")
    plt.show()


# =========================================================
# Table: Compare calibration candidates for V1 MLP
# =========================================================
def summarize_v1_mlp_calibration(df):
    subset = df[(df["Visit"] == "V1") & (df["Model"] == "MultiLayer Perceptron")].copy()

    results = []
    for _, row in subset.iterrows():
        try:
            n_feats = count_selected_features(row["Selected Features"])
            y_true = np.array(parse_literal(row["All y True"]))
            y_prob = np.array(parse_literal(row["All y Proba"]))

            brier = brier_score_loss(y_true, y_prob)
            auc_value = row["CV ROC AUC"]

            results.append({
                "n_features": n_feats,
                "Brier Score": brier,
                "CV ROC AUC": auc_value
            })
        except Exception:
            continue

    results_df = pd.DataFrame(results)

    print("Top 5 by Brier Score:")
    print(results_df.sort_values("Brier Score").head(5))

    print("\nTop 5 by CV ROC AUC:")
    print(results_df.sort_values("CV ROC AUC", ascending=False).head(5))

    return results_df

# Figure: V1 calibration trade-off comparison
def plot_v1_calibration_tradeoff(df, selected_features=20, calibration_features=13):
    y_true_sel, y_prob_sel, auc_sel = get_model_data_by_feature_count(
        df, "V1", "MultiLayer Perceptron", selected_features
    )
    y_true_cal, y_prob_cal, auc_cal = get_model_data_by_feature_count(
        df, "V1", "MultiLayer Perceptron", calibration_features
    )

    if y_true_sel is None or y_true_cal is None:
        print("Required V1 comparison data could not be found.")
        return

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated", alpha=0.8)

    prob_true_sel, prob_pred_sel = calibration_curve(y_true_sel, y_prob_sel, n_bins=10, strategy="quantile")
    brier_sel = brier_score_loss(y_true_sel, y_prob_sel)
    ax1.plot(
        prob_pred_sel,
        prob_true_sel,
        "s-",
        color="crimson",
        linewidth=2,
        label=f"{selected_features} Features\nAUC = {auc_sel:.3f}, Brier = {brier_sel:.3f}"
    )

    prob_true_cal, prob_pred_cal = calibration_curve(y_true_cal, y_prob_cal, n_bins=10, strategy="quantile")
    brier_cal = brier_score_loss(y_true_cal, y_prob_cal)
    ax1.plot(
        prob_pred_cal,
        prob_true_cal,
        "o--",
        color="royalblue",
        linewidth=2,
        label=f"{calibration_features} Features\nAUC = {auc_cal:.3f}, Brier = {brier_cal:.3f}"
    )

    ax1.set_ylabel("Fraction of Positives")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_title("Third Trimester: Calibration Curve Trade-off")
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.hist(
        y_prob_sel[y_true_sel == 0],
        bins=20,
        range=(0, 1),
        alpha=0.5,
        label="Class 0",
        color="blue",
        density=True,
        hatch="//"
    )
    ax2.hist(
        y_prob_sel[y_true_sel == 1],
        bins=20,
        range=(0, 1),
        alpha=0.5,
        label="Class 1",
        color="red",
        density=True
    )

    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Density")
    ax2.legend(loc="upper center", fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/V1_Comparison_With_Hist.png", dpi=600, bbox_inches="tight")
    plt.show()

# Figure: Supplemental calibration plots
def plot_supplemental_calibration_figures(df, targets):
    for target in targets:
        visit = target["Visit"]
        model_name = target["Model"]
        n_features = target["Features"]
        output_file = target["Output"]

        y_true, y_prob, auc_value = get_model_data_by_feature_count(df, visit, model_name, n_features)

        if y_true is None:
            print(f"Skipping {visit}: required data not found.")
            continue

        brier = brier_score_loss(y_true, y_prob)

        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)

        ax1 = fig.add_subplot(gs[0])
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated", alpha=0.8)

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        ax1.plot(
            prob_pred,
            prob_true,
            "s-",
            linewidth=2,
            color="#2ca02c",
            label=f"{model_name} ({n_features} features)\nAUC = {auc_value:.3f}, Brier = {brier:.3f}"
        )
        visit_label = VISIT_LABEL_MAP.get(visit, visit).replace("\n", " ")
        ax1.set_ylabel("Fraction of Positives")
        ax1.set_xlabel("Mean Predicted Probability")
        ax1.set_title(f"Calibration Curve for {visit_label}")
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1])
        ax2.hist(
            y_prob[y_true == 0],
            bins=15,
            range=(0, 1),
            alpha=0.6,
            label="Class 0",
            color="blue",
            density=True
        )
        ax2.hist(
            y_prob[y_true == 1],
            bins=15,
            range=(0, 1),
            alpha=0.6,
            label="Class 1",
            color="red",
            density=True
        )

        ax2.set_xlabel("Predicted Probability")
        ax2.set_ylabel("Density")
        ax2.legend(loc="upper center", fontsize=9, ncol=2)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=600, bbox_inches="tight")
        plt.close()

        print(f"Generated: {output_file}")

if __name__ == "__main__":
    df = load_results(INPUT_FILE)

    plot_model_performance_over_visits(df, metric="CV ROC AUC")
    plot_roc_curves_best_models(df, BEST_MODELS)
    summarize_v1_mlp_calibration(df)
    plot_v1_calibration_tradeoff(df, selected_features=20, calibration_features=13)
    plot_supplemental_calibration_figures(df, SUPPLEMENTAL_TARGETS)