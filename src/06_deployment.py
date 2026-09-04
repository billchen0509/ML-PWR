import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from scipy.stats import ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Constants
ONE_HOT_COLS = ['vig_activity']
BINARY_COLS = ['race_1', 'race_2']
CONTINUOUS_COLS = ['wt_gain', 'weight_change', 'age_delivery_range2', 'num_alc_weekly']
EXPLICIT_COLS = ['caffeine', 'smoke_current', 'ppg_BMI', 'first_BMI']

META_COLS = ONE_HOT_COLS + BINARY_COLS + CONTINUOUS_COLS + EXPLICIT_COLS
NON_FEATURE_COLS = ['participant_id', 'visit', 'pwr_current']

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULT_DIR = REPO_ROOT / "test" / "result"
BASE_DIR = RESULT_DIR / "deploy"
TRAIN_DATA_DIR = RESULT_DIR / "data" / "meta+2dnmr"
DEPLOY_TEST_DATA_DIR = RESULT_DIR / "data" / "deploy"

DEPLOYMENT_PLAN = {
    1: {"MultiLayer Perceptron": 20},
    2: {"MultiLayer Perceptron": 10},
    3: {"Logistic Regression": 5},
    4: {"Random Forest": 15},
    5: {"Logistic Regression": 8},
}

PREDICTION_PLAN = {
    1: ("MultiLayer Perceptron", 20),
    2: ("MultiLayer Perceptron", 10),
    3: ("Logistic Regression", 5),
    4: ("Random Forest", 15),
    5: ("Logistic Regression", 8),
}

class NMRTopNSelector(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=10):
        self.top_n = top_n
        self.selected_cols_ = None

    def fit(self, X, y=None):
        pvals = {}
        X0 = X[y == 0]
        X1 = X[y == 1]

        for col in X.columns:
            try:
                _, p = ttest_ind(X0[col], X1[col], equal_var=False)
            except Exception:
                p = np.nan
            pvals[col] = p

        cols_sorted = sorted(pvals, key=pvals.get)
        self.selected_cols_ = cols_sorted[: self.top_n]
        return self

    def transform(self, X):
        return X[self.selected_cols_]

# Training
def final_tune_and_refit(df, model_name, estimator, param_grid, top_n, outdir,
                         cv=10, random_state=42):
    X = df.drop(columns=['pwr_current', 'participant_id', 'visit'])
    y = df['pwr_current']

    all_nmr_cols = [c for c in df.columns if c not in META_COLS + NON_FEATURE_COLS]

    nmr_block = Pipeline([
        ('select', NMRTopNSelector(top_n=top_n))
    ])

    pre = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ONE_HOT_COLS),
        ('explicit', OneHotEncoder(handle_unknown='ignore'), EXPLICIT_COLS),
        ('binary', 'passthrough', BINARY_COLS),
        ('num', StandardScaler(), CONTINUOUS_COLS),
        ('nmr', nmr_block, all_nmr_cols),
    ])

    pipe = Pipeline([
        ('preprocessor', pre),
        ('model', estimator)
    ])

    grid = param_grid[model_name]

    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        pipe,
        grid,
        cv=cv_split,
        scoring='roc_auc',
        n_jobs=-1,
        refit=True
    )
    gs.fit(X, y)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = outdir / f"{model_name}_N{top_n}_final_model.pkl"
    params_path = outdir / f"{model_name}_N{top_n}_best_params.json"
    selected_nmr_path = outdir / f"{model_name}_N{top_n}_selected_nmr.json"

    joblib.dump(gs.best_estimator_, model_path)

    with params_path.open("w") as f:
        json.dump(gs.best_params_, f, indent=2)

    nmr_selected = (
        gs.best_estimator_
        .named_steps['preprocessor']
        .named_transformers_['nmr']
        .named_steps['select']
        .selected_cols_
    )

    with selected_nmr_path.open("w") as f:
        json.dump(nmr_selected, f, indent=2)

    return {
        "best_params": gs.best_params_,
        "cv_best_score": gs.best_score_,
        "selected_nmr": nmr_selected,
        "final_model_path": model_path
    }


def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'MultiLayer Perceptron': MLPClassifier(max_iter=1000, random_state=42),
    }


def get_param_grid():
    return {
        'Logistic Regression': [
            {
                'model__C': [0.01, 0.05, 0.1, 1, 10],
                'model__penalty': ['l1', 'l2'],
                'model__solver': ['liblinear']
            },
            {
                'model__C': [0.01, 0.05, 0.1, 1, 10],
                'model__penalty': ['l2'],
                'model__solver': ['lbfgs']
            },
            {
                'model__C': [0.01, 0.05, 0.1, 1, 10],
                'model__penalty': ['elasticnet'],
                'model__l1_ratio': [0.1, 0.5, 0.9],
                'model__solver': ['saga']
            },
        ],
        'Random Forest': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10],
            'model__min_samples_split': [2, 5],
            'model__max_features': ['sqrt', 1.0],
            'model__class_weight': ['balanced'],
            'model__bootstrap': [True],
        },
        'MultiLayer Perceptron': {
            'model__hidden_layer_sizes': [(100,), (150,), (150, 50)],
            'model__activation': ['relu', 'tanh'],
            'model__alpha': [1e-4, 1e-3],
            'model__solver': ['adam', 'lbfgs'],
            'model__learning_rate_init': [0.001, 0.01],
        },
    }


def run_training():
    models = get_models()
    param_grid = get_param_grid()

    for visit in range(1, 6):
        print(f"[Deployment] Visit {visit}")
        train_file = TRAIN_DATA_DIR / f"DF{visit}_filtered_2d.csv"
        if not train_file.exists():
            raise FileNotFoundError(f"Missing training input file: {train_file}")
        df = pd.read_csv(train_file)
        outdir = BASE_DIR / f"V{visit}"

        for model_name, top_n in DEPLOYMENT_PLAN.get(visit, {}).items():
            info = final_tune_and_refit(
                df=df,
                model_name=model_name,
                estimator=models[model_name],
                param_grid=param_grid,
                top_n=top_n,
                outdir=outdir,
                cv=10,
                random_state=42,
            )
            print(f"V{visit} {model_name} N={top_n} — AUC(cv)={info['cv_best_score']:.3f}")
            print("Best params:", info["best_params"])
            print("Selected NMR:", info["selected_nmr"][:5], "…")
            print("Saved model →", info["final_model_path"])

# Prediction utilities
def load_final_pipeline(base_dir, visit, model_name, top_n):
    path = Path(base_dir) / f"V{visit}" / f"{model_name}_N{top_n}_final_model.pkl"
    pipe = joblib.load(path)
    return pipe


def make_X_test(df):
    cols_to_drop = [c for c in NON_FEATURE_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    return X


def predict_with_threshold(pipe, X, threshold=0.5):
    proba = pipe.predict_proba(X)[:, 1]
    yhat = (proba >= threshold).astype(int)
    return proba, yhat


def run_prediction():
    prediction_outdir = BASE_DIR / "prediction"
    prediction_outdir.mkdir(parents=True, exist_ok=True)

    for visit in range(1, 5):
        test_file = DEPLOY_TEST_DATA_DIR / f"V{visit}_test.csv"
        if not test_file.exists():
            raise FileNotFoundError(
                f"Missing deployment test dataset: {test_file}. "
                "Prepare the private deploy test data before running predictions."
            )
        test_df = pd.read_csv(test_file)
        model_name, top_n = PREDICTION_PLAN[visit]

        pipe = load_final_pipeline(BASE_DIR, visit, model_name, top_n)
        X_test = make_X_test(test_df)

        proba, yhat = predict_with_threshold(pipe, X_test, threshold=0.5)

        out = test_df.copy()
        out["pred_proba"] = proba
        out["pred_label"] = yhat

        print(out[["participant_id", "pred_proba", "pred_label"]])

        out_df = out[["participant_id", "pred_proba", "pred_label"]]
        out_path = prediction_outdir / f"V{visit}_{model_name.replace(' ', '')}_N{top_n}_predictions.csv"
        out_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    run_training()
    run_prediction()