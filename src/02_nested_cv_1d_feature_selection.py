from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import os

def select_top_nmr_features(X, y, other_cols, top_n):
    p_values = {}
    for col in other_cols:
        group0 = X[y == 0][col]
        group1 = X[y == 1][col]
        try:
            t_stat, p = ttest_ind(group0, group1, equal_var=False)
        except Exception as e:
            print(f"Error calculating t-test for {col}: {e}")
            p = np.nan
        p_values[col] = p
    sorted_features = sorted(p_values, key=p_values.get)
    return sorted_features[:top_n]

def train_with_cv(df, cv=10, inner_cv=10, random_state=42, all_nmr_cols=None,top_n=None,):
    one_hot_cols = ['vig_activity']
    binary_cols = ['race_1', 'race_2']
    continuous_cols = ['wt_gain','weight_change','age_delivery_range2','num_alc_weekly']
    explicit_cols = ['caffeine','smoke_current','ppg_BMI', 'first_BMI']
    meta_cols = one_hot_cols + binary_cols + continuous_cols + explicit_cols
    non_feature_cols = ['participant_id', 'visit', 'pwr_current']

    if all_nmr_cols is None:
        all_nmr_cols = [col for col in df.columns if col not in meta_cols + non_feature_cols]
 
    X_all = df.drop(columns=['pwr_current','participant_id','visit'])
    y_all = df['pwr_current']


    models = {
        'Logistic Regression': LogisticRegression(max_iter=5000, random_state=random_state),
        'GaussianNB': GaussianNB(),
        'BernoulliNB': BernoulliNB(),
        'Support Vector Machine': SVC(probability=True, random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'K-Nearest Neighbors': KNeighborsClassifier(n_jobs=1),
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'MultiLayer Perceptron': MLPClassifier(max_iter=1000, random_state=random_state),
    }

    param_grid = {
        'Logistic Regression': [
            {'model__C': [0.01, 0.05, 0.1, 1, 10],'model__penalty': ['l1', 'l2'],'model__solver': ['liblinear']},
            {'model__C': [0.01, 0.05, 0.1, 1, 10],'model__penalty': ['l2'],'model__solver': ['lbfgs']},
            {'model__C': [0.01, 0.05, 0.1, 1, 10],'model__penalty': ['elasticnet'],'model__l1_ratio': [0.1, 0.5, 0.9],'model__solver': ['saga']},
        ],
        'Support Vector Machine':[
            {'model__C': [0.01, 0.1, 1, 10], 'model__kernel': ['linear']},
            {'model__C': [0.01, 0.1, 1, 10], 'model__kernel': ['rbf'], 'model__gamma': ['scale', 0.001, 0.01, 0.1, 1]},
        ],
        'K-Nearest Neighbors':{
            'model__n_neighbors': [1, 2, 3, 4, 5],
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['euclidean', 'manhattan'],
        },
        'Random Forest':{
            'model__n_estimators': [50, 100, 200,300],
            'model__max_depth': [None, 10],
            'model__min_samples_split':[2, 5, 10],
            'model__max_features': ['sqrt', 1.0],
            'model__class_weight': ['balanced'],
            'model__bootstrap': [True],
        },
        'XGBoost':{
            'model__n_estimators': [50, 100, 200,300],
            'model__learning_rate': [0.01, 0.05,0.1, 0.2],
            'model__gamma': [0, 0.1],
            'model__reg_alpha': [0.01,0.1],
            'model__reg_lambda': [0.01,0.1],
        },
        'MultiLayer Perceptron':{
            'model__hidden_layer_sizes': [(100,), (150,),(150,50),(100,50),(50,25)],
            'model__activation': ['relu','tanh'],
            'model__alpha': [0.0001, 0.001],
            'model__solver': ['adam', 'lbfgs'],
            'model__learning_rate_init': [0.001, 0.01, 0.1],
        },
        'GaussianNB':{
            'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6],
        },
        'BernoulliNB':{
            'model__alpha': [1e-5, 0.1, 0.5, 1.0],
        },
        'Decision Tree':{
            'model__max_depth': [5, 10, None],
            'model__max_features': [None, 'sqrt', 'log2', 0.8],
            'model__criterion': ['gini', 'entropy'],
        }
    }

    outer_kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    all_results = []
    model_fold_params = {}

    # selected features per model per fold
    selected_features_record = {model_name: [] for model_name in models}

    for model_name, model in models.items():
        print(f"Training model: {model_name}")
        y_true, y_pred, y_proba = [], [], []
        fold_scores = []
        fold_best_params = []

        for fold_id,(train_idx, test_idx) in enumerate(outer_kf.split(X_all, y_all)):
            # Split raw data
            df_train = df.iloc[train_idx].copy()
            df_test = df.iloc[test_idx].copy()

            # Recompute categories from TRAIN ONLY (strict)
            explicit_categories = [sorted(set(df_train[col].dropna().unique())) for col in explicit_cols]

            # Build training feature/label tables
            X_train = df_train.drop(columns=['pwr_current','participant_id','visit'])
            y_train = df_train['pwr_current']
            X_test  = df_test.drop(columns=['pwr_current','participant_id','visit'])
            y_test  = df_test['pwr_current']

            # Feature selection on TRAIN ONLY (per fold)
            current_nmr_cols = all_nmr_cols
            if (top_n is not None) and (top_n > 0):
                selected_nmr = select_top_nmr_features(X_train, y_train, current_nmr_cols, top_n)
            else:
                selected_nmr = current_nmr_cols
            selected_features_record[model_name].append({
                'fold': fold_id,
                'selected_nmr': selected_nmr
            })

            # ColumnTransformer using TRAIN-ONLY categories and selected NMRs
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), one_hot_cols),
                    ('explicit', OneHotEncoder(categories=explicit_categories, handle_unknown='ignore'), explicit_cols),
                    ('binary', 'passthrough', binary_cols),
                    ('num', StandardScaler(), continuous_cols),
                    ('other', 'passthrough', selected_nmr)
                ]
            )

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            # Inner CV on TRAIN ONLY
            if model_name in param_grid:
                grid = GridSearchCV(pipeline, param_grid[model_name], cv=inner_cv, scoring='roc_auc', n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                fold_best_params.append(grid.best_params_)
            else:
                best_model = pipeline.fit(X_train, y_train)
                fold_best_params.append({})

            # Evaluate on TEST ONLY
            y_pred_fold = best_model.predict(X_test)
            try:
                y_proba_fold = best_model.predict_proba(X_test)[:, 1]
                roc = roc_auc_score(y_test, y_proba_fold)
            except:
                y_proba_fold = None
                roc = None

            fold_scores.append({
                'CV Accuracy': accuracy_score(y_test, y_pred_fold),
                'f1': f1_score(y_test, y_pred_fold),
                'precision': precision_score(y_test, y_pred_fold),
                'recall': recall_score(y_test, y_pred_fold),
                'CV ROC AUC': roc
            })

            y_true.extend(y_test.tolist())
            y_pred.extend(y_pred_fold.tolist())
            y_proba.extend(y_proba_fold.tolist() if y_proba_fold is not None else [None]*len(y_test))

        # Aggregate results
        df_result = pd.DataFrame(fold_scores)
        avg_result = df_result.mean().to_dict()
        avg_result['Model'] = model_name
        avg_result['Std CV ROC AUC'] = df_result['CV ROC AUC'].std()
        avg_result['SE CV ROC AUC'] = df_result['CV ROC AUC'].std() / np.sqrt(len(df_result)) 
        avg_result['Overall ROC AUC'] = roc_auc_score(y_true, y_proba) if None not in y_proba else None
        avg_result['Overall Confusion Matrix'] = confusion_matrix(y_true, y_pred)
        avg_result['Std CV Accuracy'] = df_result['CV Accuracy'].std()
        avg_result['SE CV Accuracy'] = df_result['CV Accuracy'].std() / np.sqrt(len(df_result))
        avg_result['SE CV F1'] = df_result['f1'].std() / np.sqrt(len(df_result))
        avg_result['SE CV Precision'] = df_result['precision'].std() / np.sqrt(len(df_result))
        avg_result['SE CV Recall'] = df_result['recall'].std() / np.sqrt(len(df_result))
        avg_result['All y True'] = y_true
        avg_result['All y Pred'] = y_pred
        avg_result['All y Proba'] = y_proba
        avg_result['Selected Features'] = selected_features_record[model_name]
        all_results.append(avg_result)
        model_fold_params[model_name] = fold_best_params

    return pd.DataFrame(all_results), model_fold_params

def run_nmr_feature_selection(df, n_start=5, n_end=20, **kwargs):
    all_results = []
    one_hot_cols = ['vig_activity']
    binary_cols = ['race_1', 'race_2']
    continuous_cols = ['wt_gain','weight_change','age_delivery_range2','num_alc_weekly']
    explicit_cols = ['caffeine','smoke_current','ppg_BMI', 'first_BMI']
    meta_cols = one_hot_cols + binary_cols + continuous_cols + explicit_cols
    non_feature_cols = ['participant_id', 'visit', 'pwr_current']
    all_nmr_cols = [col for col in df.columns if col not in meta_cols + non_feature_cols]

    all_results = []
    for N in range(n_start, n_end +1):
        print(f"Selecting top {N} NMR features")
        result_df, model_fold_params = train_with_cv(
            df,
            all_nmr_cols=all_nmr_cols,
            top_n=N,
            **kwargs,
        )
        result_df['Top NMR Features'] = N
        all_results.append(result_df)

    return pd.concat(all_results, ignore_index=True), model_fold_params

if __name__ == "__main__":
    input_dir = "../test/result/data/meta+1dnmr"
    output_dir = "../test/result/data/meta+1dnmr"
    os.makedirs(output_dir, exist_ok=True)

    visit_dfs = []
    feature_dfs = []
    all_param_dicts = {}

    for visit in range(1, 6):
        print(f"Processing visit {visit}")

        # Input: DFx_filtered_1d.csv
        df = pd.read_csv(f"{input_dir}/DF{visit}_filtered_1d.csv")

        result_df, param_dict = run_nmr_feature_selection(
            df,
            n_start=5,
            n_end=25,
            cv=10,
            inner_cv=10,
            random_state=42
        )

        # Save per-visit results
        result_path = f"{output_dir}/v{visit}_result_meta+selected1dnmr.csv"
        result_df.to_csv(result_path, index=False)

        # Save per-visit parameters
        param_path = f"{output_dir}/v{visit}_param_meta+selected1dnmr.pkl"
        joblib.dump(param_dict, param_path)

        # Save selected features
        features_df = result_df["Selected Features"].explode().apply(pd.Series)
        features_df["Visit"] = f"V{visit}"
        features_path = f"{output_dir}/v{visit}_selected_features_meta+selected1dnmr.csv"
        features_df.to_csv(features_path, index=False)

        # Collect results
        result_df["Visit"] = f"V{visit}"
        visit_dfs.append(result_df)
        feature_dfs.append(features_df)
        all_param_dicts[f"V{visit}"] = param_dict

    # Combine all visits
    all_visits_results = pd.concat(visit_dfs, ignore_index=True)
    all_visits_results.to_csv(
        f"{output_dir}/model_results_all_visits_selected_features_meta+selected1dnmr.csv",
        index=False
    )

    feature_dfs_combined = pd.concat(feature_dfs, ignore_index=True)
    feature_dfs_combined.to_csv(
        f"{output_dir}/model_results_all_visits_selected_features_meta+selected1dnmr.csv",
        index=False
    )

    joblib.dump(
        all_param_dicts,
        f"{output_dir}/all_param_meta+selected1dnmr.pkl"
    )

    print("Done.")