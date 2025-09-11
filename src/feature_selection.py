
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

def train_with_cv(df, cv=10, inner_cv=10, random_state=42, other_cols=None):
    one_hot_cols = ['vig_activity']
    binary_cols = ['race_1', 'race_2']
    continuous_cols = ['wt_gain','weight_change','age_delivery_range2','num_alc_weekly']
    explicit_cols = ['caffeine','smoke_current','ppg_BMI', 'first_BMI']
    meta_cols = one_hot_cols + binary_cols + continuous_cols + explicit_cols
    non_feature_cols = ['participant_id', 'visit', 'pwr_current']

    if other_cols is None:
        other_cols = [col for col in df.columns if col not in meta_cols + non_feature_cols]
    explicit_categories = [sorted(set(df[col].dropna().unique())) for col in explicit_cols]

    X = df.drop(columns=['pwr_current','participant_id','visit'])
    y = df['pwr_current']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), one_hot_cols),
            ('explicit', OneHotEncoder(categories=explicit_categories, handle_unknown='ignore'), explicit_cols),
            ('binary', 'passthrough', binary_cols),
            ('num', StandardScaler(), continuous_cols),
            ('other', 'passthrough', other_cols)
        ]
    )

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

    for model_name, model in models.items():
        print(f"Training model: {model_name}")
        y_true, y_pred, y_proba = [], [], []
        fold_scores = []
        fold_best_params = []

        for train_idx, test_idx in outer_kf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            if model_name in param_grid:
                grid = GridSearchCV(pipeline, param_grid[model_name], cv=inner_cv, scoring='roc_auc', n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                fold_best_params.append(grid.best_params_)
            else:
                best_model = pipeline.fit(X_train, y_train)
                fold_best_params.append({})

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
        all_results.append(avg_result)
        model_fold_params[model_name] = fold_best_params

    return pd.DataFrame(all_results), model_fold_params

def run_nmr_feature_selection(df, n_start=5, n_end=20, **kwargs):
    all_results = []
    top_nmr_feature_dict = {}
    all_param_dict = {}
    one_hot_cols = ['vig_activity']
    binary_cols = ['race_1', 'race_2']
    continuous_cols = ['wt_gain','weight_change','age_delivery_range2','num_alc_weekly']
    explicit_cols = ['caffeine','smoke_current','ppg_BMI', 'first_BMI']
    meta_cols = one_hot_cols + binary_cols + continuous_cols + explicit_cols
    non_feature_cols = ['participant_id', 'visit', 'pwr_current']
    all_nmr_cols = [col for col in df.columns if col not in meta_cols + non_feature_cols]

    for N in range(n_start, n_end +1):
        print(f"Selecting top {N} NMR features")
        X_tmp = df.drop(columns=['pwr_current', 'participant_id', 'visit'])
        y_tmp = df['pwr_current']
        top_n_nmr = select_top_nmr_features(X_tmp, y_tmp, all_nmr_cols, N)
        top_nmr_feature_dict[N] = top_n_nmr
        selected_cols = meta_cols + top_n_nmr + non_feature_cols
        df_subset = df[selected_cols].copy()
        result_df, model_fold_params = train_with_cv(df_subset, other_cols=top_n_nmr, **kwargs)
        result_df['Top NMR Features'] = N
        all_results.append(result_df)
        all_param_dict[N] = model_fold_params

    return pd.concat(all_results, ignore_index=True), top_nmr_feature_dict, all_param_dict

if __name__ == "__main__":
    for visit in range(1,6):
        print(f"Processing visit {visit}")
        df = pd.read_csv(f"/zhome/08/f/202291/result/data/meta+nmr/selection/DF{visit}_filtered.csv")
        result_df, feature_dict, param_dict = run_nmr_feature_selection(df,n_start=5, n_end=25, cv=10, inner_cv=10, random_state=42)
        result_df.to_csv(f"/zhome/08/f/202291/result/data/meta+nmr/selection/v{visit}_result_selection.csv", index=False)
        feature_dict_df = pd.DataFrame.from_dict(feature_dict, orient='index').reset_index()
        feature_dict_df.columns = ['Top NMR Features'] + [f'Feature {i+5}' for i in range(feature_dict_df.shape[1] - 1)]
        feature_dict_df.to_csv(f"/zhome/08/f/202291/result/data/meta+nmr/selection/{visit}_top_nmr_feature.csv", index=False)
        joblib.dump(param_dict, f"/zhome/08/f/202291/result/data/meta+nmr/selection/v{visit}_param_selection.pkl")

    visit_dfs = []
    for v in range(1, 6):
        df = pd.read_csv(f"/zhome/08/f/202291/result/data/meta+nmr/selection/v{v}_result_selection.csv")
        df["Visit"] = f"V{v}"
        visit_dfs.append(df)

    all_visits_results = pd.concat(visit_dfs, ignore_index=True)
    all_visits_results.to_csv("/zhome/08/f/202291/result/data/meta+nmr/selection/model_results_all_visits_selection.csv", index=False)
