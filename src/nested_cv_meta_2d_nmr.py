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
import pandas as pd
import numpy as np
import joblib

def train_with_cv(df,cv=10,inner_cv =10,random_state=42):
    # list of columns to analyze
    one_hot_cols = ['vig_activity']
    binary_cols = ['race_1', 'race_2']
    continuous_cols = ['wt_gain','weight_change','age_delivery_range2','num_alc_weekly']
    explicit_cols = ['caffeine','smoke_current','ppg_BMI', 'first_BMI']

    # Drop the target and identifier columns
    X = df.drop(columns = ['pwr_current','participant_id','visit'])
    y = df['pwr_current']
    other_cols = [col for col in X.columns if col not in one_hot_cols + binary_cols + continuous_cols + explicit_cols]

    # Get the explicit categories
    explicit_categories = [sorted(set(X[col].dropna().unique())) for col in explicit_cols]
    # Convert categorical columns to one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop = 'first',handle_unknown='ignore'), one_hot_cols),
            ('explicit', OneHotEncoder(categories=explicit_categories, handle_unknown='ignore'), explicit_cols),
            ('binary', 'passthrough', binary_cols),
            ('num', StandardScaler(), continuous_cols),
            ('other', 'passthrough', other_cols)  # Keep other columns as they are
        ]
    )

    param_grid = {
        'Logistic Regression': [
        # Combinations for 'liblinear' solver
        {'model__C': [0.01, 0.05, 0.1, 1, 10],'model__penalty': ['l1', 'l2'],'model__solver': ['liblinear']},
        # Combinations for 'lbfgs' solver
        {'model__C': [0.01, 0.05, 0.1, 1, 10],'model__penalty': ['l2'],'model__solver': ['lbfgs']},
        # Combinations for 'elastic net'
        {'model__C': [0.01, 0.05, 0.1, 1, 10],'model__penalty': ['elasticnet'],'model__l1_ratio': [0.1, 0.5, 0.9],'model__solver': ['saga']},
        
        ],
        
        'Support Vector Machine':[
        # Linear kernel: C only
        {'model__C': [0.01, 0.1, 1, 10], 'model__kernel': ['linear']},
        # RBF kernel: C and gamma
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
            'model__max_features': ['sqrt', 1.0], # 1.0 means all
            'model__class_weight': ['balanced'],
            'model__bootstrap': [True],
        },

        'XGBoost':{
            'model__n_estimators': [50, 100, 200,300],
            'model__learning_rate': [0.01, 0.05,0.1, 0.2],
            'model__gamma': [0, 0.1],
            'model__reg_alpha': [0.01,0.1], # L1 regularization
            'model__reg_lambda': [0.01,0.1], # L2 regularization
            },
        
        'MultiLayer Perceptron':{
            'model__hidden_layer_sizes': [(100,), (150,),(150,50),(100,50),(50,25)],
            'model__activation': ['relu','tanh'],
            'model__alpha': [0.0001, 0.001],
            'model__solver': ['adam', 'lbfgs'], # 'lbfgs' is for smaller datasets
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
    # prepare the models
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

    all_results = []
    outer_kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    best_models = {}
    # Loop through each outer fold
    for model_name, model in models.items():
        print(f"Training on the model:{model_name}")
        fold_metrics = []
        all_y_test_outer = []
        all_y_pred_outer = []
        all_y_proba_outer = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_kf.split(X, y)):
            print(f"This is fold:{fold_idx + 1} for model:{model_name}")
            X_train, X_test_outer_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test_outer_fold = y.iloc[train_idx], y.iloc[test_idx]

            # Define the pipeline
            clf = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            if model_name in param_grid:
                grid = GridSearchCV(
                    clf,
                    param_grid=param_grid[model_name],
                    cv=StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=random_state),
                    scoring='roc_auc',
                    n_jobs=4,
                )
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
            else:
                best_model = clf
                best_model.fit(X_train, y_train)
            
            # Predict and Evaluate on the outer fold test set
            y_pred_outer_fold = best_model.predict(X_test_outer_fold)

            try:
                y_proba_outer_fold = best_model.predict_proba(X_test_outer_fold)[:, 1]
                roc = roc_auc_score(y_test_outer_fold, y_proba_outer_fold)
            except:
                y_proba_outer_fold = None
                roc = None
            
            # Store the results for this fold
            fold_metrics.append({
                'accuracy': accuracy_score(y_test_outer_fold, y_pred_outer_fold),
                'f1': f1_score(y_test_outer_fold, y_pred_outer_fold, zero_division=0),
                'precision': precision_score(y_test_outer_fold, y_pred_outer_fold, zero_division=0),
                'recall': recall_score(y_test_outer_fold, y_pred_outer_fold, zero_division=0),
                'roc_auc': roc,
                'confusion_matrix': confusion_matrix(y_test_outer_fold, y_pred_outer_fold),
            })

            # Store predictions and probabilities for later analysis
            all_y_test_outer.extend(y_test_outer_fold.tolist())
            all_y_pred_outer.extend(y_pred_outer_fold.tolist())
            
            if y_proba_outer_fold is not None:
                all_y_proba_outer.extend(y_proba_outer_fold.tolist())
            else:
                all_y_proba_outer.extend([None] * len(y_test_outer_fold))
        
        # Calculate mean metrics across all folds
        overall_roc_auc = roc_auc_score(all_y_test_outer, all_y_proba_outer) if all_y_proba_outer else None
        overall_confusion_matrix = confusion_matrix(all_y_test_outer, all_y_pred_outer)
        
        # Store the average results for this model
        n_folds = len(fold_metrics)

        avg_results = {
            'Model': model_name,
            'Average CV Accuracy': np.mean([fold['accuracy'] for fold in fold_metrics]),
            'Std CV Accuracy': np.std([fold['accuracy'] for fold in fold_metrics]),
            'SE CV Accuracy': np.std([fold['accuracy'] for fold in fold_metrics]) / np.sqrt(n_folds),

            'Average CV F1 Score': np.mean([fold['f1'] for fold in fold_metrics]),
            'Std CV F1 Score': np.std([fold['f1'] for fold in fold_metrics]),
            'SE CV F1 Score': np.std([fold['f1'] for fold in fold_metrics]) / np.sqrt(n_folds),

            'Average CV Precision': np.mean([fold['precision'] for fold in fold_metrics]),
            'Std CV Precision': np.std([fold['precision'] for fold in fold_metrics]),
            'SE CV Precision': np.std([fold['precision'] for fold in fold_metrics]) / np.sqrt(n_folds),

            'Average CV Recall': np.mean([fold['recall'] for fold in fold_metrics]),
            'Std CV Recall': np.std([fold['recall'] for fold in fold_metrics]),
            'SE CV Recall': np.std([fold['recall'] for fold in fold_metrics]) / np.sqrt(n_folds),

            'Average CV ROC AUC': np.mean([fold['roc_auc'] for fold in fold_metrics if fold['roc_auc'] is not None]),
            'Std CV ROC AUC': np.std([fold['roc_auc'] for fold in fold_metrics if fold['roc_auc'] is not None]),
            'SE CV ROC AUC': np.std([fold['roc_auc'] for fold in fold_metrics if fold['roc_auc'] is not None]) / np.sqrt(n_folds),
            
            'Overall ROC AUC': overall_roc_auc,
            'Overall Confusion Matrix': overall_confusion_matrix,
            'All Fold Confusion Matrices': [fold['confusion_matrix'] for fold in fold_metrics],
        }
        avg_results['All y True'] = all_y_test_outer
        avg_results['All y Proba'] = all_y_proba_outer
        all_results.append(avg_results)
        best_models[model_name] = {
            'estimator': best_model
        }
    return pd.DataFrame(all_results), all_results, best_models

if __name__ == "__main__":
    # Training and evaluating models for each visit

    # Dictionary to save all visits' all_results 
    all_visit_raw_results = {}
    all_best_models = {}
    for v in range(1,6): 
        df = pd.read_csv(f"/zhome/08/f/202291/result/data/meta+nmr/DF{v}.csv")
        print(f"Results for Visit V{v}:")
        result, all_results, best_models = train_with_cv(df)
        # Save the results to excel files
        result.to_csv(f"/zhome/08/f/202291/result/data/meta+nmr/model_results_v{v}_hyper.csv", index=True)
        joblib.dump(all_results,f"/zhome/08/f/202291/result/data/meta+nmr/all_results_v{v}.pkl")

        all_visit_raw_results[f"V{v}"] = all_results
        all_best_models[f"V{v}"] = best_models
    # Save combined all_results
    joblib.dump(all_visit_raw_results,"/zhome/08/f/202291/result/data/meta+nmr/all_results_combined.pkl")
    joblib.dump(all_best_models,"/zhome/08/f/202291/result/data/meta+nmr/all_best_models.pkl")
    print("\nTraining and evaluation complete for all files.")