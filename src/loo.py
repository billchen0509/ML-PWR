import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
import xgboost as xgb

def train_with_cv(df,inner_cv = 10,random_state=42):
    # Drop the target and identifier columns
    X = df.drop(columns = ['pwr_current'])
    y = df['pwr_current']

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
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'GaussianNB': GaussianNB(),
        'BernoulliNB': BernoulliNB(),
        'Support Vector Machine': SVC(probability=True, random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'K-Nearest Neighbors': KNeighborsClassifier(n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'MultiLayer Perceptron': MLPClassifier(max_iter=1000, random_state=random_state),
    }    

    all_results = []
    outer_kf = LeaveOneOut()
    best_models = {}
    # Loop through each outer fold
    for model_name, model in models.items():
        print(f"Training on the model:{model_name}")
        all_y_test_outer = []
        all_y_pred_outer = []
        all_y_proba_outer = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_kf.split(X, y)):
            print(f"{model_name} Leave-One-Out iteration {fold_idx + 1} / {len(X)} (Test index: {test_idx[0]})")
            X_train, X_test_outer_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test_outer_fold = y.iloc[train_idx], y.iloc[test_idx]

            # Define the pipeline
            clf = Pipeline([
                ('model', model)
            ])

            if model_name in param_grid:
                grid = GridSearchCV(
                    clf,
                    param_grid=param_grid[model_name],
                    cv=StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=random_state),
                    scoring='f1',
                    n_jobs=-1,
                )
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
            else:
                best_model = clf
                best_model.fit(X_train, y_train)
            
            # Predict and Evaluate on the outer fold test set
            X_test_fixed = (
            np.ascontiguousarray(X_test_outer_fold.values)
            if model_name == "K-Nearest Neighbors"
            else X_test_outer_fold
            )

            y_pred_outer_fold = best_model.predict(X_test_fixed)

            try:
                y_proba_outer_fold = best_model.predict_proba(X_test_fixed)[:, 1]
            except:
                y_proba_outer_fold = None
            
            # Store predictions and probabilities for later analysis
            all_y_test_outer.extend(y_test_outer_fold.tolist())
            all_y_pred_outer.extend(y_pred_outer_fold.tolist())
            
            if y_proba_outer_fold is not None:
                all_y_proba_outer.extend(y_proba_outer_fold.tolist())
            else:
                all_y_proba_outer.extend([None] * len(y_test_outer_fold))
        
        valid_roc_pairs = [(yt, yp) for yt, yp in zip(all_y_test_outer, all_y_proba_outer) if yp is not None]
        if len(valid_roc_pairs) > 0:
            y_true_valid, y_proba_valid = zip(*valid_roc_pairs)
            if len(set(y_true_valid)) > 1:
                overall_roc_auc = roc_auc_score(y_true_valid, y_proba_valid)
            else:
                overall_roc_auc = None  
        else:
            overall_roc_auc = None  

        avg_results = {
            'Model':model_name,
            'Overall Accuracy': accuracy_score(all_y_test_outer, all_y_pred_outer),
            'Overall F1': f1_score(all_y_test_outer, all_y_pred_outer, zero_division=0),
            'Overall Precision': precision_score(all_y_test_outer, all_y_pred_outer, zero_division=0),
            'Overall Recall': recall_score(all_y_test_outer, all_y_pred_outer, zero_division=0),
            'Overall ROC AUC': overall_roc_auc,
            'All y True': all_y_test_outer,
            'All y Proba': all_y_proba_outer,
            'Overall Confusion Matrix': confusion_matrix(all_y_test_outer, all_y_pred_outer),
        }
        all_results.append(avg_results)
        best_models[model_name] = {
            'estimator': best_model
        }
    return pd.DataFrame(all_results), all_results, best_models

if __name__ == "__main__":
    # Dictionary to save all visits' all_results 
    all_visit_raw_results = {}
    all_best_models = {}
    for v in range(1,6): 
        df = pd.read_csv(f"../result/data/nmr_only/DF{v}_nmr_only.csv")
        print(f"Results for Visit V{v}:")
        result, all_results, best_models = train_with_cv(df)
        # Save the results to excel files
        result.to_excel(f"../result/data/nmr_only/loo/model_results_v{v}_hyper_nmr.xlsx", index=True)
        joblib.dump(all_results,f"../result/data/nmr_only/loo/all_results_v{v}_nmr.pkl")
    
        all_visit_raw_results[f"V{v}"] = all_results
        all_best_models[f"V{v}"] = best_models
    # Save combined all_results
    joblib.dump(all_visit_raw_results,"../result/data/nmr_only/loo/all_results_combined_nmr.pkl")
    joblib.dump(all_best_models,"../result/data/nmr_only/loo/all_best_models_nmr.pkl")
    print("\nTraining and evaluation complete for all files.")
