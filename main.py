import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


# -----------------------------
# Evaluation function
# -----------------------------
def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_tr = model.predict(X_train)
    y_te = model.predict(X_test)

    acc_tr = accuracy_score(y_train, y_tr)
    acc_te = accuracy_score(y_test, y_te)

    pr_tr, rc_tr, f1_tr, _ = precision_recall_fscore_support(
        y_train, y_tr, average='weighted', zero_division=0
    )
    pr_te, rc_te, f1_te, _ = precision_recall_fscore_support(
        y_test, y_te, average='weighted', zero_division=0
    )

    return dict(
        Model=name,
        Train_Accuracy=acc_tr, Test_Accuracy=acc_te,
        Train_Precision_w=pr_tr, Test_Precision_w=pr_te,
        Train_Recall_w=rc_tr, Test_Recall_w=rc_te,
        Train_F1_w=f1_tr, Test_F1_w=f1_te
    )


# -----------------------------
# 1) Load data
# -----------------------------
SPECTRAL_FILE = '20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv'
assert Path(SPECTRAL_FILE).exists(), f"File not found: {SPECTRAL_FILE}"

spectral = pd.read_csv(SPECTRAL_FILE)

spec_cols = [
    c for c in spectral.columns
    if c.startswith('V') or c.startswith('sprs')
    or c in ['Fw1','Fw2','Fw3','Fw4','Mw1','Mw2','Mw3','Mw4',
             'F1','F2','F3','F4','M1','M2','M3','M4']
]

df = spectral.dropna(subset=['Context2']).copy()
X = df[spec_cols].fillna(0)
y = df['Context2'].astype(str)

# -----------------------------
# 2) Split + CV
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scaler = StandardScaler()

rows = []

# -----------------------------
# 3) Models + RandomizedSearchCV
# -----------------------------
# SVM
svm = Pipeline([('scaler', scaler), ('clf', SVC(kernel='rbf'))])
svm_param = {'clf__C': [0.1, 1, 10, 30], 'clf__gamma': ['scale', 0.01, 0.001]}
svm_cv = RandomizedSearchCV(svm, svm_param, n_iter=6, cv=cv, n_jobs=-1,
                            random_state=42, scoring='f1_weighted')
rows.append(evaluate('SVM (RS)', svm_cv, X_train, X_test, y_train, y_test))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt_param = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_cv = RandomizedSearchCV(dt, dt_param, n_iter=6, cv=cv, n_jobs=-1,
                           random_state=42, scoring='f1_weighted')
rows.append(evaluate('DecisionTree (RS)', dt_cv, X_train, X_test, y_train, y_test))

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf_param = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', None]
}
rf_cv = RandomizedSearchCV(rf, rf_param, n_iter=8, cv=cv, n_jobs=-1,
                           random_state=42, scoring='f1_weighted')
rows.append(evaluate('RandomForest (RS)', rf_cv, X_train, X_test, y_train, y_test))

# AdaBoost
ada = AdaBoostClassifier(random_state=42)
ada_param = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.03, 0.1, 0.3, 1.0]
}
ada_cv = RandomizedSearchCV(ada, ada_param, n_iter=6, cv=cv, n_jobs=-1,
                            random_state=42, scoring='f1_weighted')
rows.append(evaluate('AdaBoost (RS)', ada_cv, X_train, X_test, y_train, y_test))

# GaussianNB
nb = Pipeline([('scaler', scaler), ('clf', GaussianNB())])
rows.append(evaluate('GaussianNB', nb, X_train, X_test, y_train, y_test))

# MLP
mlp = Pipeline([('scaler', scaler), ('clf', MLPClassifier(max_iter=400, random_state=42))])
mlp_param = {
    'clf__hidden_layer_sizes': [(64,), (128,), (64, 32)],
    'clf__alpha': [1e-4, 1e-3, 1e-2],
    'clf__learning_rate_init': [1e-3, 3e-3, 1e-2]
}
mlp_cv = RandomizedSearchCV(mlp, mlp_param, n_iter=6, cv=cv, n_jobs=-1,
                            random_state=42, scoring='f1_weighted')
rows.append(evaluate('MLP (RS)', mlp_cv, X_train, X_test, y_train, y_test))

# Optional: XGBoost
try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    xgb_param = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 8],
        'learning_rate': [0.03, 0.1, 0.2],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0]
    }
    xgb_cv = RandomizedSearchCV(xgb, xgb_param, n_iter=8, cv=cv, n_jobs=-1,
                                random_state=42, scoring='f1_weighted')
    rows.append(evaluate('XGBoost (RS)', xgb_cv, X_train, X_test, y_train, y_test))
except Exception:
    pass

# Optional: CatBoost
try:
    from catboost import CatBoostClassifier
    cat = CatBoostClassifier(verbose=False, random_state=42)
    cat_param = {
        'depth': [4, 6, 8],
        'learning_rate': [0.03, 0.1, 0.2],
        'iterations': [100, 200],
        'l2_leaf_reg': [1, 3, 5]
    }
    cat_cv = RandomizedSearchCV(cat, cat_param, n_iter=6, cv=cv, n_jobs=-1,
                                random_state=42, scoring='f1_weighted')
    rows.append(evaluate('CatBoost (RS)', cat_cv, X_train, X_test, y_train, y_test))
except Exception:
    pass


# -----------------------------
# MAIN: outputs & graphs
# -----------------------------
if __name__ == "__main__":
    # Results table
    res = pd.DataFrame(rows).sort_values('Test_F1_w', ascending=False)
    print(res)

    # Save results
    res.to_csv('model_cv_results.csv', index=False)
    print("\nSaved: model_cv_results.csv")

    # Show best params (if available)
    try:
        print("\nSVM best params:", svm_cv.best_params_)
    except Exception:
        pass
    try:
        print("RF best params:", rf_cv.best_params_)
    except Exception:
        pass

    # Plot bar chart
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.bar(res['Model'], res['Test_F1_w'], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Weighted F1 (Test)")
    plt.title("Model Comparison (Test F1 Score)")
    plt.tight_layout()
    plt.show()

