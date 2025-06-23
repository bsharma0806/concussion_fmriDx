import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score, learning_curve
)
from sklearn import metrics
from tqdm import tqdm

# load_data
def load_data(file):
    """
    Load a CSV where the last column is the target and
    all preceding columns are features.
    """
    import pandas as pd

    df = pd.read_csv(file)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y, X.columns.tolist()

# main as in the ipynb
def main(file):
    start_time = time.time()
    np.random.seed(89)

    X, y, feature_names = load_data(file)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=89, stratify=y
    )

    safe_k = max(1, min(3, minority_class_count - 1))
    print(f"Using k_neighbors={safe_k} for SMOTE to avoid errors")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=89, k_neighbors=safe_k, sampling_strategy='auto')),
        ('classifier', GradientBoostingClassifier(random_state=89))
    ])

    param_distributions = {
        'smote__k_neighbors': [1, 2, 3], 
        'classifier__n_estimators': [100, 200, 300, 400, 500],
        'classifier__learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [2, 3, 4],
        'classifier__min_samples_split': [2, 5, 10, 15, 20],
        'classifier__min_samples_leaf': [2, 4, 6, 8],
        'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'classifier__max_features': ['sqrt', 'log2', None]
    }

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=50,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=89),
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=42,
        error_score='raise'
    )

    print("\nStarting RandomizedSearchCV...")
    with tqdm(total=100, desc="Training Progress") as pbar:
        random_search.fit(X_train, y_train)
        pbar.update(100)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=89)
    cv_scores = cross_val_score(random_search.best_estimator_, X_train, y_train, cv=cv, scoring='f1')
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    best_pipeline = random_search.best_estimator_
    X_train_resampled, y_train_resampled = best_pipeline.named_steps['smote'].fit_resample(
        best_pipeline.named_steps['scaler'].fit_transform(X_train), 
        y_train
    )

    # results
    results = {
        'best_parameters': random_search.best_params_,
        'best_cv_score': float(random_search.best_score_),
        'cv_scores': cv_scores.tolist(),
        'training_time': time.time() - start_time,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'class_distribution_before': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        'class_distribution_after_smote': {str(k): int(v) for k, v in zip(*np.unique(y_train_resampled, return_counts=True))},
        'best_k_neighbors': random_search.best_params_['smote__k_neighbors']
    }

    # best model
    best_model = random_search.best_estimator_

    # test metrics
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    # additional metrics
    results['test_accuracy'] = float(metrics.accuracy_score(y_test, y_pred))
    results['test_auc'] = float(metrics.roc_auc_score(y_test, y_pred_proba[:, 1]))
    results['test_f1'] = float(metrics.f1_score(y_test, y_pred))
    results['test_precision'] = float(metrics.precision_score(y_test, y_pred))
    results['test_recall'] = float(metrics.recall_score(y_test, y_pred))
    results['classification_report'] = metrics.classification_report(y_test, y_pred, output_dict=True)

    # print results
    print("\n=== Model Performance ===")
    print(f"Best parameters: {results['best_parameters']}")
    print(f"Best CV score: {results['best_cv_score']:.4f}")
    print(f"Test accuracy: {results['test_accuracy']:.4f}")
    print(f"Test AUC: {results['test_auc']:.4f}")
    print(f"Test F1-score: {results['test_f1']:.4f}")
    print(f"Test Precision: {results['test_precision']:.4f}")
    print(f"Test Recall: {results['test_recall']:.4f}")
    print(f"\nTraining time: {results['training_time']:.2f} seconds")
    
    best_model = random_search.best_estimator_
    return best_model, feature_names