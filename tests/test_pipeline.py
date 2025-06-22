from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def create_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=89, k_neighbors=2)),
        ('classifier', GradientBoostingClassifier(random_state=89))
    ])

def test_pipeline_structure():
    pipeline = create_pipeline()
    steps = dict(pipeline.named_steps)

    assert 'scaler' in steps
    assert isinstance(steps['scaler'], StandardScaler)

    assert 'smote' in steps
    assert isinstance(steps['smote'], SMOTE)

    assert 'classifier' in steps
    assert isinstance(steps['classifier'], GradientBoostingClassifier)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def test_evaluation_metrics_are_valid():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    y_scores = [0.1, 0.9, 0.4, 0.2, 0.8]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)

    assert isinstance(acc, float) and 0 <= acc <= 1
    assert isinstance(f1, float) and 0 <= f1 <= 1
    assert isinstance(auc, float) and 0 <= auc <= 1