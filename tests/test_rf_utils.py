import pandas as pd
from rf_utils import load_data, load_model, load_splits

def test_load_data_returns_correct_shapes():
    X, y, feature_names = load_data("synthetic_data.csv")
    assert isinstance(X, pd.DataFrame)
    assert len(X) == len(y), "X and y must have the same number of rows."
    assert list(X.columns) == feature_names, "Feature names must match column names."

def test_load_model_can_predict():
    model = load_model("best_model.pkl")
    assert hasattr(model, "predict"), "Model should have a .predict() method."

def test_load_splits_shapes_match():
    X_train, X_test, y_train, y_test = load_splits("data/splits.pkl")
    assert len(X_train) == len(y_train), "Train features and labels should match."
    assert len(X_test) == len(y_test), "Test features and labels should match."