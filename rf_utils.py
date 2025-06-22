import pandas as pd
import joblib

def load_data(file="synthetic_data.csv"):
    """Load CSV; last column is target."""
    df = pd.read_csv(file)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y, X.columns.tolist()

def load_model(path="best_model.pkl"):
    """Load your trained pipeline."""
    return joblib.load(path)

def load_splits(path="data/splits.pkl"):
    """Load the exact train/test splits you saved."""
    return joblib.load(path)  # returns X_train, X_test, y_train, y_test