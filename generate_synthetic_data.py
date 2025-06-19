import numpy as np
import pandas as pd

def generate_synthetic_data(real_data_path: str, output_path: str = "synthetic_data.csv", random_seed: int = 42):
    np.random.seed(random_seed)
    df = pd.read_csv(real_data_path)
    df.columns = df.columns.str.strip()
    label_col = 'Label'
    features = df.drop(columns=label_col)
    labels = df[label_col]
    means = features.mean()
    stds = features.std()
    label_distribution = labels.value_counts(normalize=True)
    synthetic_features = np.random.normal(
        loc=means.values,
        scale=stds.values,
        size=(df.shape[0], features.shape[1])
    )
    synthetic_df = pd.DataFrame(synthetic_features, columns=features.columns)
    synthetic_df.insert(0, label_col, np.random.choice(
        label_distribution.index,
        size=df.shape[0],
        p=label_distribution.values
    ))
    synthetic_df.to_csv(output_path, index=False)
    print(f"Synthetic data saved to: {output_path}")

if __name__ == "__main__":
    generate_synthetic_data("data.csv")
