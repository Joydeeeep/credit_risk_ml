import pandas as pd

def load_data(path, sample_size=None):
    df = pd.read_csv(path)

    # Optional sampling for faster iteration
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    return df