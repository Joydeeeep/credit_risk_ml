import pandas as pd

def preprocess(df):
    # Target
    y = df["loan_status"]

    # Features
    X = df.drop(columns=["loan_status"])

    # Separate numeric & categorical
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # Fill missing values
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    X[cat_cols] = X[cat_cols].fillna("Unknown")

    # One-hot encoding
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y

def preprocess_single_input(input_df, feature_columns):
    # Same logic as training
    num_cols = input_df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = input_df.select_dtypes(include=["object"]).columns

    input_df[num_cols] = input_df[num_cols].fillna(input_df[num_cols].median())
    input_df[cat_cols] = input_df[cat_cols].fillna("Unknown")

    input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

    # Align with training features
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure same column order
    input_df = input_df[feature_columns]

    return input_df