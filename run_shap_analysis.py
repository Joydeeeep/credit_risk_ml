import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.preprocessing import preprocess


def main():
    print("🔹 Loading model...")
    model = joblib.load("model.pkl")

    print("🔹 Loading data...")
    df = load_data("data/credit_risk_dataset.csv")

    print("🔹 Preprocessing...")
    X, _ = preprocess(df)

    print("🔹 Sampling data for SHAP...")
    X_sample = X.sample(n=500, random_state=42)

    print("🔹 Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model)

    print("🔹 Computing SHAP values...")
    shap_values = explainer.shap_values(X_sample)

    print("🔹 Generating summary plot...")
    shap.summary_plot(shap_values, X_sample)

    print("🔹 Generating feature importance bar plot...")
    shap.summary_plot(shap_values, X_sample, plot_type="bar")

    print("✅ SHAP analysis complete.")


if __name__ == "__main__":
    main()