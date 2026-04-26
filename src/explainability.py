import shap


def explain_model(model, X_sample):
    """
    Generate SHAP explanations for XGBoost model
    """

    print("\nGenerating SHAP explanations...")

    # TreeExplainer works best for XGBoost
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    shap.summary_plot(shap_values, X_sample)

    return shap_values