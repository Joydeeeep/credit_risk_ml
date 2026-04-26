from sklearn.metrics import roc_auc_score, classification_report

def evaluate(model, scaler, X_test, y_test):
    if scaler:
        X_test = scaler.transform(X_test)

    preds_proba = model.predict_proba(X_test)[:, 1]
    preds = (preds_proba > 0.5).astype(int)

    roc = roc_auc_score(y_test, preds_proba)

    report = classification_report(y_test, preds, output_dict=True)

    precision_default = report['1']['precision']
    recall_default = report['1']['recall']

    print("ROC-AUC:", roc)
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    return {
        "roc_auc": roc,
        "precision_default": precision_default,
        "recall_default": recall_default
    }