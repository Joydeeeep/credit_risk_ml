from src.data_loader import load_data
from src.preprocessing import preprocess
from src.train import split_data, train_model, train_xgboost
from src.ab_testing import ab_test
from src.evaluate import evaluate
import joblib


def main():
    print("🔹 Loading data...")
    df = load_data("data/credit_risk_dataset.csv")

    print("🔹 Preprocessing data...")
    X, y = preprocess(df)
    print(f"Feature shape: {X.shape}")

    print("🔹 Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ---------------- LOGISTIC REGRESSION ----------------
    print("\n🔹 Training Logistic Regression...")
    log_model, scaler = train_model(X_train, y_train)

    print("🔹 Evaluating Logistic Regression...")
    log_metrics = evaluate(log_model, scaler, X_test, y_test)

    print(f"""
Logistic Regression Results:
ROC-AUC: {log_metrics['roc_auc']:.4f}
Precision (Default): {log_metrics['precision_default']:.4f}
Recall (Default): {log_metrics['recall_default']:.4f}
""")

    # ---------------- XGBOOST ----------------
    print("\n🔹 Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)

    print("🔹 Evaluating XGBoost...")
    xgb_metrics = evaluate(xgb_model, None, X_test, y_test)

    print(f"""
XGBoost Results:
ROC-AUC: {xgb_metrics['roc_auc']:.4f}
Precision (Default): {xgb_metrics['precision_default']:.4f}
Recall (Default): {xgb_metrics['recall_default']:.4f}
""")

    # ---------------- A/B TEST ----------------
    print("\n🔹 Running A/B Test Simulation...")

    lr_probs = log_model.predict_proba(scaler.transform(X_test))[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    ab_test(y_test, lr_probs, xgb_probs)

    # ---------------- SAVE ARTIFACTS ----------------
    print("\n🔹 Saving artifacts...")

    joblib.dump(xgb_model, "model.pkl")
    joblib.dump(log_model, "logistic_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(X.columns.tolist(), "features.pkl")

    metrics = {
        "logistic": log_metrics,
        "xgboost": xgb_metrics
    }

    joblib.dump(metrics, "metrics.pkl")

    print("✅ Pipeline complete. Models and metrics saved.")


if __name__ == "__main__":
    main()