import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------- LOAD ARTIFACTS ----------------
log_model = joblib.load("logistic_model.pkl")
xgb_model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("features.pkl")
metrics = joblib.load("metrics.pkl")

explainer = shap.TreeExplainer(xgb_model)

st.title("💳 Credit Risk Prediction System")

st.header("📊 Model Evaluation (Offline Results)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Logistic Regression")
    st.write(f"ROC-AUC: {metrics['logistic']['roc_auc']:.2f}")
    st.write(f"Precision (Default): {metrics['logistic']['precision_default']:.2f}")
    st.write(f"Recall (Default): {metrics['logistic']['recall_default']:.2f}")

with col2:
    st.subheader("XGBoost")
    st.write(f"ROC-AUC: {metrics['xgboost']['roc_auc']:.2f}")
    st.write(f"Precision (Default): {metrics['xgboost']['precision_default']:.2f}")
    st.write(f"Recall (Default): {metrics['xgboost']['recall_default']:.2f}")

# ---------------- INPUTS ----------------
st.header("📋 Borrower Details")

age = st.number_input("Age", 18, 100, 30)
income = st.number_input("Income", 1000, 500000, 50000)
loan_amount = st.number_input("Loan Amount", 500, 100000, 10000)
interest_rate = st.number_input("Interest Rate", 1.0, 30.0, 10.0)
emp_length = st.number_input("Employment Length", 0, 40, 5)
cred_hist = st.number_input("Credit History Length", 0, 30, 5)

home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
default_on_file = st.selectbox("Previous Default", ["Y", "N"])

loan_percent_income = loan_amount / income if income > 0 else 0

# ---------------- MODEL SELECT ----------------
st.header("⚙️ Model Selection")

model_choice = st.selectbox(
    "Select Model",
    ["XGBoost (Recommended)", "Logistic Regression"]
)

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    # ---------------- PREPROCESS ----------------
    input_data = pd.DataFrame([{
        "person_age": age,
        "person_income": income,
        "person_home_ownership": home_ownership,
        "person_emp_length": emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amount,
        "loan_int_rate": interest_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": default_on_file,
        "cb_person_cred_hist_length": cred_hist
    }])

    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)

    # ---------------- PREDICTIONS ----------------
    lr_prob = log_model.predict_proba(scaler.transform(input_data))[0][1]
    xgb_prob = xgb_model.predict_proba(input_data)[0][1]

    # Select model output
    if "Logistic" in model_choice:
        prob = lr_prob
        model_used = "Logistic Regression"
    else:
        prob = xgb_prob
        model_used = "XGBoost"

    # ---------------- RESULT ----------------
    st.header("📊 Prediction Result")

    st.subheader(f"Model Used: {model_used}")
    st.metric("Default Risk", f"{prob*100:.2f}%")

    if prob > 0.5:
        st.error("❌ High Risk - Reject Loan")
    else:
        st.success("✅ Low Risk - Approve Loan")

    # ---------------- MODEL COMPARISON ----------------
    st.header("🔁 Model Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Logistic Regression", f"{lr_prob*100:.2f}%")

    with col2:
        st.metric("XGBoost", f"{xgb_prob*100:.2f}%")

    # ---------------- DECISION INSIGHT ----------------
    st.header("🧠 Decision Insight")

    diff = lr_prob - xgb_prob

    if diff > 0:
        direction = "lower"
    else:
        direction = "higher"

    st.write(f"""
### 📊 Model Behavior:

- Logistic Regression Risk: **{lr_prob*100:.2f}%**
- XGBoost Risk: **{xgb_prob*100:.2f}%**

### 🔍 Key Difference:

XGBoost predicts **{abs(diff)*100:.2f}% {direction} risk** compared to Logistic Regression.

### 💡 Interpretation:

- XGBoost captures complex feature interactions better  
- It separates low-risk and high-risk borrowers more effectively  
- This leads to improved decision quality  

### 🎯 Conclusion:

👉 XGBoost provides more reliable predictions for this applicant
""")

    # ---------------- SHAP ----------------
    if "XGBoost" in model_choice:

        st.header("🔍 Explainability (SHAP)")

        shap_values = explainer.shap_values(input_data)

        shap_df = pd.DataFrame({
            "feature": input_data.columns,
            "impact": shap_values[0]
        })

        shap_df["abs_impact"] = shap_df["impact"].abs()
        shap_df = shap_df.sort_values(by="abs_impact", ascending=False).head(5)

        st.subheader("Top Factors Influencing Prediction")

        for _, row in shap_df.iterrows():
            direction = "increases risk" if row["impact"] > 0 else "decreases risk"
            st.write(f"**{row['feature']}** → {direction}")

        fig, ax = plt.subplots()
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value,
            shap_values[0],
            feature_names=input_data.columns,
            max_display=6,
            show=False
        )
        st.pyplot(fig)

    else:
        st.info("SHAP explanation available only for XGBoost model.")