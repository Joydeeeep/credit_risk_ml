# 💳 Credit Risk Prediction System with Model Comparison & Explainability

An end-to-end Machine Learning project for predicting loan default risk, comparing models, and providing interpretable insights using SHAP.  
The system includes a full ML pipeline, model evaluation, decision-level comparison, and an interactive Streamlit UI.

---

## 🚀 Overview

This project builds a credit risk prediction system to classify whether a borrower is likely to default on a loan.

It focuses on:
- Strong ML fundamentals
- Model comparison (Logistic Regression vs XGBoost)
- Business-oriented evaluation
- Explainability (SHAP)
- Interactive deployment (Streamlit)

---

## 📊 Key Results

| Metric | Logistic Regression | XGBoost |
|-------|--------------------|---------|
| ROC-AUC | 0.87 | **0.95** |
| Precision (Default) | 0.55 | **0.85** |
| Recall (Default) | 0.78 | 0.80 |

### 💡 Insights:
- XGBoost significantly improves classification performance  
- ~10% higher approval rate  
- ~1.4% reduction in default risk  
- Better separation between low-risk and high-risk borrowers  

---

## 🧠 Features

### ✅ ML Pipeline
- Data preprocessing (handling categorical + numerical features)
- Train/test split
- Logistic Regression (baseline)
- XGBoost (advanced model)

### ✅ Evaluation
- ROC-AUC
- Precision & Recall
- Classification report
- Model comparison

### ✅ Decision-Level Analysis
- Per-instance comparison between models
- Business interpretation of predictions

### ✅ Explainability
- SHAP (global + local)
- Feature importance analysis
- Per-user explanation in UI

### ✅ Deployment
- Interactive Streamlit app
- Real-time prediction + explanation

---

## 🏗️ Project Structure
