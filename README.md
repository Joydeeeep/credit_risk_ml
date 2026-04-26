# 💳 Credit Risk Prediction System with Model Comparison & Explainability

## 🚀 Overview

This project builds an end-to-end **credit risk prediction system** to estimate the probability of loan default.
It compares **Logistic Regression** and **XGBoost**, evaluates their performance, and deploys the best model with an interactive UI and explainability.

The system is designed to demonstrate:

* ML pipeline design
* Model evaluation and comparison
* Business-oriented decision analysis
* Model explainability (SHAP)
* Deployment via Streamlit

---

## 🧠 Problem Statement

Financial institutions must decide whether to approve or reject loan applications.
The challenge is to:

* Minimize default risk
* Maximize approval rates

This project simulates that decision system using machine learning.

---

## 📊 Dataset

* ~32K loan records
* Features include:

  * Borrower details (age, income, employment length)
  * Loan attributes (amount, interest rate, intent, grade)
  * Credit history

---

## ⚙️ Project Architecture

```
Credit_Risk_ML/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── ab_testing.py
│   └── explainability.py
│
├── main.py                  # Training + evaluation pipeline
├── run_shap_analysis.py     # Global SHAP analysis
├── app.py                   # Streamlit UI
├── requirements.txt
└── README.md
```

---

## 🔧 ML Pipeline

1. Data Loading
2. Preprocessing

   * Missing value handling
   * One-hot encoding
3. Model Training

   * Logistic Regression (baseline)
   * XGBoost (advanced model)
4. Evaluation

   * ROC-AUC
   * Precision / Recall
5. Model Comparison
6. Artifact Saving (models, features, metrics)

---

## 📈 Model Performance

| Model               | ROC-AUC   | Precision (Default) | Recall (Default) |
| ------------------- | --------- | ------------------- | ---------------- |
| Logistic Regression | ~0.87     | ~0.55               | ~0.78            |
| XGBoost             | **~0.95** | **~0.85**           | ~0.80            |

### 🔥 Key Insight

XGBoost significantly improves:

* Risk discrimination
* Precision (fewer false positives)
* Overall decision quality

---

## 🧪 Decision-Level Comparison

Instead of relying only on metrics, the project compares models at the **decision level**:

* XGBoost predicts lower risk for safe borrowers
* Leads to higher approvals
* Reduces unnecessary rejections

---

## 🔍 Explainability (SHAP)

### Global Explainability

* Identifies most important features:

  * Loan-to-income ratio
  * Income
  * Interest rate

### Local Explainability (UI)

* Explains **why a prediction was made**
* Shows feature impact per user

---

## 🖥️ Streamlit UI

The UI allows users to:

* Input borrower details
* Select model (Logistic vs XGBoost)
* View predicted default probability
* Compare both models
* See feature-level explanations (SHAP)

---

## ▶️ How to Run

### 1. Clone the repository

```
git clone <your-repo-link>
cd Credit_Risk_ML
```

---

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

### 4. Train models

```
python main.py
```

---

### 5. Run SHAP analysis

```
python run_shap_analysis.py
```

---

### 6. Launch UI

```
streamlit run app.py
```

---

## 🧠 Key Learnings

* Importance of **proper evaluation beyond accuracy**
* Trade-offs between **precision and recall** in risk systems
* Benefits of **tree-based models for non-linear data**
* Role of **explainability in real-world ML systems**
* Separation of **training, evaluation, and inference layers**

---

## 📌 Future Improvements

* Threshold tuning for business-specific risk tolerance
* Profit/loss simulation
* Model monitoring and drift detection
* API deployment (FastAPI)

---

## 👤 Author

**Joydeep Debsinha**

---

## ⭐ If you found this useful

Give the repo a star!
