# 🏦 Bank Customer Churn Prediction ML System

An end-to-end Machine Learning system for predicting bank customer churn using CatBoost, SMOTE, FastAPI, and Docker.

This project goes beyond notebook experimentation and demonstrates a full ML workflow from data analysis to production-ready deployment.

---

## 🚀 Project Overview

Customer churn is one of the most critical challenges in the banking industry.  
Acquiring a new customer is significantly more expensive than retaining an existing one.

This project builds a complete churn prediction system including:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Comparison
- Hyperparameter Optimization
- Threshold Optimization
- API Deployment
- Docker Containerization

---

## 🧠 Machine Learning Pipeline

### 1️⃣ Exploratory Data Analysis
- Distribution analysis
- Churn segmentation
- Age group analysis
- Missing value inspection

### 2️⃣ Feature Engineering (6 engineered features)
- Balance_Salary_Ratio
- Tenure_Age_Ratio
- Is_Senior
- CreditScore_Age_Ratio
- Products_x_Active
- IsBalanceZero

### 3️⃣ Handling Class Imbalance
- SMOTE applied on training data only

### 4️⃣ Model Comparison
Compared multiple algorithms:

- Logistic Regression
- KNN
- SVM
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

### 5️⃣ Hyperparameter Tuning
- RandomizedSearchCV (5-fold cross-validation)
- Optimized for F1 score

### 6️⃣ Threshold Optimization
Default threshold (0.5) was replaced with mathematically optimized threshold for improved F1 performance.

---

## 📊 Final Model Performance

| Metric   | Score |
|----------|-------|
| ROC-AUC  | 0.865 |
| F1 Score | 0.635 |

**Final Algorithm:** CatBoost Classifier

---

## 🏗 Production Architecture

This project includes a production-ready structure:

- FastAPI backend
- Swagger documentation
- Custom web interface
- Docker containerization
- Structured project architecture
- Scaler & encoder persistence
- Feature order consistency via JSON contract

---

## 📂 Project Structure
bank-churn-prediction-ml-system/
│
├── app/
│ ├── main.py
│ ├── model.joblib
│ ├── scaler.joblib
│ ├── label_encoder.joblib
│ ├── feature_columns.json
│ └── static/
│ └── index.html
│
├── notebooks/
│ └── Churn_Prediction_CatBoost_SMOTE.ipynb
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── LICENSE
└── README.md


---

## 🐳 Run Locally with Docker

```bash
docker-compose up --build

Then open:

http://localhost:8000


## 📡 API Example

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "credit_score": 650,
    "geography": "Germany",
    "gender": "Female",
    "age": 42,
    "tenure": 3,
    "balance": 120000,
    "num_of_products": 1,
    "has_cr_card": 1,
    "is_active_member": 0,
    "estimated_salary": 95000
  }'


📎 Related Links

Kaggle Notebook: (https://www.kaggle.com/code/mesutkaragz/churn-prediction-catboost-smote)

👤 Author

Mesut Karagöz
Data Analyst | Data Scientist

📌 License

This project is licensed under the MIT License.

