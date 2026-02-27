# 🏦 Bank Customer Churn Predictor

> FastAPI + CatBoost ML model deployed as a containerized web application.

| Metric     | Score  |
|------------|--------|
| ROC-AUC    | 0.865  |
| F1 Score   | 0.635  |
| Model      | CatBoost |
| Balancing  | SMOTE  |

---

## 🚀 Quick Start

### Option 1 — Docker Compose (Recommended)

```bash
# 1. Clone / copy this folder
cd churn_app

# 2. Build & run
docker-compose up --build

# 3. Open browser
open http://localhost:8000
```

### Option 2 — Docker only

```bash
docker build -t churn-predictor .
docker run -p 8000:8000 churn-predictor
```

### Option 3 — Run locally (no Docker)

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

---

## 📡 API Usage

### Predict endpoint

```bash
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
```

#### Response

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.7823,
  "risk_level": "HIGH",
  "message": "This customer is at high risk of churning. Immediate action recommended."
}
```

### Other endpoints

| Endpoint    | Method | Description           |
|-------------|--------|-----------------------|
| `/`         | GET    | Web UI                |
| `/predict`  | POST   | Prediction endpoint   |
| `/health`   | GET    | Health check          |
| `/docs`     | GET    | Swagger UI            |
| `/redoc`    | GET    | ReDoc documentation   |

---

## 🗂️ Project Structure

```
churn_app/
├── main.py                          # FastAPI application
├── model.joblib                     # Trained CatBoost model
├── scaler.joblib                    # StandardScaler
├── label_encoder.joblib             # LabelEncoder (Gender)
├── feature_columns.json             # Feature column order
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
└── static/
    └── index.html                   # Web UI
```

---

## 🌐 Deploy to Cloud

### Render.com (Free tier)

1. Push this folder to a GitHub repo
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repo
4. Set: **Runtime** = Docker, **Port** = 8000
5. Deploy ✅

### Railway.app

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login & deploy
railway login
railway init
railway up
```

### Fly.io

```bash
# Install flyctl
# https://fly.io/docs/hands-on/install-flyctl/

fly launch
fly deploy
```

---

## 🧠 Model Details

- **Algorithm:** CatBoost Classifier
- **Features:** 17 (including 6 engineered features)
- **Class Imbalance:** SMOTE applied on train set only
- **Tuning:** RandomizedSearchCV (5-fold CV, F1 scoring)
- **Threshold:** Mathematically optimized for best F1

### Engineered Features

| Feature | Formula |
|---|---|
| Balance_Salary_Ratio | Balance / (EstimatedSalary + 1) |
| Tenure_Age_Ratio | Tenure / (Age + 1) |
| Is_Senior | Age >= 60 |
| CreditScore_Age_Ratio | CreditScore / (Age + 1) |
| Products_x_Active | NumOfProducts × IsActiveMember |
| IsBalanceZero | Balance == 0 |

---

*Built by Mesut Karagöz — Data Analyst | Data Scientist*
