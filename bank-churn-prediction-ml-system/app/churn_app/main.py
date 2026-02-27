from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
import joblib, json, numpy as np, pandas as pd
from pathlib import Path

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="Predict whether a bank customer will churn using CatBoost ML model.",
    version="1.0.0"
)

# ── Load Artifacts ────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
model    = joblib.load(BASE / "model.joblib")
scaler   = joblib.load(BASE / "scaler.joblib")
le       = joblib.load(BASE / "label_encoder.joblib")
FEATURES = json.load(open(BASE / "feature_columns.json"))

# ── Request Schema ────────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    credit_score:       int   = Field(..., ge=300, le=850,  example=650)
    geography:          str   = Field(...,                  example="France")
    gender:             str   = Field(...,                  example="Male")
    age:                int   = Field(..., ge=18, le=100,   example=35)
    tenure:             int   = Field(..., ge=0,  le=10,    example=5)
    balance:            float = Field(..., ge=0,            example=75000.0)
    num_of_products:    int   = Field(..., ge=1, le=4,      example=2)
    has_cr_card:        int   = Field(..., ge=0, le=1,      example=1)
    is_active_member:   int   = Field(..., ge=0, le=1,      example=1)
    estimated_salary:   float = Field(..., ge=0,            example=100000.0)

    @validator('geography')
    def validate_geography(cls, v):
        if v not in ['France', 'Germany', 'Spain']:
            raise ValueError("geography must be one of: France, Germany, Spain")
        return v

    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError("gender must be Male or Female")
        return v

class PredictionResponse(BaseModel):
    churn_prediction:   int
    churn_probability:  float
    risk_level:         str
    message:            str

# ── Helper ────────────────────────────────────────────────────────────────────
def build_features(inp: CustomerInput) -> pd.DataFrame:
    row = {
        'CreditScore':      inp.credit_score,
        'Gender':           le.transform([inp.gender])[0],
        'Age':              inp.age,
        'Tenure':           inp.tenure,
        'Balance':          inp.balance,
        'NumOfProducts':    inp.num_of_products,
        'HasCrCard':        inp.has_cr_card,
        'IsActiveMember':   inp.is_active_member,
        'EstimatedSalary':  inp.estimated_salary,
        # Feature engineering
        'Balance_Salary_Ratio':  inp.balance / (inp.estimated_salary + 1),
        'Tenure_Age_Ratio':      inp.tenure / (inp.age + 1),
        'Is_Senior':             int(inp.age >= 60),
        'CreditScore_Age_Ratio': inp.credit_score / (inp.age + 1),
        'Products_x_Active':     inp.num_of_products * inp.is_active_member,
        'IsBalanceZero':         int(inp.balance == 0),
        'Geography_Germany':     int(inp.geography == 'Germany'),
        'Geography_Spain':       int(inp.geography == 'Spain'),
    }
    df = pd.DataFrame([row])[FEATURES]
    df_scaled = pd.DataFrame(scaler.transform(df), columns=FEATURES)
    return df_scaled

def risk_label(prob: float) -> tuple[str, str]:
    if prob < 0.30:
        return "LOW",    "This customer is unlikely to churn. Focus on maintaining satisfaction."
    elif prob < 0.60:
        return "MEDIUM", "This customer shows moderate churn risk. Consider a retention offer."
    else:
        return "HIGH",   "This customer is at high risk of churning. Immediate action recommended."

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return FileResponse(BASE / "static" / "index.html")

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerInput):
    try:
        features  = build_features(customer)
        pred      = int(model.predict(features)[0])
        prob      = float(model.predict_proba(features)[0][1])
        risk, msg = risk_label(prob)
        return PredictionResponse(
            churn_prediction=pred,
            churn_probability=round(prob, 4),
            risk_level=risk,
            message=msg
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model": "CatBoost", "version": "1.0.0"}

# ── Static files ──────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
