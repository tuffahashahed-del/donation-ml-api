
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd, numpy as np

model     = joblib.load("model.pkl")
scaler    = joblib.load("scaler.pkl")
imputer   = joblib.load("imputer.pkl")
target_le = joblib.load("target_le.pkl")
FEATURES  = joblib.load("features.pkl")

app = FastAPI()

class Applicant(BaseModel):
    category: str
    age: int
    monthly_income_jod: float
    family_size: int
    dependents_count: int
    children_in_public_school: int
    is_camp_resident: int
    has_chronic_disease: int
    is_emergency: int
    previous_aid_count: int
    debt_level_jod: float
    housing_score: int

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/predict")
def predict(data: Applicant):
    df = pd.DataFrame([data.dict()])
    cat_dummies = pd.get_dummies(df["category"])
    for col in ["أسرة فقيرة","طالب","مديون","مريض","يتيم"]:
        if col not in cat_dummies:
            cat_dummies[col] = 0
    df = pd.concat([df.drop(columns="category"), cat_dummies], axis=1)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURES]
    df = pd.DataFrame(imputer.transform(df), columns=FEATURES)
    df_scaled = scaler.transform(df)
    label      = target_le.inverse_transform(model.predict(df_scaled))[0]
    confidence = round(float(model.predict_proba(df_scaled).max()), 3)
    return {
        "priority":       label,
        "confidence":     confidence,
        "recommendation": {
            "عالية":  "تدخل فوري مطلوب",
            "متوسطة": "أدرج في قائمة الانتظار",
            "منخفضة": "لا تتوفر معايير الأولوية حالياً",
        }[label]
    }
