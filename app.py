from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Heart Disease Prediction API", version="1.0")

try:
    preprocessor = joblib.load("models/preprocessor.pkl")
    xgb_model = joblib.load("models/xgboost_model.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load models. Ensure .pkl files exist. Error: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class PatientData(BaseModel):
    age: float
    sex: str
    cp: str
    trestbps: float
    chol: float
    fbs: str
    restecg: str
    thalch: float
    exang: str
    oldpeak: float
    ca: float

    class Config:
        json_schema_extra = {
            "example": {
                "age": 63,
                "sex": "male",
                "cp": "asymptomatic",
                "trestbps": 145,
                "chol": 233,
                "fbs": "true",
                "restecg": "lv hypertrophy",
                "thalch": 150,
                "exang": "false",
                "oldpeak": 2.3,
                "ca": 0
            }
        }

@app.get("/predict")
def read_root():
    return {"message": "Hello World"}

@app.post("/predict")
def predict_heart_disease(patient: PatientData):
    try:
        data = patient.model_dump()

        data['sex'] = 1 if data['sex'].strip().lower() == 'male' else 0
        data['fbs'] = 1 if data['fbs'].strip().lower() == 'true' else 0
        data['exang'] = 1 if data['exang'].strip().lower() == 'true' else 0

        df = pd.DataFrame([data])

        X_processed = preprocessor.transform(df)

        prediction = xgb_model.predict(X_processed)[0]
        probability = xgb_model.predict_proba(X_processed)[0][1]

        return {
            "prediction": int(prediction),
            "disease_detected": bool(prediction == 1),
            "probability": round(float(probability), 4),
            "risk_level": "High" if probability > 0.5 else "Low"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))