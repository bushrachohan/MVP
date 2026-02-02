from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import pandas as pd

# Load model & metadata
MODEL_PATH = "decision_model_4class_uk_predictive.pkl"
SCHEMA_PATH = "model_schema_4class_uk_predictive.json"
LABEL_MAP_PATH = "label_map_4class_uk_predictive.json"

model = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH) as f:
    schema = json.load(f)

with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)

expected_features = schema["expected_features"]

app = FastAPI(title="GlobalEDU Pathway Decision API")

class UserInput(BaseModel):
    Level: str
    Duration_Years: float
    Total_Budget: float
    loan_eligibility: int
    family_support: int
    career_switch: int
    uk_psw_available: int
    uk_skill_shortage: int

@app.post("/predict")
def predict_pathway(user: UserInput):
    data = pd.DataFrame([user.dict()])

    # ensure all expected features exist
    for col in expected_features:
        if col not in data.columns:
            data[col] = 0   # safe default

    # reorder columns
    data = data[expected_features]


    prediction = model.predict(data)[0]
    label = label_map[str(prediction)]

    return {
        "prediction": int(prediction),
        "label": label
    }
