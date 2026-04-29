from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

class InputData(BaseModel):
    average_daily_temperature: float
    rainfall_mm: float
    soil_moisture: float
    soil_ph: float
    nitrogen_content: float
    phosphorus_content: float
    potassium_content: float
    soil_type: str
    crop_type: str
    fertilizer_amount: float
    irrigation: str
    erosion_risk: str
    sunlight_hours: float
    previous_year_yield: float

@app.get("/")
def root():
    return {"message": "ML service is running"}


model = None

try:
    model = joblib.load("modelRF4.pkl")
    print("MODEL LOADED SUCCESSFULLY")
except Exception as e:
    print("MODEL LOAD ERROR:", e)

@app.post("/predict")
def predict(data: InputData):

    if model is None:
        return {"error": "Model not loaded"}

    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)

    return {"prediction": float(prediction[0])}