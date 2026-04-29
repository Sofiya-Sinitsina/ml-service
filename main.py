from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI(
    title="Crop Yield ML Service",
    description="ML service for predicting crop yield",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputDataN(BaseModel):
    crop: str
    area_ha: float
    rainfall_7d_sum: float
    avg_temp_7d: float
    humidity: float
    uv_index: float = Field(..., alias="uv-index")
    day_of_year: int
    month: int

@app.get("/")
def root():
    return {"message": "ML service is running"}


model_ndvi = None

try:
    model_ndvi = joblib.load("model.pkl")
    print("MODEL LOADED SUCCESSFULLY")
except Exception as e:
    print("MODEL LOAD ERROR:", e)

@app.post("/predictNVDI")
def predict(data: InputDataN):

    if model_ndvi is None:
        return {"error": "Model not loaded"}

    df = pd.DataFrame([data.dict()])
    df.rename(columns={"uv_index": "uv-index"}, inplace=True)
    prediction = model_ndvi.predict(df)

    return {"prediction": float(prediction[0]), "data": data}


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

    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)

    return {
        "prediction": float(prediction[0]),
        "data": data.model_dump()
    }