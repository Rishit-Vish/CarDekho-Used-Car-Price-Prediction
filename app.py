from pathlib import Path
from fastapi import FastAPI, HTTPException
import joblib 
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from Schema.prediction_response import PredictionResponse
from Schema.schemas import Features
from Config.config import (
    car_transmission_type,
    car_seller_type,
    car_names,
    car_model,
    car_fuel_type,
    car_brand,
    COLUMNS,
    car_catalog,
    df
)

import logging
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins =["*"],
    allow_methods =["*"],
    allow_headers =["*"]
)

model = joblib.load("best_random_forest.pkl") 

print("Loaded model from:", Path("best_random_forest.pkl").resolve())

@app.get('/options')
def get_options():
    return {
    "Car_names" : sorted(car_names),
    "Brands" : sorted(car_brand),
    "Model" : sorted(car_model),
    "Seller_type" : sorted(car_seller_type),
    "fuel_type": sorted(car_fuel_type),
    "Transmission_type": sorted(car_transmission_type),
    "Catalog": car_catalog
    }


@app.post('/predict', response_model = PredictionResponse)
def predict(features: Features):
    try:
        row = pd.DataFrame([features.model_dump()], columns = COLUMNS) 
        prediction = model.predict(row)
        return{"predicted_selling_price": float(prediction[0])}
    
    except Exception as e:
        logger.exception("Car price prediction Failed")

        raise HTTPException(
            status_code = 500,
            detail = "Unable to predict the car price right now",
        ) from e 