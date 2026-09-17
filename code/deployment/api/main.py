from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Paths inside the Docker container
model = joblib.load("/app/models/model.pkl")
scaler = joblib.load("/app/models/scaler.pkl")

class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

@app.post("/predict")
def predict(features: WineFeatures):
    try:
        input_data = np.array([[getattr(features, f) for f in WineFeatures.model_fields]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))