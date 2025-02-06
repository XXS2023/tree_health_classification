from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib  # Для загрузки модели и лейблов

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    model = joblib.load('model.joblib')
    prediction = predict(model, np.array(data.features).reshape(1, -1))
    return {"prediction": prediction.tolist()}