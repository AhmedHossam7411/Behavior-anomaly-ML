from fastapi import APIRouter, Request
from pydantic import BaseModel
import numpy as np

router = APIRouter()

class PredictionRequest(BaseModel):
    data: list

@router.post("/predict")
def predict(request: PredictionRequest, req: Request):
    model = req.app.state.model   
    X = np.array(request.data)
    prediction = model.predict(X)
    return {"prediction": prediction.tolist()}