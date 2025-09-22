from fastapi import APIRouter, HTTPException
import time
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from api.models import HousePriceRequest, HousePriceResponse, HousePriceBatchRequest, BatchResponse
from api.monitoring import record_prediction_metrics
from models.house_price.predict import HousePricePredictor

router = APIRouter()
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = HousePricePredictor()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    return predictor

@router.post("/predict", response_model=HousePriceResponse)
async def predict_house_price(request: HousePriceRequest):
    start_time = time.time()
    try:
        pred = get_predictor()
        features = [
            request.med_inc, request.house_age, request.ave_rooms, request.ave_bedrms,
            request.population, request.ave_occup, request.latitude, request.longitude
        ]
        result = pred.predict(features)
        duration = time.time() - start_time
        record_prediction_metrics("house_price", "single", duration)
        return HousePriceResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info")
async def get_model_info():
    return {
        "model_name": "house_price_predictor",
        "model_type": "regression",
        "description": "XGBoost regressor for California house price prediction"
    }