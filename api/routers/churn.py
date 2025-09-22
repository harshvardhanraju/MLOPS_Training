from fastapi import APIRouter, HTTPException
import time
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from api.models import ChurnRequest, ChurnResponse, ChurnBatchRequest, BatchResponse
from api.monitoring import record_prediction_metrics
from models.churn.predict import ChurnPredictor

router = APIRouter()
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = ChurnPredictor()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    return predictor

@router.post("/predict", response_model=ChurnResponse)
async def predict_churn(request: ChurnRequest):
    start_time = time.time()
    try:
        pred = get_predictor()
        features = request.dict()
        result = pred.predict(features)
        duration = time.time() - start_time
        record_prediction_metrics("churn", "single", duration)
        return ChurnResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info")
async def get_model_info():
    return {
        "model_name": "churn_predictor",
        "model_type": "binary_classification",
        "classes": ["no_churn", "churn"],
        "description": "LightGBM model for customer churn prediction"
    }