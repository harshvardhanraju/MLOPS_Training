from fastapi import APIRouter, HTTPException
import time
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from api.models import SentimentRequest, SentimentResponse, SentimentBatchRequest, BatchResponse
from api.monitoring import record_prediction_metrics
from models.sentiment.predict import SentimentPredictor

router = APIRouter()
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = SentimentPredictor()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    return predictor

@router.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    start_time = time.time()
    try:
        pred = get_predictor()
        result = pred.predict(request.text)
        duration = time.time() - start_time
        record_prediction_metrics("sentiment", "single", duration)
        return SentimentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info")
async def get_model_info():
    return {
        "model_name": "sentiment_analyzer",
        "model_type": "text_classification",
        "classes": ["negative", "neutral", "positive"],
        "description": "DistilBERT model for sentiment analysis"
    }