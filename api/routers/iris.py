"""
Iris classification router
"""

from fastapi import APIRouter, HTTPException
import time
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from api.models import IrisRequest, IrisResponse, IrisBatchRequest, BatchResponse
from api.monitoring import record_prediction_metrics
from models.iris.predict import IrisPredictor

router = APIRouter()

# Initialize predictor (will be lazy-loaded)
predictor = None

def get_predictor():
    """Lazy load the predictor"""
    global predictor
    if predictor is None:
        try:
            predictor = IrisPredictor()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    return predictor

@router.post("/predict", response_model=IrisResponse)
async def predict_iris(request: IrisRequest):
    """Predict iris flower species"""
    start_time = time.time()

    try:
        pred = get_predictor()
        features = [
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width
        ]

        result = pred.predict(features)

        # Record metrics
        duration = time.time() - start_time
        record_prediction_metrics("iris", "single", duration)

        return IrisResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch", response_model=BatchResponse)
async def predict_iris_batch(request: IrisBatchRequest):
    """Predict iris flower species for multiple samples"""
    start_time = time.time()

    try:
        pred = get_predictor()
        features_list = [
            [sample.sepal_length, sample.sepal_width, sample.petal_length, sample.petal_width]
            for sample in request.samples
        ]

        results = pred.predict_batch(features_list)

        # Record metrics
        duration = time.time() - start_time
        record_prediction_metrics("iris", "batch", duration)

        return BatchResponse(
            predictions=results,
            total_predictions=len(results),
            processing_time_ms=duration * 1000
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info")
async def get_model_info():
    """Get iris model information"""
    return {
        "model_name": "iris_classifier",
        "model_type": "multiclass_classification",
        "classes": ["setosa", "versicolor", "virginica"],
        "features": [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ],
        "description": "Random Forest classifier for iris flower species prediction"
    }