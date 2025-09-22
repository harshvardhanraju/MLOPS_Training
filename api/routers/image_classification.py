from fastapi import APIRouter, HTTPException
import time
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from api.models import ImageRequest, ImageResponse, ImageBatchRequest, BatchResponse
from api.monitoring import record_prediction_metrics
from models.image_classification.predict import ImageClassifier

router = APIRouter()
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = ImageClassifier()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    return predictor

@router.post("/predict", response_model=ImageResponse)
async def predict_image(request: ImageRequest):
    start_time = time.time()
    try:
        pred = get_predictor()
        image_array = np.array(request.image_data)
        result = pred.predict(image_array)
        duration = time.time() - start_time
        record_prediction_metrics("image", "single", duration)
        return ImageResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info")
async def get_model_info():
    return {
        "model_name": "image_classifier",
        "model_type": "image_classification",
        "classes": ["red_dominant", "green_dominant", "blue_dominant"],
        "input_shape": [32, 32, 3],
        "description": "CNN model for color-based image classification"
    }

@router.get("/sample")
async def get_sample_image():
    """Generate a sample image for testing"""
    try:
        pred = get_predictor()
        sample_image = pred.create_sample_image('red_dominant')
        return {
            "image_data": sample_image.tolist(),
            "description": "Sample red-dominant image for testing",
            "shape": list(sample_image.shape)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))