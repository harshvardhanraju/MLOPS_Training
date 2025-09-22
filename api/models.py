"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import numpy as np

# Iris Classification Models
class IrisRequest(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, le=10, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, le=10, description="Petal width in cm")

class IrisResponse(BaseModel):
    prediction: str
    prediction_id: int
    probabilities: Dict[str, float]
    confidence: float

# House Price Prediction Models
class HousePriceRequest(BaseModel):
    med_inc: float = Field(..., description="Median income")
    house_age: float = Field(..., ge=0, description="House age")
    ave_rooms: float = Field(..., gt=0, description="Average rooms")
    ave_bedrms: float = Field(..., gt=0, description="Average bedrooms")
    population: float = Field(..., gt=0, description="Population")
    ave_occup: float = Field(..., gt=0, description="Average occupancy")
    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")

class HousePriceResponse(BaseModel):
    predicted_price: float
    predicted_price_formatted: str
    model_output: float
    confidence_interval: Dict[str, float]

# Sentiment Analysis Models
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Text to analyze")

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]

# Customer Churn Models
class ChurnRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Customer age")
    tenure_months: int = Field(..., ge=0, le=100, description="Tenure in months")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges")
    total_charges: float = Field(..., ge=0, description="Total charges")
    contract_type: str = Field(..., description="Contract type")
    payment_method: str = Field(..., description="Payment method")
    internet_service: str = Field(..., description="Internet service type")
    online_security: str = Field(..., description="Online security")
    tech_support: str = Field(..., description="Tech support")
    streaming_tv: str = Field(..., description="Streaming TV")
    paperless_billing: str = Field(..., description="Paperless billing")
    senior_citizen: int = Field(..., ge=0, le=1, description="Senior citizen flag")
    partner: str = Field(..., description="Has partner")
    dependents: str = Field(..., description="Has dependents")

class ChurnResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
    confidence: float
    recommendations: List[str]

# Image Classification Models
class ImageRequest(BaseModel):
    image_data: List[List[List[float]]] = Field(..., description="32x32x3 image array")

    @validator('image_data')
    def validate_image_shape(cls, v):
        if len(v) != 32:
            raise ValueError("Image must be 32x32 pixels")
        for row in v:
            if len(row) != 32:
                raise ValueError("Image must be 32x32 pixels")
            for pixel in row:
                if len(pixel) != 3:
                    raise ValueError("Image must have 3 channels (RGB)")
                for channel in pixel:
                    if not 0 <= channel <= 1:
                        raise ValueError("Pixel values must be between 0 and 1")
        return v

class ImageResponse(BaseModel):
    prediction: str
    prediction_id: int
    confidence: float
    probabilities: Dict[str, float]
    input_shape: List[int]

# Batch prediction models
class IrisBatchRequest(BaseModel):
    samples: List[IrisRequest]

class HousePriceBatchRequest(BaseModel):
    samples: List[HousePriceRequest]

class SentimentBatchRequest(BaseModel):
    samples: List[SentimentRequest]

class ChurnBatchRequest(BaseModel):
    samples: List[ChurnRequest]

class ImageBatchRequest(BaseModel):
    samples: List[ImageRequest]

# Generic response models
class BatchResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    total_predictions: int
    processing_time_ms: float

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: float