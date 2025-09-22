"""
FastAPI application for serving ML models
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import time
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from api.models import *
from api.routers import iris, house_price, sentiment, churn, image_classification
from api.monitoring import setup_monitoring, get_model_metrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MLOps Demo API",
    description="API for serving 5 different ML models with monitoring and observability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup monitoring
setup_monitoring()

# Include routers
app.include_router(iris.router, prefix="/api/v1/iris", tags=["Iris Classification"])
app.include_router(house_price.router, prefix="/api/v1/house-price", tags=["House Price Prediction"])
app.include_router(sentiment.router, prefix="/api/v1/sentiment", tags=["Sentiment Analysis"])
app.include_router(churn.router, prefix="/api/v1/churn", tags=["Customer Churn"])
app.include_router(image_classification.router, prefix="/api/v1/image", tags=["Image Classification"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MLOps Demo API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "metrics": "/metrics",
            "health": "/health",
            "models": {
                "iris": "/api/v1/iris",
                "house_price": "/api/v1/house-price",
                "sentiment": "/api/v1/sentiment",
                "churn": "/api/v1/churn",
                "image": "/api/v1/image"
            }
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models": {
            "iris": "ready",
            "house_price": "ready",
            "sentiment": "ready",
            "churn": "ready",
            "image": "ready"
        }
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/api/v1/models")
async def list_models():
    """List all available models"""
    return {
        "models": [
            {
                "name": "iris",
                "description": "Iris flower classification",
                "type": "multiclass_classification",
                "endpoint": "/api/v1/iris/predict"
            },
            {
                "name": "house_price",
                "description": "California house price prediction",
                "type": "regression",
                "endpoint": "/api/v1/house-price/predict"
            },
            {
                "name": "sentiment",
                "description": "Text sentiment analysis",
                "type": "text_classification",
                "endpoint": "/api/v1/sentiment/predict"
            },
            {
                "name": "churn",
                "description": "Customer churn prediction",
                "type": "binary_classification",
                "endpoint": "/api/v1/churn/predict"
            },
            {
                "name": "image",
                "description": "Image classification",
                "type": "image_classification",
                "endpoint": "/api/v1/image/predict"
            }
        ]
    }

@app.get("/api/v1/model-metrics")
async def model_metrics():
    """Get aggregated model metrics"""
    return get_model_metrics()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )