"""
Test suite for the MLOps API
"""

import pytest
import requests
from fastapi.testclient import TestClient
import sys
from pathlib import Path
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from api.main import app

client = TestClient(app)

class TestAPIEndpoints:
    """Test API endpoints functionality"""

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "MLOps Demo API"

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models" in data

    def test_models_list_endpoint(self):
        """Test models list endpoint"""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 5

    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

class TestIrisModel:
    """Test Iris classification model"""

    def test_iris_predict_valid_input(self):
        """Test iris prediction with valid input"""
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = client.post("/api/v1/iris/predict", json=payload)

        # Skip if model not available (CI environment)
        if response.status_code == 503:
            pytest.skip("Model not available in test environment")

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert data["prediction"] in ["setosa", "versicolor", "virginica"]

    def test_iris_predict_invalid_input(self):
        """Test iris prediction with invalid input"""
        payload = {
            "sepal_length": -1,  # Invalid negative value
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = client.post("/api/v1/iris/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_iris_info_endpoint(self):
        """Test iris model info endpoint"""
        response = client.get("/api/v1/iris/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "iris_classifier"
        assert data["model_type"] == "multiclass_classification"

class TestHousePriceModel:
    """Test house price prediction model"""

    def test_house_price_predict_valid_input(self):
        """Test house price prediction with valid input"""
        payload = {
            "med_inc": 8.3252,
            "house_age": 41.0,
            "ave_rooms": 6.984127,
            "ave_bedrms": 1.023810,
            "population": 322.0,
            "ave_occup": 2.555556,
            "latitude": 37.88,
            "longitude": -122.23
        }
        response = client.post("/api/v1/house-price/predict", json=payload)

        # Skip if model not available (CI environment)
        if response.status_code == 503:
            pytest.skip("Model not available in test environment")

        assert response.status_code == 200
        data = response.json()
        assert "predicted_price" in data
        assert "predicted_price_formatted" in data
        assert "confidence_interval" in data

    def test_house_price_info_endpoint(self):
        """Test house price model info endpoint"""
        response = client.get("/api/v1/house-price/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "house_price_predictor"
        assert data["model_type"] == "regression"

class TestSentimentModel:
    """Test sentiment analysis model"""

    def test_sentiment_predict_valid_input(self):
        """Test sentiment prediction with valid input"""
        payload = {
            "text": "I love this product!"
        }
        response = client.post("/api/v1/sentiment/predict", json=payload)

        # Skip if model not available (CI environment)
        if response.status_code == 503:
            pytest.skip("Model not available in test environment")

        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert data["sentiment"] in ["negative", "neutral", "positive"]

    def test_sentiment_predict_empty_text(self):
        """Test sentiment prediction with empty text"""
        payload = {
            "text": ""
        }
        response = client.post("/api/v1/sentiment/predict", json=payload)
        assert response.status_code == 422  # Validation error

class TestChurnModel:
    """Test customer churn prediction model"""

    def test_churn_predict_valid_input(self):
        """Test churn prediction with valid input"""
        payload = {
            "age": 45,
            "tenure_months": 36,
            "monthly_charges": 75.5,
            "total_charges": 2500.0,
            "contract_type": "Two year",
            "payment_method": "Credit card",
            "internet_service": "Fiber optic",
            "online_security": "Yes",
            "tech_support": "Yes",
            "streaming_tv": "Yes",
            "paperless_billing": "Yes",
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "Yes"
        }
        response = client.post("/api/v1/churn/predict", json=payload)

        # Skip if model not available (CI environment)
        if response.status_code == 503:
            pytest.skip("Model not available in test environment")

        assert response.status_code == 200
        data = response.json()
        assert "churn_prediction" in data
        assert "churn_probability" in data
        assert "risk_level" in data
        assert "recommendations" in data

class TestImageModel:
    """Test image classification model"""

    def test_image_predict_valid_input(self):
        """Test image prediction with valid input"""
        # Create a 32x32x3 test image
        test_image = np.random.rand(32, 32, 3).tolist()
        payload = {
            "image_data": test_image
        }
        response = client.post("/api/v1/image/predict", json=payload)

        # Skip if model not available (CI environment)
        if response.status_code == 503:
            pytest.skip("Model not available in test environment")

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data

    def test_image_predict_invalid_shape(self):
        """Test image prediction with invalid image shape"""
        # Wrong shape image
        test_image = np.random.rand(28, 28, 3).tolist()
        payload = {
            "image_data": test_image
        }
        response = client.post("/api/v1/image/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_image_sample_endpoint(self):
        """Test image sample generation endpoint"""
        response = client.get("/api/v1/image/sample")

        # Skip if model not available (CI environment)
        if response.status_code == 503:
            pytest.skip("Model not available in test environment")

        assert response.status_code == 200
        data = response.json()
        assert "image_data" in data
        assert "description" in data
        assert "shape" in data

if __name__ == "__main__":
    pytest.main([__file__])