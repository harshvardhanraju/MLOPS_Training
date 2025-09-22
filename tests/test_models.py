"""
Test suite for individual ML models
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def check_model_artifacts(model_name: str) -> bool:
    """Check if model artifacts exist"""
    artifacts_dir = f"models/{model_name}/artifacts"
    return os.path.exists(artifacts_dir) and len(os.listdir(artifacts_dir)) > 0

class TestIrisModel:
    """Test Iris classification model"""

    @pytest.fixture
    def iris_predictor(self):
        """Fixture for iris predictor"""
        if not check_model_artifacts("iris"):
            pytest.skip("Iris model artifacts not found")

        from models.iris.predict import IrisPredictor
        return IrisPredictor()

    def test_iris_prediction_shape(self, iris_predictor):
        """Test iris prediction output shape"""
        features = [5.1, 3.5, 1.4, 0.2]
        result = iris_predictor.predict(features)

        assert isinstance(result, dict)
        assert "prediction" in result
        assert "probabilities" in result
        assert "confidence" in result
        assert len(result["probabilities"]) == 3

    def test_iris_prediction_values(self, iris_predictor):
        """Test iris prediction values are valid"""
        features = [5.1, 3.5, 1.4, 0.2]
        result = iris_predictor.predict(features)

        assert result["prediction"] in ["setosa", "versicolor", "virginica"]
        assert 0 <= result["confidence"] <= 1
        assert all(0 <= prob <= 1 for prob in result["probabilities"].values())
        assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01

    def test_iris_batch_prediction(self, iris_predictor):
        """Test iris batch prediction"""
        features_list = [
            [5.1, 3.5, 1.4, 0.2],
            [6.2, 2.8, 4.8, 1.8],
            [7.7, 2.6, 6.9, 2.3]
        ]
        results = iris_predictor.predict_batch(features_list)

        assert len(results) == 3
        assert all(isinstance(result, dict) for result in results)

class TestHousePriceModel:
    """Test house price prediction model"""

    @pytest.fixture
    def house_price_predictor(self):
        """Fixture for house price predictor"""
        if not check_model_artifacts("house_price"):
            pytest.skip("House price model artifacts not found")

        from models.house_price.predict import HousePricePredictor
        return HousePricePredictor()

    def test_house_price_prediction_shape(self, house_price_predictor):
        """Test house price prediction output shape"""
        features = [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]
        result = house_price_predictor.predict(features)

        assert isinstance(result, dict)
        assert "predicted_price" in result
        assert "confidence_interval" in result

    def test_house_price_prediction_values(self, house_price_predictor):
        """Test house price prediction values are reasonable"""
        features = [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]
        result = house_price_predictor.predict(features)

        # Price should be positive and reasonable (for California housing)
        assert result["predicted_price"] > 0
        assert result["predicted_price"] < 10000000  # Less than $10M
        assert result["confidence_interval"]["lower"] < result["confidence_interval"]["upper"]

class TestSentimentModel:
    """Test sentiment analysis model"""

    @pytest.fixture
    def sentiment_predictor(self):
        """Fixture for sentiment predictor"""
        if not check_model_artifacts("sentiment"):
            pytest.skip("Sentiment model artifacts not found")

        from models.sentiment.predict import SentimentPredictor
        return SentimentPredictor()

    def test_sentiment_prediction_shape(self, sentiment_predictor):
        """Test sentiment prediction output shape"""
        result = sentiment_predictor.predict("I love this product!")

        assert isinstance(result, dict)
        assert "sentiment" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_sentiment_prediction_values(self, sentiment_predictor):
        """Test sentiment prediction values are valid"""
        result = sentiment_predictor.predict("I love this product!")

        assert result["sentiment"] in ["negative", "neutral", "positive"]
        assert 0 <= result["confidence"] <= 1
        assert "negative" in result["probabilities"]
        assert "neutral" in result["probabilities"]
        assert "positive" in result["probabilities"]

    def test_sentiment_different_texts(self, sentiment_predictor):
        """Test sentiment prediction on different text types"""
        texts = [
            "I love this!",
            "This is terrible.",
            "It's okay.",
            "Best product ever!",
            "Worst experience."
        ]

        for text in texts:
            result = sentiment_predictor.predict(text)
            assert result["sentiment"] in ["negative", "neutral", "positive"]
            assert 0 <= result["confidence"] <= 1

class TestChurnModel:
    """Test customer churn prediction model"""

    @pytest.fixture
    def churn_predictor(self):
        """Fixture for churn predictor"""
        if not check_model_artifacts("churn"):
            pytest.skip("Churn model artifacts not found")

        from models.churn.predict import ChurnPredictor
        return ChurnPredictor()

    def test_churn_prediction_shape(self, churn_predictor):
        """Test churn prediction output shape"""
        features = {
            'age': 45,
            'tenure_months': 36,
            'monthly_charges': 75.5,
            'total_charges': 2500.0,
            'contract_type': 'Two year',
            'payment_method': 'Credit card',
            'internet_service': 'Fiber optic',
            'online_security': 'Yes',
            'tech_support': 'Yes',
            'streaming_tv': 'Yes',
            'paperless_billing': 'Yes',
            'senior_citizen': 0,
            'partner': 'Yes',
            'dependents': 'Yes'
        }
        result = churn_predictor.predict(features)

        assert isinstance(result, dict)
        assert "churn_prediction" in result
        assert "churn_probability" in result
        assert "risk_level" in result
        assert "recommendations" in result

    def test_churn_prediction_values(self, churn_predictor):
        """Test churn prediction values are valid"""
        features = {
            'age': 45,
            'tenure_months': 36,
            'monthly_charges': 75.5,
            'total_charges': 2500.0,
            'contract_type': 'Two year',
            'payment_method': 'Credit card',
            'internet_service': 'Fiber optic',
            'online_security': 'Yes',
            'tech_support': 'Yes',
            'streaming_tv': 'Yes',
            'paperless_billing': 'Yes',
            'senior_citizen': 0,
            'partner': 'Yes',
            'dependents': 'Yes'
        }
        result = churn_predictor.predict(features)

        assert result["churn_prediction"] in [0, 1]
        assert 0 <= result["churn_probability"] <= 1
        assert result["risk_level"] in ["Low", "Medium", "High"]
        assert isinstance(result["recommendations"], list)

class TestImageModel:
    """Test image classification model"""

    @pytest.fixture
    def image_classifier(self):
        """Fixture for image classifier"""
        if not check_model_artifacts("image_classification"):
            pytest.skip("Image model artifacts not found")

        from models.image_classification.predict import ImageClassifier
        return ImageClassifier()

    def test_image_prediction_shape(self, image_classifier):
        """Test image prediction output shape"""
        test_image = np.random.rand(32, 32, 3)
        result = image_classifier.predict(test_image)

        assert isinstance(result, dict)
        assert "prediction" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_image_prediction_values(self, image_classifier):
        """Test image prediction values are valid"""
        test_image = np.random.rand(32, 32, 3)
        result = image_classifier.predict(test_image)

        assert result["prediction"] in ["red_dominant", "green_dominant", "blue_dominant"]
        assert 0 <= result["confidence"] <= 1
        assert len(result["probabilities"]) == 3
        assert all(0 <= prob <= 1 for prob in result["probabilities"].values())

    def test_image_sample_generation(self, image_classifier):
        """Test sample image generation"""
        for class_type in ["red_dominant", "green_dominant", "blue_dominant"]:
            sample_image = image_classifier.create_sample_image(class_type)
            assert sample_image.shape == (32, 32, 3)
            assert 0 <= sample_image.min() <= sample_image.max() <= 1

class TestDataProcessing:
    """Test data processing functions"""

    def test_dataset_creation(self):
        """Test dataset creation script"""
        # This would test if datasets can be created without errors
        try:
            from data.create_datasets import create_iris_dataset
            df = create_iris_dataset()
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'target' in df.columns
        except ImportError:
            pytest.skip("Dataset creation modules not available")

if __name__ == "__main__":
    pytest.main([__file__])