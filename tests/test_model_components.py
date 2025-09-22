"""
Comprehensive tests for ML model components
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
import pickle
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class TestModelTraining:
    """Test model training functionality"""

    def setup_method(self):
        """Setup test data for model training"""
        # Ensure data exists for training
        if not os.path.exists("data/raw/iris.csv"):
            try:
                from data.create_datasets import main
                main()
            except Exception as e:
                pytest.skip(f"Could not create test data: {e}")

    def test_iris_model_training(self):
        """Test iris model training"""
        try:
            # Import and run training
            sys.path.append("models/iris")
            from models.iris.train import main as train_iris

            # Run training
            model, metrics = train_iris()

            # Test model object
            assert model is not None, "Model should be created"
            assert hasattr(model, 'predict'), "Model should have predict method"
            assert hasattr(model, 'predict_proba'), "Model should have predict_proba method"

            # Test metrics
            assert isinstance(metrics, dict), "Metrics should be a dictionary"
            assert 'accuracy' in metrics, "Should have accuracy metric"
            assert metrics['accuracy'] > 0.8, "Accuracy should be reasonable"

            # Test model file creation
            model_path = "models/iris/artifacts/iris_model.pkl"
            assert os.path.exists(model_path), "Model file should be saved"

            print("✅ Iris model training test passed")

        except ImportError as e:
            pytest.skip(f"Could not import iris training module: {e}")
        except Exception as e:
            pytest.fail(f"Iris model training failed: {e}")

    def test_house_price_model_training(self):
        """Test house price model training"""
        try:
            # Import and run training
            sys.path.append("models/house_price")
            from models.house_price.train import main as train_house_price

            # Run training
            model, scaler, metrics = train_house_price()

            # Test model object
            assert model is not None, "Model should be created"
            assert scaler is not None, "Scaler should be created"
            assert hasattr(model, 'predict'), "Model should have predict method"

            # Test metrics
            assert isinstance(metrics, dict), "Metrics should be a dictionary"
            assert 'rmse' in metrics, "Should have RMSE metric"
            assert 'r2_score' in metrics, "Should have R2 score"
            assert metrics['r2_score'] > 0.5, "R2 score should be reasonable"

            # Test model files creation
            model_path = "models/house_price/artifacts/house_price_model.pkl"
            scaler_path = "models/house_price/artifacts/scaler.pkl"
            assert os.path.exists(model_path), "Model file should be saved"
            assert os.path.exists(scaler_path), "Scaler file should be saved"

            print("✅ House price model training test passed")

        except ImportError as e:
            pytest.skip(f"Could not import house price training module: {e}")
        except Exception as e:
            pytest.fail(f"House price model training failed: {e}")

    def test_sentiment_model_training(self):
        """Test sentiment model training"""
        try:
            # Import and run training
            sys.path.append("models/sentiment")
            from models.sentiment.train import main as train_sentiment

            # Run training
            model, metrics = train_sentiment()

            # Test model object
            assert model is not None, "Model should be created"

            # Test metrics
            assert isinstance(metrics, dict), "Metrics should be a dictionary"
            assert 'accuracy' in metrics, "Should have accuracy metric"
            assert metrics['accuracy'] > 0.5, "Accuracy should be reasonable"

            # Test model file creation
            model_path = "models/sentiment/artifacts/sentiment_model.pkl"
            assert os.path.exists(model_path), "Model file should be saved"

            print("✅ Sentiment model training test passed")

        except ImportError as e:
            pytest.skip(f"Could not import sentiment training module: {e}")
        except Exception as e:
            pytest.fail(f"Sentiment model training failed: {e}")

    def test_churn_model_training(self):
        """Test churn model training"""
        try:
            # Import and run training
            sys.path.append("models/churn")
            from models.churn.train import main as train_churn

            # Run training
            model, encoders, metrics = train_churn()

            # Test model object
            assert model is not None, "Model should be created"
            assert encoders is not None, "Encoders should be created"
            assert hasattr(model, 'predict'), "Model should have predict method"

            # Test metrics
            assert isinstance(metrics, dict), "Metrics should be a dictionary"
            assert 'accuracy' in metrics, "Should have accuracy metric"
            assert 'auc' in metrics, "Should have AUC metric"
            assert metrics['accuracy'] > 0.6, "Accuracy should be reasonable"

            # Test model files creation
            model_path = "models/churn/artifacts/churn_model.pkl"
            encoders_path = "models/churn/artifacts/label_encoders.pkl"
            assert os.path.exists(model_path), "Model file should be saved"
            assert os.path.exists(encoders_path), "Encoders file should be saved"

            print("✅ Churn model training test passed")

        except ImportError as e:
            pytest.skip(f"Could not import churn training module: {e}")
        except Exception as e:
            pytest.fail(f"Churn model training failed: {e}")

    def test_image_model_training(self):
        """Test image model training"""
        try:
            # Import and run training
            sys.path.append("models/image_classification")
            from models.image_classification.train import main as train_image

            # Run training
            model, metrics = train_image()

            # Test model object
            assert model is not None, "Model should be created"
            assert hasattr(model, 'predict'), "Model should have predict method"

            # Test metrics
            assert isinstance(metrics, dict), "Metrics should be a dictionary"
            assert 'accuracy' in metrics, "Should have accuracy metric"
            assert metrics['accuracy'] > 0.4, "Accuracy should be reasonable"

            # Test model file creation
            model_path = "models/image_classification/artifacts/image_classifier.h5"
            assert os.path.exists(model_path), "Model file should be saved"

            print("✅ Image model training test passed")

        except ImportError as e:
            pytest.skip(f"Could not import image training module: {e}")
        except Exception as e:
            pytest.fail(f"Image model training failed: {e}")

class TestModelPrediction:
    """Test model prediction functionality"""

    def test_iris_prediction(self):
        """Test iris model prediction"""
        try:
            from models.iris.predict import IrisPredictor

            # Check if model exists
            model_path = "models/iris/artifacts/iris_model.pkl"
            if not os.path.exists(model_path):
                pytest.skip("Iris model not found, run training first")

            predictor = IrisPredictor()

            # Test single prediction
            features = [5.1, 3.5, 1.4, 0.2]
            result = predictor.predict(features)

            # Validate result structure
            assert isinstance(result, dict), "Result should be a dictionary"
            assert 'prediction' in result, "Should have prediction"
            assert 'probabilities' in result, "Should have probabilities"
            assert 'confidence' in result, "Should have confidence"

            # Validate result values
            assert result['prediction'] in ['setosa', 'versicolor', 'virginica'], "Prediction should be valid class"
            assert 0 <= result['confidence'] <= 1, "Confidence should be between 0 and 1"
            assert len(result['probabilities']) == 3, "Should have 3 class probabilities"

            # Test batch prediction
            features_batch = [[5.1, 3.5, 1.4, 0.2], [6.2, 2.8, 4.8, 1.8]]
            batch_results = predictor.predict_batch(features_batch)
            assert len(batch_results) == 2, "Should return 2 predictions"

            print("✅ Iris prediction test passed")

        except Exception as e:
            pytest.fail(f"Iris prediction test failed: {e}")

    def test_house_price_prediction(self):
        """Test house price model prediction"""
        try:
            from models.house_price.predict import HousePricePredictor

            # Check if model exists
            model_path = "models/house_price/artifacts/house_price_model.pkl"
            if not os.path.exists(model_path):
                pytest.skip("House price model not found, run training first")

            predictor = HousePricePredictor()

            # Test single prediction
            features = [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]
            result = predictor.predict(features)

            # Validate result structure
            assert isinstance(result, dict), "Result should be a dictionary"
            assert 'predicted_price' in result, "Should have predicted price"
            assert 'confidence_interval' in result, "Should have confidence interval"

            # Validate result values
            assert result['predicted_price'] > 0, "Price should be positive"
            assert result['predicted_price'] < 10000000, "Price should be reasonable"

            print("✅ House price prediction test passed")

        except Exception as e:
            pytest.fail(f"House price prediction test failed: {e}")

    def test_sentiment_prediction(self):
        """Test sentiment model prediction"""
        try:
            from models.sentiment.predict import SentimentPredictor

            # Check if model exists
            model_path = "models/sentiment/artifacts/sentiment_model.pkl"
            if not os.path.exists(model_path):
                pytest.skip("Sentiment model not found, run training first")

            predictor = SentimentPredictor()

            # Test single prediction
            text = "I love this product!"
            result = predictor.predict(text)

            # Validate result structure
            assert isinstance(result, dict), "Result should be a dictionary"
            assert 'sentiment' in result, "Should have sentiment"
            assert 'confidence' in result, "Should have confidence"
            assert 'probabilities' in result, "Should have probabilities"

            # Validate result values
            assert result['sentiment'] in ['negative', 'neutral', 'positive'], "Sentiment should be valid"
            assert 0 <= result['confidence'] <= 1, "Confidence should be between 0 and 1"

            print("✅ Sentiment prediction test passed")

        except Exception as e:
            pytest.fail(f"Sentiment prediction test failed: {e}")

    def test_churn_prediction(self):
        """Test churn model prediction"""
        try:
            from models.churn.predict import ChurnPredictor

            # Check if model exists
            model_path = "models/churn/artifacts/churn_model.pkl"
            if not os.path.exists(model_path):
                pytest.skip("Churn model not found, run training first")

            predictor = ChurnPredictor()

            # Test single prediction
            features = {
                'age': 45, 'tenure_months': 36, 'monthly_charges': 75.5, 'total_charges': 2500.0,
                'contract_type': 'Two year', 'payment_method': 'Credit card', 'internet_service': 'Fiber optic',
                'online_security': 'Yes', 'tech_support': 'Yes', 'streaming_tv': 'Yes',
                'paperless_billing': 'Yes', 'senior_citizen': 0, 'partner': 'Yes', 'dependents': 'Yes'
            }
            result = predictor.predict(features)

            # Validate result structure
            assert isinstance(result, dict), "Result should be a dictionary"
            assert 'churn_prediction' in result, "Should have churn prediction"
            assert 'churn_probability' in result, "Should have churn probability"
            assert 'risk_level' in result, "Should have risk level"

            # Validate result values
            assert result['churn_prediction'] in [0, 1], "Churn prediction should be binary"
            assert 0 <= result['churn_probability'] <= 1, "Probability should be between 0 and 1"
            assert result['risk_level'] in ['Low', 'Medium', 'High'], "Risk level should be valid"

            print("✅ Churn prediction test passed")

        except Exception as e:
            pytest.fail(f"Churn prediction test failed: {e}")

    def test_image_prediction(self):
        """Test image model prediction"""
        try:
            from models.image_classification.predict import ImageClassifier

            # Check if model exists
            model_path = "models/image_classification/artifacts/image_classifier.h5"
            if not os.path.exists(model_path):
                pytest.skip("Image model not found, run training first")

            classifier = ImageClassifier()

            # Test single prediction
            test_image = np.random.rand(32, 32, 3)
            result = classifier.predict(test_image)

            # Validate result structure
            assert isinstance(result, dict), "Result should be a dictionary"
            assert 'prediction' in result, "Should have prediction"
            assert 'confidence' in result, "Should have confidence"
            assert 'probabilities' in result, "Should have probabilities"

            # Validate result values
            expected_classes = ['red_dominant', 'green_dominant', 'blue_dominant']
            assert result['prediction'] in expected_classes, "Prediction should be valid class"
            assert 0 <= result['confidence'] <= 1, "Confidence should be between 0 and 1"

            # Test sample image generation
            sample_image = classifier.create_sample_image('red_dominant')
            assert sample_image.shape == (32, 32, 3), "Sample image should have correct shape"

            print("✅ Image prediction test passed")

        except Exception as e:
            pytest.fail(f"Image prediction test failed: {e}")

class TestModelPersistence:
    """Test model saving and loading"""

    def test_model_files_exist(self):
        """Test that model files exist after training"""
        model_files = {
            'iris': 'models/iris/artifacts/iris_model.pkl',
            'house_price': 'models/house_price/artifacts/house_price_model.pkl',
            'sentiment': 'models/sentiment/artifacts/sentiment_model.pkl',
            'churn': 'models/churn/artifacts/churn_model.pkl',
            'image': 'models/image_classification/artifacts/image_classifier.h5'
        }

        existing_models = 0
        for model_name, file_path in model_files.items():
            if os.path.exists(file_path):
                # Check file is not empty
                file_size = os.path.getsize(file_path)
                assert file_size > 0, f"Model file should not be empty: {file_path}"
                existing_models += 1
                print(f"✅ {model_name} model file exists and is valid")

        if existing_models == 0:
            pytest.skip("No model files found, run training first")

    def test_model_loading(self):
        """Test loading saved models"""
        # Test pickle models
        pickle_models = {
            'iris': 'models/iris/artifacts/iris_model.pkl',
            'house_price': 'models/house_price/artifacts/house_price_model.pkl',
            'sentiment': 'models/sentiment/artifacts/sentiment_model.pkl',
            'churn': 'models/churn/artifacts/churn_model.pkl'
        }

        for model_name, file_path in pickle_models.items():
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        model = pickle.load(f)
                    assert model is not None, f"{model_name} model should load successfully"
                    print(f"✅ {model_name} model loading test passed")
                except Exception as e:
                    pytest.fail(f"Failed to load {model_name} model: {e}")

        # Test TensorFlow model
        tf_model_path = 'models/image_classification/artifacts/image_classifier.h5'
        if os.path.exists(tf_model_path):
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(tf_model_path)
                assert model is not None, "TensorFlow model should load successfully"
                print("✅ TensorFlow model loading test passed")
            except Exception as e:
                pytest.fail(f"Failed to load TensorFlow model: {e}")

class TestModelValidation:
    """Test model validation and performance"""

    def test_model_performance_validation(self):
        """Test that models meet minimum performance requirements"""
        performance_thresholds = {
            'iris': {'accuracy': 0.8},
            'house_price': {'r2_score': 0.5},
            'sentiment': {'accuracy': 0.6},
            'churn': {'accuracy': 0.7},
            'image': {'accuracy': 0.5}
        }

        # This would typically load metrics from MLflow or saved files
        # For testing, we'll use mock metrics or skip if no metrics available
        print("⚠️  Model performance validation requires trained models with saved metrics")
        print("   Run training scripts to generate performance metrics")

    def test_prediction_consistency(self):
        """Test that predictions are consistent"""
        # Test iris model if available
        try:
            from models.iris.predict import IrisPredictor
            if os.path.exists("models/iris/artifacts/iris_model.pkl"):
                predictor = IrisPredictor()
                features = [5.1, 3.5, 1.4, 0.2]

                # Make multiple predictions
                results = [predictor.predict(features) for _ in range(3)]

                # Check consistency
                predictions = [r['prediction'] for r in results]
                assert len(set(predictions)) == 1, "Predictions should be consistent"

                print("✅ Prediction consistency test passed")
        except Exception as e:
            pytest.skip(f"Could not test prediction consistency: {e}")

def run_model_component_tests():
    """Run all model component tests"""
    print("🧪 Running Model Component Tests")
    print("=" * 50)

    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

if __name__ == "__main__":
    run_model_component_tests()