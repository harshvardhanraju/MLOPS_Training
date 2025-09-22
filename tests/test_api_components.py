"""
Comprehensive tests for API components
"""

import pytest
import requests
import time
import json
import numpy as np
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class TestAPIService:
    """Test API service availability and basic functionality"""

    def setup_method(self):
        """Setup for API tests"""
        self.base_url = "http://localhost:8000"
        self.timeout = 10

    def test_api_server_running(self):
        """Test if API server is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            assert response.status_code == 200, "API server should be running"
            print("✅ API server is running")
        except requests.ConnectionError:
            pytest.skip("API server is not running. Start with: docker-compose up -d")

    def test_api_health_endpoint(self):
        """Test API health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            assert response.status_code == 200

            data = response.json()
            assert 'status' in data, "Health response should have status"
            assert data['status'] == 'healthy', "API should be healthy"
            assert 'models' in data, "Health response should include models status"

            print("✅ API health endpoint test passed")
        except Exception as e:
            pytest.fail(f"API health test failed: {e}")

    def test_api_root_endpoint(self):
        """Test API root endpoint"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=self.timeout)
            assert response.status_code == 200

            data = response.json()
            assert 'message' in data, "Root response should have message"
            assert 'version' in data, "Root response should have version"
            assert 'endpoints' in data, "Root response should have endpoints"

            print("✅ API root endpoint test passed")
        except Exception as e:
            pytest.fail(f"API root test failed: {e}")

    def test_api_models_list_endpoint(self):
        """Test API models list endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/models", timeout=self.timeout)
            assert response.status_code == 200

            data = response.json()
            assert 'models' in data, "Response should have models list"
            assert len(data['models']) == 5, "Should have 5 models"

            # Check each model has required fields
            for model in data['models']:
                assert 'name' in model, "Model should have name"
                assert 'description' in model, "Model should have description"
                assert 'type' in model, "Model should have type"
                assert 'endpoint' in model, "Model should have endpoint"

            print("✅ API models list endpoint test passed")
        except Exception as e:
            pytest.fail(f"API models list test failed: {e}")

    def test_api_metrics_endpoint(self):
        """Test API metrics endpoint"""
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=self.timeout)
            assert response.status_code == 200
            assert "text/plain" in response.headers.get("content-type", "")

            # Check for some expected Prometheus metrics
            content = response.text
            assert "api_requests_total" in content, "Should have request metrics"
            print("✅ API metrics endpoint test passed")
        except Exception as e:
            pytest.fail(f"API metrics test failed: {e}")

class TestIrisAPI:
    """Test Iris model API endpoints"""

    def setup_method(self):
        """Setup for Iris API tests"""
        self.base_url = "http://localhost:8000"
        self.endpoint = f"{self.base_url}/api/v1/iris"
        self.timeout = 10

    def test_iris_info_endpoint(self):
        """Test iris model info endpoint"""
        try:
            response = requests.get(f"{self.endpoint}/info", timeout=self.timeout)
            assert response.status_code == 200

            data = response.json()
            assert data['model_name'] == 'iris_classifier'
            assert data['model_type'] == 'multiclass_classification'
            assert 'classes' in data
            assert 'features' in data

            print("✅ Iris info endpoint test passed")
        except Exception as e:
            pytest.fail(f"Iris info test failed: {e}")

    def test_iris_prediction_endpoint(self):
        """Test iris prediction endpoint"""
        try:
            payload = {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }

            response = requests.post(
                f"{self.endpoint}/predict",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 503:
                pytest.skip("Iris model not available")

            assert response.status_code == 200, f"Request failed: {response.text}"

            data = response.json()
            assert 'prediction' in data
            assert 'confidence' in data
            assert 'probabilities' in data

            # Validate prediction values
            assert data['prediction'] in ['setosa', 'versicolor', 'virginica']
            assert 0 <= data['confidence'] <= 1
            assert len(data['probabilities']) == 3

            print("✅ Iris prediction endpoint test passed")
        except Exception as e:
            pytest.fail(f"Iris prediction test failed: {e}")

    def test_iris_prediction_validation(self):
        """Test iris prediction input validation"""
        try:
            # Test invalid input (negative values)
            invalid_payload = {
                "sepal_length": -1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }

            response = requests.post(
                f"{self.endpoint}/predict",
                json=invalid_payload,
                timeout=self.timeout
            )

            assert response.status_code == 422, "Should reject invalid input"
            print("✅ Iris input validation test passed")
        except Exception as e:
            pytest.fail(f"Iris validation test failed: {e}")

class TestHousePriceAPI:
    """Test House Price model API endpoints"""

    def setup_method(self):
        """Setup for House Price API tests"""
        self.base_url = "http://localhost:8000"
        self.endpoint = f"{self.base_url}/api/v1/house-price"
        self.timeout = 10

    def test_house_price_prediction_endpoint(self):
        """Test house price prediction endpoint"""
        try:
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

            response = requests.post(
                f"{self.endpoint}/predict",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 503:
                pytest.skip("House price model not available")

            assert response.status_code == 200, f"Request failed: {response.text}"

            data = response.json()
            assert 'predicted_price' in data
            assert 'predicted_price_formatted' in data
            assert 'confidence_interval' in data

            # Validate prediction values
            assert data['predicted_price'] > 0
            assert data['predicted_price'] < 10000000  # Reasonable upper bound

            print("✅ House price prediction endpoint test passed")
        except Exception as e:
            pytest.fail(f"House price prediction test failed: {e}")

class TestSentimentAPI:
    """Test Sentiment Analysis API endpoints"""

    def setup_method(self):
        """Setup for Sentiment API tests"""
        self.base_url = "http://localhost:8000"
        self.endpoint = f"{self.base_url}/api/v1/sentiment"
        self.timeout = 10

    def test_sentiment_prediction_endpoint(self):
        """Test sentiment prediction endpoint"""
        try:
            payload = {"text": "I love this product!"}

            response = requests.post(
                f"{self.endpoint}/predict",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 503:
                pytest.skip("Sentiment model not available")

            assert response.status_code == 200, f"Request failed: {response.text}"

            data = response.json()
            assert 'sentiment' in data
            assert 'confidence' in data
            assert 'probabilities' in data

            # Validate prediction values
            assert data['sentiment'] in ['negative', 'neutral', 'positive']
            assert 0 <= data['confidence'] <= 1

            print("✅ Sentiment prediction endpoint test passed")
        except Exception as e:
            pytest.fail(f"Sentiment prediction test failed: {e}")

    def test_sentiment_empty_text_validation(self):
        """Test sentiment API with empty text"""
        try:
            payload = {"text": ""}

            response = requests.post(
                f"{self.endpoint}/predict",
                json=payload,
                timeout=self.timeout
            )

            assert response.status_code == 422, "Should reject empty text"
            print("✅ Sentiment empty text validation test passed")
        except Exception as e:
            pytest.fail(f"Sentiment validation test failed: {e}")

class TestChurnAPI:
    """Test Customer Churn API endpoints"""

    def setup_method(self):
        """Setup for Churn API tests"""
        self.base_url = "http://localhost:8000"
        self.endpoint = f"{self.base_url}/api/v1/churn"
        self.timeout = 10

    def test_churn_prediction_endpoint(self):
        """Test churn prediction endpoint"""
        try:
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

            response = requests.post(
                f"{self.endpoint}/predict",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 503:
                pytest.skip("Churn model not available")

            assert response.status_code == 200, f"Request failed: {response.text}"

            data = response.json()
            assert 'churn_prediction' in data
            assert 'churn_probability' in data
            assert 'risk_level' in data
            assert 'recommendations' in data

            # Validate prediction values
            assert data['churn_prediction'] in [0, 1]
            assert 0 <= data['churn_probability'] <= 1
            assert data['risk_level'] in ['Low', 'Medium', 'High']

            print("✅ Churn prediction endpoint test passed")
        except Exception as e:
            pytest.fail(f"Churn prediction test failed: {e}")

class TestImageAPI:
    """Test Image Classification API endpoints"""

    def setup_method(self):
        """Setup for Image API tests"""
        self.base_url = "http://localhost:8000"
        self.endpoint = f"{self.base_url}/api/v1/image"
        self.timeout = 15  # Longer timeout for image processing

    def test_image_sample_endpoint(self):
        """Test image sample generation endpoint"""
        try:
            response = requests.get(f"{self.endpoint}/sample", timeout=self.timeout)

            if response.status_code == 503:
                pytest.skip("Image model not available")

            assert response.status_code == 200, f"Request failed: {response.text}"

            data = response.json()
            assert 'image_data' in data
            assert 'description' in data
            assert 'shape' in data

            # Validate image data
            assert len(data['shape']) == 3
            assert data['shape'] == [32, 32, 3]

            print("✅ Image sample endpoint test passed")
        except Exception as e:
            pytest.fail(f"Image sample test failed: {e}")

    def test_image_prediction_endpoint(self):
        """Test image prediction endpoint"""
        try:
            # Create a test image (32x32x3)
            test_image = np.random.rand(32, 32, 3).tolist()
            payload = {"image_data": test_image}

            response = requests.post(
                f"{self.endpoint}/predict",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 503:
                pytest.skip("Image model not available")

            assert response.status_code == 200, f"Request failed: {response.text}"

            data = response.json()
            assert 'prediction' in data
            assert 'confidence' in data
            assert 'probabilities' in data

            # Validate prediction values
            expected_classes = ['red_dominant', 'green_dominant', 'blue_dominant']
            assert data['prediction'] in expected_classes
            assert 0 <= data['confidence'] <= 1

            print("✅ Image prediction endpoint test passed")
        except Exception as e:
            pytest.fail(f"Image prediction test failed: {e}")

    def test_image_invalid_shape_validation(self):
        """Test image API with invalid image shape"""
        try:
            # Create invalid image (wrong shape)
            invalid_image = np.random.rand(28, 28, 3).tolist()
            payload = {"image_data": invalid_image}

            response = requests.post(
                f"{self.endpoint}/predict",
                json=payload,
                timeout=self.timeout
            )

            assert response.status_code == 422, "Should reject invalid image shape"
            print("✅ Image shape validation test passed")
        except Exception as e:
            pytest.fail(f"Image validation test failed: {e}")

class TestAPIPerformance:
    """Test API performance and reliability"""

    def setup_method(self):
        """Setup for performance tests"""
        self.base_url = "http://localhost:8000"
        self.timeout = 30

    def test_api_response_time(self):
        """Test API response time"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            response_time = time.time() - start_time

            assert response.status_code == 200
            assert response_time < 5.0, f"Response time too slow: {response_time:.2f}s"

            print(f"✅ API response time test passed ({response_time:.2f}s)")
        except Exception as e:
            pytest.fail(f"API performance test failed: {e}")

    def test_api_concurrent_requests(self):
        """Test API with concurrent requests"""
        import concurrent.futures
        import threading

        def make_request():
            try:
                response = requests.get(f"{self.base_url}/health", timeout=10)
                return response.status_code == 200
            except:
                return False

        try:
            # Make 5 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request) for _ in range(5)]
                results = [future.result() for future in futures]

            success_rate = sum(results) / len(results)
            assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2%}"

            print(f"✅ Concurrent requests test passed ({success_rate:.2%} success)")
        except Exception as e:
            pytest.fail(f"Concurrent requests test failed: {e}")

class TestAPIIntegration:
    """Test API integration with other components"""

    def setup_method(self):
        """Setup for integration tests"""
        self.base_url = "http://localhost:8000"
        self.timeout = 10

    def test_api_model_metrics_endpoint(self):
        """Test API model metrics integration"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/model-metrics", timeout=self.timeout)
            assert response.status_code == 200

            data = response.json()
            assert 'model_accuracy' in data
            assert 'total_predictions' in data
            assert 'system_metrics' in data

            print("✅ API model metrics integration test passed")
        except Exception as e:
            pytest.fail(f"API metrics integration test failed: {e}")

def run_api_component_tests():
    """Run all API component tests"""
    print("🧪 Running API Component Tests")
    print("=" * 50)

    # Check if API is running first
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("⚠️  API server is not healthy. Please check the service.")
            return False
    except requests.ConnectionError:
        print("❌ API server is not running. Please start with: docker-compose up -d")
        return False

    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
    return True

if __name__ == "__main__":
    run_api_component_tests()