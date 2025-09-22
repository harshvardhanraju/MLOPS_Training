"""
Pytest configuration and fixtures
"""

import pytest
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture for test data directory"""
    return Path(__file__).parent / "test_data"

@pytest.fixture(scope="session")
def create_test_datasets():
    """Fixture to create test datasets"""
    # Ensure test data exists
    try:
        from data.create_datasets import main
        main()
        return True
    except Exception as e:
        pytest.skip(f"Could not create test datasets: {e}")
        return False

@pytest.fixture
def sample_iris_data():
    """Sample iris data for testing"""
    return {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

@pytest.fixture
def sample_house_data():
    """Sample house data for testing"""
    return {
        "med_inc": 8.3252,
        "house_age": 41.0,
        "ave_rooms": 6.984127,
        "ave_bedrms": 1.023810,
        "population": 322.0,
        "ave_occup": 2.555556,
        "latitude": 37.88,
        "longitude": -122.23
    }

@pytest.fixture
def sample_churn_data():
    """Sample churn data for testing"""
    return {
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

def pytest_configure(config):
    """Configure pytest"""
    # Add custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "model: marks tests that require trained models")

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Skip model tests if models are not available
    skip_model = pytest.mark.skip(reason="Trained models not available")

    for item in items:
        if "model" in item.keywords:
            # Check if any model artifacts exist
            models_exist = any(
                os.path.exists(f"models/{model}/artifacts")
                for model in ["iris", "house_price", "sentiment", "churn", "image_classification"]
            )
            if not models_exist:
                item.add_marker(skip_model)