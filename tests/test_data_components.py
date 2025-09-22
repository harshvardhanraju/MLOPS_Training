"""
Comprehensive tests for data components
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class TestDataCreation:
    """Test data creation functionality"""

    def test_create_datasets_script_exists(self):
        """Test that the data creation script exists"""
        script_path = "data/create_datasets.py"
        assert os.path.exists(script_path), f"Data creation script not found: {script_path}"

    def test_create_iris_dataset(self):
        """Test iris dataset creation"""
        try:
            from data.create_datasets import create_iris_dataset
            df = create_iris_dataset()

            # Basic structure tests
            assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
            assert len(df) > 0, "Dataset should not be empty"
            assert 'target' in df.columns, "Should have target column"
            assert 'target_name' in df.columns, "Should have target_name column"

            # Data quality tests
            assert df.isnull().sum().sum() == 0, "Should not have null values"
            assert len(df['target'].unique()) == 3, "Should have 3 classes"
            assert set(df['target_name'].unique()) == {'setosa', 'versicolor', 'virginica'}, "Should have correct class names"

            # Feature tests
            feature_columns = [col for col in df.columns if col not in ['target', 'target_name']]
            assert len(feature_columns) == 4, "Should have 4 feature columns"

            print("✅ Iris dataset creation test passed")

        except ImportError as e:
            pytest.skip(f"Could not import create_iris_dataset: {e}")

    def test_create_housing_dataset(self):
        """Test housing dataset creation"""
        try:
            from data.create_datasets import create_housing_dataset
            df = create_housing_dataset()

            # Basic structure tests
            assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
            assert len(df) > 0, "Dataset should not be empty"
            assert 'price' in df.columns, "Should have price column"

            # Data quality tests
            assert df.isnull().sum().sum() == 0, "Should not have null values"
            assert (df['price'] > 0).all(), "All prices should be positive"

            # Feature tests
            expected_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
            for feature in expected_features:
                assert feature in df.columns, f"Should have {feature} column"

            print("✅ Housing dataset creation test passed")

        except ImportError as e:
            pytest.skip(f"Could not import create_housing_dataset: {e}")

    def test_create_churn_dataset(self):
        """Test churn dataset creation"""
        try:
            from data.create_datasets import create_churn_dataset
            df = create_churn_dataset()

            # Basic structure tests
            assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
            assert len(df) > 0, "Dataset should not be empty"
            assert 'churn' in df.columns, "Should have churn column"

            # Data quality tests
            assert df.isnull().sum().sum() == 0, "Should not have null values"
            assert set(df['churn'].unique()) == {0, 1}, "Churn should be binary"

            # Feature tests
            expected_features = ['age', 'tenure_months', 'monthly_charges', 'contract_type']
            for feature in expected_features:
                assert feature in df.columns, f"Should have {feature} column"

            # Business logic tests
            assert (df['age'] >= 18).all(), "All customers should be adults"
            assert (df['monthly_charges'] > 0).all(), "All charges should be positive"

            print("✅ Churn dataset creation test passed")

        except ImportError as e:
            pytest.skip(f"Could not import create_churn_dataset: {e}")

    def test_create_sentiment_dataset(self):
        """Test sentiment dataset creation"""
        try:
            from data.create_datasets import create_sentiment_dataset
            df = create_sentiment_dataset()

            # Basic structure tests
            assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
            assert len(df) > 0, "Dataset should not be empty"
            assert 'text' in df.columns, "Should have text column"
            assert 'label' in df.columns, "Should have label column"
            assert 'sentiment' in df.columns, "Should have sentiment column"

            # Data quality tests
            assert df.isnull().sum().sum() == 0, "Should not have null values"
            assert set(df['label'].unique()) == {0, 1, 2}, "Should have 3 sentiment classes"
            assert set(df['sentiment'].unique()) == {'negative', 'neutral', 'positive'}, "Should have correct sentiment names"

            # Text quality tests
            assert (df['text'].str.len() > 0).all(), "All texts should be non-empty"

            print("✅ Sentiment dataset creation test passed")

        except ImportError as e:
            pytest.skip(f"Could not import create_sentiment_dataset: {e}")

    def test_create_image_dataset(self):
        """Test image dataset creation"""
        try:
            from data.create_datasets import create_image_dataset
            df = create_image_dataset()

            # Basic structure tests
            assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
            assert len(df) > 0, "Dataset should not be empty"
            assert 'class' in df.columns, "Should have class column"

            # Data quality tests
            assert df.isnull().sum().sum() == 0, "Should not have null values"
            expected_classes = {'red_dominant', 'green_dominant', 'blue_dominant'}
            assert set(df['class'].unique()) == expected_classes, "Should have correct class names"

            # Image metadata tests
            assert 'width' in df.columns and 'height' in df.columns, "Should have image dimensions"
            assert (df['width'] == 32).all() and (df['height'] == 32).all(), "All images should be 32x32"

            print("✅ Image dataset creation test passed")

        except ImportError as e:
            pytest.skip(f"Could not import create_image_dataset: {e}")

class TestDataPersistence:
    """Test data saving and loading"""

    def test_create_all_datasets_script(self):
        """Test running the complete dataset creation script"""
        try:
            # Ensure data directories exist
            os.makedirs("data/raw", exist_ok=True)
            os.makedirs("data/processed", exist_ok=True)

            # Import and run the main function
            from data.create_datasets import main
            main()

            # Check that files were created
            expected_files = [
                "data/raw/iris.csv",
                "data/raw/housing.csv",
                "data/raw/churn.csv",
                "data/raw/sentiment.csv",
                "data/raw/image_metadata.csv",
                "data/raw/data_summary.txt"
            ]

            for file_path in expected_files:
                assert os.path.exists(file_path), f"Expected file not created: {file_path}"

                # Check file is not empty
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    assert len(df) > 0, f"Dataset file should not be empty: {file_path}"
                elif file_path.endswith('.txt'):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    assert len(content) > 0, f"Summary file should not be empty: {file_path}"

            print("✅ All datasets creation and persistence test passed")

        except Exception as e:
            pytest.fail(f"Dataset creation script failed: {e}")

    def test_data_loading(self):
        """Test loading created datasets"""
        data_files = {
            "iris": "data/raw/iris.csv",
            "housing": "data/raw/housing.csv",
            "churn": "data/raw/churn.csv",
            "sentiment": "data/raw/sentiment.csv",
            "image": "data/raw/image_metadata.csv"
        }

        for dataset_name, file_path in data_files.items():
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    assert isinstance(df, pd.DataFrame), f"Should load {dataset_name} as DataFrame"
                    assert len(df) > 0, f"{dataset_name} dataset should not be empty"
                    print(f"✅ {dataset_name} dataset loading test passed")
                except Exception as e:
                    pytest.fail(f"Failed to load {dataset_name} dataset: {e}")
            else:
                pytest.skip(f"Dataset file not found: {file_path}")

class TestDataValidation:
    """Test data validation and quality checks"""

    def test_data_quality_iris(self):
        """Test iris data quality"""
        if os.path.exists("data/raw/iris.csv"):
            df = pd.read_csv("data/raw/iris.csv")

            # Check for duplicates
            assert df.duplicated().sum() == 0, "Should not have duplicate rows"

            # Check feature ranges
            feature_cols = [col for col in df.columns if col not in ['target', 'target_name']]
            for col in feature_cols:
                assert df[col].min() > 0, f"Feature {col} should be positive"
                assert df[col].max() < 100, f"Feature {col} should be reasonable"

            # Check class distribution
            class_counts = df['target'].value_counts()
            assert len(class_counts) == 3, "Should have 3 classes"

            print("✅ Iris data quality test passed")
        else:
            pytest.skip("Iris dataset not found")

    def test_data_quality_housing(self):
        """Test housing data quality"""
        if os.path.exists("data/raw/housing.csv"):
            df = pd.read_csv("data/raw/housing.csv")

            # Check for missing values
            assert df.isnull().sum().sum() == 0, "Should not have missing values"

            # Check geographical constraints (California)
            assert df['Latitude'].between(32, 42).all(), "Latitude should be within California range"
            assert df['Longitude'].between(-125, -114).all(), "Longitude should be within California range"

            # Check reasonable values
            assert (df['HouseAge'] >= 0).all(), "House age should be non-negative"
            assert (df['AveRooms'] > 0).all(), "Average rooms should be positive"
            assert (df['price'] > 0).all(), "Prices should be positive"

            print("✅ Housing data quality test passed")
        else:
            pytest.skip("Housing dataset not found")

    def test_data_consistency(self):
        """Test data consistency across datasets"""
        datasets = {}

        # Load all available datasets
        data_files = {
            "iris": "data/raw/iris.csv",
            "housing": "data/raw/housing.csv",
            "churn": "data/raw/churn.csv",
            "sentiment": "data/raw/sentiment.csv"
        }

        loaded_count = 0
        for name, file_path in data_files.items():
            if os.path.exists(file_path):
                datasets[name] = pd.read_csv(file_path)
                loaded_count += 1

        if loaded_count > 0:
            # Check that all datasets have reasonable sizes
            for name, df in datasets.items():
                assert len(df) >= 100, f"{name} dataset should have at least 100 samples"
                assert len(df.columns) >= 2, f"{name} dataset should have at least 2 columns"

            print(f"✅ Data consistency test passed for {loaded_count} datasets")
        else:
            pytest.skip("No datasets found for consistency testing")

def run_data_component_tests():
    """Run all data component tests"""
    print("🧪 Running Data Component Tests")
    print("=" * 50)

    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

if __name__ == "__main__":
    run_data_component_tests()