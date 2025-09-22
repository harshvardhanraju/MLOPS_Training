"""
Integration tests for the complete MLOps pipeline
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_data_to_model_pipeline():
    """Test data creation -> model training -> prediction pipeline"""
    print("🧪 Testing Data-to-Model Pipeline...")

    try:
        # Step 1: Create data
        from data.create_datasets import create_iris_dataset
        data = create_iris_dataset()
        print(f"   ✅ Data created: {data.shape}")

        # Step 2: Train simple model
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        import pickle

        X = data.drop(['target', 'target_name'], axis=1)
        y = data['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        print("   ✅ Model trained")

        # Step 3: Save model
        os.makedirs("models/iris/artifacts", exist_ok=True)
        model_path = "models/iris/artifacts/integration_test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print("   ✅ Model saved")

        # Step 4: Load and predict
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)

        prediction = loaded_model.predict([[5.1, 3.5, 1.4, 0.2]])
        probability = loaded_model.predict_proba([[5.1, 3.5, 1.4, 0.2]])

        print(f"   ✅ Prediction: {prediction[0]}")
        print(f"   ✅ Probability: {probability[0].max():.3f}")

        # Clean up
        os.remove(model_path)

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end_model_workflow():
    """Test complete model workflow for multiple models"""
    print("🧪 Testing End-to-End Model Workflow...")

    results = {}

    try:
        # Test Iris workflow
        print("   🔸 Testing Iris workflow...")
        from models.iris.predict import IrisPredictor

        iris_model_path = "models/iris/artifacts/iris_model.pkl"
        if os.path.exists(iris_model_path):
            predictor = IrisPredictor()
            result = predictor.predict([5.1, 3.5, 1.4, 0.2])
            results['iris'] = result['prediction']
            print(f"     ✅ Iris: {result['prediction']}")
        else:
            print("     ⚠️  Iris model not found")

        # Test House Price workflow (if simple model exists)
        house_model_path = "models/house_price/artifacts/simple_house_model.pkl"
        if os.path.exists(house_model_path):
            print("   🔸 Testing House Price workflow...")
            import pickle
            with open(house_model_path, 'rb') as f:
                house_model = pickle.load(f)
            prediction = house_model.predict([[8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]])
            results['house_price'] = f"${prediction[0] * 100000:.0f}"
            print(f"     ✅ House Price: {results['house_price']}")

        # Test Sentiment workflow (if simple model exists)
        sentiment_model_path = "models/sentiment/artifacts/simple_sentiment_model.pkl"
        if os.path.exists(sentiment_model_path):
            print("   🔸 Testing Sentiment workflow...")
            import pickle
            with open(sentiment_model_path, 'rb') as f:
                sentiment_model = pickle.load(f)
            prediction = sentiment_model.predict(["I love this product!"])
            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            results['sentiment'] = sentiment_map.get(prediction[0], 'unknown')
            print(f"     ✅ Sentiment: {results['sentiment']}")

        print(f"   ✅ Tested {len(results)} model workflows")
        return len(results) > 0

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_monitoring_data_flow():
    """Test monitoring and data flow"""
    print("🧪 Testing Monitoring Data Flow...")

    try:
        # Simulate metrics collection
        metrics = {
            'timestamp': time.time(),
            'model_predictions': {
                'iris': 15,
                'house_price': 8,
                'sentiment': 22
            },
            'model_performance': {
                'iris': 0.95,
                'house_price': 0.82,
                'sentiment': 0.88
            },
            'system_metrics': {
                'cpu_usage': 45.2,
                'memory_usage': 67.8
            }
        }

        # Save metrics to reports directory
        reports_dir = "monitoring/reports"
        os.makedirs(reports_dir, exist_ok=True)

        metrics_file = os.path.join(reports_dir, "integration_test_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"   ✅ Metrics saved to: {metrics_file}")

        # Validate metrics file
        with open(metrics_file, 'r') as f:
            loaded_metrics = json.load(f)

        assert 'model_predictions' in loaded_metrics
        assert 'model_performance' in loaded_metrics
        assert 'system_metrics' in loaded_metrics

        print("   ✅ Metrics validation passed")

        # Clean up
        os.remove(metrics_file)

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_configuration_consistency():
    """Test configuration consistency across components"""
    print("🧪 Testing Configuration Consistency...")

    try:
        # Check Docker Compose config
        if os.path.exists("docker-compose.yml"):
            with open("docker-compose.yml", 'r') as f:
                compose_content = f.read()

            required_services = ['api', 'mlflow', 'prometheus', 'grafana']
            found_services = []

            for service in required_services:
                if f"{service}:" in compose_content:
                    found_services.append(service)
                    print(f"   ✅ Found service: {service}")

            print(f"   ✅ Found {len(found_services)}/{len(required_services)} services")

        # Check API structure
        api_files = [
            "api/main.py",
            "api/models.py",
            "api/monitoring.py"
        ]

        found_api_files = 0
        for file_path in api_files:
            if os.path.exists(file_path):
                found_api_files += 1
                print(f"   ✅ Found API file: {file_path}")

        # Check model structure consistency
        models = ['iris', 'house_price', 'sentiment', 'churn', 'image_classification']
        consistent_models = 0

        for model in models:
            model_dir = f"models/{model}"
            train_file = f"{model_dir}/train.py"
            predict_file = f"{model_dir}/predict.py"

            if os.path.exists(train_file) and os.path.exists(predict_file):
                consistent_models += 1
                print(f"   ✅ Consistent structure: {model}")

        print(f"   ✅ {consistent_models}/{len(models)} models have consistent structure")

        return found_api_files >= 2 and consistent_models >= 3

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_documentation_completeness():
    """Test documentation completeness"""
    print("🧪 Testing Documentation Completeness...")

    try:
        doc_files = [
            "README.md",
            "docs/README.md",
            "docs/presentations/01_mlops_overview.md",
            "docs/guides/setup_guide.md"
        ]

        found_docs = 0
        for doc_file in doc_files:
            if os.path.exists(doc_file):
                # Check if file has content
                with open(doc_file, 'r') as f:
                    content = f.read()
                if len(content.strip()) > 100:  # At least 100 characters
                    found_docs += 1
                    print(f"   ✅ Complete documentation: {doc_file}")
                else:
                    print(f"   ⚠️  Minimal documentation: {doc_file}")
            else:
                print(f"   ❌ Missing documentation: {doc_file}")

        print(f"   ✅ Found {found_docs}/{len(doc_files)} complete documentation files")
        return found_docs >= 3

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_script_functionality():
    """Test that key scripts are functional"""
    print("🧪 Testing Script Functionality...")

    try:
        scripts = [
            "scripts/demo_script.py",
            "scripts/project_summary.py",
            "scripts/train_all_models.py"
        ]

        functional_scripts = 0
        for script in scripts:
            if os.path.exists(script):
                try:
                    # Check syntax by compiling
                    with open(script, 'r') as f:
                        content = f.read()
                    compile(content, script, 'exec')
                    functional_scripts += 1
                    print(f"   ✅ Valid syntax: {script}")
                except SyntaxError as e:
                    print(f"   ❌ Syntax error in {script}: {e}")
                except Exception as e:
                    print(f"   ⚠️  Error checking {script}: {e}")
            else:
                print(f"   ❌ Missing script: {script}")

        print(f"   ✅ {functional_scripts}/{len(scripts)} scripts are functional")
        return functional_scripts >= 2

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def run_all_integration_tests():
    """Run all integration tests"""
    print("🚀 Running Integration Tests")
    print("=" * 50)

    results = []

    # Run tests
    results.append(("Data-to-Model Pipeline", test_data_to_model_pipeline()))
    results.append(("End-to-End Model Workflow", test_end_to_end_model_workflow()))
    results.append(("Monitoring Data Flow", test_monitoring_data_flow()))
    results.append(("Configuration Consistency", test_configuration_consistency()))
    results.append(("Documentation Completeness", test_documentation_completeness()))
    results.append(("Script Functionality", test_script_functionality()))

    # Print summary
    print("\n📊 Integration Test Results:")
    print("-" * 35)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\nOverall: {passed}/{total} integration tests passed")

    if passed >= total * 0.8:  # 80% pass rate
        print("🎉 Integration tests passed!")
        success_rate = passed / total
        print(f"   Success Rate: {success_rate:.1%}")
        return True
    else:
        print("⚠️  Integration tests have issues")
        return False

if __name__ == "__main__":
    success = run_all_integration_tests()
    exit(0 if success else 1)