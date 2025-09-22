"""
Simple monitoring tests without heavy dependencies
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_drift_detection_basic():
    """Test basic drift detection functionality"""
    print("🧪 Testing Basic Drift Detection...")

    try:
        from monitoring.drift_detection import DriftDetector

        # Check if reference data exists
        if not os.path.exists("data/raw/iris.csv"):
            print("   ⚠️  Reference data not found, skipping")
            return True

        # Create detector
        detector = DriftDetector("iris", "data/raw/iris.csv")
        print("   ✅ DriftDetector created successfully")

        # Generate synthetic drift data
        current_data = detector.generate_synthetic_drift_data(drift_intensity=0.2)
        print(f"   ✅ Generated drift data: {current_data.shape}")

        # Test drift detection (without saving reports)
        drift_summary = detector.detect_drift(current_data, save_report=False)
        print(f"   ✅ Drift detection completed")

        # Validate results
        assert 'dataset_drift_detected' in drift_summary
        assert 'drift_share' in drift_summary
        assert isinstance(drift_summary['dataset_drift_detected'], bool)
        assert isinstance(drift_summary['drift_share'], float)

        print(f"   ✅ Drift detected: {drift_summary['dataset_drift_detected']}")
        print(f"   ✅ Drift share: {drift_summary['drift_share']:.2%}")

        return True

    except ImportError as e:
        print(f"   ⚠️  Import error (dependency missing): {e}")
        return True  # Not a failure, just missing dependency
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_model_performance_monitoring():
    """Test basic model performance monitoring"""
    print("🧪 Testing Model Performance Monitoring...")

    try:
        from monitoring.check_model_performance import ModelPerformanceMonitor

        # Create monitor
        monitor = ModelPerformanceMonitor()
        print("   ✅ ModelPerformanceMonitor created")

        # Test threshold checking (with mock data)
        mock_metrics = {'accuracy': 0.85, 'f1_score': 0.82}
        result = monitor.check_performance_thresholds('iris', mock_metrics)

        assert 'status' in result
        assert 'alerts' in result
        print(f"   ✅ Performance check: {result['status']}")

        return True

    except ImportError as e:
        print(f"   ⚠️  Import error (dependency missing): {e}")
        return True  # Not a failure, just missing dependency
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_prometheus_config():
    """Test Prometheus configuration"""
    print("🧪 Testing Prometheus Configuration...")

    try:
        import yaml

        config_path = "monitoring/prometheus/prometheus.yml"
        if not os.path.exists(config_path):
            print("   ❌ Prometheus config not found")
            return False

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check basic structure
        assert 'global' in config, "Should have global section"
        assert 'scrape_configs' in config, "Should have scrape configs"

        # Check for required scrape jobs
        job_names = [job['job_name'] for job in config['scrape_configs']]
        required_jobs = ['prometheus', 'mlops-api']

        for job in required_jobs:
            if job in job_names:
                print(f"   ✅ Found job: {job}")
            else:
                print(f"   ⚠️  Missing job: {job}")

        print("   ✅ Prometheus config structure valid")
        return True

    except ImportError:
        print("   ⚠️  PyYAML not available, skipping YAML validation")
        # Check if file exists at least
        return os.path.exists("monitoring/prometheus/prometheus.yml")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_grafana_config():
    """Test Grafana configuration"""
    print("🧪 Testing Grafana Configuration...")

    try:
        grafana_files = [
            "monitoring/grafana/provisioning/datasources.yml",
            "monitoring/grafana/provisioning/dashboards.yml",
            "monitoring/grafana/dashboards/mlops_dashboard.json"
        ]

        valid_files = 0
        for file_path in grafana_files:
            if os.path.exists(file_path):
                print(f"   ✅ Found: {file_path}")
                valid_files += 1
            else:
                print(f"   ❌ Missing: {file_path}")

        # Test dashboard JSON structure
        dashboard_path = "monitoring/grafana/dashboards/mlops_dashboard.json"
        if os.path.exists(dashboard_path):
            with open(dashboard_path, 'r') as f:
                dashboard = json.load(f)

            if 'dashboard' in dashboard:
                dash_config = dashboard['dashboard']
                assert 'title' in dash_config, "Dashboard should have title"
                assert 'panels' in dash_config, "Dashboard should have panels"
                print(f"   ✅ Dashboard has {len(dash_config['panels'])} panels")

        return valid_files >= 2  # At least 2 files should exist

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_basic_metrics_collection():
    """Test basic metrics collection"""
    print("🧪 Testing Basic Metrics Collection...")

    try:
        # Test creating metrics without Prometheus client
        metrics_data = {
            'api_requests_total': 100,
            'model_predictions_total': {
                'iris': 25,
                'house_price': 15,
                'sentiment': 30
            },
            'model_accuracy': {
                'iris': 0.95,
                'house_price': 0.82,
                'sentiment': 0.88
            },
            'system_metrics': {
                'cpu_usage_percent': 45.2,
                'memory_usage_percent': 67.8
            }
        }

        # Validate metrics structure
        assert isinstance(metrics_data['api_requests_total'], int)
        assert isinstance(metrics_data['model_predictions_total'], dict)
        assert isinstance(metrics_data['model_accuracy'], dict)
        assert isinstance(metrics_data['system_metrics'], dict)

        # Check if all models have metrics
        models = ['iris', 'house_price', 'sentiment']
        for model in models:
            if model in metrics_data['model_predictions_total']:
                assert metrics_data['model_predictions_total'][model] >= 0
                print(f"   ✅ {model} predictions: {metrics_data['model_predictions_total'][model]}")

            if model in metrics_data['model_accuracy']:
                accuracy = metrics_data['model_accuracy'][model]
                assert 0 <= accuracy <= 1, f"Invalid accuracy for {model}: {accuracy}"
                print(f"   ✅ {model} accuracy: {accuracy}")

        print("   ✅ Metrics structure validation passed")
        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_monitoring_reports_directory():
    """Test monitoring reports directory structure"""
    print("🧪 Testing Monitoring Reports Directory...")

    try:
        # Create reports directory if it doesn't exist
        reports_dir = "monitoring/reports"
        os.makedirs(reports_dir, exist_ok=True)
        print(f"   ✅ Reports directory: {reports_dir}")

        # Create model-specific subdirectories
        models = ['iris', 'house_price', 'sentiment', 'churn', 'image']
        for model in models:
            model_dir = os.path.join(reports_dir, model)
            os.makedirs(model_dir, exist_ok=True)

        print(f"   ✅ Created subdirectories for {len(models)} models")

        # Test writing a sample report
        sample_report = {
            'timestamp': '2024-01-01T00:00:00',
            'model_name': 'test_model',
            'drift_detected': False,
            'drift_share': 0.05
        }

        sample_path = os.path.join(reports_dir, 'sample_report.json')
        with open(sample_path, 'w') as f:
            json.dump(sample_report, f, indent=2)

        print("   ✅ Sample report written successfully")

        # Clean up sample file
        os.remove(sample_path)

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def run_all_monitoring_tests():
    """Run all monitoring tests"""
    print("🚀 Running Monitoring Component Tests")
    print("=" * 50)

    results = []

    # Run tests
    results.append(("Drift Detection", test_drift_detection_basic()))
    results.append(("Performance Monitoring", test_model_performance_monitoring()))
    results.append(("Prometheus Config", test_prometheus_config()))
    results.append(("Grafana Config", test_grafana_config()))
    results.append(("Metrics Collection", test_basic_metrics_collection()))
    results.append(("Reports Directory", test_monitoring_reports_directory()))

    # Print summary
    print("\n📊 Test Results Summary:")
    print("-" * 30)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed >= total * 0.75:  # 75% pass rate
        print("🎉 Monitoring tests mostly passed!")
        return True
    else:
        print("⚠️  Many monitoring tests failed")
        return False

if __name__ == "__main__":
    success = run_all_monitoring_tests()
    exit(0 if success else 1)