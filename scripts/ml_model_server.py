#!/usr/bin/env python3
"""
ML Model Server with Prometheus Metrics
Provides a REST API for model predictions with comprehensive monitoring
"""

from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import numpy as np
import time
import pickle
import threading
import random
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_wine
from datetime import datetime

app = Flask(__name__)

# Prometheus metrics definitions
model_predictions_total = Counter(
    'ml_model_predictions_total',
    'Total number of model predictions',
    ['model_name', 'model_version', 'prediction_class', 'status']
)

model_prediction_latency = Histogram(
    'ml_model_prediction_duration_seconds',
    'Time spent on model predictions',
    ['model_name', 'model_version'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0]
)

model_accuracy = Gauge(
    'ml_model_accuracy',
    'Current model accuracy score',
    ['model_name', 'model_version']
)

data_drift_score = Gauge(
    'ml_data_drift_score',
    'Data drift detection score (0-1)',
    ['feature_name', 'model_name']
)

prediction_confidence = Histogram(
    'ml_prediction_confidence',
    'Prediction confidence distribution',
    ['model_name'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

model_memory_usage = Gauge(
    'ml_model_memory_usage_bytes',
    'Memory usage of ML models',
    ['model_name']
)

active_requests = Gauge(
    'ml_active_requests',
    'Number of requests currently being processed',
    ['model_name']
)

request_size_bytes = Histogram(
    'ml_request_size_bytes',
    'Size of prediction requests in bytes',
    ['model_name'],
    buckets=[10, 50, 100, 500, 1000, 5000, 10000]
)

# Business metrics
daily_predictions = Counter(
    'ml_daily_predictions_total',
    'Daily prediction volume',
    ['model_name', 'date']
)

model_usage_by_time = Counter(
    'ml_model_usage_by_hour',
    'Model usage patterns by hour',
    ['model_name', 'hour']
)

# Load and prepare models
class MLModelManager:
    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        """Load ML models for serving"""
        # Load Iris model
        iris = load_iris()
        X_iris, y_iris = iris.data, iris.target
        iris_model = RandomForestClassifier(n_estimators=100, random_state=42)
        iris_model.fit(X_iris, y_iris)

        self.models['iris_classifier'] = {
            'model': iris_model,
            'version': 'v1.0',
            'class_names': iris.target_names.tolist(),
            'feature_names': iris.feature_names,
            'accuracy': 0.95,
            'created': datetime.now()
        }

        # Load Wine model
        wine = load_wine()
        X_wine, y_wine = wine.data, wine.target
        wine_model = RandomForestClassifier(n_estimators=100, random_state=42)
        wine_model.fit(X_wine, y_wine)

        self.models['wine_classifier'] = {
            'model': wine_model,
            'version': 'v2.1',
            'class_names': wine.target_names.tolist(),
            'feature_names': wine.feature_names,
            'accuracy': 0.92,
            'created': datetime.now()
        }

        # Initialize model metrics
        for model_name, model_info in self.models.items():
            model_accuracy.labels(
                model_name=model_name,
                model_version=model_info['version']
            ).set(model_info['accuracy'])

model_manager = MLModelManager()

# Simulate data drift monitoring
class DataDriftMonitor:
    def __init__(self):
        self.running = True
        self.drift_thread = threading.Thread(target=self._monitor_drift, daemon=True)
        self.drift_thread.start()

    def _monitor_drift(self):
        """Simulate data drift detection in background"""
        while self.running:
            for model_name, model_info in model_manager.models.items():
                for feature in model_info['feature_names']:
                    # Simulate drift scores with realistic patterns
                    base_drift = np.random.beta(2, 8)  # Usually low drift

                    # Occasionally simulate higher drift
                    if random.random() < 0.05:  # 5% chance
                        base_drift = np.random.beta(8, 2)  # Higher drift

                    data_drift_score.labels(
                        feature_name=feature,
                        model_name=model_name
                    ).set(base_drift)

            # Update model accuracy with small variations
            for model_name, model_info in model_manager.models.items():
                current_accuracy = model_info['accuracy']
                # Add small random variation
                variation = random.gauss(0, 0.02)  # 2% standard deviation
                new_accuracy = max(0.7, min(1.0, current_accuracy + variation))

                model_accuracy.labels(
                    model_name=model_name,
                    model_version=model_info['version']
                ).set(new_accuracy)

            time.sleep(30)  # Update every 30 seconds

# Start drift monitoring
drift_monitor = DataDriftMonitor()

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    """ML model prediction endpoint with comprehensive metrics"""

    if model_name not in model_manager.models:
        model_predictions_total.labels(
            model_name=model_name,
            model_version='unknown',
            prediction_class='error',
            status='model_not_found'
        ).inc()
        return jsonify({'error': f'Model {model_name} not found'}), 404

    model_info = model_manager.models[model_name]
    model = model_info['model']
    version = model_info['version']

    # Track active requests
    active_requests.labels(model_name=model_name).inc()

    start_time = time.time()

    try:
        # Get request data
        data = request.get_json()
        if not data or 'features' not in data:
            model_predictions_total.labels(
                model_name=model_name,
                model_version=version,
                prediction_class='error',
                status='invalid_request'
            ).inc()
            return jsonify({'error': 'Features required in request'}), 400

        features = np.array(data['features']).reshape(1, -1)

        # Track request size
        request_size = len(str(data).encode('utf-8'))
        request_size_bytes.labels(model_name=model_name).observe(request_size)

        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))

        prediction_class = model_info['class_names'][prediction]

        # Record metrics
        model_predictions_total.labels(
            model_name=model_name,
            model_version=version,
            prediction_class=prediction_class,
            status='success'
        ).inc()

        # Record confidence distribution
        prediction_confidence.labels(model_name=model_name).observe(confidence)

        # Record business metrics
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_hour = datetime.now().hour

        daily_predictions.labels(
            model_name=model_name,
            date=current_date
        ).inc()

        model_usage_by_time.labels(
            model_name=model_name,
            hour=str(current_hour)
        ).inc()

        # Calculate and record latency
        latency = time.time() - start_time
        model_prediction_latency.labels(
            model_name=model_name,
            model_version=version
        ).observe(latency)

        # Simulate occasional errors for demo purposes
        if random.random() < 0.02:  # 2% error rate
            model_predictions_total.labels(
                model_name=model_name,
                model_version=version,
                prediction_class='error',
                status='processing_error'
            ).inc()
            raise Exception("Simulated processing error")

        response = {
            'model_name': model_name,
            'model_version': version,
            'prediction': prediction_class,
            'prediction_id': int(prediction),
            'confidence': confidence,
            'probabilities': {
                model_info['class_names'][i]: float(prob)
                for i, prob in enumerate(probabilities)
            },
            'latency_ms': latency * 1000,
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        model_predictions_total.labels(
            model_name=model_name,
            model_version=version,
            prediction_class='error',
            status='processing_error'
        ).inc()

        return jsonify({
            'error': str(e),
            'model_name': model_name,
            'timestamp': datetime.now().isoformat()
        }), 500

    finally:
        active_requests.labels(model_name=model_name).dec()

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    models_info = {}
    for name, info in model_manager.models.items():
        models_info[name] = {
            'version': info['version'],
            'accuracy': info['accuracy'],
            'classes': info['class_names'],
            'features': info['feature_names'],
            'created': info['created'].isoformat()
        }

    return jsonify({
        'models': models_info,
        'total_models': len(models_info)
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(model_manager.models),
        'uptime_seconds': time.time() - start_time_global
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    # Update memory usage metrics
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()

    for model_name in model_manager.models:
        # Simulate per-model memory usage
        base_memory = memory_info.rss / len(model_manager.models)
        model_memory_usage.labels(model_name=model_name).set(base_memory)

    return Response(generate_latest(), mimetype='text/plain')

# Synthetic load generator for demo purposes
class LoadGenerator:
    def __init__(self):
        self.running = True
        self.load_thread = threading.Thread(target=self._generate_load, daemon=True)
        self.load_thread.start()

    def _generate_load(self):
        """Generate synthetic prediction requests for demo"""
        while self.running:
            try:
                # Random delays to simulate real usage patterns
                sleep_time = random.exponential(2.0)  # Average 2 seconds between requests
                time.sleep(min(sleep_time, 10))  # Cap at 10 seconds

                # Choose random model and generate request
                model_name = random.choice(list(model_manager.models.keys()))
                model_info = model_manager.models[model_name]

                # Generate realistic feature data
                if model_name == 'iris_classifier':
                    features = [
                        random.uniform(4.0, 8.0),  # sepal_length
                        random.uniform(2.0, 4.5),  # sepal_width
                        random.uniform(1.0, 7.0),  # petal_length
                        random.uniform(0.1, 2.5)   # petal_width
                    ]
                elif model_name == 'wine_classifier':
                    features = [random.uniform(10, 15) for _ in range(13)]  # Wine has 13 features

                # Make internal prediction (simulates external API call)
                with app.test_client() as client:
                    response = client.post(
                        f'/predict/{model_name}',
                        json={'features': features}
                    )

            except Exception as e:
                # Ignore errors in synthetic load generation
                pass

# Start load generator only if not in production
if os.getenv('GENERATE_LOAD', 'true').lower() == 'true':
    load_generator = LoadGenerator()

# Global start time for uptime calculation
start_time_global = time.time()

if __name__ == '__main__':
    print("🚀 Starting ML Model Server with Prometheus Metrics")
    print("=" * 60)
    print(f"📊 Models loaded: {len(model_manager.models)}")
    print(f"🔍 Metrics endpoint: http://localhost:8001/metrics")
    print(f"🏥 Health check: http://localhost:8001/health")
    print(f"📋 Models list: http://localhost:8001/models")
    print("=" * 60)

    for model_name, model_info in model_manager.models.items():
        print(f"🤖 {model_name} v{model_info['version']}: http://localhost:8001/predict/{model_name}")

    print("\n🎯 Example prediction request:")
    print("curl -X POST http://localhost:8001/predict/iris_classifier \\")
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"features": [5.1, 3.5, 1.4, 0.2]}\'')

    app.run(host='0.0.0.0', port=8001, debug=False, threaded=True)