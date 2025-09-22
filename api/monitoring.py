"""
Monitoring and metrics collection for the API
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time
import psutil
import logging

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'model_predictions_total',
    'Total number of model predictions',
    ['model_name', 'prediction_type']
)

PREDICTION_DURATION = Histogram(
    'model_prediction_duration_seconds',
    'Model prediction duration in seconds',
    ['model_name']
)

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Model accuracy score',
    ['model_name']
)

SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_percent',
    'System memory usage percentage'
)

API_INFO = Info(
    'api_info',
    'API information'
)

def setup_monitoring():
    """Setup monitoring and initialize metrics"""
    API_INFO.info({
        'version': '1.0.0',
        'description': 'MLOps Demo API',
        'models': 'iris,house_price,sentiment,churn,image'
    })

    # Initialize model accuracy metrics (these would be updated from actual model performance)
    MODEL_ACCURACY.labels(model_name='iris').set(0.95)
    MODEL_ACCURACY.labels(model_name='house_price').set(0.82)
    MODEL_ACCURACY.labels(model_name='sentiment').set(0.88)
    MODEL_ACCURACY.labels(model_name='churn').set(0.86)
    MODEL_ACCURACY.labels(model_name='image').set(0.78)

    logger.info("Monitoring setup completed")

def update_system_metrics():
    """Update system resource metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent

        SYSTEM_CPU_USAGE.set(cpu_percent)
        SYSTEM_MEMORY_USAGE.set(memory_percent)
    except Exception as e:
        logger.error(f"Error updating system metrics: {e}")

def record_request_metrics(method: str, endpoint: str, status_code: int, duration: float):
    """Record request metrics"""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

def record_prediction_metrics(model_name: str, prediction_type: str, duration: float):
    """Record prediction metrics"""
    PREDICTION_COUNT.labels(model_name=model_name, prediction_type=prediction_type).inc()
    PREDICTION_DURATION.labels(model_name=model_name).observe(duration)

def get_model_metrics():
    """Get aggregated model metrics"""
    return {
        "total_predictions": {
            "iris": PREDICTION_COUNT.labels(model_name='iris', prediction_type='single')._value.get(),
            "house_price": PREDICTION_COUNT.labels(model_name='house_price', prediction_type='single')._value.get(),
            "sentiment": PREDICTION_COUNT.labels(model_name='sentiment', prediction_type='single')._value.get(),
            "churn": PREDICTION_COUNT.labels(model_name='churn', prediction_type='single')._value.get(),
            "image": PREDICTION_COUNT.labels(model_name='image', prediction_type='single')._value.get(),
        },
        "model_accuracy": {
            "iris": MODEL_ACCURACY.labels(model_name='iris')._value.get(),
            "house_price": MODEL_ACCURACY.labels(model_name='house_price')._value.get(),
            "sentiment": MODEL_ACCURACY.labels(model_name='sentiment')._value.get(),
            "churn": MODEL_ACCURACY.labels(model_name='churn')._value.get(),
            "image": MODEL_ACCURACY.labels(model_name='image')._value.get(),
        },
        "system_metrics": {
            "cpu_usage_percent": SYSTEM_CPU_USAGE._value.get(),
            "memory_usage_percent": SYSTEM_MEMORY_USAGE._value.get(),
        },
        "timestamp": time.time()
    }