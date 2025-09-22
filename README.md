# MLOps Complete Demo

A comprehensive MLOps demonstration covering the entire machine learning lifecycle with 5 different models, showcasing:

- Data Versioning (DVC)
- Model Versioning & Experiment Tracking (MLflow)
- Model Serving (FastAPI)
- Monitoring & Observability (Prometheus + Grafana)
- CI/CD Pipelines (GitHub Actions)
- Containerization (Docker)
- Data Drift Detection (Evidently AI)
- Automated Testing

## Models Included

1. **Iris Classification** - Scikit-learn multiclass classification
2. **House Price Prediction** - XGBoost regression
3. **Sentiment Analysis** - Transformers NLP
4. **Image Classification** - TensorFlow/Keras computer vision
5. **Customer Churn** - LightGBM binary classification

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd mlops-demo

# Install dependencies
pip install -r requirements.txt

# Start services
docker-compose up -d

# Access services
# MLflow UI: http://localhost:5000
# API Documentation: http://localhost:8000/docs
# Grafana Dashboard: http://localhost:3000
```

## Project Structure

```
mlops-demo/
├── data/                    # Raw and processed datasets
├── models/                  # Model implementations
├── api/                     # FastAPI serving layer
├── monitoring/              # Monitoring and drift detection
├── tests/                   # Test suites
├── docs/                    # Documentation for KT sessions
├── .github/workflows/       # CI/CD pipelines
├── docker/                  # Docker configurations
└── scripts/                 # Utility scripts
```

## Documentation

See the `docs/` directory for detailed documentation and presentation materials for knowledge transfer sessions.