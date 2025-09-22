FROM python:3.9-slim

# Install MLflow and dependencies
RUN pip install mlflow==2.6.0 psycopg2-binary boto3

# Create MLflow directories
RUN mkdir -p /mlflow/mlruns /mlflow/mlartifacts

# Set working directory
WORKDIR /mlflow

# Expose MLflow port
EXPOSE 5000

# Set environment variables
ENV MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/mlartifacts

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start MLflow server
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/mlflow/mlartifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]