"""
MLflow configuration for the MLOps demo
"""

import os
import mlflow
from pathlib import Path

# MLflow configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_ARTIFACT_ROOT = "./mlartifacts"
MLFLOW_DEFAULT_ARTIFACT_ROOT = "./mlartifacts"

def setup_mlflow():
    """Setup MLflow tracking"""
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Create artifacts directory
    Path(MLFLOW_ARTIFACT_ROOT).mkdir(exist_ok=True)

    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"MLflow artifacts root: {MLFLOW_ARTIFACT_ROOT}")

def get_latest_model_version(model_name: str, stage: str = "Production"):
    """Get the latest model version from MLflow Model Registry"""
    client = mlflow.tracking.MlflowClient()
    try:
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        if latest_version:
            return latest_version[0].version
        else:
            # If no production model, get the latest version regardless of stage
            all_versions = client.search_model_versions(f"name='{model_name}'")
            if all_versions:
                return max([int(v.version) for v in all_versions])
    except Exception as e:
        print(f"Error getting latest model version for {model_name}: {e}")
    return None

def promote_model_to_production(model_name: str, version: str):
    """Promote a model version to production"""
    client = mlflow.tracking.MlflowClient()
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        print(f"Promoted {model_name} version {version} to Production")
    except Exception as e:
        print(f"Error promoting model to production: {e}")

def archive_model_version(model_name: str, version: str):
    """Archive a model version"""
    client = mlflow.tracking.MlflowClient()
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Archived"
        )
        print(f"Archived {model_name} version {version}")
    except Exception as e:
        print(f"Error archiving model: {e}")

if __name__ == "__main__":
    setup_mlflow()