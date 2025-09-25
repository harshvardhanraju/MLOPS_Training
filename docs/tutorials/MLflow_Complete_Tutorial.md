# MLflow Complete Tutorial: Experiment Tracking & Model Management

## Table of Contents

1. [Introduction to MLflow](#introduction-to-mlflow)
2. [MLflow Architecture Overview](#mlflow-architecture-overview)
3. [Setting Up MLflow](#setting-up-mlflow)
4. [MLflow Tracking - Experiment Management](#mlflow-tracking---experiment-management)
5. [MLflow Models - Model Packaging & Serving](#mlflow-models---model-packaging--serving)
6. [MLflow Model Registry - Model Lifecycle](#mlflow-model-registry---model-lifecycle)
7. [MLflow Projects - Reproducible ML Code](#mlflow-projects---reproducible-ml-code)
8. [Advanced MLflow Features](#advanced-mlflow-features)
9. [MLflow UI Deep Dive](#mlflow-ui-deep-dive)
10. [Integration Patterns](#integration-patterns)
11. [Production Best Practices](#production-best-practices)
12. [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Introduction to MLflow

### What is MLflow?

**MLflow** is an open-source platform designed to manage the complete machine learning lifecycle. It addresses four primary areas:

1. **Experiment Tracking**: Log and compare experiments and runs
2. **Model Packaging**: Package ML models for deployment
3. **Model Registry**: Centralized model store with versioning and stage transitions
4. **Project Management**: Reproducible and reusable ML workflows

### Why MLflow is Essential for MLOps

#### The Problem MLflow Solves

Before MLflow, data science teams faced several challenges:

- **Experiment Chaos**: No standardized way to track experiments
- **Reproducibility Issues**: Difficulty recreating successful experiments
- **Model Deployment Complexity**: Different deployment patterns for different frameworks
- **Collaboration Barriers**: Hard to share and compare work across team members
- **Version Control Nightmares**: No clear model versioning strategy

#### MLflow's Solution Approach

```
┌─────────────────────────────────────────────────────────────┐
│                    MLflow Ecosystem                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   MLflow    │    │   MLflow    │    │   MLflow    │     │
│  │  Tracking   │────│   Models    │────│  Registry   │     │
│  │             │    │             │    │             │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                   │                   │          │
│         └───────────────────┼───────────────────┘          │
│                             │                              │
│                    ┌─────────────┐                         │
│                    │   MLflow    │                         │
│                    │  Projects   │                         │
│                    │             │                         │
│                    └─────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Benefits

- **🔍 Experiment Tracking**: Systematic logging of parameters, metrics, and artifacts
- **🔄 Reproducibility**: Consistent environment and dependency management
- **📦 Model Packaging**: Framework-agnostic model packaging and deployment
- **🏷️ Model Versioning**: Centralized model registry with lifecycle management
- **👥 Team Collaboration**: Shared experiment tracking and model sharing
- **🚀 Deployment Flexibility**: Multiple deployment options (REST API, batch, cloud)

---

## MLflow Architecture Overview

### Core Components Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MLflow Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────┐                                          │
│  │   Client Code     │  (Python, R, Java, REST API)            │
│  │                   │                                          │
│  │  ┌─────────────┐  │    ┌─────────────────────────────────┐   │
│  │  │   MLflow    │  │    │        MLflow Server            │   │
│  │  │   Client    │◄─┼────┤                                 │   │
│  │  │             │  │    │  ┌─────────────┐                │   │
│  │  └─────────────┘  │    │  │ Tracking    │                │   │
│  │                   │    │  │   Server    │                │   │
│  └───────────────────┘    │  └─────────────┘                │   │
│                           │         │                       │   │
│                           │  ┌─────────────┐                │   │
│                           │  │   Model     │                │   │
│                           │  │  Registry   │                │   │
│                           │  └─────────────┘                │   │
│                           └─────────────────────────────────┘   │
│                                      │                          │
│                           ┌─────────────────────────────────┐   │
│                           │       Storage Layer             │   │
│                           │                                 │   │
│                           │  ┌─────────────┐                │   │
│                           │  │ Metadata    │                │   │
│                           │  │Store (DB)   │                │   │
│                           │  └─────────────┘                │   │
│                           │                                 │   │
│                           │  ┌─────────────┐                │   │
│                           │  │ Artifact    │                │   │
│                           │  │Store (S3/FS)│                │   │
│                           │  └─────────────┘                │   │
│                           └─────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Storage Backend Options

#### Metadata Store (Backend Store)
- **Local File System**: `file:///path/to/mlruns`
- **SQLAlchemy**: PostgreSQL, MySQL, SQLite
- **Example**: `postgresql://user:password@host:port/database`

#### Artifact Store
- **Local File System**: `/path/to/artifacts`
- **Amazon S3**: `s3://bucket-name/path`
- **Azure Blob Storage**: `abfss://container@account.dfs.core.windows.net/path`
- **Google Cloud Storage**: `gs://bucket-name/path`

---

## Setting Up MLflow

### Installation Options

#### Basic Installation
```bash
# Install MLflow
pip install mlflow

# Install with extra dependencies
pip install mlflow[extras]

# Install with specific database support
pip install mlflow[postgres]
pip install mlflow[mysql]
```

#### Docker Installation
```bash
# Pull MLflow Docker image
docker pull mlflow/mlflow

# Run MLflow server in Docker
docker run -p 5000:5000 mlflow/mlflow mlflow server --host 0.0.0.0
```

### Configuration Options

#### Local Setup (Development)
```bash
# Start local MLflow server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

#### Production Setup with PostgreSQL and S3
```bash
# Production MLflow server
mlflow server \
  --backend-store-uri postgresql://user:password@host:5432/mlflow_db \
  --default-artifact-root s3://mlflow-bucket/artifacts \
  --host 0.0.0.0 \
  --port 5000
```

#### Environment Variables Configuration
```bash
# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Set default experiment
export MLFLOW_EXPERIMENT_NAME=my-experiment

# AWS credentials for S3 artifact store
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
```

### Docker Compose Setup for Production

```yaml
# docker-compose.yml for MLflow with PostgreSQL
version: '3.8'

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow123
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  mlflow:
    image: mlflow/mlflow:latest
    environment:
      - BACKEND_STORE_URI=postgresql://mlflow:mlflow123@postgres:5432/mlflow
      - DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    ports:
      - "5000:5000"
    depends_on:
      - postgres
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:mlflow123@postgres:5432/mlflow
      --default-artifact-root s3://mlflow-artifacts
      --host 0.0.0.0
      --port 5000

volumes:
  postgres_data:
```

---

## MLflow Tracking - Experiment Management

### Core Concepts

#### Experiments
- **Purpose**: Group related runs together
- **Use Case**: Different projects, models, or approaches
- **Default**: MLflow creates a "Default" experiment

#### Runs
- **Purpose**: Individual execution of your ML code
- **Contains**: Parameters, metrics, artifacts, metadata
- **Lifecycle**: Start → Log data → End

#### Parameters
- **Purpose**: Input values that don't change during run
- **Examples**: learning_rate, n_estimators, batch_size
- **Type**: Key-value pairs (string keys, basic value types)

#### Metrics
- **Purpose**: Values that can change during run execution
- **Examples**: accuracy, loss, precision, recall
- **Type**: Numeric values with optional step/timestamp

#### Artifacts
- **Purpose**: Output files from your runs
- **Examples**: Models, plots, datasets, reports
- **Storage**: File system, S3, Azure, GCS

### Basic Tracking Example

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Set or create experiment
mlflow.set_experiment("iris_classification")

def train_iris_model(n_estimators, max_depth, random_state=42):
    """Train iris classification model with MLflow tracking"""

    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=random_state
    )

    # Start MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", 0.2)

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Log additional metrics for each epoch/step
        for i in range(10):  # Simulate training epochs
            train_loss = np.random.uniform(0.1, 1.0) * (0.9 ** i)
            val_loss = train_loss + np.random.uniform(0.01, 0.1)
            mlflow.log_metric("train_loss", train_loss, step=i)
            mlflow.log_metric("val_loss", val_loss, step=i)

        # Create and log artifacts

        # 1. Feature importance plot
        feature_importance = model.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.bar(iris.feature_names, feature_importance)
        plt.title('Feature Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()

        # 2. Confusion matrix
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=iris.target_names,
                   yticklabels=iris.target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # 3. Model predictions sample
        predictions_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'correct': y_test == y_pred
        })
        predictions_df.to_csv("predictions.csv", index=False)
        mlflow.log_artifact("predictions.csv")

        # 4. Log model
        mlflow.sklearn.log_model(
            model,
            "iris_model",
            registered_model_name="iris_classifier",
            input_example=X_train[:5],
            signature=mlflow.models.infer_signature(X_train, y_pred)
        )

        # Log tags for better organization
        mlflow.set_tag("model_type", "random_forest")
        mlflow.set_tag("dataset", "iris")
        mlflow.set_tag("framework", "sklearn")
        mlflow.set_tag("developer", "ml_engineer")

        # Log run information
        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")
        print(f"Accuracy: {accuracy:.4f}")

        return model, run.info.run_id

# Run experiments with different parameters
experiments = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 150, "max_depth": 7},
    {"n_estimators": 200, "max_depth": 10}
]

print("Running MLflow experiments...")
for i, params in enumerate(experiments):
    print(f"\nExperiment {i+1}: {params}")
    model, run_id = train_iris_model(**params)
```

### Advanced Tracking Features

#### Custom Metrics and Parameters

```python
# Log complex parameters
hyperparameters = {
    "learning_rate": 0.01,
    "optimizer": "adam",
    "layer_sizes": [128, 64, 32],
    "dropout_rate": 0.2,
    "batch_size": 32
}

with mlflow.start_run():
    # Log nested parameters
    for key, value in hyperparameters.items():
        if isinstance(value, list):
            mlflow.log_param(key, str(value))
        else:
            mlflow.log_param(key, value)

    # Log metrics with custom steps
    for epoch in range(100):
        train_accuracy = 0.7 + 0.3 * (1 - np.exp(-epoch/20))
        val_accuracy = train_accuracy - 0.1 + 0.05 * np.sin(epoch/10)

        mlflow.log_metrics({
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "epoch": epoch
        }, step=epoch)

    # Log system metrics
    import psutil
    mlflow.log_metric("cpu_percent", psutil.cpu_percent())
    mlflow.log_metric("memory_percent", psutil.virtual_memory().percent)
```

#### Nested Runs (Parent-Child Relationships)

```python
# Parent run for hyperparameter tuning
with mlflow.start_run(run_name="hyperparameter_tuning") as parent_run:
    mlflow.log_param("tuning_strategy", "grid_search")
    mlflow.log_param("search_space_size", len(experiments))

    best_accuracy = 0
    best_run_id = None

    for i, params in enumerate(experiments):
        # Child run for each experiment
        with mlflow.start_run(run_name=f"experiment_{i+1}", nested=True) as child_run:
            model, _ = train_iris_model(**params)

            # Get accuracy from logged metrics
            client = mlflow.tracking.MlflowClient()
            run_data = client.get_run(child_run.info.run_id)
            accuracy = run_data.data.metrics.get("accuracy", 0)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_run_id = child_run.info.run_id

    # Log best results to parent run
    mlflow.log_metric("best_accuracy", best_accuracy)
    mlflow.log_param("best_run_id", best_run_id)
    mlflow.set_tag("tuning_complete", "true")
```

#### Automatic Logging

```python
# Enable autologging for sklearn
mlflow.sklearn.autolog()

# Train model - MLflow automatically logs parameters, metrics, and model
model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# Enable autologging for specific frameworks
import tensorflow as tf
mlflow.tensorflow.autolog()

import pytorch_lightning as pl
mlflow.pytorch.autolog()

import xgboost as xgb
mlflow.xgboost.autolog()
```

---

## MLflow UI Deep Dive

### Main Dashboard

When you navigate to `http://localhost:5000`, you'll see the MLflow UI with several key sections:

#### 1. Experiments List View
```
┌─────────────────────────────────────────────────────────────────┐
│                    MLflow Experiments                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─ Search Experiments ─────────────────────────────────────┐   │
│  │  [Search box]  [Filter] [Sort]                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Experiment Name    │ # Runs │ Last Updated      │ Actions      │
│  ─────────────────────────────────────────────────────────────│ │
│  iris_classification │   15   │ 2 hours ago      │ [View][Del] │ │
│  house_pricing      │    8   │ 1 day ago        │ [View][Del] │ │
│  sentiment_analysis │    5   │ 3 days ago       │ [View][Del] │ │
│  Default            │    2   │ 1 week ago       │ [View][Del] │ │
│                                                                 │
│  [Create Experiment]                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Why This View is Important:**
- **Organization**: Easily navigate between different projects/experiments
- **Overview**: Quick stats on number of runs per experiment
- **Management**: Create, delete, and organize experiments
- **Search**: Find specific experiments quickly

#### 2. Experiment Runs View
```
┌─────────────────────────────────────────────────────────────────┐
│              iris_classification - Runs                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─ Search & Filter ─────────────────────────────────────────┐  │
│  │ Search: [accuracy > 0.9]  [Filter Tags] [Date Range]     │  │
│  │ Columns: ☑ metrics.accuracy ☑ params.n_estimators       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Run Name        │Status│ Accuracy │N_Est│Max_D│ Start Time    │ │
│  ────────────────────────────────────────────────────────────│ │
│  ○ run_20240322_1│ ✓   │  0.9667  │ 200 │ 10  │ 2h ago       │ │
│  ○ run_20240322_2│ ✓   │  0.9333  │ 150 │  7  │ 2h ago       │ │
│  ○ run_20240322_3│ ✓   │  0.9000  │ 100 │  5  │ 2h ago       │ │
│  ○ run_20240322_4│ ✓   │  0.8667  │  50 │  3  │ 2h ago       │ │
│                                                                 │
│  [Compare Selected]  [Download CSV]  [Chart View]              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features Explained:**

- **Search & Filter**: Advanced filtering capabilities
  - Metric filters: `accuracy > 0.9`
  - Parameter filters: `params.n_estimators = 100`
  - Tag filters: `tags.model_type = "random_forest"`
  - Date range filters for time-based analysis

- **Column Customization**: Choose which parameters/metrics to display
- **Status Indicators**: Running (⟳), Finished (✓), Failed (✗)
- **Interactive Selection**: Select multiple runs for comparison

#### 3. Individual Run Detail View
```
┌─────────────────────────────────────────────────────────────────┐
│                    Run: run_20240322_1                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Overview] [Parameters] [Metrics] [Artifacts] [Models]        │
│                                                                 │
│  ┌─ Run Info ──────────────────────────────────────────────┐   │
│  │ Run ID: abc123...                                       │   │
│  │ Status: FINISHED                                        │   │
│  │ Start: 2024-03-22 14:30:15                             │   │
│  │ End:   2024-03-22 14:32:48                             │   │
│  │ Duration: 2m 33s                                        │   │
│  │ User: ml_engineer                                       │   │
│  │ Source: /path/to/script.py                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─ Parameters ─────────────────────────────────────────────┐  │
│  │ n_estimators    : 200                                   │  │
│  │ max_depth       : 10                                    │  │
│  │ random_state    : 42                                    │  │
│  │ test_size       : 0.2                                   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Metrics ───────────────────────────────────────────────┐   │
│  │ accuracy        : 0.9667                               │   │
│  │ precision       : 0.9672                               │   │
│  │ recall          : 0.9667                               │   │
│  │ train_loss      : [View Chart]                         │   │
│  │ val_loss        : [View Chart]                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Why Each Section Matters:**

- **Run Info**: Reproducibility and debugging information
- **Parameters**: Understand experiment configuration
- **Metrics**: Evaluate model performance
- **Artifacts**: Access generated files, plots, and datasets
- **Models**: Direct access to trained models

#### 4. Metrics Visualization
```
┌─────────────────────────────────────────────────────────────────┐
│                    Metrics: train_loss                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Loss ▲                                                        │
│   1.0 │\                                                       │
│       │ \                                                      │
│   0.8 │  \                                                     │
│       │   \                                                    │
│   0.6 │    \                                                   │
│       │     \___                                               │
│   0.4 │         \___                                           │
│       │             \___                                       │
│   0.2 │                 \_____                                 │
│       │                       \_____________                   │
│   0.0 └─────────────────────────────────────────────────► Step│
│       0    1    2    3    4    5    6    7    8    9         │
│                                                                 │
│  [Linear] [Log Scale] [Download] [Compare Runs]                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Visualization Features:**
- **Interactive Charts**: Zoom, pan, hover for details
- **Scale Options**: Linear vs logarithmic scales
- **Multi-run Comparison**: Overlay multiple runs on same chart
- **Download Options**: Export charts and data

#### 5. Model Registry View
```
┌─────────────────────────────────────────────────────────────────┐
│                    Model Registry                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Model Name          │ Latest Ver │ Stage      │ Description    │
│  ──────────────────────────────────────────────────────────────│ │
│  iris_classifier     │     v4     │ Production │ RF classifier  │ │
│    ├─ Version 4      │            │ Production │ Best model     │ │
│    ├─ Version 3      │            │ Archived   │ Previous       │ │
│    ├─ Version 2      │            │ Staging    │ Testing        │ │
│    └─ Version 1      │            │ None       │ Initial        │ │
│                                                                 │
│  house_predictor     │     v2     │ Staging    │ Price model    │ │
│    ├─ Version 2      │            │ Staging    │ Current test   │ │
│    └─ Version 1      │            │ Production │ Stable version │ │
│                                                                 │
│  [Register New Model]                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Model Registry Benefits:**
- **Version Control**: Track all model versions
- **Stage Management**: Organize models by deployment stage
- **Collaboration**: Team members can see model status
- **Deployment Ready**: Direct integration with deployment tools

---

## MLflow Models - Model Packaging & Serving

### Model Format and Structure

MLflow Models are saved in a standardized directory structure:

```
model/
├── MLmodel                  # Model metadata file
├── model.pkl               # Serialized model (format varies)
├── conda.yaml             # Conda environment specification
├── python_env.yaml        # Python environment specification
├── requirements.txt        # Python dependencies
└── input_example.json     # Sample input for testing
```

#### MLmodel File Structure
```yaml
# MLmodel file example
artifact_path: iris_model
flavors:
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 1.3.0
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
    python_version: 3.10.0
model_size_bytes: 157824
model_uuid: a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
run_id: abc123def456ghi789
signature:
  inputs: '[{"name": "sepal_length", "type": "double"}, {"name": "sepal_width", "type": "double"}, {"name": "petal_length", "type": "double"}, {"name": "petal_width", "type": "double"}]'
  outputs: '[{"type": "tensor", "tensor-spec": {"dtype": "int64", "shape": [-1]}}]'
time_created: '2024-03-22T14:32:48.123456'
```

### Saving Models with Different Frameworks

#### Scikit-learn Models
```python
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save model with signature and example
signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
input_example = X_train[:5]

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="iris_model",
    signature=signature,
    input_example=input_example,
    registered_model_name="iris_classifier"
)
```

#### TensorFlow/Keras Models
```python
import mlflow.tensorflow
import tensorflow as tf

# Create and train model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, validation_split=0.2)

# Log model
mlflow.tensorflow.log_model(
    model=model,
    artifact_path="keras_model",
    signature=mlflow.models.infer_signature(X_train, model.predict(X_train))
)
```

#### PyTorch Models
```python
import mlflow.pytorch
import torch
import torch.nn as nn

class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train model (simplified)
model = IrisNet()
# ... training code ...

# Log model
mlflow.pytorch.log_model(
    pytorch_model=model,
    artifact_path="pytorch_model",
    signature=signature,
    input_example=input_example
)
```

#### Custom Python Function Models
```python
import mlflow.pyfunc
import pandas as pd

class IrisWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, sklearn_model):
        self.sklearn_model = sklearn_model

    def predict(self, context, model_input):
        """Custom prediction logic"""
        # Preprocessing
        if isinstance(model_input, pd.DataFrame):
            features = model_input.values
        else:
            features = model_input

        # Make prediction
        predictions = self.sklearn_model.predict(features)

        # Post-processing
        class_names = ['setosa', 'versicolor', 'virginica']
        probabilities = self.sklearn_model.predict_proba(features)

        result = []
        for i, pred in enumerate(predictions):
            result.append({
                'prediction': class_names[pred],
                'confidence': float(max(probabilities[i])),
                'probabilities': {
                    class_names[j]: float(prob)
                    for j, prob in enumerate(probabilities[i])
                }
            })

        return pd.DataFrame(result)

# Wrap and log custom model
wrapped_model = IrisWrapper(trained_sklearn_model)

mlflow.pyfunc.log_model(
    artifact_path="custom_iris_model",
    python_model=wrapped_model,
    signature=signature,
    input_example=input_example
)
```

### Model Serving Options

#### 1. Local Model Serving
```bash
# Serve model locally
mlflow models serve \
    -m "models:/iris_classifier/Production" \
    -p 8080 \
    --host 0.0.0.0

# Test the served model
curl -X POST http://localhost:8080/invocations \
    -H 'Content-Type: application/json' \
    -d '{
        "dataframe_records": [
            {"sepal_length": 5.1, "sepal_width": 3.5,
             "petal_length": 1.4, "petal_width": 0.2}
        ]
    }'
```

#### 2. Docker Container Serving
```bash
# Build Docker image for model
mlflow models build-docker \
    -m "models:/iris_classifier/Production" \
    -n iris-model-container

# Run Docker container
docker run -p 8080:8080 iris-model-container

# Alternative: Generate Dockerfile
mlflow models generate-dockerfile \
    -m "models:/iris_classifier/Production" \
    -d ./model-dockerfile
```

#### 3. Cloud Deployment

##### AWS SageMaker
```bash
# Deploy to SageMaker
mlflow sagemaker deploy \
    -a iris-classifier-app \
    -m "models:/iris_classifier/Production" \
    --region-name us-west-2 \
    --mode create
```

##### Azure ML
```bash
# Deploy to Azure ML
mlflow azureml deploy \
    -m "models:/iris_classifier/Production" \
    --model-name iris-classifier \
    --service-name iris-service
```

##### Google Cloud Run
```bash
# Build and deploy to Cloud Run
mlflow models build-docker \
    -m "models:/iris_classifier/Production" \
    -n gcr.io/project-id/iris-model

docker push gcr.io/project-id/iris-model

gcloud run deploy iris-service \
    --image gcr.io/project-id/iris-model \
    --platform managed \
    --region us-central1
```

#### 4. Kubernetes Deployment
```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iris-model-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: iris-model
  template:
    metadata:
      labels:
        app: iris-model
    spec:
      containers:
      - name: iris-model
        image: iris-model-container:latest
        ports:
        - containerPort: 8080
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-server:5000"
---
apiVersion: v1
kind: Service
metadata:
  name: iris-model-service
spec:
  selector:
    app: iris-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

### Model Evaluation and Validation

#### Batch Model Evaluation
```python
import mlflow
from mlflow.models import MetricThreshold

# Define evaluation metrics
def custom_accuracy_metric(eval_df, builtin_metrics):
    """Custom metric function"""
    return {
        "custom_accuracy": (eval_df["prediction"] == eval_df["target"]).mean()
    }

# Load model for evaluation
model_uri = "models:/iris_classifier/Staging"
model = mlflow.pyfunc.load_model(model_uri)

# Evaluate model
results = mlflow.evaluate(
    model=model_uri,
    data=test_data,
    targets="target",
    model_type="classifier",
    evaluators=["default"],
    custom_metrics=[custom_accuracy_metric],
    validation_thresholds={
        "accuracy_score": MetricThreshold(threshold=0.8, min_absolute_change=0.05),
        "precision_score": MetricThreshold(threshold=0.8)
    }
)

print(f"Evaluation results: {results.metrics}")
```

#### A/B Testing Framework
```python
class ABTestManager:
    def __init__(self, model_a_uri, model_b_uri, traffic_split=0.5):
        self.model_a = mlflow.pyfunc.load_model(model_a_uri)
        self.model_b = mlflow.pyfunc.load_model(model_b_uri)
        self.traffic_split = traffic_split
        self.results = {"a": [], "b": []}

    def predict(self, data, user_id=None):
        """Route traffic between models"""
        import hashlib

        # Deterministic routing based on user_id
        if user_id:
            hash_value = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
            use_model_a = (hash_value % 100) < (self.traffic_split * 100)
        else:
            use_model_a = np.random.random() < self.traffic_split

        if use_model_a:
            prediction = self.model_a.predict(data)
            self.results["a"].append({
                "prediction": prediction,
                "timestamp": time.time(),
                "user_id": user_id
            })
            return prediction, "model_a"
        else:
            prediction = self.model_b.predict(data)
            self.results["b"].append({
                "prediction": prediction,
                "timestamp": time.time(),
                "user_id": user_id
            })
            return prediction, "model_b"

    def get_statistics(self):
        """Get A/B test statistics"""
        return {
            "model_a_requests": len(self.results["a"]),
            "model_b_requests": len(self.results["b"]),
            "split_ratio": len(self.results["a"]) /
                          (len(self.results["a"]) + len(self.results["b"]))
        }

# Usage
ab_tester = ABTestManager(
    "models:/iris_classifier/Production",
    "models:/iris_classifier/Staging",
    traffic_split=0.2  # 20% to new model
)

prediction, model_used = ab_tester.predict(test_sample, user_id="user123")
```

---

## MLflow Model Registry - Model Lifecycle

### Model Registry Concepts

The Model Registry provides centralized model management with:

- **Model Versioning**: Automatic versioning of registered models
- **Stage Transitions**: Move models through lifecycle stages
- **Model Lineage**: Track model source and dependencies
- **Annotations**: Add descriptions and metadata
- **Access Control**: Manage permissions (MLflow Enterprise)

### Model Lifecycle Stages

```
┌─────────────────────────────────────────────────────────────────┐
│                   Model Lifecycle Stages                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  None ──────► Staging ──────► Production ──────► Archived      │
│    │                              │                             │
│    │                              │                             │
│    └──────────────────────────────┘                             │
│                                                                 │
│  Stage Descriptions:                                            │
│  • None: Initial state, not yet promoted                       │
│  • Staging: Model under testing, pre-production validation     │
│  • Production: Model actively serving production traffic       │
│  • Archived: Deprecated model, kept for historical reference   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Registering Models Programmatically

#### During Training (Automatic Registration)
```python
import mlflow
import mlflow.sklearn

# Train and automatically register model
with mlflow.start_run() as run:
    # Training code...
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Calculate validation metrics
    val_accuracy = accuracy_score(y_val, model.predict(X_val))
    mlflow.log_metric("val_accuracy", val_accuracy)

    # Conditional registration based on performance
    if val_accuracy > 0.85:
        mlflow.sklearn.log_model(
            model,
            "iris_model",
            registered_model_name="iris_classifier",
            signature=mlflow.models.infer_signature(X_train, model.predict(X_train))
        )
        print(f"Model registered with validation accuracy: {val_accuracy:.4f}")
    else:
        print(f"Model not registered - accuracy too low: {val_accuracy:.4f}")
```

#### Post-Training Registration
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register existing model from run
run_id = "abc123def456"  # Run ID from training
model_uri = f"runs:/{run_id}/iris_model"

registered_model = client.create_registered_model(
    name="iris_classifier_v2",
    description="Improved iris classification model with feature engineering"
)

# Create model version
model_version = client.create_model_version(
    name="iris_classifier_v2",
    source=model_uri,
    description="Model trained with enhanced feature set",
    run_id=run_id
)

print(f"Registered model version {model_version.version}")
```

### Managing Model Versions and Stages

#### Stage Transitions
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Promote model to staging
client.transition_model_version_stage(
    name="iris_classifier",
    version=3,
    stage="Staging",
    description="Promoting to staging for validation testing"
)

# After validation, promote to production
client.transition_model_version_stage(
    name="iris_classifier",
    version=3,
    stage="Production",
    description="Validation successful, deploying to production"
)

# Archive old production model
client.transition_model_version_stage(
    name="iris_classifier",
    version=2,
    stage="Archived",
    description="Replaced by version 3"
)
```

#### Model Version Comparison
```python
def compare_model_versions(model_name, version_1, version_2, test_data):
    """Compare two model versions on test data"""

    client = MlflowClient()

    # Load both models
    model_1_uri = f"models:/{model_name}/{version_1}"
    model_2_uri = f"models:/{model_name}/{version_2}"

    model_1 = mlflow.pyfunc.load_model(model_1_uri)
    model_2 = mlflow.pyfunc.load_model(model_2_uri)

    # Make predictions
    pred_1 = model_1.predict(test_data.drop('target', axis=1))
    pred_2 = model_2.predict(test_data.drop('target', axis=1))

    # Calculate metrics
    acc_1 = accuracy_score(test_data['target'], pred_1)
    acc_2 = accuracy_score(test_data['target'], pred_2)

    # Get model metadata
    version_1_info = client.get_model_version(model_name, version_1)
    version_2_info = client.get_model_version(model_name, version_2)

    comparison = {
        "version_1": {
            "version": version_1,
            "accuracy": acc_1,
            "stage": version_1_info.current_stage,
            "created": version_1_info.creation_timestamp
        },
        "version_2": {
            "version": version_2,
            "accuracy": acc_2,
            "stage": version_2_info.current_stage,
            "created": version_2_info.creation_timestamp
        },
        "accuracy_improvement": acc_2 - acc_1
    }

    return comparison

# Compare versions
comparison = compare_model_versions("iris_classifier", "2", "3", test_df)
print(f"Accuracy improvement: {comparison['accuracy_improvement']:.4f}")
```

### Model Registry Webhooks and Automation

#### Setting Up Webhooks for Stage Transitions
```python
from mlflow.utils.rest_utils import http_request
import json

def setup_webhook(registry_webhook_url, model_name):
    """Setup webhook for model stage transitions"""

    webhook_config = {
        "model_name": model_name,
        "events": ["MODEL_VERSION_TRANSITIONED_STAGE"],
        "http_url_spec": {
            "url": registry_webhook_url,
            "authorization": "Bearer your-auth-token"
        }
    }

    # Register webhook (this is conceptual - actual implementation varies)
    response = http_request(
        host_creds=None,
        endpoint="/api/2.0/mlflow/registry-webhooks/create",
        method="POST",
        json=webhook_config
    )

    return response

# Webhook handler example
def handle_model_transition(webhook_data):
    """Handle model stage transition webhook"""

    event = webhook_data.get("event")
    model_name = webhook_data.get("model_name")
    version = webhook_data.get("version")
    to_stage = webhook_data.get("to_stage")

    if to_stage == "Production":
        # Trigger production deployment
        deploy_to_production(model_name, version)

        # Update monitoring dashboards
        update_monitoring_config(model_name, version)

        # Send notification
        send_notification(f"Model {model_name} v{version} deployed to production")

    elif to_stage == "Staging":
        # Run validation tests
        run_validation_tests(model_name, version)

        # Update staging environment
        deploy_to_staging(model_name, version)
```

#### Automated Model Promotion Pipeline
```python
class ModelPromotionPipeline:
    def __init__(self, model_name, test_data):
        self.model_name = model_name
        self.test_data = test_data
        self.client = MlflowClient()

    def validate_model_version(self, version, min_accuracy=0.9):
        """Validate model version meets requirements"""

        model_uri = f"models:/{self.model_name}/{version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Run predictions on test data
        predictions = model.predict(self.test_data.drop('target', axis=1))
        accuracy = accuracy_score(self.test_data['target'], predictions)

        # Additional validation checks
        validation_results = {
            "accuracy": accuracy,
            "meets_threshold": accuracy >= min_accuracy,
            "prediction_distribution": np.bincount(predictions),
            "model_size_mb": self.get_model_size(model_uri)
        }

        return validation_results

    def promote_if_valid(self, version):
        """Promote model version if validation passes"""

        validation = self.validate_model_version(version)

        if validation["meets_threshold"]:
            # Get current production version
            current_prod = self.get_latest_production_version()

            # Promote new version to production
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage="Production",
                description=f"Auto-promoted with accuracy {validation['accuracy']:.4f}"
            )

            # Archive previous production version
            if current_prod:
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=current_prod.version,
                    stage="Archived",
                    description=f"Replaced by version {version}"
                )

            return True, f"Model v{version} promoted to production"
        else:
            return False, f"Model v{version} failed validation (accuracy: {validation['accuracy']:.4f})"

    def get_latest_production_version(self):
        """Get current production model version"""
        versions = self.client.get_latest_versions(
            self.model_name,
            stages=["Production"]
        )
        return versions[0] if versions else None

# Usage
pipeline = ModelPromotionPipeline("iris_classifier", test_data)
success, message = pipeline.promote_if_valid("4")
print(message)
```

---

## MLflow Projects - Reproducible ML Code

### MLproject File Structure

MLflow Projects are defined by an `MLproject` file in the root directory:

```yaml
# MLproject
name: iris-classification-project

python_env: python_env.yaml
# OR: conda_env: conda.yaml

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 5}
      test_size: {type: float, default: 0.2}
      random_state: {type: int, default: 42}
    command: "python train.py --n-estimators {n_estimators} --max-depth {max_depth} --test-size {test_size} --random-state {random_state}"

  evaluate:
    parameters:
      model_uri: {type: str}
      data_path: {type: str, default: "data/test.csv"}
    command: "python evaluate.py --model-uri {model_uri} --data-path {data_path}"

  hyperparameter_tuning:
    parameters:
      search_space: {type: str, default: "config/search_space.json"}
      n_trials: {type: int, default: 50}
    command: "python tune_hyperparameters.py --search-space {search_space} --n-trials {n_trials}"

  data_preparation:
    parameters:
      raw_data_path: {type: str}
      output_path: {type: str, default: "data/processed"}
    command: "python prepare_data.py --raw-data-path {raw_data_path} --output-path {output_path}"
```

### Environment Specification

#### Python Environment (python_env.yaml)
```yaml
# python_env.yaml
python: "3.10.0"
build_dependencies:
  - pip
dependencies:
  - scikit-learn==1.3.0
  - pandas==2.0.0
  - numpy==1.24.0
  - matplotlib==3.7.0
  - seaborn==0.12.0
  - mlflow==2.8.0
  - optuna==3.4.0  # for hyperparameter tuning
  - click==8.1.0   # for CLI
```

#### Conda Environment (conda.yaml)
```yaml
# conda.yaml
name: iris-classification
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - scikit-learn=1.3.0
  - pandas=2.0.0
  - numpy=1.24.0
  - matplotlib=3.7.0
  - seaborn=0.12.0
  - pip
  - pip:
    - mlflow==2.8.0
    - optuna==3.4.0
    - click==8.1.0
```

### Project Structure Example

```
iris-classification-project/
├── MLproject                 # Project definition
├── python_env.yaml          # Python environment
├── conda.yaml               # Alternative: Conda environment
├── train.py                 # Main training script
├── evaluate.py              # Model evaluation script
├── tune_hyperparameters.py  # Hyperparameter tuning
├── prepare_data.py          # Data preparation
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── data_utils.py
│   ├── model_utils.py
│   └── visualization.py
├── config/                  # Configuration files
│   └── search_space.json
├── data/                    # Data directory
│   ├── raw/
│   └── processed/
└── tests/                   # Test files
    ├── test_data_utils.py
    └── test_model_utils.py
```

### Running MLflow Projects

#### Local Project Execution
```bash
# Run main entry point with default parameters
mlflow run .

# Run with custom parameters
mlflow run . -P n_estimators=200 -P max_depth=10

# Run specific entry point
mlflow run . -e evaluate -P model_uri="models:/iris_classifier/Production"

# Run with experiment tracking
mlflow run . --experiment-name iris_experiments
```

#### Remote Project Execution
```bash
# Run from GitHub repository
mlflow run https://github.com/username/iris-classification-project.git

# Run specific commit/branch
mlflow run https://github.com/username/iris-classification-project.git \
    --version main

# Run with parameters
mlflow run https://github.com/username/iris-classification-project.git \
    -P n_estimators=500 -P max_depth=15
```

#### Docker-based Project Execution
```bash
# Run project in Docker container
mlflow run . --backend docker --backend-config '{"image": "python:3.10"}'

# Use custom Docker image
mlflow run . --backend docker --backend-config '{"image": "my-ml-image:latest"}'
```

### Advanced Project Features

#### Parameterized Training Script (train.py)
```python
import click
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.data_utils import load_and_preprocess_data
from src.model_utils import evaluate_model
from src.visualization import create_plots

@click.command()
@click.option("--n-estimators", type=int, default=100, help="Number of trees")
@click.option("--max-depth", type=int, default=5, help="Maximum tree depth")
@click.option("--test-size", type=float, default=0.2, help="Test set proportion")
@click.option("--random-state", type=int, default=42, help="Random seed")
@click.option("--data-path", type=str, default=None, help="Custom data path")
def train(n_estimators, max_depth, test_size, random_state, data_path):
    """Train iris classification model"""

    # Set experiment
    mlflow.set_experiment("iris_classification")

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "test_size": test_size,
            "random_state": random_state
        })

        # Load data
        if data_path:
            X, y = load_and_preprocess_data(data_path)
        else:
            iris = load_iris()
            X, y = iris.data, iris.target

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Create and log visualizations
        plots = create_plots(model, X_test, y_test)
        for plot_name, plot_path in plots.items():
            mlflow.log_artifact(plot_path)

        # Log model
        mlflow.sklearn.log_model(
            model,
            "iris_model",
            signature=mlflow.models.infer_signature(X_train, model.predict(X_train)),
            input_example=X_train[:5]
        )

        print(f"Model trained with accuracy: {metrics['accuracy']:.4f}")
        print(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train()
```

#### Hyperparameter Tuning Entry Point (tune_hyperparameters.py)
```python
import json
import click
import mlflow
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

@click.command()
@click.option("--search-space", type=str, help="Path to search space JSON")
@click.option("--n-trials", type=int, default=50, help="Number of trials")
def tune_hyperparameters(search_space, n_trials):
    """Hyperparameter tuning using Optuna"""

    # Load search space
    with open(search_space, 'r') as f:
        space_config = json.load(f)

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Set experiment
    mlflow.set_experiment("iris_hyperparameter_tuning")

    def objective(trial):
        with mlflow.start_run(nested=True) as run:
            # Suggest parameters
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    space_config["n_estimators"]["low"],
                    space_config["n_estimators"]["high"]
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    space_config["max_depth"]["low"],
                    space_config["max_depth"]["high"]
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split",
                    space_config["min_samples_split"]["low"],
                    space_config["min_samples_split"]["high"]
                )
            }

            # Log parameters
            mlflow.log_params(params)

            # Create and evaluate model
            model = RandomForestClassifier(**params, random_state=42)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

            # Log metrics
            mean_cv_score = cv_scores.mean()
            mlflow.log_metric("cv_accuracy_mean", mean_cv_score)
            mlflow.log_metric("cv_accuracy_std", cv_scores.std())

            return mean_cv_score

    # Run optimization
    with mlflow.start_run() as parent_run:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        # Log best results
        best_params = study.best_params
        best_value = study.best_value

        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_accuracy", best_value)
        mlflow.log_metric("n_trials", n_trials)

        # Train final model with best parameters
        final_model = RandomForestClassifier(**best_params, random_state=42)
        final_model.fit(X, y)

        mlflow.sklearn.log_model(
            final_model,
            "best_model",
            signature=mlflow.models.infer_signature(X, final_model.predict(X))
        )

        print(f"Best parameters: {best_params}")
        print(f"Best CV accuracy: {best_value:.4f}")

if __name__ == "__main__":
    tune_hyperparameters()
```

#### Search Space Configuration (config/search_space.json)
```json
{
  "n_estimators": {
    "low": 10,
    "high": 500
  },
  "max_depth": {
    "low": 1,
    "high": 20
  },
  "min_samples_split": {
    "low": 2,
    "high": 20
  },
  "min_samples_leaf": {
    "low": 1,
    "high": 20
  }
}
```

### Project Workflow Orchestration

#### Multi-step Pipeline
```bash
#!/bin/bash
# run_full_pipeline.sh

echo "Starting MLflow pipeline..."

# Step 1: Data preparation
echo "Step 1: Preparing data..."
mlflow run . -e data_preparation \
    -P raw_data_path=data/raw/iris.csv \
    -P output_path=data/processed

# Step 2: Hyperparameter tuning
echo "Step 2: Hyperparameter tuning..."
TUNING_RUN=$(mlflow run . -e hyperparameter_tuning \
    -P n_trials=100 | grep "Run ID:" | awk '{print $3}')

# Step 3: Train final model with best parameters
echo "Step 3: Training final model..."
TRAIN_RUN=$(mlflow run . -e main \
    -P n_estimators=150 -P max_depth=8 | grep "Run ID:" | awk '{print $3}')

# Step 4: Model evaluation
echo "Step 4: Evaluating model..."
mlflow run . -e evaluate \
    -P model_uri="runs:/${TRAIN_RUN}/iris_model"

echo "Pipeline completed successfully!"
echo "Training run: ${TRAIN_RUN}"
echo "Tuning run: ${TUNING_RUN}"
```

---

## Advanced MLflow Features

### MLflow with Kubernetes

#### Kubernetes Tracking Server Deployment
```yaml
# mlflow-tracking-server.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-tracking-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mlflow-tracking-server
  template:
    metadata:
      labels:
        app: mlflow-tracking-server
    spec:
      containers:
      - name: mlflow-tracking-server
        image: mlflow/mlflow:latest
        command:
          - mlflow
          - server
          - --backend-store-uri
          - postgresql://user:password@postgres:5432/mlflow
          - --default-artifact-root
          - s3://mlflow-artifacts
          - --host
          - 0.0.0.0
          - --port
          - "5000"
        ports:
        - containerPort: 5000
        env:
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: access-key-id
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: secret-access-key
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-tracking-service
spec:
  selector:
    app: mlflow-tracking-server
  ports:
  - protocol: TCP
    port: 5000
    targetPort: 5000
  type: LoadBalancer
```

#### MLflow Job Runner
```yaml
# mlflow-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: mlflow-training-job
spec:
  template:
    spec:
      containers:
      - name: mlflow-job
        image: python:3.10
        command: ["sh", "-c"]
        args:
          - |
            pip install mlflow scikit-learn
            mlflow run https://github.com/user/iris-project.git \
              -P n_estimators=200 -P max_depth=10 \
              --experiment-name kubernetes-experiments
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-tracking-service:5000"
      restartPolicy: Never
  backoffLimit: 3
```

### Custom Authentication and Authorization

#### Custom Authentication Plugin
```python
# custom_auth.py
from mlflow.server import get_app
from mlflow.server.auth import AuthenticationProvider
import jwt
import requests

class CustomAuthProvider(AuthenticationProvider):
    def authenticate_request(self, request):
        """Custom authentication logic"""
        auth_header = request.headers.get('Authorization')

        if not auth_header or not auth_header.startswith('Bearer '):
            return None

        token = auth_header.split(' ')[1]

        try:
            # Verify JWT token with external service
            payload = jwt.decode(token, options={"verify_signature": False})
            user_id = payload.get('user_id')

            # Verify with user service
            response = requests.get(
                f"https://user-service.company.com/verify/{user_id}",
                headers={'Authorization': auth_header}
            )

            if response.status_code == 200:
                return {
                    'user_id': user_id,
                    'username': payload.get('username'),
                    'roles': payload.get('roles', [])
                }
        except Exception as e:
            print(f"Authentication error: {e}")

        return None

# Register custom auth provider
def create_app():
    app = get_app()
    app.auth_provider = CustomAuthProvider()
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000)
```

#### Role-Based Access Control
```python
from functools import wraps
from flask import request, jsonify

def require_role(required_role):
    """Decorator for role-based access control"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user = request.user  # Set by auth middleware

            if not user or required_role not in user.get('roles', []):
                return jsonify({'error': 'Insufficient permissions'}), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Usage in MLflow server extension
@require_role('data_scientist')
def create_experiment(request):
    """Only data scientists can create experiments"""
    pass

@require_role('ml_engineer')
def transition_model_stage(request):
    """Only ML engineers can transition model stages"""
    pass
```

### MLflow with Apache Airflow

#### Airflow DAG for MLflow Pipeline
```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.http.sensors.http import HttpSensor
import mlflow
from mlflow.tracking import MlflowClient

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'mlflow_training_pipeline',
    default_args=default_args,
    description='MLflow training pipeline',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['mlflow', 'machine-learning']
)

def check_data_quality(**context):
    """Check if new data meets quality requirements"""
    # Data quality checks
    data_path = context['params']['data_path']
    # ... quality check logic ...
    return True

def run_mlflow_training(**context):
    """Run MLflow training job"""
    mlflow.set_tracking_uri("http://mlflow-server:5000")

    # Run training
    run_result = mlflow.run(
        uri="https://github.com/company/ml-project.git",
        parameters={
            "n_estimators": context['params']['n_estimators'],
            "max_depth": context['params']['max_depth']
        },
        experiment_name="airflow_experiments"
    )

    return run_result.run_id

def promote_model_if_valid(**context):
    """Promote model to staging if validation passes"""
    run_id = context['task_instance'].xcom_pull(task_ids='train_model')

    client = MlflowClient("http://mlflow-server:5000")
    run = client.get_run(run_id)

    accuracy = run.data.metrics.get('accuracy', 0)

    if accuracy > 0.9:  # Promotion threshold
        # Register and promote model
        model_uri = f"runs:/{run_id}/model"

        model_version = client.create_model_version(
            name="production_model",
            source=model_uri,
            run_id=run_id
        )

        client.transition_model_version_stage(
            name="production_model",
            version=model_version.version,
            stage="Staging"
        )

        return True

    return False

# Define tasks
check_mlflow_server = HttpSensor(
    task_id='check_mlflow_server',
    http_conn_id='mlflow_server',
    endpoint='health',
    timeout=20,
    poke_interval=30,
    dag=dag
)

data_quality_check = PythonOperator(
    task_id='data_quality_check',
    python_callable=check_data_quality,
    params={'data_path': '/data/latest/'},
    dag=dag
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=run_mlflow_training,
    params={
        'n_estimators': 200,
        'max_depth': 10
    },
    dag=dag
)

model_promotion = PythonOperator(
    task_id='model_promotion',
    python_callable=promote_model_if_valid,
    dag=dag
)

# Set task dependencies
check_mlflow_server >> data_quality_check >> train_model >> model_promotion
```

### MLflow Plugins and Extensions

#### Custom MLflow Plugin
```python
# mlflow_custom_plugin.py
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric
import time

class CustomMetricsLogger:
    """Custom plugin for advanced metrics logging"""

    def __init__(self, tracking_uri=None):
        self.client = MlflowClient(tracking_uri)

    def log_system_metrics(self, run_id):
        """Log system performance metrics"""
        import psutil
        import GPUtil

        timestamp = int(time.time() * 1000)

        metrics = [
            Metric("system_cpu_percent", psutil.cpu_percent(), timestamp, 0),
            Metric("system_memory_percent", psutil.virtual_memory().percent, timestamp, 0),
            Metric("system_disk_usage", psutil.disk_usage('/').percent, timestamp, 0)
        ]

        # Add GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                metrics.extend([
                    Metric(f"gpu_{i}_utilization", gpu.load * 100, timestamp, 0),
                    Metric(f"gpu_{i}_memory_util", gpu.memoryUtil * 100, timestamp, 0),
                    Metric(f"gpu_{i}_temperature", gpu.temperature, timestamp, 0)
                ])
        except:
            pass

        self.client.log_batch(run_id, metrics=metrics)

    def log_data_drift_metrics(self, run_id, reference_data, current_data):
        """Log data drift metrics"""
        from scipy.stats import ks_2samp

        timestamp = int(time.time() * 1000)
        drift_metrics = []

        for column in reference_data.columns:
            if reference_data[column].dtype in ['float64', 'int64']:
                ks_stat, p_value = ks_2samp(
                    reference_data[column],
                    current_data[column]
                )

                drift_metrics.extend([
                    Metric(f"drift_{column}_ks_stat", ks_stat, timestamp, 0),
                    Metric(f"drift_{column}_p_value", p_value, timestamp, 0)
                ])

        self.client.log_batch(run_id, metrics=drift_metrics)

# Register plugin
def register_mlflow_plugin():
    """Register custom plugin with MLflow"""
    import mlflow

    # Add custom methods to MLflow
    mlflow.custom_logger = CustomMetricsLogger()

    # Custom autolog decorator
    def custom_autolog(func):
        def wrapper(*args, **kwargs):
            with mlflow.start_run() as run:
                # Log system metrics before training
                mlflow.custom_logger.log_system_metrics(run.info.run_id)

                # Execute original function
                result = func(*args, **kwargs)

                # Log system metrics after training
                mlflow.custom_logger.log_system_metrics(run.info.run_id)

                return result
        return wrapper

    mlflow.custom_autolog = custom_autolog

# Usage
register_mlflow_plugin()

@mlflow.custom_autolog
def train_model():
    # Training code here
    pass
```

---

## Production Best Practices

### High Availability MLflow Setup

#### Load Balancer Configuration (NGINX)
```nginx
# nginx.conf
upstream mlflow_servers {
    server mlflow-server-1:5000 weight=3;
    server mlflow-server-2:5000 weight=3;
    server mlflow-server-3:5000 weight=2;
}

server {
    listen 80;
    server_name mlflow.company.com;

    location / {
        proxy_pass http://mlflow_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Health check
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://mlflow_servers/health;
        access_log off;
    }
}
```

#### Database Connection Pooling
```python
# mlflow_server_config.py
import os
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

def create_mlflow_db_engine():
    """Create database engine with connection pooling"""

    database_url = os.getenv(
        'MLFLOW_DATABASE_URL',
        'postgresql://mlflow:password@postgres:5432/mlflow'
    )

    engine = create_engine(
        database_url,
        poolclass=QueuePool,
        pool_size=20,          # Number of connections to maintain
        max_overflow=30,       # Additional connections when needed
        pool_pre_ping=True,    # Validate connections before use
        pool_recycle=3600,     # Recycle connections every hour
        echo=False             # Set to True for SQL debugging
    )

    return engine

# Environment configuration
os.environ['MLFLOW_BACKEND_STORE_URI'] = 'postgresql://...'
os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = 's3://mlflow-artifacts'
```

### Security Best Practices

#### TLS/SSL Configuration
```yaml
# docker-compose-prod.yml
version: '3.8'

services:
  mlflow-server:
    image: mlflow/mlflow:latest
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://...
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://...
    volumes:
      - ./ssl/cert.pem:/app/cert.pem
      - ./ssl/key.pem:/app/key.pem
    command: >
      mlflow server
      --backend-store-uri postgresql://...
      --default-artifact-root s3://...
      --host 0.0.0.0
      --port 5000
      --ssl-keyfile /app/key.pem
      --ssl-certfile /app/cert.pem

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl/:/etc/nginx/ssl/
    depends_on:
      - mlflow-server
```

#### Secrets Management
```python
# secrets_manager.py
import boto3
import json
from botocore.exceptions import ClientError

class SecretsManager:
    """Manage secrets from AWS Secrets Manager"""

    def __init__(self, region_name='us-west-2'):
        self.client = boto3.client('secretsmanager', region_name=region_name)

    def get_secret(self, secret_name):
        """Retrieve secret from AWS Secrets Manager"""
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            return json.loads(response['SecretString'])
        except ClientError as e:
            print(f"Error retrieving secret {secret_name}: {e}")
            return None

    def setup_mlflow_config(self):
        """Setup MLflow configuration from secrets"""
        secrets = self.get_secret('mlflow-production-secrets')

        if secrets:
            os.environ['MLFLOW_BACKEND_STORE_URI'] = secrets['database_url']
            os.environ['AWS_ACCESS_KEY_ID'] = secrets['aws_access_key']
            os.environ['AWS_SECRET_ACCESS_KEY'] = secrets['aws_secret_key']

# Usage
secrets_manager = SecretsManager()
secrets_manager.setup_mlflow_config()
```

#### Audit Logging
```python
# audit_logger.py
import logging
import json
from datetime import datetime
from mlflow.tracking import MlflowClient

class MLflowAuditLogger:
    """Audit logger for MLflow operations"""

    def __init__(self, log_file='/var/log/mlflow_audit.log'):
        self.logger = logging.getLogger('mlflow_audit')
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_experiment_created(self, user_id, experiment_name, experiment_id):
        """Log experiment creation"""
        audit_entry = {
            'action': 'experiment_created',
            'user_id': user_id,
            'experiment_name': experiment_name,
            'experiment_id': experiment_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(audit_entry))

    def log_model_registered(self, user_id, model_name, version, source_run_id):
        """Log model registration"""
        audit_entry = {
            'action': 'model_registered',
            'user_id': user_id,
            'model_name': model_name,
            'version': version,
            'source_run_id': source_run_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(audit_entry))

    def log_stage_transition(self, user_id, model_name, version, from_stage, to_stage):
        """Log model stage transition"""
        audit_entry = {
            'action': 'stage_transition',
            'user_id': user_id,
            'model_name': model_name,
            'version': version,
            'from_stage': from_stage,
            'to_stage': to_stage,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(audit_entry))

# Integration with MLflow operations
audit_logger = MLflowAuditLogger()

class AuditedMlflowClient(MlflowClient):
    """MLflow client with audit logging"""

    def __init__(self, tracking_uri=None, user_id=None):
        super().__init__(tracking_uri)
        self.user_id = user_id
        self.audit_logger = MLflowAuditLogger()

    def create_experiment(self, name, artifact_location=None, tags=None):
        experiment_id = super().create_experiment(name, artifact_location, tags)
        self.audit_logger.log_experiment_created(
            self.user_id, name, experiment_id
        )
        return experiment_id

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions=False, description=None):
        # Get current stage
        current_version = self.get_model_version(name, version)
        from_stage = current_version.current_stage

        # Perform transition
        result = super().transition_model_version_stage(
            name, version, stage, archive_existing_versions, description
        )

        # Log transition
        self.audit_logger.log_stage_transition(
            self.user_id, name, version, from_stage, stage
        )

        return result
```

### Performance Optimization

#### Database Optimization
```sql
-- PostgreSQL optimization for MLflow metadata store

-- Indexes for common queries
CREATE INDEX CONCURRENTLY idx_runs_experiment_id ON runs(experiment_id);
CREATE INDEX CONCURRENTLY idx_runs_start_time ON runs(start_time);
CREATE INDEX CONCURRENTLY idx_runs_status ON runs(status);
CREATE INDEX CONCURRENTLY idx_metrics_run_uuid ON metrics(run_uuid);
CREATE INDEX CONCURRENTLY idx_metrics_key ON metrics(key);
CREATE INDEX CONCURRENTLY idx_params_run_uuid ON params(run_uuid);

-- Composite indexes for complex queries
CREATE INDEX CONCURRENTLY idx_runs_exp_status_start
ON runs(experiment_id, status, start_time);

CREATE INDEX CONCURRENTLY idx_metrics_run_key_step
ON metrics(run_uuid, key, step);

-- Partitioning for large tables (PostgreSQL 10+)
CREATE TABLE runs_partitioned (
    LIKE runs INCLUDING ALL
) PARTITION BY RANGE (start_time);

CREATE TABLE runs_2024_01 PARTITION OF runs_partitioned
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE runs_2024_02 PARTITION OF runs_partitioned
FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Vacuum and analyze regularly
CREATE OR REPLACE FUNCTION maintain_mlflow_db()
RETURNS void AS $$
BEGIN
    VACUUM ANALYZE runs;
    VACUUM ANALYZE experiments;
    VACUUM ANALYZE metrics;
    VACUUM ANALYZE params;
    VACUUM ANALYZE tags;
END;
$$ LANGUAGE plpgsql;

-- Schedule maintenance (use pg_cron extension)
SELECT cron.schedule('mlflow-maintenance', '0 2 * * *', 'SELECT maintain_mlflow_db();');
```

#### Artifact Storage Optimization
```python
# artifact_optimization.py
import mlflow
import os
from concurrent.futures import ThreadPoolExecutor
import gzip
import pickle

class OptimizedArtifactLogger:
    """Optimized artifact logging for MLflow"""

    def __init__(self, max_workers=4):
        self.max_workers = max_workers

    def log_compressed_artifact(self, obj, artifact_path, compression_level=6):
        """Log artifact with compression"""

        # Serialize and compress
        serialized_data = pickle.dumps(obj)
        compressed_data = gzip.compress(serialized_data, compresslevel=compression_level)

        # Save compressed artifact
        temp_path = f"/tmp/{artifact_path}.pkl.gz"
        with open(temp_path, 'wb') as f:
            f.write(compressed_data)

        # Log to MLflow
        mlflow.log_artifact(temp_path, artifact_path)

        # Cleanup
        os.remove(temp_path)

        print(f"Compressed artifact logged: {artifact_path}")
        print(f"Original size: {len(serialized_data):,} bytes")
        print(f"Compressed size: {len(compressed_data):,} bytes")
        print(f"Compression ratio: {len(serialized_data)/len(compressed_data):.2f}x")

    def log_artifacts_batch(self, artifacts_dict):
        """Log multiple artifacts in parallel"""

        def log_single_artifact(item):
            path, obj = item
            self.log_compressed_artifact(obj, path)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(log_single_artifact, artifacts_dict.items())

    def load_compressed_artifact(self, run_id, artifact_path):
        """Load compressed artifact from MLflow"""

        # Download artifact
        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(run_id, f"{artifact_path}.pkl.gz")

        # Load and decompress
        with open(local_path, 'rb') as f:
            compressed_data = f.read()

        serialized_data = gzip.decompress(compressed_data)
        obj = pickle.loads(serialized_data)

        return obj

# Usage example
optimizer = OptimizedArtifactLogger()

with mlflow.start_run():
    # Log multiple artifacts efficiently
    artifacts = {
        "large_model": trained_model,
        "feature_importance": feature_importance_array,
        "predictions": prediction_results,
        "metadata": model_metadata
    }

    optimizer.log_artifacts_batch(artifacts)
```

### Monitoring and Alerting

#### MLflow Health Monitoring
```python
# health_monitor.py
import requests
import time
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from mlflow.tracking import MlflowClient

class MLflowHealthMonitor:
    """Monitor MLflow server health and performance"""

    def __init__(self, tracking_uri, alert_email=None):
        self.tracking_uri = tracking_uri
        self.alert_email = alert_email
        self.client = MlflowClient(tracking_uri)

    def check_server_health(self):
        """Check if MLflow server is responding"""
        try:
            response = requests.get(f"{self.tracking_uri}/health", timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    def check_database_connectivity(self):
        """Check database connectivity"""
        try:
            experiments = self.client.list_experiments(max_results=1)
            return True
        except Exception as e:
            print(f"Database connectivity check failed: {e}")
            return False

    def check_artifact_store_connectivity(self):
        """Check artifact store connectivity"""
        try:
            # Try to create a temporary run to test artifact store
            with mlflow.start_run() as run:
                mlflow.log_param("health_check", "test")

            # Try to download the logged parameter (indirectly tests artifact store)
            run_data = self.client.get_run(run.info.run_id)
            return "health_check" in run_data.data.params
        except Exception as e:
            print(f"Artifact store connectivity check failed: {e}")
            return False

    def check_performance_metrics(self):
        """Check MLflow performance metrics"""
        try:
            start_time = time.time()

            # Measure response time for common operations
            experiments = self.client.list_experiments(max_results=10)
            experiment_list_time = time.time() - start_time

            start_time = time.time()
            if experiments:
                runs = self.client.search_runs(
                    experiment_ids=[experiments[0].experiment_id],
                    max_results=10
                )
            runs_search_time = time.time() - start_time

            metrics = {
                "experiment_list_time": experiment_list_time,
                "runs_search_time": runs_search_time,
                "total_experiments": len(experiments) if experiments else 0
            }

            return metrics
        except Exception as e:
            print(f"Performance metrics check failed: {e}")
            return {}

    def send_alert(self, subject, message):
        """Send email alert"""
        if not self.alert_email:
            return

        try:
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = 'mlflow-monitor@company.com'
            msg['To'] = self.alert_email

            with smtplib.SMTP('localhost') as server:
                server.send_message(msg)
        except Exception as e:
            print(f"Failed to send alert: {e}")

    def run_health_check(self):
        """Run complete health check"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n=== MLflow Health Check - {timestamp} ===")

        results = {
            "timestamp": timestamp,
            "server_healthy": self.check_server_health(),
            "database_connected": self.check_database_connectivity(),
            "artifact_store_connected": self.check_artifact_store_connectivity(),
            "performance_metrics": self.check_performance_metrics()
        }

        # Print results
        for key, value in results.items():
            if key != "performance_metrics":
                status = "✅ PASS" if value else "❌ FAIL"
                print(f"{key}: {status}")

        # Print performance metrics
        if results["performance_metrics"]:
            print("\nPerformance Metrics:")
            for metric, value in results["performance_metrics"].items():
                print(f"  {metric}: {value}")

        # Check for alerts
        failures = [k for k, v in results.items()
                   if k != "performance_metrics" and k != "timestamp" and not v]

        if failures:
            alert_subject = f"MLflow Health Check Failed - {timestamp}"
            alert_message = f"The following checks failed:\n" + "\n".join(f"- {failure}" for failure in failures)
            self.send_alert(alert_subject, alert_message)

        # Check performance thresholds
        perf = results["performance_metrics"]
        if perf.get("experiment_list_time", 0) > 5.0:  # 5 second threshold
            self.send_alert(
                "MLflow Performance Alert",
                f"Experiment listing taking too long: {perf['experiment_list_time']:.2f}s"
            )

        return results

# Continuous monitoring script
def continuous_monitor():
    monitor = MLflowHealthMonitor(
        tracking_uri="http://mlflow-server:5000",
        alert_email="mlops-team@company.com"
    )

    while True:
        try:
            results = monitor.run_health_check()

            # Log results to file
            with open("/var/log/mlflow_health.log", "a") as f:
                f.write(f"{results}\n")

            # Wait 5 minutes before next check
            time.sleep(300)

        except KeyboardInterrupt:
            print("Monitoring stopped by user")
            break
        except Exception as e:
            print(f"Monitoring error: {e}")
            time.sleep(60)  # Wait 1 minute before retry

if __name__ == "__main__":
    continuous_monitor()
```

---

## Troubleshooting & FAQ

### Common Issues and Solutions

#### 1. Connection Issues

**Problem**: "Connection refused" when accessing MLflow UI
```bash
requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=5000)
```

**Solutions**:
```bash
# Check if MLflow server is running
ps aux | grep mlflow

# Check port binding
netstat -tlnp | grep 5000

# Start MLflow server if not running
mlflow server --host 0.0.0.0 --port 5000

# Check firewall rules
sudo ufw status
sudo iptables -L -n | grep 5000
```

**Docker-specific checks**:
```bash
# Check container status
docker ps | grep mlflow

# Check container logs
docker logs mlflow-container

# Check port mapping
docker port mlflow-container
```

#### 2. Database Issues

**Problem**: "Database connection failed"
```
sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) could not connect to server
```

**Solutions**:
```bash
# Test database connection directly
psql -h localhost -U mlflow -d mlflow_db

# Check database server status
systemctl status postgresql

# Verify connection parameters
echo $MLFLOW_BACKEND_STORE_URI

# Test with telnet
telnet postgres-host 5432
```

**Database maintenance**:
```sql
-- Check database size
SELECT pg_size_pretty(pg_database_size('mlflow_db'));

-- Check table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Clean up old runs (careful!)
DELETE FROM runs WHERE start_time < NOW() - INTERVAL '90 days';
```

#### 3. Artifact Storage Issues

**Problem**: "Artifact download failed"
```
mlflow.exceptions.MlflowException: Unable to download artifacts
```

**S3 troubleshooting**:
```python
# Test S3 connectivity
import boto3
s3_client = boto3.client('s3')
response = s3_client.list_objects_v2(Bucket='mlflow-artifacts', MaxKeys=1)
print(response)

# Check credentials
import os
print("AWS_ACCESS_KEY_ID:", os.getenv('AWS_ACCESS_KEY_ID', 'Not set'))
print("AWS_SECRET_ACCESS_KEY:", "Set" if os.getenv('AWS_SECRET_ACCESS_KEY') else "Not set")

# Test bucket permissions
s3_client.head_bucket(Bucket='mlflow-artifacts')
```

**Local filesystem troubleshooting**:
```bash
# Check permissions
ls -la /path/to/mlruns/

# Check disk space
df -h /path/to/mlruns/

# Fix permissions
chmod -R 755 /path/to/mlruns/
chown -R mlflow:mlflow /path/to/mlruns/
```

#### 4. Model Loading Issues

**Problem**: "Model not found" or "Serialization error"
```
mlflow.exceptions.MlflowException: Model with name 'my_model' not found
```

**Debugging steps**:
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# List all registered models
models = client.list_registered_models()
for model in models:
    print(f"Model: {model.name}")
    for version in model.latest_versions:
        print(f"  Version {version.version}: {version.current_stage}")

# Check specific model version
try:
    model_version = client.get_model_version("my_model", "1")
    print(f"Source: {model_version.source}")
    print(f"Status: {model_version.status}")
except Exception as e:
    print(f"Error: {e}")

# Test model loading
import mlflow.pyfunc
try:
    model = mlflow.pyfunc.load_model("models:/my_model/Production")
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")
```

#### 5. Performance Issues

**Problem**: MLflow UI is slow or unresponsive

**Database optimization**:
```sql
-- Check for missing indexes
SELECT
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE schemaname = 'public'
    AND tablename IN ('runs', 'metrics', 'params')
ORDER BY tablename, attname;

-- Check query performance
EXPLAIN ANALYZE SELECT * FROM runs WHERE experiment_id = '1' ORDER BY start_time DESC LIMIT 100;

-- Add indexes if needed
CREATE INDEX CONCURRENTLY idx_runs_experiment_start ON runs(experiment_id, start_time DESC);
```

**Server configuration**:
```bash
# Increase worker processes for gunicorn
mlflow server \
    --backend-store-uri postgresql://... \
    --default-artifact-root s3://... \
    --workers 4 \
    --host 0.0.0.0 \
    --port 5000
```

### Frequently Asked Questions

#### Q: How do I migrate MLflow data from local to remote storage?

**A: Migration steps:**

```python
# migration_script.py
from mlflow.tracking import MlflowClient
import mlflow
import shutil
import os

def migrate_to_remote():
    # Source client (local)
    source_client = MlflowClient("sqlite:///mlflow.db")

    # Destination client (remote)
    dest_client = MlflowClient("postgresql://user:pass@host:5432/mlflow")

    # Migrate experiments
    source_experiments = source_client.list_experiments()

    for exp in source_experiments:
        print(f"Migrating experiment: {exp.name}")

        # Create experiment in destination
        if exp.name != "Default":
            dest_exp_id = dest_client.create_experiment(exp.name)
        else:
            dest_exp_id = "0"  # Default experiment

        # Migrate runs
        runs = source_client.search_runs([exp.experiment_id])

        for run in runs:
            print(f"  Migrating run: {run.info.run_id}")

            # Create new run in destination
            with mlflow.start_run(experiment_id=dest_exp_id) as new_run:
                # Copy parameters
                for key, value in run.data.params.items():
                    mlflow.log_param(key, value)

                # Copy metrics
                for key, value in run.data.metrics.items():
                    mlflow.log_metric(key, value)

                # Copy tags
                for key, value in run.data.tags.items():
                    mlflow.set_tag(key, value)

                # Copy artifacts
                artifacts = source_client.list_artifacts(run.info.run_id)
                for artifact in artifacts:
                    local_path = source_client.download_artifacts(
                        run.info.run_id,
                        artifact.path
                    )
                    mlflow.log_artifact(local_path)

if __name__ == "__main__":
    migrate_to_remote()
```

#### Q: How do I backup and restore MLflow data?

**A: Backup strategy:**

```bash
#!/bin/bash
# backup_mlflow.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/mlflow_${BACKUP_DATE}"

mkdir -p ${BACKUP_DIR}

# Backup database
pg_dump mlflow_db > ${BACKUP_DIR}/mlflow_db_${BACKUP_DATE}.sql

# Backup artifacts (if using local storage)
if [ -d "/path/to/mlruns" ]; then
    tar -czf ${BACKUP_DIR}/artifacts_${BACKUP_DATE}.tar.gz /path/to/mlruns/
fi

# Backup artifacts from S3
if [ ! -z "$MLFLOW_S3_BUCKET" ]; then
    aws s3 sync s3://${MLFLOW_S3_BUCKET}/ ${BACKUP_DIR}/s3_artifacts/
fi

# Create backup metadata
cat > ${BACKUP_DIR}/backup_info.txt << EOF
Backup Date: ${BACKUP_DATE}
MLflow Version: $(mlflow --version)
Database: PostgreSQL
Artifact Store: ${MLFLOW_DEFAULT_ARTIFACT_ROOT:-"local"}
EOF

echo "Backup completed: ${BACKUP_DIR}"
```

**Restore procedure:**
```bash
#!/bin/bash
# restore_mlflow.sh

BACKUP_DIR=$1

if [ -z "$BACKUP_DIR" ]; then
    echo "Usage: $0 <backup_directory>"
    exit 1
fi

# Restore database
psql mlflow_db < ${BACKUP_DIR}/mlflow_db_*.sql

# Restore artifacts
if [ -f "${BACKUP_DIR}/artifacts_*.tar.gz" ]; then
    tar -xzf ${BACKUP_DIR}/artifacts_*.tar.gz -C /
fi

# Restore S3 artifacts
if [ -d "${BACKUP_DIR}/s3_artifacts" ]; then
    aws s3 sync ${BACKUP_DIR}/s3_artifacts/ s3://${MLFLOW_S3_BUCKET}/
fi

echo "Restore completed from: ${BACKUP_DIR}"
```

#### Q: How do I handle MLflow in a multi-tenant environment?

**A: Multi-tenant setup:**

```python
# multi_tenant_mlflow.py
from mlflow.tracking import MlflowClient
import mlflow
from functools import wraps

class TenantAwareMLflowClient:
    """MLflow client with tenant isolation"""

    def __init__(self, tracking_uri, tenant_id):
        self.client = MlflowClient(tracking_uri)
        self.tenant_id = tenant_id
        self.tenant_prefix = f"tenant_{tenant_id}_"

    def create_experiment(self, name, **kwargs):
        """Create tenant-specific experiment"""
        tenant_name = f"{self.tenant_prefix}{name}"
        return self.client.create_experiment(tenant_name, **kwargs)

    def list_experiments(self):
        """List only tenant's experiments"""
        all_experiments = self.client.list_experiments()
        return [exp for exp in all_experiments
                if exp.name.startswith(self.tenant_prefix)]

    def get_experiment_by_name(self, name):
        """Get tenant-specific experiment"""
        tenant_name = f"{self.tenant_prefix}{name}"
        return self.client.get_experiment_by_name(tenant_name)

def tenant_context(tenant_id):
    """Context manager for tenant isolation"""
    class TenantContext:
        def __init__(self, tenant_id):
            self.tenant_id = tenant_id
            self.original_client = None

        def __enter__(self):
            # Replace default MLflow client
            self.original_client = mlflow.tracking._get_store()
            tenant_client = TenantAwareMLflowClient(
                mlflow.get_tracking_uri(),
                self.tenant_id
            )
            mlflow.tracking._tracking_service = tenant_client
            return tenant_client

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore original client
            mlflow.tracking._tracking_service = self.original_client

    return TenantContext(tenant_id)

# Usage
with tenant_context("company_a") as client:
    mlflow.set_experiment("ml_project")  # Creates "tenant_company_a_ml_project"

    with mlflow.start_run():
        mlflow.log_param("algorithm", "random_forest")
        mlflow.log_metric("accuracy", 0.95)
```

#### Q: How do I set up MLflow with custom authentication?

**A: Custom authentication implementation:**

```python
# custom_auth_server.py
from flask import Flask, request, jsonify
from mlflow.server import get_app
import jwt
import os

class CustomAuthenticatedApp:
    def __init__(self, secret_key):
        self.secret_key = secret_key
        self.mlflow_app = get_app()

    def authenticate_request(self, request):
        """Authenticate incoming request"""
        auth_header = request.headers.get('Authorization', '')

        if not auth_header.startswith('Bearer '):
            return False, "Missing or invalid authorization header"

        token = auth_header.split(' ')[1]

        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            request.user = {
                'user_id': payload.get('user_id'),
                'username': payload.get('username'),
                'roles': payload.get('roles', [])
            }
            return True, None
        except jwt.ExpiredSignatureError:
            return False, "Token has expired"
        except jwt.InvalidTokenError:
            return False, "Invalid token"

    def create_app(self):
        """Create Flask app with authentication"""
        app = Flask(__name__)

        @app.before_request
        def before_request():
            # Skip authentication for login endpoint
            if request.endpoint == 'login':
                return

            is_authenticated, error = self.authenticate_request(request)

            if not is_authenticated:
                return jsonify({'error': error}), 401

        @app.route('/login', methods=['POST'])
        def login():
            """Login endpoint to get JWT token"""
            username = request.json.get('username')
            password = request.json.get('password')

            # Validate credentials (implement your logic)
            if self.validate_credentials(username, password):
                token = jwt.encode(
                    {
                        'user_id': username,
                        'username': username,
                        'roles': self.get_user_roles(username)
                    },
                    self.secret_key,
                    algorithm='HS256'
                )
                return jsonify({'token': token})
            else:
                return jsonify({'error': 'Invalid credentials'}), 401

        # Mount MLflow app
        app.register_blueprint(self.mlflow_app, url_prefix='/api/2.0/mlflow')

        return app

    def validate_credentials(self, username, password):
        """Implement your credential validation logic"""
        # This is a simple example - use your actual authentication system
        valid_users = {
            'data_scientist': 'password123',
            'ml_engineer': 'secure_password'
        }
        return valid_users.get(username) == password

    def get_user_roles(self, username):
        """Get user roles"""
        user_roles = {
            'data_scientist': ['read', 'experiment_create'],
            'ml_engineer': ['read', 'write', 'model_deploy']
        }
        return user_roles.get(username, [])

# Start authenticated MLflow server
if __name__ == "__main__":
    secret_key = os.getenv('JWT_SECRET_KEY', 'your-secret-key-here')
    auth_app = CustomAuthenticatedApp(secret_key)
    app = auth_app.create_app()
    app.run(host='0.0.0.0', port=5000)
```

**Client-side authentication:**
```python
# authenticated_client.py
import requests
import mlflow
from mlflow.tracking.client import MlflowClient

class AuthenticatedMlflowClient:
    def __init__(self, tracking_uri, username, password):
        self.tracking_uri = tracking_uri
        self.token = self._login(username, password)
        self.client = MlflowClient(tracking_uri)

        # Set authentication headers
        import mlflow.tracking._tracking_service.utils
        original_request = mlflow.tracking._tracking_service.utils._get_request_header

        def authenticated_request(*args, **kwargs):
            headers = original_request(*args, **kwargs)
            headers['Authorization'] = f'Bearer {self.token}'
            return headers

        mlflow.tracking._tracking_service.utils._get_request_header = authenticated_request

    def _login(self, username, password):
        """Get JWT token from MLflow server"""
        response = requests.post(
            f"{self.tracking_uri}/login",
            json={'username': username, 'password': password}
        )

        if response.status_code == 200:
            return response.json()['token']
        else:
            raise Exception(f"Authentication failed: {response.json()}")

# Usage
auth_client = AuthenticatedMlflowClient(
    'http://localhost:5000',
    'data_scientist',
    'password123'
)

# Now use MLflow normally - all requests will be authenticated
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('authenticated_experiment')

with mlflow.start_run():
    mlflow.log_param('authenticated', True)
    mlflow.log_metric('test_metric', 1.0)
```

---

This completes the comprehensive MLflow tutorial. The document covers everything from basic setup to advanced production configurations, providing practical examples and real-world solutions for common challenges.

Would you like me to proceed with creating the Prometheus and Grafana tutorial document next?