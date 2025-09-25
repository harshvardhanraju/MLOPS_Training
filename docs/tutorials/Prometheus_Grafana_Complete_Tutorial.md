# Prometheus & Grafana Complete Tutorial: Monitoring & Observability for MLOps

## Table of Contents

1. [Introduction to Prometheus & Grafana](#introduction-to-prometheus--grafana)
2. [Architecture Overview](#architecture-overview)
3. [Setting Up Prometheus](#setting-up-prometheus)
4. [Setting Up Grafana](#setting-up-grafana)
5. [Prometheus Fundamentals](#prometheus-fundamentals)
6. [PromQL - Prometheus Query Language](#promql---prometheus-query-language)
7. [Grafana Dashboard Creation](#grafana-dashboard-creation)
8. [MLOps-Specific Monitoring](#mlops-specific-monitoring)
9. [Advanced Features & Integrations](#advanced-features--integrations)
10. [Production Best Practices](#production-best-practices)
11. [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Introduction to Prometheus & Grafana

### What is Prometheus?

**Prometheus** is an open-source monitoring and alerting toolkit designed for reliability and scalability. It excels at collecting and storing time-series data as metrics.

#### Key Features:
- **Time Series Database**: Efficient storage and querying of metrics
- **Pull-based Model**: Actively scrapes metrics from configured targets
- **PromQL**: Powerful query language for metrics analysis
- **Service Discovery**: Automatic discovery of monitoring targets
- **Alerting**: Built-in alerting capabilities with Alertmanager
- **Multi-dimensional Data**: Labels for flexible metric organization

### What is Grafana?

**Grafana** is an open-source analytics and interactive visualization web application. It provides charts, graphs, and alerts when connected to supported data sources.

#### Key Features:
- **Rich Visualizations**: Multiple chart types and dashboard layouts
- **Data Source Agnostic**: Supports 60+ data sources
- **Dashboard Sharing**: Team collaboration and dashboard templates
- **Alerting**: Advanced alerting with multiple notification channels
- **User Management**: Role-based access control and teams
- **Plugin Ecosystem**: Extensive community plugins

### Why Prometheus & Grafana for MLOps?

#### The Monitoring Challenge in MLOps

```
┌─────────────────────────────────────────────────────────────────┐
│                    MLOps Monitoring Challenges                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Model Performance     │  Infrastructure      │  Data Quality   │
│  • Accuracy drift     │  • CPU/Memory usage  │  • Schema changes│
│  • Latency issues     │  • Request rates     │  • Data drift    │
│  • Error rates        │  • Service health    │  • Missing values│
│  • Prediction dist.   │  • Scaling needs     │  • Outliers      │
│                        │                      │                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Prometheus & Grafana Solution

- **Comprehensive Metrics**: Track model, infrastructure, and business metrics
- **Real-time Monitoring**: Immediate visibility into system health
- **Historical Analysis**: Trend analysis and capacity planning
- **Alerting**: Proactive issue detection and notification
- **Scalability**: Handle metrics from distributed ML systems

---

## Architecture Overview

### Complete Monitoring Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                   Prometheus & Grafana Stack                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────┐    ┌───────────────────┐                │
│  │   ML Applications │    │  Infrastructure   │                │
│  │                   │    │                   │                │
│  │  ┌─────────────┐  │    │  ┌─────────────┐  │                │
│  │  │   FastAPI   │  │    │  │Node Exporter│  │                │
│  │  │   Metrics   │  │    │  │   Metrics   │  │                │
│  │  └─────────────┘  │    │  └─────────────┘  │                │
│  │         │         │    │         │         │                │
│  └─────────┼─────────┘    └─────────┼─────────┘                │
│            │                        │                          │
│            └────────────┬───────────┘                          │
│                         │                                      │
│                ┌─────────────────┐                             │
│                │   Prometheus    │                             │
│                │     Server      │                             │
│                │                 │                             │
│                │ ┌─────────────┐ │                             │
│                │ │Time Series  │ │                             │
│                │ │  Database   │ │                             │
│                │ └─────────────┘ │                             │
│                └─────────────────┘                             │
│                         │                                      │
│                ┌─────────────────┐                             │
│                │     Grafana     │                             │
│                │   Dashboards    │                             │
│                │                 │                             │
│                │ ┌─────────────┐ │    ┌─────────────────┐      │
│                │ │ Visualization│◄┼────┤   Alert Manager │      │
│                │ │  & Alerting │ │    │                 │      │
│                │ └─────────────┘ │    └─────────────────┘      │
│                └─────────────────┘                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Metric Generation**: Applications expose metrics endpoints
2. **Metric Collection**: Prometheus scrapes metrics at regular intervals
3. **Data Storage**: Time-series data stored in Prometheus database
4. **Querying**: Grafana queries Prometheus using PromQL
5. **Visualization**: Dashboards display metrics as charts and graphs
6. **Alerting**: Rules trigger alerts based on metric thresholds

### Component Interactions

```
Application Metrics ──┐
                      │
Infrastructure Metrics─┼──► Prometheus ──► Grafana ──► Dashboards
                      │      (Storage)     (Queries)    (Visualization)
Custom Metrics ───────┘           │
                                  │
                                  ▼
                            Alert Manager ──► Notifications
                            (Rules Engine)    (Email, Slack, etc.)
```

---

## Setting Up Prometheus

### Installation Options

#### 1. Binary Installation
```bash
# Download Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.47.0/prometheus-2.47.0.linux-amd64.tar.gz

# Extract
tar xvfz prometheus-2.47.0.linux-amd64.tar.gz
cd prometheus-2.47.0.linux-amd64

# Run Prometheus
./prometheus --config.file=prometheus.yml
```

#### 2. Docker Installation
```bash
# Run Prometheus in Docker
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus:latest
```

#### 3. Docker Compose Setup
```yaml
# docker-compose.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/rules:/etc/prometheus/rules
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'

volumes:
  prometheus_data:
```

### Prometheus Configuration

#### Basic Configuration File (prometheus.yml)
```yaml
# prometheus.yml
global:
  scrape_interval: 15s      # Default scrape interval
  evaluation_interval: 15s  # Rule evaluation interval

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

# Load rules once and periodically evaluate them
rule_files:
  - "rules/*.yml"

# Scrape configurations
scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 5s

  # Node Exporter for system metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
    scrape_interval: 10s

  # MLOps API metrics
  - job_name: 'mlops-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  # MLflow metrics
  - job_name: 'mlflow'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  # Custom application metrics
  - job_name: 'ml-models'
    static_configs:
      - targets: ['localhost:8001', 'localhost:8002']
    metrics_path: '/api/metrics'
    scrape_interval: 15s
```

#### Advanced Configuration Features

##### Service Discovery
```yaml
# EC2 Service Discovery
scrape_configs:
  - job_name: 'ec2-discovery'
    ec2_sd_configs:
      - region: us-west-2
        port: 9100
        filters:
          - name: tag:Environment
            values: [production]
    relabel_configs:
      - source_labels: [__meta_ec2_instance_id]
        target_label: instance_id
      - source_labels: [__meta_ec2_tag_Name]
        target_label: instance_name
```

##### Kubernetes Service Discovery
```yaml
# Kubernetes Service Discovery
scrape_configs:
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
```

### Prometheus Metrics Types

#### 1. Counter
Monotonically increasing value (resets to zero on restart)
```python
from prometheus_client import Counter

# Example: API request counter
api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code']
)

# Usage
api_requests_total.labels(method='POST', endpoint='/predict', status_code='200').inc()
```

#### 2. Gauge
Value that can increase or decrease
```python
from prometheus_client import Gauge

# Example: Model accuracy gauge
model_accuracy = Gauge(
    'model_accuracy',
    'Current model accuracy',
    ['model_name', 'version']
)

# Usage
model_accuracy.labels(model_name='iris_classifier', version='v1.0').set(0.95)
```

#### 3. Histogram
Samples observations and counts them in configurable buckets
```python
from prometheus_client import Histogram

# Example: Request duration histogram
request_duration = Histogram(
    'request_duration_seconds',
    'Time spent processing requests',
    ['method', 'endpoint']
)

# Usage
with request_duration.labels(method='POST', endpoint='/predict').time():
    # Process request
    result = model.predict(data)
```

#### 4. Summary
Similar to histogram but calculates configurable quantiles
```python
from prometheus_client import Summary

# Example: Request latency summary
request_latency = Summary(
    'request_processing_seconds',
    'Time spent processing requests',
    ['endpoint']
)

# Usage
@request_latency.labels(endpoint='/predict').time()
def predict_endpoint():
    return model.predict(data)
```

### MLOps-Specific Prometheus Setup

#### Model Serving Metrics
```python
# ml_metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
import psutil

# Model performance metrics
model_predictions_total = Counter(
    'ml_model_predictions_total',
    'Total number of model predictions',
    ['model_name', 'model_version', 'prediction_class']
)

model_prediction_latency = Histogram(
    'ml_model_prediction_duration_seconds',
    'Time spent on model predictions',
    ['model_name'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

model_accuracy = Gauge(
    'ml_model_accuracy',
    'Current model accuracy score',
    ['model_name', 'model_version']
)

# Data quality metrics
data_drift_score = Gauge(
    'ml_data_drift_score',
    'Data drift detection score',
    ['feature_name', 'model_name']
)

# Infrastructure metrics
model_memory_usage = Gauge(
    'ml_model_memory_usage_bytes',
    'Memory usage of ML model',
    ['model_name']
)

# Business metrics
prediction_confidence = Histogram(
    'ml_prediction_confidence',
    'Confidence score distribution',
    ['model_name'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

class MLMetricsCollector:
    def __init__(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version

    def record_prediction(self, prediction_class, confidence, latency):
        """Record a model prediction with metrics"""
        # Increment prediction counter
        model_predictions_total.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            prediction_class=str(prediction_class)
        ).inc()

        # Record latency
        model_prediction_latency.labels(
            model_name=self.model_name
        ).observe(latency)

        # Record confidence distribution
        prediction_confidence.labels(
            model_name=self.model_name
        ).observe(confidence)

    def update_model_accuracy(self, accuracy):
        """Update model accuracy metric"""
        model_accuracy.labels(
            model_name=self.model_name,
            model_version=self.model_version
        ).set(accuracy)

    def update_data_drift(self, feature_drifts):
        """Update data drift scores"""
        for feature, drift_score in feature_drifts.items():
            data_drift_score.labels(
                feature_name=feature,
                model_name=self.model_name
            ).set(drift_score)

    def update_resource_usage(self):
        """Update resource usage metrics"""
        process = psutil.Process()
        memory_usage = process.memory_info().rss

        model_memory_usage.labels(
            model_name=self.model_name
        ).set(memory_usage)

# Flask/FastAPI integration
from flask import Flask, Response
app = Flask(__name__)

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), mimetype='text/plain')
```

---

## Setting Up Grafana

### Installation Options

#### 1. Binary Installation
```bash
# Download and install Grafana
wget https://dl.grafana.com/oss/release/grafana-10.1.0.linux-amd64.tar.gz
tar -zxvf grafana-10.1.0.linux-amd64.tar.gz
cd grafana-10.1.0

# Start Grafana
./bin/grafana-server
```

#### 2. Docker Installation
```bash
# Run Grafana in Docker
docker run -d \
  --name=grafana \
  -p 3000:3000 \
  -v grafana-storage:/var/lib/grafana \
  grafana/grafana:latest
```

#### 3. Docker Compose Integration
```yaml
# Add to docker-compose.yml
services:
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SECURITY_ALLOW_EMBEDDING=true
    depends_on:
      - prometheus

volumes:
  grafana_data:
```

### Initial Grafana Setup

#### 1. Access Grafana
- **URL**: http://localhost:3000
- **Default Login**: admin/admin
- **First Login**: You'll be prompted to change the password

#### 2. Add Prometheus Data Source

**Step-by-step Configuration:**

1. **Navigate to Configuration** → **Data Sources**
2. **Click "Add data source"**
3. **Select "Prometheus"**
4. **Configure settings:**
   - **URL**: `http://localhost:9090` (or `http://prometheus:9090` in Docker)
   - **Access**: Server (default)
   - **Scrape interval**: 15s

**Data Source Configuration JSON:**
```json
{
  "id": 1,
  "orgId": 1,
  "name": "Prometheus",
  "type": "prometheus",
  "typeName": "Prometheus",
  "typeLogoUrl": "public/app/plugins/datasource/prometheus/img/prometheus_logo.svg",
  "access": "proxy",
  "url": "http://localhost:9090",
  "password": "",
  "user": "",
  "database": "",
  "basicAuth": false,
  "isDefault": true,
  "jsonData": {
    "httpMethod": "POST",
    "timeInterval": "15s"
  }
}
```

#### 3. Verify Connection
Click **"Save & Test"** - you should see "Data source is working"

### Grafana Dashboard Configuration

#### Dashboard Structure
```
Dashboard
├── Panels (Individual visualizations)
├── Rows (Grouping mechanism)
├── Variables (Dynamic filtering)
├── Annotations (Event markers)
└── Settings (Time range, refresh, etc.)
```

#### Creating Your First Dashboard

**Manual Creation Steps:**

1. **Create New Dashboard**
   - Click **"+"** → **"Dashboard"**
   - Click **"Add new panel"**

2. **Configure Query**
   - Data source: Prometheus
   - Query: `up`
   - Legend: `{{job}} - {{instance}}`

3. **Customize Visualization**
   - Panel type: Time series
   - Title: "Service Uptime"
   - Y-axis: 0-1 range

#### Dashboard Provisioning (Recommended)

**Directory Structure:**
```
monitoring/grafana/
├── provisioning/
│   ├── datasources/
│   │   └── prometheus.yml
│   └── dashboards/
│       └── dashboards.yml
└── dashboards/
    ├── mlops-overview.json
    ├── model-performance.json
    └── infrastructure.json
```

**Datasource Provisioning (prometheus.yml):**
```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    orgId: 1
    url: http://prometheus:9090
    basicAuth: false
    isDefault: true
    version: 1
    editable: true
    jsonData:
      httpMethod: POST
      timeInterval: 15s
```

**Dashboard Provisioning (dashboards.yml):**
```yaml
apiVersion: 1

providers:
  - name: 'MLOps Dashboards'
    orgId: 1
    folder: 'MLOps'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
```

---

## PromQL - Prometheus Query Language

### PromQL Fundamentals

#### Basic Syntax
```promql
# Simple metric query
up

# Metric with labels
up{job="prometheus"}

# Label matching operators
up{job="prometheus", instance="localhost:9090"}
up{job=~"prometheus|node.*"}  # Regex match
up{job!="prometheus"}         # Not equal
```

#### Time Series Selectors
```promql
# Instant vector (current value)
http_requests_total

# Range vector (values over time)
http_requests_total[5m]  # Last 5 minutes
http_requests_total[1h]  # Last 1 hour
http_requests_total[1d]  # Last 1 day
```

### Common PromQL Functions

#### Rate and Increase
```promql
# Rate: per-second average rate of increase
rate(http_requests_total[5m])

# Increase: total increase over time range
increase(http_requests_total[5m])

# irate: instantaneous rate (based on last two data points)
irate(http_requests_total[5m])
```

#### Aggregation Functions
```promql
# Sum across all instances
sum(http_requests_total)

# Sum by job
sum by (job) (http_requests_total)

# Average, min, max
avg(cpu_usage_percent)
min(memory_available_bytes)
max(disk_usage_percent)

# Count number of instances
count(up == 1)  # Number of healthy instances
```

#### Mathematical Operations
```promql
# Basic arithmetic
cpu_usage_percent * 100
memory_used_bytes / 1024 / 1024  # Convert to MB

# Comparison operators
cpu_usage_percent > 0.8  # High CPU usage
up == 0                  # Down services

# Logical operators
up == 1 and cpu_usage_percent < 0.9  # Healthy and not overloaded
```

### MLOps-Specific PromQL Queries

#### Model Performance Monitoring
```promql
# Prediction rate (predictions per second)
rate(ml_model_predictions_total[5m])

# Average prediction latency
rate(ml_model_prediction_duration_seconds_sum[5m]) /
rate(ml_model_prediction_duration_seconds_count[5m])

# 95th percentile latency
histogram_quantile(0.95,
  rate(ml_model_prediction_duration_seconds_bucket[5m])
)

# Model accuracy over time
ml_model_accuracy{model_name="iris_classifier"}

# Error rate
rate(ml_model_predictions_total{prediction_class="error"}[5m]) /
rate(ml_model_predictions_total[5m]) * 100
```

#### Data Quality Monitoring
```promql
# Data drift detection
ml_data_drift_score > 0.3  # Alert threshold

# Missing data rate
rate(ml_missing_data_total[5m])

# Data schema violations
ml_schema_violations_total

# Feature distribution changes
delta(ml_feature_mean[1h])  # Change in feature mean over 1 hour
```

#### Infrastructure Monitoring
```promql
# CPU usage
100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory usage percentage
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100

# Disk usage
(node_filesystem_size_bytes - node_filesystem_avail_bytes) /
node_filesystem_size_bytes * 100

# Network I/O
rate(node_network_receive_bytes_total[5m])
rate(node_network_transmit_bytes_total[5m])
```

#### Business Metrics
```promql
# Daily active predictions
sum(increase(ml_model_predictions_total[1d]))

# Model usage by class
sum by (prediction_class) (rate(ml_model_predictions_total[5m]))

# Revenue impact (if available)
sum(ml_prediction_value_total) by (model_name)

# User engagement
rate(ml_user_interactions_total[5m])
```

### Advanced PromQL Patterns

#### Alerting Queries
```promql
# Service down alert
up{job="mlops-api"} == 0

# High error rate alert
(
  rate(http_requests_total{status=~"5.."}[5m]) /
  rate(http_requests_total[5m])
) > 0.05

# Model accuracy degradation
(
  ml_model_accuracy -
  ml_model_accuracy offset 1d
) < -0.1

# Resource exhaustion
node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes < 0.1
```

#### Forecasting and Trending
```promql
# Predict future values using linear regression
predict_linear(cpu_usage_percent[1h], 3600)  # Predict 1 hour ahead

# Derivative (rate of change)
deriv(ml_model_accuracy[1h])  # How fast accuracy is changing

# Moving average
avg_over_time(cpu_usage_percent[10m])
```

---

## Grafana Dashboard Creation

### MLOps Dashboard Design Principles

#### 1. **Hierarchical Information**
```
Overview Dashboard (High-level KPIs)
    ├── Service Health Dashboard
    ├── Model Performance Dashboard
    ├── Infrastructure Dashboard
    └── Business Metrics Dashboard
```

#### 2. **USER Framework**
- **U**tilization: Resource usage metrics
- **S**aturation: Capacity and limits
- **E**rrors: Error rates and failures
- **R**ate: Request rates and throughput

#### 3. **The Four Golden Signals**
- **Latency**: Response time
- **Traffic**: Request volume
- **Errors**: Error rate
- **Saturation**: Resource utilization

### MLOps Overview Dashboard

#### Dashboard JSON Structure
```json
{
  "dashboard": {
    "id": null,
    "title": "MLOps Overview",
    "tags": ["mlops", "overview"],
    "timezone": "browser",
    "refresh": "5s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "id": 1,
        "title": "Service Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"mlops-api\"}",
            "legendFormat": "{{instance}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "green", "value": 1}
              ]
            }
          }
        }
      }
    ]
  }
}
```

#### Key Panel Types for MLOps

##### 1. **Stat Panels** (Single Value Displays)
```json
{
  "title": "Model Accuracy",
  "type": "stat",
  "targets": [
    {
      "expr": "ml_model_accuracy{model_name=\"iris_classifier\"}",
      "legendFormat": "Current Accuracy"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "unit": "percentunit",
      "min": 0,
      "max": 1,
      "thresholds": {
        "steps": [
          {"color": "red", "value": 0},
          {"color": "yellow", "value": 0.8},
          {"color": "green", "value": 0.9}
        ]
      }
    }
  }
}
```

##### 2. **Time Series Panels** (Line Charts)
```json
{
  "title": "Prediction Rate",
  "type": "timeseries",
  "targets": [
    {
      "expr": "rate(ml_model_predictions_total[5m])",
      "legendFormat": "{{model_name}} - {{instance}}"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "unit": "reqps",
      "custom": {
        "drawStyle": "line",
        "lineInterpolation": "linear",
        "lineWidth": 2,
        "fillOpacity": 10
      }
    }
  }
}
```

##### 3. **Histogram Panels** (Distribution Analysis)
```json
{
  "title": "Prediction Latency Distribution",
  "type": "histogram",
  "targets": [
    {
      "expr": "histogram_quantile(0.95, rate(ml_model_prediction_duration_seconds_bucket[5m]))",
      "legendFormat": "95th percentile"
    },
    {
      "expr": "histogram_quantile(0.50, rate(ml_model_prediction_duration_seconds_bucket[5m]))",
      "legendFormat": "50th percentile"
    }
  ]
}
```

##### 4. **Gauge Panels** (Progress Indicators)
```json
{
  "title": "CPU Usage",
  "type": "gauge",
  "targets": [
    {
      "expr": "100 - (avg by (instance) (rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "unit": "percent",
      "min": 0,
      "max": 100,
      "thresholds": {
        "steps": [
          {"color": "green", "value": 0},
          {"color": "yellow", "value": 70},
          {"color": "red", "value": 90}
        ]
      }
    }
  }
}
```

### Model Performance Dashboard

#### Panel Configuration Examples

```json
{
  "dashboard": {
    "title": "Model Performance Dashboard",
    "panels": [
      {
        "title": "Model Accuracy Trend",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "ml_model_accuracy",
            "legendFormat": "{{model_name}} v{{model_version}}"
          }
        ]
      },
      {
        "title": "Prediction Volume",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "targets": [
          {
            "expr": "rate(ml_model_predictions_total[5m])",
            "legendFormat": "{{model_name}}"
          }
        ]
      },
      {
        "title": "Data Drift Heatmap",
        "type": "heatmap",
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "ml_data_drift_score",
            "format": "heatmap",
            "legendFormat": "{{feature_name}}"
          }
        ]
      },
      {
        "title": "Error Rate by Model",
        "type": "bargauge",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
        "targets": [
          {
            "expr": "rate(ml_model_predictions_total{prediction_class=\"error\"}[5m]) / rate(ml_model_predictions_total[5m]) * 100",
            "legendFormat": "{{model_name}}"
          }
        ]
      }
    ]
  }
}
```

### Infrastructure Dashboard

```json
{
  "dashboard": {
    "title": "Infrastructure Monitoring",
    "panels": [
      {
        "title": "System Overview",
        "type": "row",
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0}
      },
      {
        "title": "CPU Usage",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 1},
        "targets": [
          {
            "expr": "100 - (avg by (instance) (rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 1},
        "targets": [
          {
            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "title": "Disk I/O",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 1},
        "targets": [
          {
            "expr": "rate(node_disk_read_bytes_total[5m])",
            "legendFormat": "Read - {{device}}"
          },
          {
            "expr": "rate(node_disk_written_bytes_total[5m])",
            "legendFormat": "Write - {{device}}"
          }
        ]
      },
      {
        "title": "Network Traffic",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 6, "x": 18, "y": 1},
        "targets": [
          {
            "expr": "rate(node_network_receive_bytes_total[5m])",
            "legendFormat": "Receive - {{device}}"
          },
          {
            "expr": "rate(node_network_transmit_bytes_total[5m])",
            "legendFormat": "Transmit - {{device}}"
          }
        ]
      }
    ]
  }
}
```

### Dashboard Variables

#### Template Variables for Dynamic Dashboards

```json
{
  "templating": {
    "list": [
      {
        "name": "model_name",
        "type": "query",
        "query": "label_values(ml_model_accuracy, model_name)",
        "refresh": "on_time_range_change",
        "includeAll": true,
        "allValue": ".*"
      },
      {
        "name": "instance",
        "type": "query",
        "query": "label_values(up, instance)",
        "refresh": "on_dashboard_load",
        "includeAll": true
      },
      {
        "name": "time_range",
        "type": "interval",
        "query": "1m,5m,15m,30m,1h,2h,6h,12h,1d",
        "current": {
          "text": "5m",
          "value": "5m"
        }
      }
    ]
  }
}
```

#### Using Variables in Queries
```promql
# Use model_name variable
ml_model_accuracy{model_name=~"$model_name"}

# Use instance variable
up{instance=~"$instance"}

# Use time_range variable
rate(http_requests_total[$time_range])
```

---

## MLOps-Specific Monitoring

### Creating a Complete ML Model Monitoring Setup

Let me create a practical implementation that you can run:

#### Step 1: ML Model with Prometheus Metrics

```python
# ml_model_server.py
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import numpy as np
import time
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import threading

app = Flask(__name__)

# Prometheus metrics
model_predictions_total = Counter(
    'ml_model_predictions_total',
    'Total number of model predictions',
    ['model_name', 'prediction_class']
)

model_prediction_latency = Histogram(
    'ml_model_prediction_duration_seconds',
    'Time spent on model predictions',
    ['model_name'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0]
)

model_accuracy = Gauge(
    'ml_model_accuracy',
    'Current model accuracy',
    ['model_name', 'version']
)

data_drift_score = Gauge(
    'ml_data_drift_score',
    'Data drift detection score',
    ['feature_name', 'model_name']
)

prediction_confidence = Histogram(
    'ml_prediction_confidence',
    'Prediction confidence distribution',
    ['model_name'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Load and prepare model
iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

class_names = iris.target_names
feature_names = iris.feature_names

# Initialize metrics
model_accuracy.labels(model_name='iris_classifier', version='v1.0').set(0.95)

# Simulate data drift monitoring
def update_data_drift():
    """Simulate data drift detection"""
    while True:
        for feature in feature_names:
            # Simulate drift scores
            drift = np.random.beta(2, 5)  # Biased towards lower values
            data_drift_score.labels(
                feature_name=feature,
                model_name='iris_classifier'
            ).set(drift)
        time.sleep(30)  # Update every 30 seconds

# Start drift monitoring thread
drift_thread = threading.Thread(target=update_data_drift, daemon=True)
drift_thread.start()

@app.route('/predict', methods=['POST'])
def predict():
    """ML model prediction endpoint with metrics"""
    start_time = time.time()

    try:
        data = request.get_json()
        features = np.array([data['features']]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities)

        # Record metrics
        prediction_class = class_names[prediction]

        model_predictions_total.labels(
            model_name='iris_classifier',
            prediction_class=prediction_class
        ).inc()

        prediction_confidence.labels(
            model_name='iris_classifier'
        ).observe(confidence)

        # Record latency
        latency = time.time() - start_time
        model_prediction_latency.labels(
            model_name='iris_classifier'
        ).observe(latency)

        return jsonify({
            'prediction': prediction_class,
            'confidence': float(confidence),
            'probabilities': {
                class_names[i]: float(prob)
                for i, prob in enumerate(probabilities)
            },
            'latency_ms': latency * 1000
        })

    except Exception as e:
        model_predictions_total.labels(
            model_name='iris_classifier',
            prediction_class='error'
        ).inc()

        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'iris_classifier'})

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=False)
```

#### Step 2: Load Testing Script

```python
# load_test.py
import requests
import time
import random
import threading
import numpy as np
from sklearn.datasets import load_iris

# Load iris dataset for realistic test data
iris = load_iris()
X, y = iris.data, iris.target

def make_prediction():
    """Make a single prediction request"""
    # Select random sample from iris dataset
    idx = random.randint(0, len(X) - 1)
    features = X[idx].tolist()

    # Add some noise to simulate real-world data
    features = [f + random.gauss(0, 0.1) for f in features]

    try:
        response = requests.post(
            'http://localhost:8001/predict',
            json={'features': features},
            timeout=5
        )
        return response.status_code == 200
    except:
        return False

def load_test_worker():
    """Worker thread for load testing"""
    while True:
        success = make_prediction()
        if not success:
            print("Failed request")

        # Random delay between requests
        time.sleep(random.uniform(0.1, 2.0))

def main():
    """Run load test with multiple workers"""
    print("Starting load test...")

    # Start multiple worker threads
    workers = []
    for i in range(5):  # 5 concurrent workers
        worker = threading.Thread(target=load_test_worker, daemon=True)
        worker.start()
        workers.append(worker)

    print("Load test running... Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping load test...")

if __name__ == '__main__':
    main()
```

#### Step 3: Complete Monitoring Stack

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  # Node Exporter for system metrics
  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.ignored-mount-points="^/(sys|proc|dev|host|etc)($$|/)"'

  # Prometheus server
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/rules:/etc/prometheus/rules
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    depends_on:
      - node-exporter

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SECURITY_ALLOW_EMBEDDING=true
      - GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel
    depends_on:
      - prometheus

  # Alert Manager for notifications
  alertmanager:
    image: prom/alertmanager:latest
    container_name: alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:
```

---

## Advanced Features & Integrations

### Alerting Configuration

#### Prometheus Alerting Rules

```yaml
# monitoring/rules/mlops.yml
groups:
  - name: mlops-alerts
    rules:
      # Model performance alerts
      - alert: ModelAccuracyDrop
        expr: ml_model_accuracy < 0.85
        for: 5m
        labels:
          severity: warning
          team: ml-ops
        annotations:
          summary: "Model accuracy dropped below threshold"
          description: "Model {{ $labels.model_name }} accuracy is {{ $value }}, below 85% threshold"

      - alert: HighPredictionLatency
        expr: histogram_quantile(0.95, rate(ml_model_prediction_duration_seconds_bucket[5m])) > 0.5
        for: 2m
        labels:
          severity: critical
          team: ml-ops
        annotations:
          summary: "High prediction latency detected"
          description: "95th percentile latency for {{ $labels.model_name }} is {{ $value }}s"

      - alert: DataDriftDetected
        expr: ml_data_drift_score > 0.7
        for: 10m
        labels:
          severity: warning
          team: data-science
        annotations:
          summary: "Data drift detected"
          description: "Feature {{ $labels.feature_name }} shows drift score of {{ $value }}"

      # Infrastructure alerts
      - alert: HighCPUUsage
        expr: 100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 90
        for: 5m
        labels:
          severity: critical
          team: infrastructure
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage on {{ $labels.instance }} is {{ $value }}%"

      - alert: HighMemoryUsage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 90
        for: 5m
        labels:
          severity: critical
          team: infrastructure
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage on {{ $labels.instance }} is {{ $value }}%"

      - alert: ServiceDown
        expr: up{job=~"mlops-.*"} == 0
        for: 1m
        labels:
          severity: critical
          team: ml-ops
        annotations:
          summary: "Service is down"
          description: "Service {{ $labels.job }} on {{ $labels.instance }} is down"
```

#### Alertmanager Configuration

```yaml
# monitoring/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alertmanager@company.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'
  routes:
    - match:
        team: ml-ops
      receiver: 'ml-ops-team'
    - match:
        team: data-science
      receiver: 'data-science-team'
    - match:
        severity: critical
      receiver: 'critical-alerts'

receivers:
  - name: 'default'
    email_configs:
      - to: 'alerts@company.com'
        subject: 'Prometheus Alert'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}

  - name: 'ml-ops-team'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#ml-ops-alerts'
        title: 'MLOps Alert'
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          *Severity:* {{ .Labels.severity }}
          {{ end }}

  - name: 'data-science-team'
    email_configs:
      - to: 'data-science@company.com'
        subject: 'Data Science Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Runbook: https://runbooks.company.com/{{ .Labels.alertname }}
          {{ end }}

  - name: 'critical-alerts'
    pagerduty_configs:
      - routing_key: 'YOUR_PAGERDUTY_INTEGRATION_KEY'
        description: '{{ .GroupLabels.alertname }}: {{ .GroupLabels.instance }}'
```

### Custom Grafana Plugins

#### Installing Plugins
```bash
# In Grafana container or environment variable
GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel,grafana-clock-panel
```

#### Custom Panel for ML Metrics
```javascript
// Custom confusion matrix panel
import { PanelPlugin } from '@grafana/data';

export const plugin = new PanelPlugin(ConfusionMatrixPanel).setPanelOptions(builder => {
  return builder
    .addTextInput({
      path: 'model_name',
      name: 'Model Name',
      description: 'Name of the ML model',
      defaultValue: 'iris_classifier',
    })
    .addSelect({
      path: 'matrix_size',
      name: 'Matrix Size',
      description: 'Size of confusion matrix',
      options: [
        { label: '3x3', value: 3 },
        { label: '5x5', value: 5 },
        { label: '10x10', value: 10 },
      ],
      defaultValue: 3,
    });
});
```

---

## Production Best Practices

### High Availability Setup

#### Prometheus Federation
```yaml
# High-level Prometheus (Global)
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'federate'
    scrape_interval: 15s
    honor_labels: true
    metrics_path: '/federate'
    params:
      'match[]':
        - '{job=~"prometheus|node.*"}'
        - '{job=~"mlops-.*"}'
    static_configs:
      - targets:
        - 'prometheus-region1:9090'
        - 'prometheus-region2:9090'
        - 'prometheus-region3:9090'
```

#### Grafana High Availability
```yaml
# Grafana with external database
services:
  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_DATABASE_TYPE=postgres
      - GF_DATABASE_HOST=postgres:5432
      - GF_DATABASE_NAME=grafana
      - GF_DATABASE_USER=grafana
      - GF_DATABASE_PASSWORD=password
      - GF_SERVER_ROOT_URL=https://grafana.company.com
    volumes:
      - ./provisioning:/etc/grafana/provisioning

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=grafana
      - POSTGRES_USER=grafana
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
```

### Security Configuration

#### Prometheus Security
```yaml
# Basic auth configuration
basic_auth_users:
  admin: $2b$12$hNf2lSsxfm0.i4a.1kVpSOVyBCfIB51VRjgBUyv6kdnyTlgWj81Ay

# TLS configuration
tls_server_config:
  cert_file: /etc/prometheus/prometheus.crt
  key_file: /etc/prometheus/prometheus.key

# Web configuration
web_config_file: /etc/prometheus/web.yml
```

#### Grafana Security
```yaml
# Grafana security settings
environment:
  - GF_SECURITY_ADMIN_PASSWORD=secure_password
  - GF_USERS_ALLOW_SIGN_UP=false
  - GF_AUTH_ANONYMOUS_ENABLED=false
  - GF_SECURITY_COOKIE_SECURE=true
  - GF_SECURITY_STRICT_TRANSPORT_SECURITY=true
  - GF_SECURITY_X_CONTENT_TYPE_OPTIONS=true
  - GF_SECURITY_X_XSS_PROTECTION=true
```

### Performance Optimization

#### Prometheus Optimization
```yaml
# Storage optimization
global:
  scrape_interval: 15s
  evaluation_interval: 15s

# Retention policies
command:
  - '--storage.tsdb.retention.time=30d'
  - '--storage.tsdb.retention.size=50GB'
  - '--storage.tsdb.wal-compression'

# Query optimization
query:
  max_concurrency: 20
  timeout: 2m
```

#### Grafana Optimization
```yaml
# Performance settings
environment:
  - GF_DATABASE_MAX_OPEN_CONN=300
  - GF_DATABASE_MAX_IDLE_CONN=10
  - GF_DATABASE_CONN_MAX_LIFETIME=14400
  - GF_RENDERING_SERVER_URL=http://renderer:8081/render
  - GF_RENDERING_CALLBACK_URL=http://grafana:3000/
```

### Backup and Recovery

#### Prometheus Backup
```bash
#!/bin/bash
# Prometheus backup script

BACKUP_DIR="/backup/prometheus/$(date +%Y%m%d_%H%M%S)"
PROMETHEUS_DATA="/prometheus"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Create snapshot
curl -XPOST http://localhost:9090/api/v1/admin/tsdb/snapshot

# Copy snapshot data
SNAPSHOT_NAME=$(curl -s http://localhost:9090/api/v1/admin/tsdb/snapshots | jq -r '.data.name')
cp -r "$PROMETHEUS_DATA/snapshots/$SNAPSHOT_NAME" "$BACKUP_DIR/"

# Compress backup
tar -czf "$BACKUP_DIR.tar.gz" -C "$BACKUP_DIR" .
rm -rf "$BACKUP_DIR"

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

#### Grafana Backup
```bash
#!/bin/bash
# Grafana backup script

BACKUP_DIR="/backup/grafana/$(date +%Y%m%d_%H%M%S)"
GRAFANA_API="http://admin:password@localhost:3000/api"

mkdir -p "$BACKUP_DIR"

# Export all dashboards
for dashboard_uid in $(curl -s "$GRAFANA_API/search" | jq -r '.[].uid'); do
  curl -s "$GRAFANA_API/dashboards/uid/$dashboard_uid" > "$BACKUP_DIR/dashboard_$dashboard_uid.json"
done

# Export data sources
curl -s "$GRAFANA_API/datasources" > "$BACKUP_DIR/datasources.json"

# Export users
curl -s "$GRAFANA_API/users" > "$BACKUP_DIR/users.json"

# Compress backup
tar -czf "$BACKUP_DIR.tar.gz" -C "$BACKUP_DIR" .
rm -rf "$BACKUP_DIR"

echo "Grafana backup completed: $BACKUP_DIR.tar.gz"
```

---

## Troubleshooting & FAQ

### Common Issues and Solutions

#### 1. Prometheus Target Discovery Issues

**Problem**: Targets showing as "DOWN" in Prometheus
```
up{job="mlops-api"} == 0
```

**Solutions**:
```bash
# Check target accessibility
curl http://localhost:8001/metrics

# Verify network connectivity
telnet localhost 8001

# Check Prometheus logs
docker logs prometheus

# Validate prometheus.yml syntax
promtool check config prometheus.yml
```

#### 2. Grafana Data Source Connection Issues

**Problem**: "Data source proxy error"

**Solutions**:
```bash
# Test Prometheus connectivity from Grafana container
docker exec -it grafana curl http://prometheus:9090/api/v1/query?query=up

# Check network configuration
docker network ls
docker network inspect <network_name>

# Verify Prometheus is accessible
curl http://localhost:9090/-/healthy
```

#### 3. Missing Metrics

**Problem**: Expected metrics not appearing in Prometheus

**Debugging Steps**:
```bash
# Check if metrics endpoint is working
curl http://localhost:8001/metrics | grep ml_model

# Verify metric naming conventions
# Metrics must match: [a-zA-Z_:][a-zA-Z0-9_:]*

# Check for label consistency
# All time series of a metric must have the same label names
```

#### 4. High Memory Usage

**Problem**: Prometheus consuming too much memory

**Solutions**:
```yaml
# Reduce retention time
command:
  - '--storage.tsdb.retention.time=7d'

# Increase scrape interval
global:
  scrape_interval: 30s

# Use recording rules for expensive queries
rule_files:
  - "recording_rules.yml"
```

**Recording Rules Example**:
```yaml
groups:
  - name: mlops_recording_rules
    interval: 30s
    rules:
      - record: job:ml_prediction_rate:rate5m
        expr: sum(rate(ml_model_predictions_total[5m])) by (job)

      - record: job:ml_error_rate:rate5m
        expr: sum(rate(ml_model_predictions_total{prediction_class="error"}[5m])) by (job) / sum(rate(ml_model_predictions_total[5m])) by (job)
```

### Performance Tuning

#### Query Optimization

```promql
# Inefficient query (avoid)
sum(rate(http_requests_total[5m])) by (job)

# Efficient query (prefer)
sum by (job) (rate(http_requests_total[5m]))

# Use recording rules for complex queries
job:http_request_rate:rate5m

# Limit time ranges for exploration
http_requests_total[1h]  # Instead of [1d] for testing
```

#### Dashboard Optimization

```json
{
  "refresh": "5s",        // Don't use "5s" in production
  "refresh": "30s",       // Better for production

  "targets": [
    {
      "expr": "up",
      "interval": "15s",    // Match or exceed scrape interval
      "maxDataPoints": 100  // Limit data points for performance
    }
  ]
}
```

### Best Practices Checklist

#### Prometheus
- [ ] Use appropriate scrape intervals (15s-1m)
- [ ] Implement proper service discovery
- [ ] Set up recording rules for complex queries
- [ ] Configure appropriate retention periods
- [ ] Monitor Prometheus itself
- [ ] Use labels consistently
- [ ] Avoid high cardinality labels

#### Grafana
- [ ] Use template variables for dynamic dashboards
- [ ] Implement proper user access controls
- [ ] Set up notification channels
- [ ] Regular dashboard backups
- [ ] Use appropriate panel types for data
- [ ] Optimize query performance
- [ ] Document dashboard purpose and usage

#### MLOps Monitoring
- [ ] Monitor model performance metrics
- [ ] Track data quality and drift
- [ ] Monitor inference latency and throughput
- [ ] Set up appropriate alerting thresholds
- [ ] Monitor resource utilization
- [ ] Track business metrics
- [ ] Implement proper logging correlation

---

## Conclusion

This tutorial provides a comprehensive foundation for implementing Prometheus and Grafana monitoring in MLOps environments. The combination of these tools enables:

- **Complete Observability**: From infrastructure to business metrics
- **Proactive Monitoring**: Alerting before issues become critical
- **Data-Driven Decisions**: Historical analysis and trend identification
- **Operational Excellence**: Standardized monitoring across all services

### Key Takeaways

1. **Start Simple**: Begin with basic metrics and gradually add complexity
2. **Focus on SLIs/SLOs**: Monitor what matters for your business
3. **Automate Everything**: Use provisioning and Infrastructure as Code
4. **Plan for Scale**: Design monitoring architecture for growth
5. **Security First**: Implement proper authentication and authorization
6. **Document Well**: Clear runbooks and dashboard documentation

### Next Steps

- Implement the monitoring stack in your environment
- Create custom dashboards for your specific use cases
- Set up alerting rules based on your SLAs
- Integrate with your existing toolchain
- Train your team on monitoring best practices

The monitoring foundation you build today will be crucial for maintaining reliable and performant ML systems in production.

---

**Resources:**
- **Prometheus Documentation**: https://prometheus.io/docs/
- **Grafana Documentation**: https://grafana.com/docs/
- **PromQL Tutorial**: https://prometheus.io/docs/prometheus/latest/querying/basics/
- **Dashboard Examples**: https://grafana.com/grafana/dashboards/