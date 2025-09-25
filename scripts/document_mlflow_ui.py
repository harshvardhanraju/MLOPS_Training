#!/usr/bin/env python3
"""
Document MLflow UI features and create summary
"""

import mlflow
import requests
import json
from datetime import datetime

def create_mlflow_ui_documentation():
    """Create documentation of MLflow UI features"""

    mlflow.set_tracking_uri("http://localhost:5000")
    client = mlflow.tracking.MlflowClient()

    print("📸 Creating MLflow UI Documentation")
    print("=" * 50)

    # Get experiments and runs data
    experiments = client.search_experiments()

    ui_documentation = {
        "mlflow_server": "http://localhost:5000",
        "timestamp": datetime.now().isoformat(),
        "experiments": [],
        "ui_features": {
            "main_dashboard": {
                "url": "http://localhost:5000",
                "description": "Main MLflow dashboard showing all experiments",
                "features": [
                    "Experiment list with run counts",
                    "Search and filter experiments",
                    "Create new experiments",
                    "Quick access to recent runs"
                ]
            },
            "experiment_view": {
                "url": "http://localhost:5000/#/experiments/{experiment_id}",
                "description": "Detailed view of individual experiment",
                "features": [
                    "Run comparison table",
                    "Parameter and metric columns",
                    "Run filtering and search",
                    "Bulk run selection",
                    "Chart view for metrics",
                    "Download runs as CSV"
                ]
            },
            "run_detail": {
                "url": "http://localhost:5000/#/experiments/{exp_id}/runs/{run_id}",
                "description": "Individual run details",
                "features": [
                    "Parameters and metrics display",
                    "Artifacts browser",
                    "Model information",
                    "Run metadata",
                    "Tags and notes",
                    "Metric charts"
                ]
            },
            "model_registry": {
                "url": "http://localhost:5000/#/models",
                "description": "Centralized model registry",
                "features": [
                    "Model versioning",
                    "Stage transitions (None/Staging/Production/Archived)",
                    "Model descriptions and annotations",
                    "Source run linking",
                    "Model deployment integration"
                ]
            },
            "compare_runs": {
                "url": "http://localhost:5000/#/experiments/{exp_id}/compare-runs",
                "description": "Side-by-side run comparison",
                "features": [
                    "Parameter comparison",
                    "Metric comparison",
                    "Artifact comparison",
                    "Parallel coordinates plot",
                    "Scatter plot view",
                    "Difference highlighting"
                ]
            }
        }
    }

    # Document each experiment
    for exp in experiments:
        if exp.name == "Default":
            continue

        runs = client.search_runs([exp.experiment_id], max_results=10)

        exp_info = {
            "name": exp.name,
            "id": exp.experiment_id,
            "url": f"http://localhost:5000/#/experiments/{exp.experiment_id}",
            "total_runs": len(client.search_runs([exp.experiment_id])),
            "sample_runs": []
        }

        # Document sample runs
        for run in runs[:3]:  # First 3 runs
            run_info = {
                "name": run.info.run_name or f"Run_{run.info.run_id[:8]}",
                "id": run.info.run_id,
                "url": f"http://localhost:5000/#/experiments/{exp.experiment_id}/runs/{run.info.run_id}",
                "status": run.info.status,
                "parameters": dict(run.data.params),
                "metrics": {k: v for k, v in run.data.metrics.items()},
                "artifacts_url": f"http://localhost:5000/#/experiments/{exp.experiment_id}/runs/{run.info.run_id}/artifacts"
            }
            exp_info["sample_runs"].append(run_info)

        ui_documentation["experiments"].append(exp_info)

    # Document registered models
    try:
        models = client.search_registered_models()
        ui_documentation["registered_models"] = []

        for model in models:
            versions = client.search_model_versions(f'name="{model.name}"')
            model_info = {
                "name": model.name,
                "url": f"http://localhost:5000/#/models/{model.name}",
                "description": getattr(model, 'description', ''),
                "versions": [
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "source_run": v.run_id,
                        "url": f"http://localhost:5000/#/models/{model.name}/versions/{v.version}"
                    }
                    for v in versions
                ]
            }
            ui_documentation["registered_models"].append(model_info)
    except Exception as e:
        print(f"Note: Could not fetch registered models: {e}")

    # Save documentation
    with open('docs/tutorials/MLflow_UI_Documentation.json', 'w') as f:
        json.dump(ui_documentation, f, indent=2)

    # Create human-readable summary
    summary = f"""# MLflow UI Features and Demo Summary

## 🌐 Server Information
- **MLflow Server**: http://localhost:5000
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Experiments Created

"""

    for exp in ui_documentation["experiments"]:
        summary += f"""### {exp['name']}
- **URL**: {exp['url']}
- **Total Runs**: {exp['total_runs']}
- **Purpose**: Demonstrates {exp['name'].lower().replace('_', ' ')}

**Sample Runs:**
"""
        for run in exp["sample_runs"]:
            summary += f"""- **{run['name']}**: {run['url']}
  - Status: {run['status']}
  - Key Metrics: {', '.join([f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in list(run['metrics'].items())[:3]])}
"""
        summary += "\n"

    if "registered_models" in ui_documentation:
        summary += """## 🏷️ Model Registry

"""
        for model in ui_documentation["registered_models"]:
            summary += f"""### {model['name']}
- **URL**: {model['url']}
- **Versions**: {len(model['versions'])}
"""
            for version in model["versions"]:
                summary += f"""  - Version {version['version']}: {version['stage'] or 'None'}
"""

    summary += f"""
## 🎯 MLflow UI Features Demonstrated

### 1. Main Dashboard (http://localhost:5000)
- Experiment overview with run counts
- Quick experiment creation and management
- Search and filter capabilities

### 2. Experiment Views
Each experiment showcases different aspects:
- **Parameter tracking**: Different model configurations
- **Metric logging**: Accuracy, loss, training time, etc.
- **Artifact storage**: Plots, models, reports
- **Run comparison**: Side-by-side analysis
- **Tagging system**: Organization and metadata

### 3. Model Registry (http://localhost:5000/#/models)
- Centralized model storage
- Version management
- Stage transitions (Development → Staging → Production)
- Model lineage and metadata

### 4. Interactive Features
- **Metric Charts**: Time-series visualization of training metrics
- **Run Comparison**: Multi-run parameter and metric comparison
- **Artifact Browser**: Direct access to generated files
- **Search & Filter**: Advanced querying capabilities
- **Export Options**: CSV download, API access

## 📈 Data Generated

- **Total Experiments**: {len(ui_documentation['experiments'])}
- **Total Runs**: {sum(exp['total_runs'] for exp in ui_documentation['experiments'])}
- **Registered Models**: {len(ui_documentation.get('registered_models', []))}
- **ML Algorithms**: Random Forest, Logistic Regression, SVM, Gradient Boosting
- **Datasets**: Iris, California Housing, Wine Classification
- **Visualizations**: Confusion matrices, feature importance, learning curves, residual plots

## 🎨 Artifacts Created

Each run includes:
- **Model files**: Serialized sklearn models
- **Visualizations**: PNG plots and charts
- **Reports**: Classification reports, prediction CSVs
- **Metadata**: Parameter configurations, timing data

## 🚀 Next Steps for Demo

1. **Navigate to http://localhost:5000**
2. **Explore Experiments**: Click on different experiments to see runs
3. **Compare Runs**: Select multiple runs and click "Compare"
4. **View Artifacts**: Click on individual runs to see generated plots
5. **Check Models**: Visit the Models tab to see registered models
6. **Try Filtering**: Use search boxes to filter runs by metrics or parameters

## 💡 Key Takeaways

This demo showcases MLflow's capabilities for:
- **Experiment Tracking**: Systematic logging of ML experiments
- **Model Management**: Centralized model storage and versioning
- **Reproducibility**: Complete parameter and environment tracking
- **Collaboration**: Shared experiment visibility and comparison tools
- **Production Readiness**: Model registry with stage management

The generated experiments provide realistic examples of how data science teams can use MLflow to manage their ML lifecycle from experimentation to production deployment.
"""

    # Save summary
    with open('docs/tutorials/MLflow_Demo_Summary.md', 'w') as f:
        f.write(summary)

    print("✅ Created MLflow UI documentation")
    print(f"   • JSON details: docs/tutorials/MLflow_UI_Documentation.json")
    print(f"   • Human summary: docs/tutorials/MLflow_Demo_Summary.md")

    # Print key URLs for manual exploration
    print("\n🔗 Key URLs to Explore:")
    print(f"   • Main Dashboard: http://localhost:5000")
    print(f"   • Model Registry: http://localhost:5000/#/models")

    for exp in ui_documentation["experiments"]:
        print(f"   • {exp['name']}: {exp['url']}")

    return ui_documentation

if __name__ == "__main__":
    create_mlflow_ui_documentation()