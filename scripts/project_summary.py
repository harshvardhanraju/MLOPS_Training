#!/usr/bin/env python3
"""
MLOps Demo Project Summary
Generates a comprehensive overview of the implemented MLOps pipeline
"""

import os
import json
from pathlib import Path
from datetime import datetime

def count_files_by_extension(directory: str, extensions: list) -> dict:
    """Count files by extension in a directory"""
    counts = {ext: 0 for ext in extensions}
    counts['total'] = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = Path(file)
            ext = file_path.suffix.lower()
            if ext in extensions:
                counts[ext] += 1
            counts['total'] += 1

    return counts

def analyze_project_structure():
    """Analyze the project structure and return statistics"""
    structure = {}

    # Core components
    components = {
        'models': 'models/',
        'api': 'api/',
        'monitoring': 'monitoring/',
        'tests': 'tests/',
        'docs': 'docs/',
        'data': 'data/',
        'scripts': 'scripts/',
        'ci_cd': '.github/workflows/',
        'docker': ['Dockerfile', 'docker-compose.yml', 'docker/']
    }

    for component, path in components.items():
        if isinstance(path, list):
            # Multiple paths/files
            structure[component] = {'exists': all(os.path.exists(p) for p in path)}
        else:
            structure[component] = {
                'exists': os.path.exists(path),
                'files': count_files_by_extension(path, ['.py', '.md', '.yml', '.yaml', '.json'])
            }

    return structure

def get_model_summary():
    """Get summary of implemented models"""
    models = {
        'iris': {
            'name': 'Iris Classification',
            'type': 'Multiclass Classification',
            'algorithm': 'Random Forest',
            'features': 4,
            'classes': 3,
            'metrics': ['accuracy', 'f1_score', 'precision', 'recall']
        },
        'house_price': {
            'name': 'House Price Prediction',
            'type': 'Regression',
            'algorithm': 'XGBoost',
            'features': 8,
            'target': 'price',
            'metrics': ['r2_score', 'rmse', 'mae']
        },
        'sentiment': {
            'name': 'Sentiment Analysis',
            'type': 'Text Classification',
            'algorithm': 'DistilBERT',
            'features': 'text',
            'classes': 3,
            'metrics': ['accuracy', 'f1_score']
        },
        'churn': {
            'name': 'Customer Churn Prediction',
            'type': 'Binary Classification',
            'algorithm': 'LightGBM',
            'features': 14,
            'classes': 2,
            'metrics': ['accuracy', 'f1_score', 'auc', 'precision', 'recall']
        },
        'image': {
            'name': 'Image Classification',
            'type': 'Computer Vision',
            'algorithm': 'CNN',
            'features': '32x32x3',
            'classes': 3,
            'metrics': ['accuracy', 'f1_score']
        }
    }

    return models

def get_technology_stack():
    """Get the complete technology stack"""
    stack = {
        'data_versioning': {
            'tool': 'DVC (Data Version Control)',
            'purpose': 'Data and pipeline versioning',
            'features': ['Data tracking', 'Pipeline automation', 'Reproducibility']
        },
        'experiment_tracking': {
            'tool': 'MLflow',
            'purpose': 'Model versioning and experiment tracking',
            'features': ['Experiment logging', 'Model registry', 'Model serving']
        },
        'model_serving': {
            'tool': 'FastAPI',
            'purpose': 'REST API for model serving',
            'features': ['Automatic docs', 'High performance', 'Type validation']
        },
        'monitoring': {
            'tool': 'Prometheus + Grafana',
            'purpose': 'System and model monitoring',
            'features': ['Metrics collection', 'Alerting', 'Dashboards']
        },
        'ci_cd': {
            'tool': 'GitHub Actions',
            'purpose': 'Continuous integration and deployment',
            'features': ['Automated testing', 'Model retraining', 'Deployment']
        },
        'containerization': {
            'tool': 'Docker + Docker Compose',
            'purpose': 'Application containerization and orchestration',
            'features': ['Consistent environments', 'Service orchestration', 'Scalability']
        },
        'drift_detection': {
            'tool': 'Evidently AI',
            'purpose': 'Data drift and model performance monitoring',
            'features': ['Statistical tests', 'Visual reports', 'Automated alerts']
        },
        'testing': {
            'tool': 'Pytest',
            'purpose': 'Automated testing framework',
            'features': ['Unit tests', 'Integration tests', 'Performance validation']
        }
    }

    return stack

def get_mlops_capabilities():
    """Get MLOps capabilities implemented"""
    capabilities = {
        'data_management': {
            'implemented': True,
            'features': [
                'Data versioning with DVC',
                'Automated dataset creation',
                'Data quality validation',
                'Data drift detection'
            ]
        },
        'model_development': {
            'implemented': True,
            'features': [
                'Multiple model types (5 models)',
                'Experiment tracking with MLflow',
                'Model versioning and registry',
                'Hyperparameter logging'
            ]
        },
        'model_deployment': {
            'implemented': True,
            'features': [
                'REST API serving',
                'Containerized deployment',
                'Health checks',
                'API documentation'
            ]
        },
        'monitoring_observability': {
            'implemented': True,
            'features': [
                'System metrics (CPU, memory)',
                'Model performance metrics',
                'Request/response monitoring',
                'Custom dashboards'
            ]
        },
        'ci_cd_automation': {
            'implemented': True,
            'features': [
                'Automated testing',
                'Model validation',
                'Automated deployment',
                'Scheduled retraining'
            ]
        },
        'governance_compliance': {
            'implemented': True,
            'features': [
                'Model lineage tracking',
                'Audit trails',
                'Version control',
                'Documentation'
            ]
        }
    }

    return capabilities

def generate_summary_report():
    """Generate comprehensive project summary"""
    report = {
        'project_info': {
            'name': 'MLOps Complete Demo',
            'description': 'Comprehensive MLOps pipeline demonstration',
            'created': datetime.now().isoformat(),
            'version': '1.0.0'
        },
        'project_structure': analyze_project_structure(),
        'models': get_model_summary(),
        'technology_stack': get_technology_stack(),
        'mlops_capabilities': get_mlops_capabilities(),
        'key_features': [
            '5 different ML models showcasing various ML paradigms',
            'Complete CI/CD pipeline with automated testing',
            'Real-time monitoring and alerting',
            'Data drift detection and model performance tracking',
            'Containerized microservices architecture',
            'Interactive API documentation',
            'Comprehensive test coverage',
            'Knowledge transfer documentation'
        ],
        'deployment_modes': {
            'development': {
                'description': 'Local development with hot reload',
                'command': 'docker-compose -f docker-compose.yml -f docker-compose.dev.yml up'
            },
            'production': {
                'description': 'Production deployment with all services',
                'command': 'docker-compose up -d'
            },
            'testing': {
                'description': 'Testing environment for CI/CD',
                'command': 'pytest tests/ -v'
            }
        },
        'access_points': {
            'api_docs': 'http://localhost:8000/docs',
            'mlflow_ui': 'http://localhost:5000',
            'grafana_dashboard': 'http://localhost:3000',
            'prometheus_metrics': 'http://localhost:9090',
            'jupyter_notebooks': 'http://localhost:8888'
        }
    }

    return report

def print_summary():
    """Print a human-readable summary"""
    report = generate_summary_report()

    print("=" * 80)
    print("🚀 MLOps Complete Demo - Project Summary")
    print("=" * 80)

    print(f"\n📋 Project: {report['project_info']['name']}")
    print(f"📝 Description: {report['project_info']['description']}")
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n🤖 Machine Learning Models:")
    print("-" * 40)
    for model_id, model_info in report['models'].items():
        print(f"  {model_info['name']}")
        print(f"    • Type: {model_info['type']}")
        print(f"    • Algorithm: {model_info['algorithm']}")
        print(f"    • Features: {model_info.get('features', 'N/A')}")

    print("\n🔧 Technology Stack:")
    print("-" * 40)
    for category, info in report['technology_stack'].items():
        print(f"  {info['tool']}")
        print(f"    • Purpose: {info['purpose']}")

    print("\n✨ Key MLOps Capabilities:")
    print("-" * 40)
    for capability, details in report['mlops_capabilities'].items():
        if details['implemented']:
            status = "✅"
        else:
            status = "❌"
        print(f"  {status} {capability.replace('_', ' ').title()}")

    print("\n🔗 Service Access Points:")
    print("-" * 40)
    for service, url in report['access_points'].items():
        print(f"  • {service.replace('_', ' ').title()}: {url}")

    print("\n🎯 Quick Start Commands:")
    print("-" * 40)
    print("  • Setup environment:     python scripts/quick_start.py")
    print("  • Run demo:              python scripts/demo_script.py")
    print("  • Start services:        docker-compose up -d")
    print("  • Run tests:             pytest tests/ -v")
    print("  • Stop services:         docker-compose down")

    print("\n📚 Documentation:")
    print("-" * 40)
    print("  • Setup Guide:           docs/guides/setup_guide.md")
    print("  • Architecture:          docs/architecture/")
    print("  • Presentations:         docs/presentations/")
    print("  • API Reference:         http://localhost:8000/docs")

    print("\n" + "=" * 80)
    print("🎉 MLOps Demo Ready for Knowledge Transfer Sessions!")
    print("=" * 80)

def save_json_report():
    """Save detailed JSON report"""
    report = generate_summary_report()

    # Create reports directory
    os.makedirs("reports", exist_ok=True)

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/project_summary_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Detailed report saved to: {filename}")

def main():
    """Main function"""
    print_summary()
    save_json_report()

if __name__ == "__main__":
    main()