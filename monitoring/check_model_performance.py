"""
Model performance monitoring and alerting
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import logging
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.mlflow_config import setup_mlflow

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPerformanceMonitor:
    def __init__(self):
        setup_mlflow()
        self.reports_dir = "monitoring/reports/performance"
        os.makedirs(self.reports_dir, exist_ok=True)

    def get_model_metrics(self, model_name: str) -> dict:
        """Get latest metrics for a model from MLflow"""
        try:
            client = mlflow.tracking.MlflowClient()

            # Get latest run for the model
            experiment = mlflow.get_experiment_by_name(self._get_experiment_name(model_name))
            if not experiment:
                logger.warning(f"Experiment not found for {model_name}")
                return {}

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )

            if not runs:
                logger.warning(f"No runs found for {model_name}")
                return {}

            latest_run = runs[0]
            metrics = latest_run.data.metrics

            return {
                'run_id': latest_run.info.run_id,
                'start_time': latest_run.info.start_time,
                'metrics': metrics,
                'status': latest_run.info.status
            }

        except Exception as e:
            logger.error(f"Error getting metrics for {model_name}: {e}")
            return {}

    def _get_experiment_name(self, model_name: str) -> str:
        """Map model names to experiment names"""
        experiment_mapping = {
            'iris': 'iris_classification',
            'house_price': 'house_price_prediction',
            'sentiment': 'sentiment_analysis',
            'churn': 'customer_churn_prediction',
            'image': 'image_classification'
        }
        return experiment_mapping.get(model_name, model_name)

    def check_performance_thresholds(self, model_name: str, metrics: dict) -> dict:
        """Check if model metrics meet performance thresholds"""
        thresholds = self._get_performance_thresholds(model_name)
        alerts = []
        status = "healthy"

        for metric_name, threshold in thresholds.items():
            if metric_name in metrics:
                metric_value = metrics[metric_name]

                if metric_name in ['accuracy', 'f1_score', 'r2_score']:
                    # Higher is better
                    if metric_value < threshold:
                        alerts.append({
                            'metric': metric_name,
                            'value': metric_value,
                            'threshold': threshold,
                            'severity': 'critical' if metric_value < threshold * 0.9 else 'warning',
                            'message': f"{metric_name} ({metric_value:.4f}) below threshold ({threshold})"
                        })
                        status = "degraded"

                elif metric_name in ['mse', 'rmse', 'mae']:
                    # Lower is better
                    if metric_value > threshold:
                        alerts.append({
                            'metric': metric_name,
                            'value': metric_value,
                            'threshold': threshold,
                            'severity': 'critical' if metric_value > threshold * 1.1 else 'warning',
                            'message': f"{metric_name} ({metric_value:.4f}) above threshold ({threshold})"
                        })
                        status = "degraded"

        return {
            'model_name': model_name,
            'status': status,
            'alerts': alerts,
            'metrics_checked': len(thresholds),
            'alerts_count': len(alerts)
        }

    def _get_performance_thresholds(self, model_name: str) -> dict:
        """Get performance thresholds for each model"""
        thresholds = {
            'iris': {
                'accuracy': 0.90,
                'f1_score': 0.90
            },
            'house_price': {
                'r2_score': 0.75,
                'rmse': 1.0,
                'mae': 0.8
            },
            'sentiment': {
                'accuracy': 0.80
            },
            'churn': {
                'accuracy': 0.80,
                'f1_score': 0.75,
                'auc': 0.80
            },
            'image': {
                'accuracy': 0.70,
                'f1_score': 0.70
            }
        }
        return thresholds.get(model_name, {})

    def generate_performance_report(self, models: list = None) -> dict:
        """Generate comprehensive performance report"""
        if models is None:
            models = ['iris', 'house_price', 'sentiment', 'churn', 'image']

        report = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'overall_status': 'healthy',
            'total_alerts': 0
        }

        for model_name in models:
            logger.info(f"Checking performance for {model_name}...")

            # Get latest metrics
            model_info = self.get_model_metrics(model_name)

            if model_info and 'metrics' in model_info:
                # Check thresholds
                performance_check = self.check_performance_thresholds(
                    model_name,
                    model_info['metrics']
                )

                report['models'][model_name] = {
                    'latest_metrics': model_info['metrics'],
                    'run_info': {
                        'run_id': model_info.get('run_id'),
                        'start_time': model_info.get('start_time'),
                        'status': model_info.get('status')
                    },
                    'performance_check': performance_check
                }

                # Update overall status
                if performance_check['status'] == 'degraded':
                    report['overall_status'] = 'degraded'

                report['total_alerts'] += performance_check['alerts_count']

            else:
                report['models'][model_name] = {
                    'status': 'no_data',
                    'message': 'No metrics available'
                }

        return report

    def save_report(self, report: dict) -> str:
        """Save performance report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.reports_dir}/performance_report_{timestamp}.json"

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Performance report saved to {report_path}")
        return report_path

    def check_and_alert(self) -> dict:
        """Main monitoring function that checks performance and creates alerts"""
        logger.info("Starting model performance monitoring...")

        # Generate report
        report = self.generate_performance_report()

        # Save report
        report_path = self.save_report(report)

        # Log summary
        logger.info(f"Performance monitoring completed:")
        logger.info(f"  Overall status: {report['overall_status']}")
        logger.info(f"  Total alerts: {report['total_alerts']}")

        # Log individual model status
        for model_name, model_data in report['models'].items():
            if 'performance_check' in model_data:
                status = model_data['performance_check']['status']
                alerts = model_data['performance_check']['alerts_count']
                logger.info(f"  {model_name}: {status} ({alerts} alerts)")
            else:
                logger.info(f"  {model_name}: {model_data.get('status', 'unknown')}")

        # Create alerts for critical issues
        critical_alerts = []
        for model_name, model_data in report['models'].items():
            if 'performance_check' in model_data:
                for alert in model_data['performance_check']['alerts']:
                    if alert['severity'] == 'critical':
                        critical_alerts.append({
                            'model': model_name,
                            **alert
                        })

        if critical_alerts:
            logger.error(f"CRITICAL ALERTS DETECTED: {len(critical_alerts)} critical performance issues")
            for alert in critical_alerts:
                logger.error(f"  {alert['model']}: {alert['message']}")

        return {
            'report': report,
            'report_path': report_path,
            'critical_alerts': critical_alerts,
            'status': report['overall_status']
        }

def main():
    """Main monitoring function"""
    monitor = ModelPerformanceMonitor()
    result = monitor.check_and_alert()

    # Exit with error code if critical alerts
    if result['critical_alerts']:
        logger.error("Exiting with error code due to critical alerts")
        exit(1)

    logger.info("Model performance monitoring completed successfully")
    return result

if __name__ == "__main__":
    main()