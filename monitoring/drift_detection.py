"""
Data drift detection using Evidently AI
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_suite import MetricSuite
from evidently.metrics import DataDriftTable, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfColumnsWithMissingValues, TestNumberOfRowsWithMissingValues, TestNumberOfConstantColumns, TestNumberOfDuplicatedRows, TestNumberOfDuplicatedColumns, TestColumnsType, TestNumberOfDriftedColumns
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, model_name: str, reference_data_path: str):
        self.model_name = model_name
        self.reference_data_path = reference_data_path
        self.reports_dir = f"monitoring/reports/{model_name}"
        os.makedirs(self.reports_dir, exist_ok=True)

        # Load reference data
        self.reference_data = pd.read_csv(reference_data_path)
        logger.info(f"Loaded reference data for {model_name}: {self.reference_data.shape}")

    def detect_drift(self, current_data: pd.DataFrame, save_report: bool = True) -> dict:
        """Detect data drift between reference and current data"""

        # Ensure column alignment
        common_columns = list(set(self.reference_data.columns) & set(current_data.columns))
        if not common_columns:
            raise ValueError("No common columns found between reference and current data")

        reference_subset = self.reference_data[common_columns]
        current_subset = current_data[common_columns]

        # Create column mapping
        column_mapping = ColumnMapping()

        # Configure based on model type
        if self.model_name == 'iris':
            column_mapping.target = 'target' if 'target' in common_columns else None
            column_mapping.numerical_features = [col for col in common_columns if col not in ['target', 'target_name']]
        elif self.model_name == 'house_price':
            column_mapping.target = 'price' if 'price' in common_columns else None
            column_mapping.numerical_features = [col for col in common_columns if col != 'price']
        elif self.model_name == 'churn':
            column_mapping.target = 'churn' if 'churn' in common_columns else None
            column_mapping.categorical_features = ['contract_type', 'payment_method', 'internet_service', 'online_security', 'tech_support', 'streaming_tv', 'paperless_billing', 'partner', 'dependents']
            column_mapping.categorical_features = [col for col in column_mapping.categorical_features if col in common_columns]
            column_mapping.numerical_features = [col for col in common_columns if col not in column_mapping.categorical_features + ['churn']]
        elif self.model_name == 'sentiment':
            column_mapping.target = 'label' if 'label' in common_columns else None
            column_mapping.text_features = ['text'] if 'text' in common_columns else []
        else:
            # Default configuration
            column_mapping.numerical_features = [col for col in common_columns if current_subset[col].dtype in ['int64', 'float64']]
            column_mapping.categorical_features = [col for col in common_columns if current_subset[col].dtype == 'object']

        # Create drift report
        drift_report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
            DatasetMissingValuesMetric()
        ])

        drift_report.run(
            reference_data=reference_subset,
            current_data=current_subset,
            column_mapping=column_mapping
        )

        # Create data quality test suite
        data_quality_suite = TestSuite(tests=[
            TestNumberOfColumnsWithMissingValues(),
            TestNumberOfRowsWithMissingValues(),
            TestNumberOfConstantColumns(),
            TestNumberOfDuplicatedRows(),
            TestNumberOfDuplicatedColumns(),
            TestColumnsType(),
            TestNumberOfDriftedColumns()
        ])

        data_quality_suite.run(
            reference_data=reference_subset,
            current_data=current_subset,
            column_mapping=column_mapping
        )

        # Extract results
        drift_results = drift_report.as_dict()
        quality_results = data_quality_suite.as_dict()

        # Parse drift metrics
        dataset_drift = drift_results['metrics'][0]['result']
        drift_summary = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'dataset_drift_detected': dataset_drift.get('dataset_drift', False),
            'drift_share': dataset_drift.get('drift_share', 0.0),
            'number_of_columns': dataset_drift.get('number_of_columns', 0),
            'number_of_drifted_columns': dataset_drift.get('number_of_drifted_columns', 0),
            'data_quality_passed': quality_results.get('summary', {}).get('all_passed', False),
            'reference_data_shape': reference_subset.shape,
            'current_data_shape': current_subset.shape
        }

        # Add individual column drift information
        drift_summary['column_drift'] = {}
        if 'metrics' in drift_results and len(drift_results['metrics']) > 1:
            drift_table = drift_results['metrics'][1]['result']
            if 'drift_by_columns' in drift_table:
                for col, drift_info in drift_table['drift_by_columns'].items():
                    drift_summary['column_drift'][col] = {
                        'drift_detected': drift_info.get('drift_detected', False),
                        'drift_score': drift_info.get('drift_score', 0.0)
                    }

        # Save reports if requested
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save drift report
            drift_report_path = f"{self.reports_dir}/drift_report_{timestamp}.html"
            drift_report.save_html(drift_report_path)

            # Save quality report
            quality_report_path = f"{self.reports_dir}/quality_report_{timestamp}.html"
            data_quality_suite.save_html(quality_report_path)

            # Save summary JSON
            summary_path = f"{self.reports_dir}/drift_summary_{timestamp}.json"
            with open(summary_path, 'w') as f:
                json.dump(drift_summary, f, indent=2)

            logger.info(f"Reports saved: {drift_report_path}, {quality_report_path}, {summary_path}")

        return drift_summary

    def generate_synthetic_drift_data(self, drift_intensity: float = 0.3) -> pd.DataFrame:
        """Generate synthetic data with drift for testing"""
        synthetic_data = self.reference_data.copy()

        # Add noise to numerical columns
        numerical_columns = synthetic_data.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if col not in ['target', 'label', 'churn']:  # Don't modify target columns
                noise = np.random.normal(0, synthetic_data[col].std() * drift_intensity, len(synthetic_data))
                synthetic_data[col] = synthetic_data[col] + noise

        # Modify categorical distributions
        categorical_columns = synthetic_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in ['target_name', 'sentiment'] and len(synthetic_data[col].unique()) > 1:
                # Randomly change some values
                mask = np.random.random(len(synthetic_data)) < drift_intensity * 0.5
                if mask.any():
                    unique_values = synthetic_data[col].unique()
                    synthetic_data.loc[mask, col] = np.random.choice(unique_values, mask.sum())

        return synthetic_data

def monitor_all_models():
    """Monitor drift for all models"""
    models_config = {
        'iris': 'data/raw/iris.csv',
        'house_price': 'data/raw/housing.csv',
        'churn': 'data/raw/churn.csv',
        'sentiment': 'data/raw/sentiment.csv',
        'image': 'data/raw/image_metadata.csv'
    }

    results = {}

    for model_name, reference_path in models_config.items():
        if os.path.exists(reference_path):
            try:
                logger.info(f"Monitoring drift for {model_name}...")
                detector = DriftDetector(model_name, reference_path)

                # Generate synthetic current data for demo
                current_data = detector.generate_synthetic_drift_data(drift_intensity=0.2)

                # Detect drift
                drift_summary = detector.detect_drift(current_data)
                results[model_name] = drift_summary

                # Log results
                if drift_summary['dataset_drift_detected']:
                    logger.warning(f"DRIFT DETECTED for {model_name}: {drift_summary['drift_share']:.2%} of features drifted")
                else:
                    logger.info(f"No drift detected for {model_name}")

            except Exception as e:
                logger.error(f"Error monitoring {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        else:
            logger.warning(f"Reference data not found for {model_name}: {reference_path}")

    # Save overall monitoring results
    os.makedirs("monitoring/reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"monitoring/reports/drift_monitoring_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Drift monitoring completed. Results saved to {results_path}")
    return results

if __name__ == "__main__":
    # Run monitoring for all models
    monitor_all_models()