"""
Data drift monitoring script
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from monitoring.drift_detection import DriftDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_data_drift() -> dict:
    """Check data drift for all models"""
    logger.info("Starting data drift monitoring...")

    models_config = {
        'iris': 'data/raw/iris.csv',
        'house_price': 'data/raw/housing.csv',
        'churn': 'data/raw/churn.csv',
        'sentiment': 'data/raw/sentiment.csv',
        'image': 'data/raw/image_metadata.csv'
    }

    results = {
        'timestamp': datetime.now().isoformat(),
        'models': {},
        'overall_status': 'no_drift',
        'drift_detected_count': 0,
        'total_models_checked': 0
    }

    for model_name, reference_path in models_config.items():
        if os.path.exists(reference_path):
            try:
                logger.info(f"Checking drift for {model_name}...")
                detector = DriftDetector(model_name, reference_path)

                # For demo purposes, generate synthetic data with varying drift levels
                # In production, this would be your actual current/incoming data
                drift_scenarios = [0.1, 0.3, 0.5]  # Light, medium, heavy drift
                scenario_results = {}

                for i, drift_level in enumerate(drift_scenarios):
                    current_data = detector.generate_synthetic_drift_data(drift_intensity=drift_level)
                    drift_summary = detector.detect_drift(current_data, save_report=False)

                    scenario_results[f'scenario_{i+1}_drift_{drift_level}'] = {
                        'drift_detected': drift_summary['dataset_drift_detected'],
                        'drift_share': drift_summary['drift_share'],
                        'drifted_columns': drift_summary['number_of_drifted_columns']
                    }

                    if drift_summary['dataset_drift_detected']:
                        results['drift_detected_count'] += 1
                        results['overall_status'] = 'drift_detected'

                # Use the medium drift scenario as the "current" state for reporting
                current_data = detector.generate_synthetic_drift_data(drift_intensity=0.2)
                main_drift_summary = detector.detect_drift(current_data, save_report=True)

                results['models'][model_name] = {
                    'status': 'drift_detected' if main_drift_summary['dataset_drift_detected'] else 'no_drift',
                    'drift_share': main_drift_summary['drift_share'],
                    'drifted_columns': main_drift_summary['number_of_drifted_columns'],
                    'total_columns': main_drift_summary['number_of_columns'],
                    'data_quality_passed': main_drift_summary['data_quality_passed'],
                    'reference_shape': main_drift_summary['reference_data_shape'],
                    'current_shape': main_drift_summary['current_data_shape'],
                    'column_drift': main_drift_summary['column_drift'],
                    'scenarios': scenario_results  # Include all scenarios for analysis
                }

                results['total_models_checked'] += 1

                # Log results
                if main_drift_summary['dataset_drift_detected']:
                    logger.warning(f"DRIFT DETECTED for {model_name}: {main_drift_summary['drift_share']:.2%} of features drifted")
                    logger.warning(f"  Drifted columns: {main_drift_summary['number_of_drifted_columns']}/{main_drift_summary['number_of_columns']}")
                else:
                    logger.info(f"No significant drift detected for {model_name}")

            except Exception as e:
                logger.error(f"Error checking drift for {model_name}: {e}")
                results['models'][model_name] = {
                    'status': 'error',
                    'error': str(e)
                }

        else:
            logger.warning(f"Reference data not found for {model_name}: {reference_path}")
            results['models'][model_name] = {
                'status': 'no_reference_data',
                'message': f'Reference data not found: {reference_path}'
            }

    # Save consolidated results
    os.makedirs("monitoring/reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"monitoring/reports/drift_check_{timestamp}.json"

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Log summary
    logger.info(f"Data drift monitoring completed:")
    logger.info(f"  Overall status: {results['overall_status']}")
    logger.info(f"  Models checked: {results['total_models_checked']}")
    logger.info(f"  Drift detected in: {results['drift_detected_count']} models")
    logger.info(f"  Results saved to: {results_path}")

    # Create alerts for models with significant drift
    drift_alerts = []
    for model_name, model_data in results['models'].items():
        if model_data.get('status') == 'drift_detected':
            drift_share = model_data.get('drift_share', 0)
            if drift_share > 0.3:  # Alert if more than 30% of features drifted
                drift_alerts.append({
                    'model': model_name,
                    'drift_share': drift_share,
                    'drifted_columns': model_data.get('drifted_columns', 0),
                    'severity': 'critical' if drift_share > 0.5 else 'warning'
                })

    if drift_alerts:
        logger.error(f"DRIFT ALERTS: {len(drift_alerts)} models require attention")
        for alert in drift_alerts:
            logger.error(f"  {alert['model']}: {alert['drift_share']:.2%} drift ({alert['severity']})")

    return {
        'results': results,
        'results_path': results_path,
        'drift_alerts': drift_alerts,
        'status': results['overall_status']
    }

def main():
    """Main drift checking function"""
    result = check_data_drift()

    # Exit with error code if significant drift detected
    if result['drift_alerts']:
        logger.error("Exiting with error code due to significant data drift")
        exit(1)

    logger.info("Data drift monitoring completed successfully")
    return result

if __name__ == "__main__":
    main()