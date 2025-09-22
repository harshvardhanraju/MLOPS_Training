#!/usr/bin/env python3
"""
Comprehensive Test Runner for MLOps Demo
Runs all component tests and provides detailed reporting
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def run_test_suite(test_module, test_name):
    """Run a test suite and capture results"""
    print(f"\n🧪 Running {test_name}...")
    print("=" * 60)

    start_time = time.time()

    try:
        # Import and run the test module
        if test_module == "data_components":
            from tests.test_data_components import TestDataCreation, TestDataPersistence

            # Run data creation tests
            test_instance = TestDataCreation()
            results = []

            try:
                test_instance.test_create_iris_dataset()
                results.append(("Iris Dataset", True))
            except Exception as e:
                results.append(("Iris Dataset", False, str(e)))

            try:
                test_instance.test_create_housing_dataset()
                results.append(("Housing Dataset", True))
            except Exception as e:
                results.append(("Housing Dataset", False, str(e)))

            try:
                test_instance.test_create_churn_dataset()
                results.append(("Churn Dataset", True))
            except Exception as e:
                results.append(("Churn Dataset", False, str(e)))

            try:
                test_instance.test_create_sentiment_dataset()
                results.append(("Sentiment Dataset", True))
            except Exception as e:
                results.append(("Sentiment Dataset", False, str(e)))

            try:
                test_instance.test_create_image_dataset()
                results.append(("Image Dataset", True))
            except Exception as e:
                results.append(("Image Dataset", False, str(e)))

        elif test_module == "model_simple":
            from tests.test_models_simple import run_all_simple_tests
            success = run_all_simple_tests()
            results = [("Model Pipeline", success)]

        elif test_module == "services_simple":
            from tests.test_services_simple import run_all_service_tests
            success = run_all_service_tests()
            results = [("Service Components", success)]

        elif test_module == "monitoring_simple":
            from tests.test_monitoring_simple import run_all_monitoring_tests
            success = run_all_monitoring_tests()
            results = [("Monitoring Components", success)]

        elif test_module == "integration_simple":
            from tests.test_integration_simple import run_all_integration_tests
            success = run_all_integration_tests()
            results = [("Integration Pipeline", success)]

        else:
            results = [("Unknown Test", False, "Test module not found")]

        end_time = time.time()
        duration = end_time - start_time

        # Count successes
        passed = sum(1 for result in results if len(result) == 2 and result[1])
        total = len(results)

        return {
            'test_name': test_name,
            'duration': duration,
            'total_tests': total,
            'passed_tests': passed,
            'failed_tests': total - passed,
            'success_rate': passed / total if total > 0 else 0,
            'details': results
        }

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        return {
            'test_name': test_name,
            'duration': duration,
            'total_tests': 1,
            'passed_tests': 0,
            'failed_tests': 1,
            'success_rate': 0,
            'error': str(e),
            'details': [("Test Execution", False, str(e))]
        }

def generate_test_report(test_results):
    """Generate a comprehensive test report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_test_suites': len(test_results),
            'total_individual_tests': sum(r['total_tests'] for r in test_results),
            'total_passed': sum(r['passed_tests'] for r in test_results),
            'total_failed': sum(r['failed_tests'] for r in test_results),
            'overall_success_rate': 0,
            'total_duration': sum(r['duration'] for r in test_results)
        },
        'test_suites': test_results
    }

    # Calculate overall success rate
    if report['summary']['total_individual_tests'] > 0:
        report['summary']['overall_success_rate'] = (
            report['summary']['total_passed'] / report['summary']['total_individual_tests']
        )

    return report

def print_test_summary(report):
    """Print a formatted test summary"""
    print("\n" + "="*80)
    print("🚀 MLOps Demo - COMPREHENSIVE TEST RESULTS")
    print("="*80)

    summary = report['summary']

    print(f"\n📊 Overall Summary:")
    print(f"   Test Suites Run: {summary['total_test_suites']}")
    print(f"   Individual Tests: {summary['total_individual_tests']}")
    print(f"   Passed: {summary['total_passed']}")
    print(f"   Failed: {summary['total_failed']}")
    print(f"   Success Rate: {summary['overall_success_rate']:.1%}")
    print(f"   Total Duration: {summary['total_duration']:.2f} seconds")

    print(f"\n📋 Test Suite Results:")
    print("-" * 60)

    for result in report['test_suites']:
        status_emoji = "✅" if result['success_rate'] >= 0.8 else "⚠️" if result['success_rate'] >= 0.5 else "❌"
        print(f"{status_emoji} {result['test_name']:<25} "
              f"{result['passed_tests']:>2}/{result['total_tests']:>2} "
              f"({result['success_rate']:>5.1%}) "
              f"{result['duration']:>6.2f}s")

    # Show detailed failures if any
    failures = []
    for suite in report['test_suites']:
        for detail in suite.get('details', []):
            if len(detail) >= 3 and not detail[1]:  # Failed test
                failures.append(f"{suite['test_name']}: {detail[0]} - {detail[2]}")

    if failures:
        print(f"\n❌ Detailed Failures:")
        print("-" * 40)
        for failure in failures[:10]:  # Show first 10 failures
            print(f"   • {failure}")
        if len(failures) > 10:
            print(f"   ... and {len(failures) - 10} more failures")

    # Overall assessment
    print(f"\n🎯 Overall Assessment:")
    if summary['overall_success_rate'] >= 0.9:
        print("   🎉 EXCELLENT! The MLOps demo is ready for production.")
    elif summary['overall_success_rate'] >= 0.8:
        print("   ✅ GOOD! The MLOps demo is mostly functional.")
    elif summary['overall_success_rate'] >= 0.6:
        print("   ⚠️  FAIR! Some components need attention.")
    else:
        print("   ❌ POOR! Major issues need to be resolved.")

    print("\n" + "="*80)

def save_test_report(report):
    """Save test report to file"""
    os.makedirs("tests/reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"tests/reports/comprehensive_test_report_{timestamp}.json"

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Detailed report saved to: {report_path}")
    return report_path

def main():
    """Main test runner"""
    print("🚀 Starting Comprehensive MLOps Demo Testing")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Define test suites
    test_suites = [
        ("data_components", "Data Components"),
        ("model_simple", "ML Model Components"),
        ("services_simple", "Service Components"),
        ("monitoring_simple", "Monitoring Components"),
        ("integration_simple", "Integration Tests")
    ]

    results = []

    # Run all test suites
    for test_module, test_name in test_suites:
        result = run_test_suite(test_module, test_name)
        results.append(result)

        # Brief status update
        status = "✅" if result['success_rate'] >= 0.8 else "⚠️" if result['success_rate'] >= 0.5 else "❌"
        print(f"\n{status} {test_name}: {result['passed_tests']}/{result['total_tests']} passed "
              f"({result['success_rate']:.1%})")

    # Generate and display final report
    report = generate_test_report(results)
    print_test_summary(report)

    # Save report
    report_path = save_test_report(report)

    # Exit with appropriate code
    overall_success = report['summary']['overall_success_rate'] >= 0.8

    if overall_success:
        print("\n🎉 Testing completed successfully!")
        exit(0)
    else:
        print("\n⚠️  Testing completed with issues. Check the report for details.")
        exit(1)

if __name__ == "__main__":
    main()