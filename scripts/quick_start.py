#!/usr/bin/env python3
"""
Quick Start Script for MLOps Demo
Sets up the complete environment and runs initial validation
"""

import subprocess
import sys
import time
import requests
import os
from pathlib import Path

def run_command(command, description, timeout=300):
    """Run a command with description and error handling"""
    print(f"⏳ {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(f"   Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def check_prerequisites():
    """Check if required software is installed"""
    print("🔍 Checking prerequisites...")

    prerequisites = {
        "docker": "docker --version",
        "docker-compose": "docker-compose --version",
        "python3": "python3 --version",
        "pip": "pip --version"
    }

    missing = []
    for name, command in prerequisites.items():
        if not run_command(command, f"Checking {name}", timeout=10):
            missing.append(name)

    if missing:
        print(f"\n❌ Missing prerequisites: {', '.join(missing)}")
        print("Please install the missing software and try again.")
        return False

    print("✅ All prerequisites are available")
    return True

def setup_environment():
    """Set up the MLOps environment"""
    print("\n🏗️  Setting up MLOps environment...")

    # Create necessary directories
    directories = [
        "data/raw", "data/processed", "models", "logs",
        "monitoring/reports", "tests/reports"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("✅ Directory structure created")
    return True

def start_services():
    """Start Docker services"""
    print("\n🚀 Starting Docker services...")

    # Stop any existing services
    run_command("docker-compose down", "Stopping existing services", timeout=60)

    # Build and start services
    if not run_command("docker-compose up --build -d", "Building and starting services", timeout=600):
        return False

    # Wait for services to be ready
    print("⏳ Waiting for services to start...")
    time.sleep(30)

    return True

def wait_for_services():
    """Wait for all services to be healthy"""
    print("\n🔍 Waiting for services to be ready...")

    services = {
        "API": "http://localhost:8000/health",
        "MLflow": "http://localhost:5000",
        "Prometheus": "http://localhost:9090",
        "Grafana": "http://localhost:3000"
    }

    max_retries = 30
    retry_interval = 10

    for service_name, url in services.items():
        print(f"   Checking {service_name}...")

        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"   ✅ {service_name} is ready")
                    break
                else:
                    print(f"   ⏳ {service_name} not ready (status: {response.status_code})")
            except requests.RequestException:
                print(f"   ⏳ {service_name} not ready (connection failed)")

            if attempt < max_retries - 1:
                time.sleep(retry_interval)
            else:
                print(f"   ❌ {service_name} failed to start after {max_retries} attempts")
                return False

    print("✅ All services are ready!")
    return True

def create_datasets():
    """Create sample datasets"""
    print("\n📊 Creating sample datasets...")

    return run_command(
        "docker-compose exec -T api python data/create_datasets.py",
        "Creating datasets",
        timeout=120
    )

def train_models():
    """Train initial models"""
    print("\n🤖 Training initial models...")

    return run_command(
        "docker-compose exec -T api python scripts/train_all_models.py",
        "Training models",
        timeout=600
    )

def run_tests():
    """Run test suite"""
    print("\n🧪 Running tests...")

    return run_command(
        "docker-compose exec -T api pytest tests/ -v --tb=short",
        "Running tests",
        timeout=300
    )

def validate_setup():
    """Validate the complete setup"""
    print("\n✅ Validating setup...")

    # Test API endpoints
    test_cases = [
        {
            "name": "Health Check",
            "url": "http://localhost:8000/health",
            "method": "GET"
        },
        {
            "name": "Models List",
            "url": "http://localhost:8000/api/v1/models",
            "method": "GET"
        },
        {
            "name": "Iris Prediction",
            "url": "http://localhost:8000/api/v1/iris/predict",
            "method": "POST",
            "data": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }
    ]

    all_passed = True
    for test in test_cases:
        try:
            if test["method"] == "GET":
                response = requests.get(test["url"], timeout=10)
            else:
                response = requests.post(test["url"], json=test.get("data"), timeout=10)

            if response.status_code == 200:
                print(f"   ✅ {test['name']}: PASSED")
            else:
                print(f"   ❌ {test['name']}: FAILED (status: {response.status_code})")
                all_passed = False
        except Exception as e:
            print(f"   ❌ {test['name']}: FAILED ({e})")
            all_passed = False

    return all_passed

def print_access_info():
    """Print access information for services"""
    print("\n" + "="*60)
    print("🎉 MLOps Demo Setup Complete!")
    print("="*60)

    print("\n🔗 Service Access URLs:")
    print("   • API Documentation: http://localhost:8000/docs")
    print("   • MLflow UI:         http://localhost:5000")
    print("   • Grafana Dashboard: http://localhost:3000 (admin/admin)")
    print("   • Prometheus:        http://localhost:9090")
    print("   • Jupyter Notebook:  http://localhost:8888")

    print("\n🎯 Quick Start Commands:")
    print("   • Run demo:          python scripts/demo_script.py")
    print("   • Run tests:         docker-compose exec api pytest tests/")
    print("   • View logs:         docker-compose logs -f api")
    print("   • Stop services:     docker-compose down")

    print("\n📚 Next Steps:")
    print("   1. Explore the API documentation")
    print("   2. Run the interactive demo")
    print("   3. Check MLflow experiments")
    print("   4. View monitoring dashboards")
    print("   5. Read the documentation in docs/")

def main():
    """Main setup function"""
    print("🚀 MLOps Demo Quick Start")
    print("=" * 50)

    steps = [
        ("Prerequisites", check_prerequisites),
        ("Environment", setup_environment),
        ("Services", start_services),
        ("Service Health", wait_for_services),
        ("Datasets", create_datasets),
        ("Models", train_models),
        ("Tests", run_tests),
        ("Validation", validate_setup),
    ]

    start_time = time.time()

    for step_name, step_function in steps:
        print(f"\n📋 Step: {step_name}")
        print("-" * 40)

        if not step_function():
            print(f"\n❌ Setup failed at step: {step_name}")
            print("Please check the error messages above and try again.")
            sys.exit(1)

    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Total setup time: {elapsed_time:.1f} seconds")

    print_access_info()

    # Ask if user wants to run the demo
    print(f"\n🎬 Would you like to run the interactive demo now? (y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice in ['y', 'yes']:
            print("\n🎭 Starting interactive demo...")
            os.system("python scripts/demo_script.py")
    except KeyboardInterrupt:
        print("\nSetup completed. You can run the demo later with: python scripts/demo_script.py")

if __name__ == "__main__":
    main()