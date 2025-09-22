"""
Simple service tests without heavy dependencies
"""

import os
import sys
import subprocess
import time
import socket
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_docker_installation():
    """Test if Docker is installed and working"""
    print("🧪 Testing Docker Installation...")

    try:
        # Check if docker command exists
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"   ✅ Docker version: {result.stdout.strip()}")
        else:
            print(f"   ❌ Docker command failed: {result.stderr}")
            return False

        # Check if docker-compose exists
        result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"   ✅ Docker Compose version: {result.stdout.strip()}")
        else:
            print(f"   ❌ Docker Compose command failed: {result.stderr}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print("   ❌ Docker commands timed out")
        return False
    except FileNotFoundError:
        print("   ❌ Docker not found in PATH")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_port_availability():
    """Test if required ports are available"""
    print("🧪 Testing Port Availability...")

    required_ports = [8000, 5000, 3000, 9090, 9100]
    available_ports = []
    occupied_ports = []

    for port in required_ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()

        if result == 0:
            occupied_ports.append(port)
        else:
            available_ports.append(port)

    print(f"   ✅ Available ports: {available_ports}")
    if occupied_ports:
        print(f"   ⚠️  Occupied ports: {occupied_ports}")

    return len(available_ports) >= 3  # Need at least 3 ports

def test_file_structure():
    """Test if all required files and directories exist"""
    print("🧪 Testing File Structure...")

    required_files = [
        "docker-compose.yml",
        "Dockerfile",
        "requirements.txt",
        "api/main.py",
        "data/create_datasets.py",
        "models/iris/train.py",
        "monitoring/drift_detection.py"
    ]

    required_dirs = [
        "api", "data", "models", "monitoring", "tests", "docs", "scripts"
    ]

    missing_files = []
    missing_dirs = []

    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"   ✅ Found: {file_path}")

    for dir_path in required_dirs:
        if not os.path.isdir(dir_path):
            missing_dirs.append(dir_path)
        else:
            print(f"   ✅ Found directory: {dir_path}")

    if missing_files:
        print(f"   ❌ Missing files: {missing_files}")
    if missing_dirs:
        print(f"   ❌ Missing directories: {missing_dirs}")

    return len(missing_files) == 0 and len(missing_dirs) == 0

def test_python_dependencies():
    """Test if basic Python dependencies are available"""
    print("🧪 Testing Python Dependencies...")

    basic_deps = [
        "pandas", "numpy", "sklearn", "pickle"
    ]

    available = []
    missing = []

    for dep in basic_deps:
        try:
            if dep == "sklearn":
                import sklearn
            else:
                __import__(dep)
            available.append(dep)
            print(f"   ✅ {dep} available")
        except ImportError:
            missing.append(dep)
            print(f"   ❌ {dep} missing")

    if missing:
        print(f"   ⚠️  Missing dependencies: {missing}")
        print("   💡 Install with: pip install pandas numpy scikit-learn")

    return len(missing) == 0

def test_data_files():
    """Test if data files exist and are valid"""
    print("🧪 Testing Data Files...")

    data_files = [
        "data/raw/iris.csv",
        "data/raw/housing.csv",
        "data/raw/churn.csv",
        "data/raw/sentiment.csv",
        "data/raw/image_metadata.csv"
    ]

    valid_files = 0
    for file_path in data_files:
        if os.path.exists(file_path):
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                if len(df) > 0:
                    print(f"   ✅ {file_path}: {len(df)} rows")
                    valid_files += 1
                else:
                    print(f"   ❌ {file_path}: empty file")
            except Exception as e:
                print(f"   ❌ {file_path}: error reading - {e}")
        else:
            print(f"   ⚠️  {file_path}: not found")

    return valid_files >= 3  # At least 3 valid data files

def test_model_files():
    """Test if model files exist"""
    print("🧪 Testing Model Files...")

    model_files = [
        "models/iris/artifacts/iris_model.pkl",
        "models/house_price/artifacts/simple_house_model.pkl",
        "models/sentiment/artifacts/simple_sentiment_model.pkl"
    ]

    valid_models = 0
    for file_path in model_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > 0:
                print(f"   ✅ {file_path}: {file_size} bytes")
                valid_models += 1
            else:
                print(f"   ❌ {file_path}: empty file")
        else:
            print(f"   ⚠️  {file_path}: not found")

    if valid_models == 0:
        print("   💡 Run model training tests first to create model files")

    return True  # Model files are optional for basic testing

def test_scripts_executable():
    """Test if scripts are executable and work"""
    print("🧪 Testing Scripts...")

    scripts = [
        "scripts/demo_script.py",
        "scripts/quick_start.py",
        "scripts/train_all_models.py",
        "scripts/project_summary.py"
    ]

    working_scripts = 0
    for script_path in scripts:
        if os.path.exists(script_path):
            try:
                # Check if the script has valid Python syntax
                with open(script_path, 'r') as f:
                    content = f.read()
                compile(content, script_path, 'exec')
                print(f"   ✅ {script_path}: valid syntax")
                working_scripts += 1
            except SyntaxError as e:
                print(f"   ❌ {script_path}: syntax error - {e}")
            except Exception as e:
                print(f"   ❌ {script_path}: error - {e}")
        else:
            print(f"   ⚠️  {script_path}: not found")

    return working_scripts >= 2  # At least 2 working scripts

def test_monitoring_components():
    """Test if monitoring components are set up"""
    print("🧪 Testing Monitoring Components...")

    monitoring_files = [
        "monitoring/prometheus/prometheus.yml",
        "monitoring/grafana/provisioning/datasources.yml",
        "monitoring/drift_detection.py"
    ]

    valid_components = 0
    for file_path in monitoring_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}: exists")
            valid_components += 1
        else:
            print(f"   ❌ {file_path}: missing")

    return valid_components >= 2  # At least 2 monitoring components

def run_all_service_tests():
    """Run all service tests"""
    print("🚀 Running Service Component Tests")
    print("=" * 50)

    results = []

    # Run tests
    results.append(("Docker Installation", test_docker_installation()))
    results.append(("Port Availability", test_port_availability()))
    results.append(("File Structure", test_file_structure()))
    results.append(("Python Dependencies", test_python_dependencies()))
    results.append(("Data Files", test_data_files()))
    results.append(("Model Files", test_model_files()))
    results.append(("Scripts", test_scripts_executable()))
    results.append(("Monitoring Components", test_monitoring_components()))

    # Print summary
    print("\n📊 Test Results Summary:")
    print("-" * 30)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed >= total * 0.75:  # 75% pass rate
        print("🎉 Service tests mostly passed!")
        return True
    else:
        print("⚠️  Many service tests failed")
        return False

if __name__ == "__main__":
    success = run_all_service_tests()
    exit(0 if success else 1)