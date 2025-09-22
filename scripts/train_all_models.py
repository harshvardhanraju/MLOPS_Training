"""
Script to train all models and track them in MLflow
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.mlflow_config import setup_mlflow

def run_training_script(script_path, model_name):
    """Run a training script and handle errors"""
    try:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} model...")
        print(f"{'='*60}")

        # Change to the script directory and run
        script_dir = os.path.dirname(script_path)
        script_name = os.path.basename(script_path)

        result = subprocess.run(
            [sys.executable, script_name],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )

        if result.returncode == 0:
            print(f"✅ {model_name} model trained successfully!")
            print("Output:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print(f"❌ {model_name} model training failed!")
            print("Error:", result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} model training timed out!")
        return False
    except Exception as e:
        print(f"🔥 {model_name} model training crashed: {e}")
        return False

def main():
    """Train all models"""
    # Setup MLflow
    setup_mlflow()

    # Create datasets first
    print("Creating datasets...")
    try:
        subprocess.run([sys.executable, "data/create_datasets.py"], check=True)
        print("✅ Datasets created successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create datasets: {e}")
        return

    # Training scripts and their corresponding model names
    training_jobs = [
        ("models/iris/train.py", "Iris Classifier"),
        ("models/house_price/train.py", "House Price Predictor"),
        ("models/sentiment/train.py", "Sentiment Analyzer"),
        ("models/churn/train.py", "Churn Predictor"),
        ("models/image_classification/train.py", "Image Classifier"),
    ]

    results = {}

    for script_path, model_name in training_jobs:
        if os.path.exists(script_path):
            success = run_training_script(script_path, model_name)
            results[model_name] = success
        else:
            print(f"❌ Training script not found: {script_path}")
            results[model_name] = False

    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")

    for model_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model_name:25} {status}")

    successful_models = sum(results.values())
    total_models = len(results)

    print(f"\nOverall: {successful_models}/{total_models} models trained successfully")

    if successful_models == total_models:
        print("🎉 All models trained successfully!")
    else:
        print("⚠️  Some models failed to train. Check the logs above.")

    print(f"\n💡 To view results, start MLflow UI:")
    print("   mlflow ui --host 0.0.0.0 --port 5000")

if __name__ == "__main__":
    main()