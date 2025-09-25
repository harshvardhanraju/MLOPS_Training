#!/usr/bin/env python3
"""
Complete MLflow Demo - Quick version to finalize experiments
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import time

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

def complete_model_comparison():
    """Complete the model comparison experiment"""

    print("🔬 Completing Model Comparison Experiment...")

    # Set experiment
    mlflow.set_experiment("Model Comparison Study")

    # Load wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Models to compare
    models = [
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Logistic Regression", LogisticRegression(random_state=42, max_iter=1000)),
        ("SVM", SVC(random_state=42, probability=True)),
    ]

    results = []

    for model_name, model in models:
        with mlflow.start_run(run_name=f"Model_{model_name.replace(' ', '_')}") as run:
            print(f"    Training {model_name}...")

            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Log parameters and metrics
            mlflow.log_params({
                "model_type": model_name,
                "dataset": "wine",
                "n_features": X.shape[1],
                "n_samples": X.shape[0]
            })

            mlflow.log_metrics({
                "accuracy": accuracy,
                "training_time": training_time
            })

            # Create confusion matrix
            plt.figure(figsize=(6, 5))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=wine.target_names, yticklabels=wine.target_names)
            plt.title(f'{model_name} - Confusion Matrix')
            plt.tight_layout()

            plot_path = f'cm_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.close()

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Tags
            mlflow.set_tags({
                "algorithm": model_name.replace(' ', ''),
                "dataset": "wine",
                "task": "classification"
            })

            results.append({
                "model": model_name,
                "accuracy": accuracy,
                "time": training_time
            })

            print(f"    ✅ {model_name}: {accuracy:.4f} accuracy")

    return results

def create_hyperparameter_tuning_demo():
    """Create hyperparameter tuning demonstration"""

    print("⚙️ Creating Hyperparameter Tuning Demo...")

    mlflow.set_experiment("Hyperparameter Tuning Demo")

    # Wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parameter combinations
    param_grid = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 150, "max_depth": 7},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 3},
        {"n_estimators": 200, "max_depth": 5},
    ]

    best_accuracy = 0
    best_params = None

    for i, params in enumerate(param_grid):
        with mlflow.start_run(run_name=f"HPTune_{i+1:02d}"):
            # Create and train model
            model = RandomForestClassifier(random_state=42, **params)
            model.fit(X_train, y_train)

            # Evaluate
            accuracy = accuracy_score(y_test, model.predict(X_test))

            # Log everything
            mlflow.log_params(params)
            mlflow.log_metrics({
                "accuracy": accuracy,
                "n_features": X.shape[1],
                "test_accuracy": accuracy
            })

            # Simulate training curves
            for epoch in range(20):
                train_loss = 1.0 * np.exp(-epoch/10) + np.random.normal(0, 0.02)
                val_loss = train_loss + 0.05 + np.random.normal(0, 0.02)
                mlflow.log_metrics({
                    "train_loss": max(0, train_loss),
                    "val_loss": max(0, val_loss)
                }, step=epoch)

            mlflow.set_tags({
                "experiment_type": "hyperparameter_tuning",
                "algorithm": "RandomForest",
                "iteration": i+1
            })

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params

            print(f"    ✅ Config {i+1}: {accuracy:.4f}")

    print(f"    🏆 Best: {best_params} with {best_accuracy:.4f}")

def register_models():
    """Register best models"""

    print("📝 Registering Models...")

    client = mlflow.tracking.MlflowClient()

    # Get experiments
    experiments = [
        ("Iris Classification", "iris_classifier"),
        ("Model Comparison Study", "wine_classifier"),
        ("Hyperparameter Tuning Demo", "tuned_wine_classifier")
    ]

    for exp_name, model_name in experiments:
        try:
            exp = client.get_experiment_by_name(exp_name)
            if exp:
                runs = client.search_runs(
                    [exp.experiment_id],
                    order_by=["metrics.accuracy DESC"]
                )

                if runs:
                    best_run = runs[0]
                    model_uri = f"runs:/{best_run.info.run_id}/model"

                    try:
                        # Register model
                        client.create_registered_model(model_name)
                        print(f"    ✅ Created model: {model_name}")
                    except:
                        print(f"    ℹ️  Model {model_name} exists")

                    # Create version
                    version = client.create_model_version(
                        model_name,
                        model_uri,
                        best_run.info.run_id
                    )

                    print(f"    ✅ Version {version.version} created")

        except Exception as e:
            print(f"    ❌ Error with {model_name}: {e}")

def create_summary_dashboard():
    """Create a summary visualization"""

    print("📊 Creating Summary Dashboard...")

    client = mlflow.tracking.MlflowClient()

    # Get all experiments
    experiments = client.search_experiments()

    summary_data = []
    for exp in experiments:
        if exp.name != "Default":
            runs = client.search_runs([exp.experiment_id])
            if runs:
                accuracies = []
                for run in runs:
                    if 'accuracy' in run.data.metrics:
                        accuracies.append(run.data.metrics['accuracy'])

                if accuracies:
                    summary_data.append({
                        'experiment': exp.name,
                        'runs': len(runs),
                        'avg_accuracy': np.mean(accuracies),
                        'max_accuracy': max(accuracies),
                        'min_accuracy': min(accuracies)
                    })

    # Create visualization
    if summary_data:
        df = pd.DataFrame(summary_data)

        plt.figure(figsize=(12, 8))

        # Subplot 1: Number of runs
        plt.subplot(2, 2, 1)
        plt.bar(range(len(df)), df['runs'], alpha=0.7, color='skyblue')
        plt.title('Runs per Experiment')
        plt.xticks(range(len(df)), [name[:15] + '...' if len(name) > 15 else name
                                  for name in df['experiment']], rotation=45)
        plt.ylabel('Number of Runs')

        # Subplot 2: Max accuracy
        plt.subplot(2, 2, 2)
        plt.bar(range(len(df)), df['max_accuracy'], alpha=0.7, color='lightgreen')
        plt.title('Best Accuracy per Experiment')
        plt.xticks(range(len(df)), [name[:15] + '...' if len(name) > 15 else name
                                  for name in df['experiment']], rotation=45)
        plt.ylabel('Max Accuracy')
        plt.ylim(0, 1)

        # Subplot 3: Accuracy range
        plt.subplot(2, 2, 3)
        plt.bar(range(len(df)), df['avg_accuracy'], alpha=0.7, color='orange',
                yerr=df['max_accuracy'] - df['min_accuracy'], capsize=5)
        plt.title('Average Accuracy with Range')
        plt.xticks(range(len(df)), [name[:15] + '...' if len(name) > 15 else name
                                  for name in df['experiment']], rotation=45)
        plt.ylabel('Average Accuracy')
        plt.ylim(0, 1)

        # Subplot 4: Summary table
        plt.subplot(2, 2, 4)
        plt.axis('off')
        table_data = []
        for _, row in df.iterrows():
            table_data.append([
                row['experiment'][:20] + '...' if len(row['experiment']) > 20 else row['experiment'],
                f"{row['runs']}",
                f"{row['max_accuracy']:.3f}",
                f"{row['avg_accuracy']:.3f}"
            ])

        table = plt.table(
            cellText=table_data,
            colLabels=['Experiment', 'Runs', 'Best', 'Avg'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        plt.title('Experiment Summary', pad=20)

        plt.tight_layout()
        plt.savefig('mlflow_summary_dashboard.png', dpi=300, bbox_inches='tight')

        # Log to a summary experiment
        mlflow.set_experiment("MLflow Demo Summary")
        with mlflow.start_run(run_name="Dashboard_Summary"):
            mlflow.log_artifact('mlflow_summary_dashboard.png')
            mlflow.log_metrics({
                "total_experiments": len(df),
                "total_runs": df['runs'].sum(),
                "overall_best_accuracy": df['max_accuracy'].max()
            })
            mlflow.set_tag("dashboard_type", "summary")

        print(f"    ✅ Created summary with {len(df)} experiments")

def main():
    """Main execution"""

    print("🚀 Completing MLflow Demo")
    print("=" * 50)

    try:
        # Complete remaining experiments
        complete_model_comparison()
        create_hyperparameter_tuning_demo()
        register_models()
        create_summary_dashboard()

        print("\n" + "=" * 50)
        print("🎉 MLflow Demo Complete!")
        print("\n📈 Demo Features:")
        print("   • Multiple experiments with different models")
        print("   • Hyperparameter tuning with parameter tracking")
        print("   • Model registry with versioning")
        print("   • Rich visualizations and artifacts")
        print("   • Comprehensive metrics tracking")
        print("\n🌐 View at: http://localhost:5000")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()