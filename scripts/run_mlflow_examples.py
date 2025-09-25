#!/usr/bin/env python3
"""
MLflow Example Script - Generate experiments, models, and visualizations
This script creates comprehensive MLflow experiments with multiple runs,
visualizations, and model artifacts for tutorial purposes.
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, fetch_california_housing, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
import time
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

def create_visualizations(model, X_test, y_test, y_pred, model_name, feature_names=None):
    """Create comprehensive visualizations for the model"""

    plots = {}

    # 1. Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(model.feature_importances_))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.bar(importance_df['feature'], importance_df['importance'])
        plt.title(f'{model_name} - Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plot_path = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plots['feature_importance'] = plot_path
        plt.close()

    # 2. Confusion Matrix (for classification)
    if len(np.unique(y_test)) < 20:  # Classification task
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        plot_path = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plots['confusion_matrix'] = plot_path
        plt.close()

    # 3. Prediction vs Actual (for regression)
    else:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name} - Predictions vs Actual')

        plot_path = f'predictions_vs_actual_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plots['predictions_vs_actual'] = plot_path
        plt.close()

    # 4. Residuals Plot (for regression)
    if len(np.unique(y_test)) >= 20:  # Regression task
        plt.figure(figsize=(8, 6))
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{model_name} - Residuals Plot')

        plot_path = f'residuals_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plots['residuals'] = plot_path
        plt.close()

    # 5. Learning Curve Simulation
    plt.figure(figsize=(10, 6))
    train_sizes = np.linspace(0.1, 1.0, 10)

    # Simulate learning curves
    np.random.seed(42)
    train_scores = 1 - np.exp(-train_sizes * 3) + np.random.normal(0, 0.02, len(train_sizes))
    val_scores = train_scores - np.random.normal(0.1, 0.05, len(train_sizes))

    plt.plot(train_sizes, train_scores, 'o-', label='Training Score', linewidth=2)
    plt.plot(train_sizes, val_scores, 'o-', label='Validation Score', linewidth=2)
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(f'{model_name} - Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = f'learning_curve_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plots['learning_curve'] = plot_path
    plt.close()

    return plots

def run_iris_classification_experiments():
    """Run comprehensive iris classification experiments"""

    print("🌸 Running Iris Classification Experiments...")

    # Set experiment
    mlflow.set_experiment("Iris Classification")

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Different models to try
    models = [
        ("Random Forest", RandomForestClassifier(random_state=42)),
        ("Logistic Regression", LogisticRegression(random_state=42, max_iter=1000)),
        ("SVM", SVC(random_state=42, probability=True)),
    ]

    # Hyperparameter variations for Random Forest
    rf_configs = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 150, "max_depth": 7},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 300, "max_depth": 15},
    ]

    best_accuracy = 0
    best_model_info = None

    # Run Random Forest experiments with different hyperparameters
    for i, config in enumerate(rf_configs):
        with mlflow.start_run(run_name=f"RF_Config_{i+1}") as run:
            # Create model with specific config
            model = RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                random_state=42
            )

            # Log parameters
            mlflow.log_params(config)
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("dataset", "iris")

            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=5)

            # Log metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "training_time": training_time,
                "n_features": X.shape[1],
                "n_samples": X.shape[0]
            })

            # Simulate training metrics over epochs
            for epoch in range(20):
                train_acc = 0.3 + 0.6 * (1 - np.exp(-epoch/5)) + np.random.normal(0, 0.02)
                val_acc = train_acc - 0.05 + np.random.normal(0, 0.02)
                mlflow.log_metrics({
                    "epoch_train_accuracy": max(0, min(1, train_acc)),
                    "epoch_val_accuracy": max(0, min(1, val_acc))
                }, step=epoch)

            # Create visualizations
            plots = create_visualizations(
                model, X_test, y_test, y_pred,
                f"Random Forest {i+1}", iris.feature_names
            )

            # Log artifacts
            for plot_name, plot_path in plots.items():
                mlflow.log_artifact(plot_path)
                os.remove(plot_path)  # Clean up

            # Create and log detailed results
            results_df = pd.DataFrame({
                'actual': y_test,
                'predicted': y_pred,
                'correct': y_test == y_pred,
                'confidence': y_pred_proba.max(axis=1)
            })
            results_df.to_csv(f'predictions_rf_{i+1}.csv', index=False)
            mlflow.log_artifact(f'predictions_rf_{i+1}.csv')
            os.remove(f'predictions_rf_{i+1}.csv')

            # Classification report
            report = classification_report(y_test, y_pred, target_names=iris.target_names)
            with open(f'classification_report_rf_{i+1}.txt', 'w') as f:
                f.write(report)
            mlflow.log_artifact(f'classification_report_rf_{i+1}.txt')
            os.remove(f'classification_report_rf_{i+1}.txt')

            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=mlflow.models.infer_signature(X_train, y_pred),
                input_example=X_train[:3]
            )

            # Set tags
            mlflow.set_tags({
                "algorithm": "RandomForest",
                "dataset": "iris",
                "task": "classification",
                "framework": "sklearn",
                "tuning_iteration": i+1
            })

            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_info = {
                    "run_id": run.info.run_id,
                    "accuracy": accuracy,
                    "config": config
                }

            print(f"    ✅ RF Config {i+1}: Accuracy = {accuracy:.4f}")

    # Run other model types
    for model_name, model in models[1:]:  # Skip RF as we already did it
        with mlflow.start_run(run_name=f"{model_name.replace(' ', '_')}") as run:
            # Log parameters
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("dataset", "iris")

            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Log metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "training_time": training_time
            })

            # Create visualizations
            plots = create_visualizations(
                model, X_test, y_test, y_pred, model_name, iris.feature_names
            )

            # Log artifacts
            for plot_name, plot_path in plots.items():
                mlflow.log_artifact(plot_path)
                os.remove(plot_path)

            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=mlflow.models.infer_signature(X_train, y_pred),
                input_example=X_train[:3]
            )

            # Set tags
            mlflow.set_tags({
                "algorithm": model_name.replace(' ', ''),
                "dataset": "iris",
                "task": "classification",
                "framework": "sklearn"
            })

            print(f"    ✅ {model_name}: Accuracy = {accuracy:.4f}")

    print(f"    🏆 Best Model: RF with accuracy {best_model_info['accuracy']:.4f}")
    return best_model_info

def run_housing_regression_experiments():
    """Run housing price prediction experiments"""

    print("🏠 Running California Housing Price Regression Experiments...")

    # Set experiment
    mlflow.set_experiment("California Housing Price Prediction")

    # Load California housing data
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Different regression models
    models = [
        ("Linear Regression", LinearRegression()),
        ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
    ]

    # Gradient Boosting configurations
    gb_configs = [
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
        {"n_estimators": 300, "learning_rate": 0.01, "max_depth": 5},
    ]

    best_r2 = -float('inf')
    best_model_info = None

    # Run Gradient Boosting experiments
    for i, config in enumerate(gb_configs):
        with mlflow.start_run(run_name=f"GB_Config_{i+1}") as run:
            model = GradientBoostingRegressor(random_state=42, **config)

            # Log parameters
            mlflow.log_params(config)
            mlflow.log_param("model_type", "GradientBoosting")
            mlflow.log_param("dataset", "housing")

            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log metrics
            mlflow.log_metrics({
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2_score": r2,
                "training_time": training_time
            })

            # Log training progress
            for epoch, loss in enumerate(model.train_score_):
                mlflow.log_metric("training_loss", loss, step=epoch)

            # Create visualizations
            plots = create_visualizations(
                model, X_test, y_test, y_pred, f"Gradient Boosting {i+1}", feature_names
            )

            # Log artifacts
            for plot_name, plot_path in plots.items():
                mlflow.log_artifact(plot_path)
                os.remove(plot_path)

            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=mlflow.models.infer_signature(X_train, y_pred),
                input_example=X_train[:3]
            )

            # Set tags
            mlflow.set_tags({
                "algorithm": "GradientBoosting",
                "dataset": "housing",
                "task": "regression",
                "framework": "sklearn"
            })

            # Track best model
            if r2 > best_r2:
                best_r2 = r2
                best_model_info = {
                    "run_id": run.info.run_id,
                    "r2_score": r2,
                    "config": config
                }

            print(f"    ✅ GB Config {i+1}: R² = {r2:.4f}")

    # Run Linear Regression
    with mlflow.start_run(run_name="Linear_Regression") as run:
        model = LinearRegression()

        # Log parameters
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("dataset", "housing")

        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metrics({
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
            "training_time": training_time
        })

        # Create visualizations (without feature importance for linear regression)
        plots = {}

        # Predictions vs Actual
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Linear Regression - Predictions vs Actual')
        plt.savefig('predictions_vs_actual_linear.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('predictions_vs_actual_linear.png')
        os.remove('predictions_vs_actual_linear.png')

        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=mlflow.models.infer_signature(X_train, y_pred),
            input_example=X_train[:3]
        )

        # Set tags
        mlflow.set_tags({
            "algorithm": "LinearRegression",
            "dataset": "housing",
            "task": "regression",
            "framework": "sklearn"
        })

        print(f"    ✅ Linear Regression: R² = {r2:.4f}")

    print(f"    🏆 Best Model: GB with R² {best_model_info['r2_score']:.4f}")
    return best_model_info

def run_model_comparison_experiment():
    """Run a comprehensive model comparison experiment"""

    print("🔬 Running Model Comparison Experiment...")

    # Set experiment
    mlflow.set_experiment("Model Comparison Study")

    # Load wine dataset for multi-class classification
    wine = load_wine()
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Models to compare
    models = [
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Logistic Regression", LogisticRegression(random_state=42, max_iter=1000)),
        ("SVM", SVC(random_state=42, probability=True)),
    ]

    results_summary = []

    # Parent run for the comparison study
    with mlflow.start_run(run_name="Model_Comparison_Study") as parent_run:
        mlflow.log_param("dataset", "wine")
        mlflow.log_param("n_models", len(models))
        mlflow.log_param("test_size", 0.3)

        for model_name, model in models:
            # Nested run for each model
            with mlflow.start_run(run_name=f"Model_{model_name.replace(' ', '_')}", nested=True) as child_run:

                # Train model
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time

                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=5)

                # Log metrics
                metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                    "training_time": training_time
                }
                mlflow.log_metrics(metrics)

                # Log parameters
                if hasattr(model, 'get_params'):
                    mlflow.log_params(model.get_params())

                # Create and log confusion matrix
                plt.figure(figsize=(8, 6))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=wine.target_names, yticklabels=wine.target_names)
                plt.title(f'{model_name} - Confusion Matrix')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')

                cm_path = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
                plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(cm_path)
                plt.close()
                os.remove(cm_path)

                # Log model
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    signature=mlflow.models.infer_signature(X_train, y_pred),
                    input_example=X_train[:3]
                )

                # Set tags
                mlflow.set_tags({
                    "algorithm": model_name.replace(' ', ''),
                    "dataset": "wine",
                    "task": "classification",
                    "framework": "sklearn",
                    "study": "comparison"
                })

                # Store results for summary
                results_summary.append({
                    "model": model_name,
                    "run_id": child_run.info.run_id,
                    **metrics
                })

                print(f"    ✅ {model_name}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")

        # Create comparison visualization
        plt.figure(figsize=(12, 8))

        models_names = [r["model"] for r in results_summary]
        accuracies = [r["accuracy"] for r in results_summary]
        f1_scores = [r["f1_score"] for r in results_summary]
        training_times = [r["training_time"] for r in results_summary]

        # Metrics comparison
        x = np.arange(len(models_names))
        width = 0.25

        plt.subplot(2, 2, 1)
        plt.bar(x - width, accuracies, width, label='Accuracy', alpha=0.8)
        plt.bar(x, f1_scores, width, label='F1 Score', alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, models_names)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.bar(models_names, training_times, alpha=0.8, color='orange')
        plt.xlabel('Models')
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # CV scores comparison
        plt.subplot(2, 2, 3)
        cv_means = [r["cv_mean"] for r in results_summary]
        cv_stds = [r["cv_std"] for r in results_summary]
        plt.bar(models_names, cv_means, yerr=cv_stds, alpha=0.8, color='green', capsize=5)
        plt.xlabel('Models')
        plt.ylabel('Cross-Validation Score')
        plt.title('Cross-Validation Performance')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Summary table
        plt.subplot(2, 2, 4)
        plt.axis('off')
        table_data = []
        for r in results_summary:
            table_data.append([
                r["model"],
                f"{r['accuracy']:.3f}",
                f"{r['f1_score']:.3f}",
                f"{r['training_time']:.2f}s"
            ])

        table = plt.table(cellText=table_data,
                         colLabels=['Model', 'Accuracy', 'F1-Score', 'Time'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        plt.title('Performance Summary')

        plt.tight_layout()
        plt.savefig('model_comparison_summary.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('model_comparison_summary.png')
        plt.close()
        os.remove('model_comparison_summary.png')

        # Log summary metrics to parent run
        best_model = max(results_summary, key=lambda x: x['accuracy'])
        mlflow.log_metrics({
            "best_accuracy": best_model['accuracy'],
            "average_accuracy": np.mean([r['accuracy'] for r in results_summary]),
            "std_accuracy": np.std([r['accuracy'] for r in results_summary])
        })

        mlflow.set_tag("best_model", best_model['model'])

        print(f"    🏆 Best Model: {best_model['model']} with accuracy {best_model['accuracy']:.4f}")

def register_best_models():
    """Register the best models in MLflow Model Registry"""

    print("📝 Registering Best Models in Model Registry...")

    client = mlflow.tracking.MlflowClient()

    # Get experiments
    experiments = {
        "iris_classification": client.get_experiment_by_name("Iris Classification"),
        "housing_prediction": client.get_experiment_by_name("California Housing Price Prediction"),
        "model_comparison": client.get_experiment_by_name("Model Comparison Study")
    }

    # Register best model from each experiment
    for exp_name, experiment in experiments.items():
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.accuracy DESC", "metrics.r2_score DESC"]
            )

            if runs:
                best_run = runs[0]
                model_uri = f"runs:/{best_run.info.run_id}/model"

                # Determine model name based on experiment
                if "iris" in exp_name.lower():
                    model_name = "iris_classifier"
                    description = "Best performing iris classification model"
                elif "housing" in exp_name.lower():
                    model_name = "housing_price_predictor"
                    description = "Best performing housing price prediction model"
                else:
                    model_name = "wine_classifier"
                    description = "Best performing wine classification model"

                try:
                    # Create registered model
                    try:
                        registered_model = client.create_registered_model(
                            name=model_name,
                            description=description
                        )
                        print(f"    ✅ Created registered model: {model_name}")
                    except Exception as e:
                        if "already exists" in str(e):
                            print(f"    ℹ️  Model {model_name} already exists")
                        else:
                            raise e

                    # Create model version
                    model_version = client.create_model_version(
                        name=model_name,
                        source=model_uri,
                        run_id=best_run.info.run_id,
                        description=f"Version created from run {best_run.info.run_id}"
                    )

                    print(f"    ✅ Registered {model_name} version {model_version.version}")

                    # Transition to Staging
                    client.transition_model_version_stage(
                        name=model_name,
                        version=model_version.version,
                        stage="Staging",
                        description="Initial model registration - moving to staging for validation"
                    )

                    print(f"    ✅ Moved {model_name} v{model_version.version} to Staging")

                except Exception as e:
                    print(f"    ❌ Error registering {model_name}: {e}")

def create_mlflow_dashboard_data():
    """Create additional data for MLflow dashboard demonstration"""

    print("📊 Creating Additional Dashboard Data...")

    # Create a hyperparameter tuning experiment
    mlflow.set_experiment("Hyperparameter Tuning Demo")

    # Simulate hyperparameter tuning with many runs
    param_combinations = [
        {"n_estimators": n_est, "max_depth": max_d, "learning_rate": lr}
        for n_est in [50, 100, 150, 200]
        for max_d in [3, 5, 7, 10]
        for lr in [0.01, 0.05, 0.1, 0.2]
    ]

    # Sample 20 combinations for demo
    np.random.seed(42)
    selected_combinations = np.random.choice(
        len(param_combinations),
        size=min(20, len(param_combinations)),
        replace=False
    )

    for i, idx in enumerate(selected_combinations):
        params = param_combinations[idx]

        with mlflow.start_run(run_name=f"HPTune_Run_{i+1:02d}"):
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", "GradientBoostingClassifier")

            # Simulate metrics based on parameters (realistic relationships)
            base_accuracy = 0.85
            n_est_factor = min(params["n_estimators"] / 200, 1) * 0.05
            depth_factor = min(params["max_depth"] / 10, 1) * 0.03
            lr_factor = -abs(params["learning_rate"] - 0.1) * 0.1

            accuracy = base_accuracy + n_est_factor + depth_factor + lr_factor
            accuracy += np.random.normal(0, 0.02)  # Add noise
            accuracy = max(0.7, min(0.98, accuracy))  # Bound the values

            # Log metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": accuracy + np.random.normal(0, 0.01),
                "recall": accuracy + np.random.normal(0, 0.01),
                "f1_score": accuracy + np.random.normal(0, 0.005),
                "training_time": np.random.uniform(10, 120),
                "model_size_mb": np.random.uniform(5, 50)
            })

            # Simulate training progress
            for epoch in range(50):
                train_loss = 2.0 * np.exp(-epoch/20) + np.random.normal(0, 0.05)
                val_loss = train_loss + 0.1 + np.random.normal(0, 0.03)
                mlflow.log_metrics({
                    "train_loss": max(0, train_loss),
                    "val_loss": max(0, val_loss)
                }, step=epoch)

            # Set tags
            mlflow.set_tags({
                "algorithm": "GradientBoosting",
                "task": "hyperparameter_tuning",
                "framework": "sklearn",
                "tuning_method": "grid_search"
            })

    print(f"    ✅ Created {len(selected_combinations)} hyperparameter tuning runs")

def main():
    """Main function to run all MLflow examples"""

    print("🚀 Starting MLflow Examples Generation")
    print("=" * 60)

    # Ensure MLflow server is running
    try:
        import requests
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code != 200:
            print("❌ MLflow server is not running. Please start it first.")
            return
    except:
        print("❌ MLflow server is not accessible. Please start it first.")
        return

    print("✅ MLflow server is running at http://localhost:5000")
    print()

    try:
        # Run all experiments
        iris_best = run_iris_classification_experiments()
        housing_best = run_housing_regression_experiments()
        run_model_comparison_experiment()
        create_mlflow_dashboard_data()
        register_best_models()

        print("\n" + "=" * 60)
        print("🎉 MLflow Examples Generation Complete!")
        print()
        print("📊 Generated Experiments:")
        print("   • Iris Classification (with multiple RF configurations)")
        print("   • Boston Housing Price Prediction")
        print("   • Model Comparison Study")
        print("   • Hyperparameter Tuning Demo")
        print()
        print("🎯 Features Demonstrated:")
        print("   • Experiment tracking with parameters and metrics")
        print("   • Artifact logging (plots, models, reports)")
        print("   • Model registry with versioning")
        print("   • Nested runs and parent-child relationships")
        print("   • Tags and metadata organization")
        print("   • Model comparison and evaluation")
        print()
        print("🌐 MLflow UI Available at: http://localhost:5000")
        print("   • Browse experiments and runs")
        print("   • Compare model performance")
        print("   • View detailed metrics and artifacts")
        print("   • Explore model registry")
        print()
        print("🏆 Best Models:")
        if iris_best:
            print(f"   • Iris: RF with {iris_best['accuracy']:.4f} accuracy")
        if housing_best:
            print(f"   • Housing: GB with {housing_best['r2_score']:.4f} R²")

    except Exception as e:
        print(f"❌ Error running MLflow examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()