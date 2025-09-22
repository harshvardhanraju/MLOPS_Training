import os
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    return X, y, iris.target_names

def train_model(X_train, y_train, random_state=42):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist()
    }

def main():
    # Set MLflow tracking
    mlflow.set_experiment("iris_classification")

    with mlflow.start_run():
        # Load and prepare data
        X, y, class_names = load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Log parameters
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("precision", metrics['classification_report']['macro avg']['precision'])
        mlflow.log_metric("recall", metrics['classification_report']['macro avg']['recall'])
        mlflow.log_metric("f1_score", metrics['classification_report']['macro avg']['f1-score'])

        # Save model locally
        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/iris_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="iris_classifier"
        )

        # Log artifacts
        mlflow.log_artifact(model_path)

        print(f"Model trained successfully!")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Model saved to: {model_path}")

        return model, metrics

if __name__ == "__main__":
    main()