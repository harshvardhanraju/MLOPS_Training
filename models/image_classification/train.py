import os
import pickle
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def create_sample_data():
    """Create a simple synthetic image dataset for demo purposes"""
    # Generate synthetic image data (32x32 RGB images)
    np.random.seed(42)

    # Create 3 classes of synthetic images
    n_samples_per_class = 500
    img_height, img_width, channels = 32, 32, 3

    # Class 0: Images with more red channel
    class_0 = np.random.rand(n_samples_per_class, img_height, img_width, channels)
    class_0[:, :, :, 0] += 0.3  # Boost red channel

    # Class 1: Images with more green channel
    class_1 = np.random.rand(n_samples_per_class, img_height, img_width, channels)
    class_1[:, :, :, 1] += 0.3  # Boost green channel

    # Class 2: Images with more blue channel
    class_2 = np.random.rand(n_samples_per_class, img_height, img_width, channels)
    class_2[:, :, :, 2] += 0.3  # Boost blue channel

    # Combine all classes
    X = np.vstack([class_0, class_1, class_2])
    y = np.hstack([
        np.zeros(n_samples_per_class),
        np.ones(n_samples_per_class),
        np.full(n_samples_per_class, 2)
    ])

    # Normalize to [0, 1]
    X = np.clip(X, 0, 1)

    return X, y

def create_model(input_shape=(32, 32, 3), num_classes=3):
    """Create a simple CNN model"""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        'accuracy': accuracy,
        'classification_report': report,
        'predictions': y_pred.tolist(),
        'probabilities': y_pred_proba.tolist()
    }

def main():
    mlflow.set_experiment("image_classification")

    with mlflow.start_run():
        # Create synthetic data
        X, y = create_sample_data()

        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Log parameters
        mlflow.log_param("input_shape", (32, 32, 3))
        mlflow.log_param("num_classes", 3)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("epochs", 10)
        mlflow.log_param("batch_size", 32)

        # Create and train model
        model = create_model()

        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("precision", metrics['classification_report']['macro avg']['precision'])
        mlflow.log_metric("recall", metrics['classification_report']['macro avg']['recall'])
        mlflow.log_metric("f1_score", metrics['classification_report']['macro avg']['f1-score'])

        # Save model
        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/image_classifier.h5"
        model.save(model_path)

        # Log model to MLflow
        mlflow.tensorflow.log_model(
            model,
            "model",
            registered_model_name="image_classifier"
        )

        # Log artifacts
        mlflow.log_artifact(model_path)

        print(f"Model trained successfully!")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Model saved to: {model_path}")

        return model, metrics

if __name__ == "__main__":
    main()