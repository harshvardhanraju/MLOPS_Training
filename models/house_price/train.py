import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import requests

def download_housing_data():
    """Download California housing dataset"""
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name='price')
    return X, y

def preprocess_data(X, y):
    # Create additional features
    X['rooms_per_household'] = X['AveRooms'] / X['AveOccup']
    X['bedrooms_per_room'] = X['AveBedrms'] / X['AveRooms']
    X['population_per_household'] = X['Population'] / X['HouseAge']

    # Remove outliers (simple approach)
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask = (y >= lower_bound) & (y <= upper_bound)
    X = X[mask]
    y = y[mask]

    return X, y

def train_model(X_train, y_train, params=None):
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'predictions': y_pred.tolist()
    }

def main():
    mlflow.set_experiment("house_price_prediction")

    with mlflow.start_run():
        # Load and preprocess data
        X, y = download_housing_data()
        X, y = preprocess_data(X, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        # Log parameters
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }

        for key, value in params.items():
            mlflow.log_param(key, value)

        mlflow.log_param("test_size", 0.2)

        # Train model
        model = train_model(X_train_scaled, y_train, params)

        # Evaluate model
        metrics = evaluate_model(model, X_test_scaled, y_test)

        # Log metrics
        mlflow.log_metric("mse", metrics['mse'])
        mlflow.log_metric("rmse", metrics['rmse'])
        mlflow.log_metric("mae", metrics['mae'])
        mlflow.log_metric("r2_score", metrics['r2_score'])

        # Save model and scaler
        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/house_price_model.pkl"
        scaler_path = "artifacts/scaler.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        # Log model to MLflow
        mlflow.xgboost.log_model(
            model,
            "model",
            registered_model_name="house_price_predictor"
        )

        # Log artifacts
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)

        print(f"Model trained successfully!")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R2 Score: {metrics['r2_score']:.4f}")
        print(f"Model saved to: {model_path}")

        return model, scaler, metrics

if __name__ == "__main__":
    main()