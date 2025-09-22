import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

def create_sample_churn_data():
    """Create a synthetic customer churn dataset"""
    np.random.seed(42)
    n_samples = 5000

    # Generate features
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 150, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
        'senior_citizen': np.random.choice([0, 1], n_samples),
        'partner': np.random.choice(['Yes', 'No'], n_samples),
        'dependents': np.random.choice(['Yes', 'No'], n_samples),
    }

    df = pd.DataFrame(data)

    # Create churn target with some logic
    churn_probability = (
        0.1 +  # Base probability
        0.3 * (df['contract_type'] == 'Month-to-month') +  # Month-to-month more likely to churn
        0.2 * (df['tenure_months'] < 12) +  # New customers more likely to churn
        0.2 * (df['monthly_charges'] > 80) +  # High charges increase churn
        0.1 * (df['senior_citizen'] == 1) +  # Senior citizens slightly more likely
        0.1 * (df['payment_method'] == 'Electronic check') -  # Electronic check increases churn
        0.2 * (df['tech_support'] == 'Yes') -  # Tech support reduces churn
        0.1 * (df['partner'] == 'Yes')  # Having partner reduces churn
    )

    # Add some randomness
    churn_probability += np.random.normal(0, 0.1, n_samples)
    churn_probability = np.clip(churn_probability, 0, 1)

    # Generate binary churn target
    df['churn'] = (np.random.random(n_samples) < churn_probability).astype(int)

    return df

def preprocess_data(df):
    """Preprocess the churn dataset"""
    # Separate features and target
    X = df.drop(['customer_id', 'churn'], axis=1)
    y = df['churn']

    # Encode categorical variables
    label_encoders = {}
    categorical_columns = X.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    return X, y, label_encoders

def train_model(X_train, y_train, params=None):
    """Train LightGBM model"""
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'random_state': 42
        }

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)

    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )

    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'probabilities': y_pred_proba.tolist()
    }

def main():
    mlflow.set_experiment("customer_churn_prediction")

    with mlflow.start_run():
        # Create and preprocess data
        df = create_sample_churn_data()
        X, y, label_encoders = preprocess_data(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Log parameters
        mlflow.log_param("objective", "binary")
        mlflow.log_param("num_leaves", 31)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("churn_rate", y.mean())

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("precision", metrics['precision'])
        mlflow.log_metric("recall", metrics['recall'])
        mlflow.log_metric("f1_score", metrics['f1_score'])
        mlflow.log_metric("auc", metrics['auc'])

        # Save model and encoders
        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/churn_model.pkl"
        encoders_path = "artifacts/label_encoders.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        with open(encoders_path, 'wb') as f:
            pickle.dump(label_encoders, f)

        # Log model to MLflow
        mlflow.lightgbm.log_model(
            model,
            "model",
            registered_model_name="churn_predictor"
        )

        # Log artifacts
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(encoders_path)

        print(f"Model trained successfully!")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Model saved to: {model_path}")

        return model, label_encoders, metrics

if __name__ == "__main__":
    main()