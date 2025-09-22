"""
Script to create sample datasets for the MLOps demo
This script generates datasets that will be versioned using DVC
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing, make_classification
import pickle

def create_iris_dataset():
    """Create Iris dataset"""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df

def create_housing_dataset():
    """Create California housing dataset"""
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['price'] = housing.target
    return df

def create_churn_dataset():
    """Create synthetic churn dataset"""
    np.random.seed(42)
    n_samples = 5000

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

    # Create churn target with logic
    churn_probability = (
        0.1 +
        0.3 * (df['contract_type'] == 'Month-to-month') +
        0.2 * (df['tenure_months'] < 12) +
        0.2 * (df['monthly_charges'] > 80) +
        0.1 * (df['senior_citizen'] == 1) +
        0.1 * (df['payment_method'] == 'Electronic check') -
        0.2 * (df['tech_support'] == 'Yes') -
        0.1 * (df['partner'] == 'Yes')
    )

    churn_probability += np.random.normal(0, 0.1, n_samples)
    churn_probability = np.clip(churn_probability, 0, 1)
    df['churn'] = (np.random.random(n_samples) < churn_probability).astype(int)

    return df

def create_sentiment_dataset():
    """Create synthetic sentiment dataset"""
    positive_texts = [
        "I love this product, it's amazing!",
        "This is the best thing I've ever bought.",
        "Absolutely fantastic quality and service.",
        "I'm so happy with this purchase.",
        "Excellent product, highly recommended!",
        "This exceeded my expectations.",
        "Outstanding quality and value.",
        "I'm thrilled with this product.",
        "Perfect in every way!",
        "This is exactly what I needed.",
    ] * 50

    negative_texts = [
        "This product is terrible.",
        "Worst purchase I've ever made.",
        "Completely disappointed with quality.",
        "This is a waste of money.",
        "Poor quality, don't buy this.",
        "Awful product, terrible service.",
        "I regret buying this.",
        "This product is useless.",
        "Very poor quality.",
        "Not worth the money.",
    ] * 50

    neutral_texts = [
        "The product is okay.",
        "It's an average product.",
        "Nothing special about this.",
        "It works as expected.",
        "Standard quality product.",
        "It's fine, nothing more.",
        "Average quality and price.",
        "This product is decent.",
        "It does the job.",
        "Acceptable quality.",
    ] * 50

    texts = positive_texts + negative_texts + neutral_texts
    labels = [2] * len(positive_texts) + [0] * len(negative_texts) + [1] * len(neutral_texts)

    df = pd.DataFrame({'text': texts, 'label': labels})
    df['sentiment'] = df['label'].map({0: 'negative', 1: 'neutral', 2: 'positive'})

    return df

def create_image_dataset():
    """Create synthetic image dataset metadata"""
    np.random.seed(42)
    n_samples = 1500

    # Simulate image metadata
    df = pd.DataFrame({
        'image_id': range(1, n_samples + 1),
        'class': np.random.choice(['red_dominant', 'green_dominant', 'blue_dominant'], n_samples),
        'width': 32,
        'height': 32,
        'channels': 3,
        'red_mean': np.random.uniform(0.3, 0.8, n_samples),
        'green_mean': np.random.uniform(0.3, 0.8, n_samples),
        'blue_mean': np.random.uniform(0.3, 0.8, n_samples),
    })

    # Adjust means based on class
    df.loc[df['class'] == 'red_dominant', 'red_mean'] += 0.2
    df.loc[df['class'] == 'green_dominant', 'green_mean'] += 0.2
    df.loc[df['class'] == 'blue_dominant', 'blue_mean'] += 0.2

    df['red_mean'] = np.clip(df['red_mean'], 0, 1)
    df['green_mean'] = np.clip(df['green_mean'], 0, 1)
    df['blue_mean'] = np.clip(df['blue_mean'], 0, 1)

    return df

def main():
    """Create all datasets and save them"""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # Create datasets
    print("Creating datasets...")

    # Iris dataset
    iris_df = create_iris_dataset()
    iris_df.to_csv('data/raw/iris.csv', index=False)
    print(f"Created iris dataset: {iris_df.shape}")

    # Housing dataset
    housing_df = create_housing_dataset()
    housing_df.to_csv('data/raw/housing.csv', index=False)
    print(f"Created housing dataset: {housing_df.shape}")

    # Churn dataset
    churn_df = create_churn_dataset()
    churn_df.to_csv('data/raw/churn.csv', index=False)
    print(f"Created churn dataset: {churn_df.shape}")

    # Sentiment dataset
    sentiment_df = create_sentiment_dataset()
    sentiment_df.to_csv('data/raw/sentiment.csv', index=False)
    print(f"Created sentiment dataset: {sentiment_df.shape}")

    # Image dataset metadata
    image_df = create_image_dataset()
    image_df.to_csv('data/raw/image_metadata.csv', index=False)
    print(f"Created image metadata: {image_df.shape}")

    # Create a data summary
    summary = {
        'datasets': {
            'iris': {'shape': iris_df.shape, 'description': 'Iris flower classification dataset'},
            'housing': {'shape': housing_df.shape, 'description': 'California housing price prediction dataset'},
            'churn': {'shape': churn_df.shape, 'description': 'Customer churn prediction dataset'},
            'sentiment': {'shape': sentiment_df.shape, 'description': 'Text sentiment analysis dataset'},
            'image_metadata': {'shape': image_df.shape, 'description': 'Image classification metadata'}
        },
        'total_records': iris_df.shape[0] + housing_df.shape[0] + churn_df.shape[0] + sentiment_df.shape[0] + image_df.shape[0]
    }

    with open('data/raw/data_summary.txt', 'w') as f:
        f.write("MLOps Demo Dataset Summary\n")
        f.write("=" * 30 + "\n\n")
        for name, info in summary['datasets'].items():
            f.write(f"{name.upper()}:\n")
            f.write(f"  Shape: {info['shape']}\n")
            f.write(f"  Description: {info['description']}\n\n")
        f.write(f"Total records across all datasets: {summary['total_records']}\n")

    print("\nAll datasets created successfully!")
    print(f"Total records: {summary['total_records']}")

if __name__ == "__main__":
    main()