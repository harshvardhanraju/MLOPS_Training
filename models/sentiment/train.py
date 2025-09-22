import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from datasets import Dataset

def create_sample_data():
    """Create a sample sentiment dataset for demo purposes"""
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
        "Great product, fast shipping.",
        "Love it! Will buy again.",
        "Superb quality and design.",
        "This product is incredible.",
        "Five stars! Excellent service."
    ] * 20  # Repeat to have more samples

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
        "Terrible experience overall.",
        "This product broke immediately.",
        "Poor design and quality.",
        "I want my money back.",
        "Completely unsatisfied."
    ] * 20  # Repeat to have more samples

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
        "It's what I expected.",
        "Standard functionality.",
        "Average customer service.",
        "Regular product quality.",
        "It's sufficient for my needs."
    ] * 20  # Repeat to have more samples

    # Create dataset
    texts = positive_texts + negative_texts + neutral_texts
    labels = [2] * len(positive_texts) + [0] * len(negative_texts) + [1] * len(neutral_texts)

    return pd.DataFrame({'text': texts, 'label': labels})

def train_sentiment_model():
    """Train a simple sentiment model using a pre-trained transformer"""
    # Use a smaller, faster model for demo
    model_name = "distilbert-base-uncased"

    # Create sample data
    df = create_sample_data()

    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    # For demo purposes, let's use a simpler approach with pipeline
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        return_all_scores=True
    )

    return classifier, train_texts, test_texts, train_labels, test_labels

def evaluate_pipeline_model(classifier, test_texts, test_labels):
    """Evaluate the pipeline model"""
    predictions = []
    confidences = []

    for text in test_texts:
        result = classifier(text)
        # Convert to our label format: negative=0, neutral=1, positive=2
        if result[0]['label'] == 'NEGATIVE':
            pred = 0 if result[0]['score'] > result[1]['score'] else 2
        else:
            pred = 2 if result[0]['score'] > result[1]['score'] else 0

        predictions.append(pred)
        confidences.append(max(result[0]['score'], result[1]['score']))

    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)

    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'confidences': confidences
    }

def main():
    mlflow.set_experiment("sentiment_analysis")

    with mlflow.start_run():
        # Train model
        classifier, train_texts, test_texts, train_labels, test_labels = train_sentiment_model()

        # Log parameters
        mlflow.log_param("model_name", "distilbert-base-uncased-finetuned-sst-2-english")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("train_samples", len(train_texts))
        mlflow.log_param("test_samples", len(test_texts))

        # Evaluate model
        metrics = evaluate_pipeline_model(classifier, test_texts, test_labels)

        # Log metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("avg_confidence", np.mean(metrics['confidences']))

        # Save model
        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/sentiment_model.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(classifier, f)

        # Log model to MLflow
        mlflow.transformers.log_model(
            transformers_model=classifier,
            artifact_path="model",
            registered_model_name="sentiment_analyzer"
        )

        # Log artifacts
        mlflow.log_artifact(model_path)

        print(f"Model trained successfully!")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Average Confidence: {np.mean(metrics['confidences']):.4f}")
        print(f"Model saved to: {model_path}")

        return classifier, metrics

if __name__ == "__main__":
    main()