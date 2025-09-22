import pickle
import numpy as np
from typing import List, Dict, Any

class SentimentPredictor:
    def __init__(self, model_path: str = "artifacts/sentiment_model.pkl"):
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)

        self.label_mapping = {
            'NEGATIVE': 0,
            'POSITIVE': 2
        }
        self.id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}

    def predict(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string")

        # Get prediction from pipeline
        result = self.classifier(text)

        # Convert to our format
        scores = {item['label']: item['score'] for item in result}

        # Determine sentiment
        if 'NEGATIVE' in scores and 'POSITIVE' in scores:
            if scores['NEGATIVE'] > scores['POSITIVE']:
                sentiment = 'negative'
                confidence = scores['NEGATIVE']
            else:
                sentiment = 'positive'
                confidence = scores['POSITIVE']
        else:
            # Fallback
            sentiment = 'neutral'
            confidence = 0.5

        # Create probability distribution
        neg_prob = scores.get('NEGATIVE', 0.5)
        pos_prob = scores.get('POSITIVE', 0.5)
        neutral_prob = 1.0 - abs(pos_prob - neg_prob)  # Simple neutral calculation

        # Normalize probabilities
        total = neg_prob + neutral_prob + pos_prob
        if total > 0:
            neg_prob /= total
            neutral_prob /= total
            pos_prob /= total

        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': float(confidence),
            'probabilities': {
                'negative': float(neg_prob),
                'neutral': float(neutral_prob),
                'positive': float(pos_prob)
            },
            'raw_output': result
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(text) for text in texts]

def main():
    predictor = SentimentPredictor()

    # Example predictions
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible, worst purchase ever.",
        "The product is okay, nothing special.",
        "Absolutely fantastic quality!",
        "I'm not sure about this product.",
        "Poor quality, very disappointed.",
        "This works as expected.",
        "Outstanding service and quality!",
        "It's an average product.",
        "I regret buying this."
    ]

    for text in test_texts:
        result = predictor.predict(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        print(f"Probabilities: {result['probabilities']}")
        print("-" * 50)

if __name__ == "__main__":
    main()