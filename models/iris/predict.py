import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class IrisPredictor:
    def __init__(self, model_path: str = "models/iris/artifacts/iris_model.pkl"):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.class_names = ['setosa', 'versicolor', 'virginica']
        self.feature_names = [
            'sepal_length', 'sepal_width',
            'petal_length', 'petal_width'
        ]

    def predict(self, features: List[float]) -> Dict[str, Any]:
        if len(features) != 4:
            raise ValueError("Expected 4 features: sepal_length, sepal_width, petal_length, petal_width")

        X = pd.DataFrame([features], columns=self.feature_names)
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        result = {
            'prediction': self.class_names[prediction],
            'prediction_id': int(prediction),
            'probabilities': {
                self.class_names[i]: float(prob)
                for i, prob in enumerate(probabilities)
            },
            'confidence': float(max(probabilities))
        }

        return result

    def predict_batch(self, features_list: List[List[float]]) -> List[Dict[str, Any]]:
        return [self.predict(features) for features in features_list]

def main():
    predictor = IrisPredictor()

    # Example predictions
    test_samples = [
        [5.1, 3.5, 1.4, 0.2],  # Setosa
        [6.2, 2.8, 4.8, 1.8],  # Versicolor
        [7.7, 2.6, 6.9, 2.3],  # Virginica
    ]

    for i, sample in enumerate(test_samples):
        result = predictor.predict(sample)
        print(f"Sample {i+1}: {sample}")
        print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
        print(f"Probabilities: {result['probabilities']}")
        print("-" * 50)

if __name__ == "__main__":
    main()