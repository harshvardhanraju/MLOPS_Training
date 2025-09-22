import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class HousePricePredictor:
    def __init__(self, model_path: str = "artifacts/house_price_model.pkl",
                 scaler_path: str = "artifacts/scaler.pkl"):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        self.feature_names = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ]

    def preprocess_features(self, features: List[float]) -> pd.DataFrame:
        if len(features) != 8:
            raise ValueError("Expected 8 features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude")

        X = pd.DataFrame([features], columns=self.feature_names)

        # Add engineered features
        X['rooms_per_household'] = X['AveRooms'] / X['AveOccup']
        X['bedrooms_per_room'] = X['AveBedrms'] / X['AveRooms']
        X['population_per_household'] = X['Population'] / X['HouseAge']

        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        return X_scaled

    def predict(self, features: List[float]) -> Dict[str, Any]:
        X_processed = self.preprocess_features(features)
        prediction = self.model.predict(X_processed)[0]

        # Convert to actual price (California housing target is in hundreds of thousands)
        price_estimate = float(prediction * 100000)  # Convert to dollars

        result = {
            'predicted_price': price_estimate,
            'predicted_price_formatted': f"${price_estimate:,.2f}",
            'model_output': float(prediction),
            'confidence_interval': {
                'lower': price_estimate * 0.85,  # Simple confidence interval
                'upper': price_estimate * 1.15
            }
        }

        return result

    def predict_batch(self, features_list: List[List[float]]) -> List[Dict[str, Any]]:
        return [self.predict(features) for features in features_list]

def main():
    predictor = HousePricePredictor()

    # Example predictions
    test_samples = [
        [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23],  # High-income area
        [3.2500, 15.0, 5.000000, 1.106383, 1551.0, 3.317073, 33.78, -117.96], # Mid-income area
        [2.5000, 30.0, 4.500000, 1.200000, 800.0, 2.800000, 34.15, -118.25],  # Lower-income area
    ]

    for i, sample in enumerate(test_samples):
        result = predictor.predict(sample)
        print(f"Sample {i+1}: {sample}")
        print(f"Predicted Price: {result['predicted_price_formatted']}")
        print(f"Confidence Interval: ${result['confidence_interval']['lower']:,.2f} - ${result['confidence_interval']['upper']:,.2f}")
        print("-" * 50)

if __name__ == "__main__":
    main()