import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class ChurnPredictor:
    def __init__(self, model_path: str = "artifacts/churn_model.pkl",
                 encoders_path: str = "artifacts/label_encoders.pkl"):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(encoders_path, 'rb') as f:
            self.label_encoders = pickle.load(f)

        self.feature_names = [
            'age', 'tenure_months', 'monthly_charges', 'total_charges',
            'contract_type', 'payment_method', 'internet_service',
            'online_security', 'tech_support', 'streaming_tv',
            'paperless_billing', 'senior_citizen', 'partner', 'dependents'
        ]

    def preprocess_features(self, features: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess input features"""
        # Create DataFrame from input
        df = pd.DataFrame([features])

        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                raise ValueError(f"Missing required feature: {feature}")

        # Reorder columns to match training data
        df = df[self.feature_names]

        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col])
                except ValueError as e:
                    # Handle unseen categories
                    print(f"Warning: Unseen category in {col}. Using most frequent category.")
                    df[col] = encoder.transform([encoder.classes_[0]])[0]

        return df

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict churn for a single customer"""
        X_processed = self.preprocess_features(features)

        # Get prediction probability
        churn_probability = self.model.predict(X_processed, num_iteration=self.model.best_iteration)[0]
        churn_prediction = int(churn_probability > 0.5)

        # Calculate risk level
        if churn_probability < 0.3:
            risk_level = "Low"
        elif churn_probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        result = {
            'churn_prediction': churn_prediction,
            'churn_probability': float(churn_probability),
            'risk_level': risk_level,
            'confidence': float(max(churn_probability, 1 - churn_probability)),
            'recommendations': self._get_recommendations(features, churn_probability)
        }

        return result

    def _get_recommendations(self, features: Dict[str, Any], churn_prob: float) -> List[str]:
        """Generate recommendations based on customer features"""
        recommendations = []

        if churn_prob > 0.5:
            if features.get('contract_type') == 'Month-to-month':
                recommendations.append("Offer long-term contract with discount")

            if features.get('tech_support') == 'No':
                recommendations.append("Provide free tech support services")

            if features.get('monthly_charges', 0) > 80:
                recommendations.append("Consider offering a discount or plan downgrade")

            if features.get('tenure_months', 0) < 12:
                recommendations.append("Implement new customer retention program")

            if not recommendations:
                recommendations.append("Schedule customer satisfaction call")

        return recommendations

    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict churn for multiple customers"""
        return [self.predict(features) for features in features_list]

def main():
    predictor = ChurnPredictor()

    # Example customer data
    test_customers = [
        {
            'age': 45,
            'tenure_months': 36,
            'monthly_charges': 75.5,
            'total_charges': 2500.0,
            'contract_type': 'Two year',
            'payment_method': 'Credit card',
            'internet_service': 'Fiber optic',
            'online_security': 'Yes',
            'tech_support': 'Yes',
            'streaming_tv': 'Yes',
            'paperless_billing': 'Yes',
            'senior_citizen': 0,
            'partner': 'Yes',
            'dependents': 'Yes'
        },
        {
            'age': 25,
            'tenure_months': 3,
            'monthly_charges': 95.0,
            'total_charges': 285.0,
            'contract_type': 'Month-to-month',
            'payment_method': 'Electronic check',
            'internet_service': 'Fiber optic',
            'online_security': 'No',
            'tech_support': 'No',
            'streaming_tv': 'No',
            'paperless_billing': 'Yes',
            'senior_citizen': 0,
            'partner': 'No',
            'dependents': 'No'
        },
        {
            'age': 65,
            'tenure_months': 60,
            'monthly_charges': 45.0,
            'total_charges': 2700.0,
            'contract_type': 'One year',
            'payment_method': 'Bank transfer',
            'internet_service': 'DSL',
            'online_security': 'Yes',
            'tech_support': 'Yes',
            'streaming_tv': 'No',
            'paperless_billing': 'No',
            'senior_citizen': 1,
            'partner': 'Yes',
            'dependents': 'No'
        }
    ]

    for i, customer in enumerate(test_customers):
        result = predictor.predict(customer)
        print(f"Customer {i+1}:")
        print(f"Churn Prediction: {'Will Churn' if result['churn_prediction'] else 'Will Stay'}")
        print(f"Churn Probability: {result['churn_probability']:.3f}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendations: {result['recommendations']}")
        print("-" * 50)

if __name__ == "__main__":
    main()