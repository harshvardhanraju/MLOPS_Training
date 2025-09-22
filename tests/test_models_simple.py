"""
Simplified model tests without heavy dependencies
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_iris_model_simple():
    """Simple test for iris model without MLflow"""
    print("🧪 Testing Iris Model (Simple)...")

    try:
        # Load data
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target, name='target')

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Test prediction
        sample_prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
        sample_proba = model.predict_proba([[5.1, 3.5, 1.4, 0.2]])

        # Save model
        os.makedirs("models/iris/artifacts", exist_ok=True)
        with open("models/iris/artifacts/iris_model.pkl", "wb") as f:
            pickle.dump(model, f)

        print(f"   ✅ Accuracy: {accuracy:.3f}")
        print(f"   ✅ Sample prediction: {sample_prediction[0]}")
        print(f"   ✅ Model saved successfully")

        assert accuracy > 0.8, f"Accuracy too low: {accuracy}"
        assert sample_prediction[0] in [0, 1, 2], "Invalid prediction"

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_housing_model_simple():
    """Simple test for housing model without XGBoost"""
    print("🧪 Testing Housing Model (Simple)...")

    try:
        # Load data
        housing = fetch_california_housing()
        X = pd.DataFrame(housing.data, columns=housing.feature_names)
        y = pd.Series(housing.target, name='price')

        # Take a smaller sample for faster testing
        X = X.head(1000)
        y = y.head(1000)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train simple linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Test prediction
        sample_data = [[8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]]
        sample_prediction = model.predict(sample_data)

        # Save model
        os.makedirs("models/house_price/artifacts", exist_ok=True)
        with open("models/house_price/artifacts/simple_house_model.pkl", "wb") as f:
            pickle.dump(model, f)

        print(f"   ✅ R2 Score: {r2:.3f}")
        print(f"   ✅ RMSE: {rmse:.3f}")
        print(f"   ✅ Sample prediction: {sample_prediction[0]:.2f}")
        print(f"   ✅ Model saved successfully")

        assert r2 > 0.1, f"R2 score too low: {r2}"
        assert sample_prediction[0] > 0, "Invalid prediction"

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_simple_text_classifier():
    """Simple test for text classification"""
    print("🧪 Testing Simple Text Classifier...")

    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline

        # Create simple sentiment data
        texts = [
            "I love this product", "This is amazing", "Great quality",
            "I hate this", "This is terrible", "Poor quality",
            "It's okay", "Average product", "Nothing special"
        ]
        labels = [2, 2, 2, 0, 0, 0, 1, 1, 1]  # 0=negative, 1=neutral, 2=positive

        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', MultinomialNB())
        ])

        # Train
        pipeline.fit(texts, labels)

        # Test prediction
        test_texts = ["I love it", "It's bad", "It's fine"]
        predictions = pipeline.predict(test_texts)
        probabilities = pipeline.predict_proba(test_texts)

        # Save model
        os.makedirs("models/sentiment/artifacts", exist_ok=True)
        with open("models/sentiment/artifacts/simple_sentiment_model.pkl", "wb") as f:
            pickle.dump(pipeline, f)

        print(f"   ✅ Predictions: {predictions}")
        print(f"   ✅ Model saved successfully")

        assert len(predictions) == 3, "Should predict for all samples"
        assert all(pred in [0, 1, 2] for pred in predictions), "Invalid predictions"

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_model_persistence():
    """Test model loading and persistence"""
    print("🧪 Testing Model Persistence...")

    try:
        # Test iris model loading
        iris_path = "models/iris/artifacts/iris_model.pkl"
        if os.path.exists(iris_path):
            with open(iris_path, "rb") as f:
                iris_model = pickle.load(f)

            # Test prediction
            test_pred = iris_model.predict([[5.1, 3.5, 1.4, 0.2]])
            print(f"   ✅ Iris model loaded and working: {test_pred[0]}")

        # Test housing model loading
        house_path = "models/house_price/artifacts/simple_house_model.pkl"
        if os.path.exists(house_path):
            with open(house_path, "rb") as f:
                house_model = pickle.load(f)

            # Test prediction
            test_pred = house_model.predict([[8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]])
            print(f"   ✅ House model loaded and working: {test_pred[0]:.2f}")

        # Test sentiment model loading
        sentiment_path = "models/sentiment/artifacts/simple_sentiment_model.pkl"
        if os.path.exists(sentiment_path):
            with open(sentiment_path, "rb") as f:
                sentiment_model = pickle.load(f)

            # Test prediction
            test_pred = sentiment_model.predict(["I love this"])
            print(f"   ✅ Sentiment model loaded and working: {test_pred[0]}")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def run_all_simple_tests():
    """Run all simple model tests"""
    print("🚀 Running Simple Model Tests")
    print("=" * 50)

    results = []

    # Run tests
    results.append(("Iris Model", test_iris_model_simple()))
    results.append(("Housing Model", test_housing_model_simple()))
    results.append(("Text Classifier", test_simple_text_classifier()))
    results.append(("Model Persistence", test_model_persistence()))

    # Print summary
    print("\n📊 Test Results Summary:")
    print("-" * 30)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All simple model tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False

if __name__ == "__main__":
    success = run_all_simple_tests()
    exit(0 if success else 1)