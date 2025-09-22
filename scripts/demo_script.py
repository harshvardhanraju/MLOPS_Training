#!/usr/bin/env python3
"""
MLOps Demo Script
Interactive demonstration of the complete MLOps pipeline
"""

import requests
import json
import time
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLOpsDemo:
    """Interactive MLOps demonstration"""

    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
        self.session = requests.Session()

    def check_services(self):
        """Check if all services are running"""
        services = {
            "API": f"{self.api_base_url}/health",
            "MLflow": "http://localhost:5000",
            "Grafana": "http://localhost:3000",
            "Prometheus": "http://localhost:9090"
        }

        print("🔍 Checking service availability...")
        all_healthy = True

        for service, url in services.items():
            try:
                response = self.session.get(url, timeout=5)
                status = "✅ UP" if response.status_code == 200 else f"❌ DOWN ({response.status_code})"
                print(f"  {service}: {status}")
                if response.status_code != 200:
                    all_healthy = False
            except Exception as e:
                print(f"  {service}: ❌ DOWN ({e})")
                all_healthy = False

        if not all_healthy:
            print("\n⚠️  Some services are not available. Please ensure all services are running.")
            print("Run: docker-compose up -d")
            return False

        print("\n✅ All services are up and running!")
        return True

    def demo_iris_prediction(self):
        """Demonstrate iris flower classification"""
        print("\n" + "="*60)
        print("🌸 DEMO 1: Iris Flower Classification")
        print("="*60)

        samples = [
            {"sample": "Setosa", "data": [5.1, 3.5, 1.4, 0.2]},
            {"sample": "Versicolor", "data": [6.2, 2.8, 4.8, 1.8]},
            {"sample": "Virginica", "data": [7.7, 2.6, 6.9, 2.3]}
        ]

        for sample in samples:
            print(f"\n📊 Predicting {sample['sample']} characteristics:")
            print(f"   Sepal Length: {sample['data'][0]}, Sepal Width: {sample['data'][1]}")
            print(f"   Petal Length: {sample['data'][2]}, Petal Width: {sample['data'][3]}")

            payload = {
                "sepal_length": sample['data'][0],
                "sepal_width": sample['data'][1],
                "petal_length": sample['data'][2],
                "petal_width": sample['data'][3]
            }

            try:
                response = self.session.post(
                    f"{self.api_base_url}/api/v1/iris/predict",
                    json=payload
                )
                if response.status_code == 200:
                    result = response.json()
                    print(f"   🔮 Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")
                    print(f"   📈 Probabilities: {json.dumps(result['probabilities'], indent=6)}")
                else:
                    print(f"   ❌ Error: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"   ❌ Request failed: {e}")

            time.sleep(1)

    def demo_house_price_prediction(self):
        """Demonstrate house price prediction"""
        print("\n" + "="*60)
        print("🏠 DEMO 2: California House Price Prediction")
        print("="*60)

        samples = [
            {
                "description": "High-income Bay Area",
                "data": [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]
            },
            {
                "description": "Mid-income Inland",
                "data": [3.2500, 15.0, 5.000000, 1.106383, 1551.0, 3.317073, 33.78, -117.96]
            },
            {
                "description": "Lower-income Rural",
                "data": [2.5000, 30.0, 4.500000, 1.200000, 800.0, 2.800000, 34.15, -118.25]
            }
        ]

        for sample in samples:
            print(f"\n🏘️  Predicting price for: {sample['description']}")
            print(f"   Median Income: ${sample['data'][0]*10000:,.0f}")
            print(f"   House Age: {sample['data'][1]} years")
            print(f"   Average Rooms: {sample['data'][2]:.1f}")

            payload = {
                "med_inc": sample['data'][0],
                "house_age": sample['data'][1],
                "ave_rooms": sample['data'][2],
                "ave_bedrms": sample['data'][3],
                "population": sample['data'][4],
                "ave_occup": sample['data'][5],
                "latitude": sample['data'][6],
                "longitude": sample['data'][7]
            }

            try:
                response = self.session.post(
                    f"{self.api_base_url}/api/v1/house-price/predict",
                    json=payload
                )
                if response.status_code == 200:
                    result = response.json()
                    print(f"   💰 Predicted Price: {result['predicted_price_formatted']}")
                    conf_lower = result['confidence_interval']['lower']
                    conf_upper = result['confidence_interval']['upper']
                    print(f"   📊 Confidence Interval: ${conf_lower:,.0f} - ${conf_upper:,.0f}")
                else:
                    print(f"   ❌ Error: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"   ❌ Request failed: {e}")

            time.sleep(1)

    def demo_sentiment_analysis(self):
        """Demonstrate sentiment analysis"""
        print("\n" + "="*60)
        print("💭 DEMO 3: Sentiment Analysis")
        print("="*60)

        texts = [
            "I absolutely love this product! Best purchase ever!",
            "This is terrible quality. Complete waste of money.",
            "The product is okay. Nothing special but does the job.",
            "Outstanding service and amazing quality! Highly recommended!",
            "Poor design and it broke after one day. Very disappointed."
        ]

        for text in texts:
            print(f"\n📝 Analyzing: \"{text}\"")

            payload = {"text": text}

            try:
                response = self.session.post(
                    f"{self.api_base_url}/api/v1/sentiment/predict",
                    json=payload
                )
                if response.status_code == 200:
                    result = response.json()
                    sentiment_emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
                    emoji = sentiment_emoji.get(result['sentiment'], "🤔")
                    print(f"   {emoji} Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2%})")
                    print(f"   📊 Breakdown: {json.dumps(result['probabilities'], indent=6)}")
                else:
                    print(f"   ❌ Error: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"   ❌ Request failed: {e}")

            time.sleep(1)

    def demo_churn_prediction(self):
        """Demonstrate customer churn prediction"""
        print("\n" + "="*60)
        print("📞 DEMO 4: Customer Churn Prediction")
        print("="*60)

        customers = [
            {
                "description": "Long-term loyal customer",
                "data": {
                    "age": 65, "tenure_months": 60, "monthly_charges": 45.0, "total_charges": 2700.0,
                    "contract_type": "Two year", "payment_method": "Bank transfer", "internet_service": "DSL",
                    "online_security": "Yes", "tech_support": "Yes", "streaming_tv": "No",
                    "paperless_billing": "No", "senior_citizen": 1, "partner": "Yes", "dependents": "No"
                }
            },
            {
                "description": "New high-value customer",
                "data": {
                    "age": 25, "tenure_months": 3, "monthly_charges": 95.0, "total_charges": 285.0,
                    "contract_type": "Month-to-month", "payment_method": "Electronic check", "internet_service": "Fiber optic",
                    "online_security": "No", "tech_support": "No", "streaming_tv": "No",
                    "paperless_billing": "Yes", "senior_citizen": 0, "partner": "No", "dependents": "No"
                }
            },
            {
                "description": "Mid-tenure customer",
                "data": {
                    "age": 45, "tenure_months": 36, "monthly_charges": 75.5, "total_charges": 2500.0,
                    "contract_type": "Two year", "payment_method": "Credit card", "internet_service": "Fiber optic",
                    "online_security": "Yes", "tech_support": "Yes", "streaming_tv": "Yes",
                    "paperless_billing": "Yes", "senior_citizen": 0, "partner": "Yes", "dependents": "Yes"
                }
            }
        ]

        for customer in customers:
            print(f"\n👤 Analyzing: {customer['description']}")
            print(f"   Age: {customer['data']['age']}, Tenure: {customer['data']['tenure_months']} months")
            print(f"   Monthly Charges: ${customer['data']['monthly_charges']}")
            print(f"   Contract: {customer['data']['contract_type']}")

            try:
                response = self.session.post(
                    f"{self.api_base_url}/api/v1/churn/predict",
                    json=customer['data']
                )
                if response.status_code == 200:
                    result = response.json()
                    churn_status = "Will Churn 📈" if result['churn_prediction'] else "Will Stay 📉"
                    risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                    risk_emoji = risk_colors.get(result['risk_level'], "⚪")

                    print(f"   🔮 Prediction: {churn_status}")
                    print(f"   📊 Churn Probability: {result['churn_probability']:.1%}")
                    print(f"   {risk_emoji} Risk Level: {result['risk_level']}")
                    if result['recommendations']:
                        print(f"   💡 Recommendations:")
                        for rec in result['recommendations']:
                            print(f"      • {rec}")
                else:
                    print(f"   ❌ Error: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"   ❌ Request failed: {e}")

            time.sleep(1)

    def demo_image_classification(self):
        """Demonstrate image classification"""
        print("\n" + "="*60)
        print("🖼️  DEMO 5: Image Classification")
        print("="*60)

        print("\n📸 Generating sample images and classifying them...")

        # Get a sample image from the API
        try:
            response = self.session.get(f"{self.api_base_url}/api/v1/image/sample")
            if response.status_code == 200:
                sample_data = response.json()
                print(f"   📏 Sample image shape: {sample_data['shape']}")
                print(f"   📝 Description: {sample_data['description']}")

                # Predict using the sample image
                payload = {"image_data": sample_data['image_data']}
                pred_response = self.session.post(
                    f"{self.api_base_url}/api/v1/image/predict",
                    json=payload
                )

                if pred_response.status_code == 200:
                    result = pred_response.json()
                    print(f"   🔮 Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")
                    print(f"   📊 Probabilities: {json.dumps(result['probabilities'], indent=6)}")
                else:
                    print(f"   ❌ Prediction error: {pred_response.status_code}")
            else:
                print(f"   ❌ Sample generation error: {response.status_code}")

        except Exception as e:
            print(f"   ❌ Request failed: {e}")

        # Create additional synthetic examples
        print("\n🎨 Testing with synthetic color-dominant images...")
        for color in ["red", "green", "blue"]:
            print(f"\n   Testing {color}-dominant image...")
            # Create a simple synthetic image (32x32x3)
            image = np.random.rand(32, 32, 3)
            if color == "red":
                image[:, :, 0] += 0.3  # Boost red channel
            elif color == "green":
                image[:, :, 1] += 0.3  # Boost green channel
            else:
                image[:, :, 2] += 0.3  # Boost blue channel

            image = np.clip(image, 0, 1)  # Ensure values are in [0, 1]

            payload = {"image_data": image.tolist()}

            try:
                response = self.session.post(
                    f"{self.api_base_url}/api/v1/image/predict",
                    json=payload
                )
                if response.status_code == 200:
                    result = response.json()
                    print(f"     🔮 Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")
                else:
                    print(f"     ❌ Error: {response.status_code}")
            except Exception as e:
                print(f"     ❌ Request failed: {e}")

            time.sleep(0.5)

    def demo_monitoring_metrics(self):
        """Demonstrate monitoring and metrics"""
        print("\n" + "="*60)
        print("📊 DEMO 6: Monitoring & Metrics")
        print("="*60)

        print("\n🔍 Fetching system metrics...")

        try:
            # Get model metrics
            response = self.session.get(f"{self.api_base_url}/api/v1/model-metrics")
            if response.status_code == 200:
                metrics = response.json()
                print("\n📈 Model Performance Metrics:")
                for model, accuracy in metrics.get('model_accuracy', {}).items():
                    print(f"   {model}: {accuracy:.1%} accuracy")

                print("\n📊 Prediction Counts:")
                for model, count in metrics.get('total_predictions', {}).items():
                    print(f"   {model}: {count} predictions")

                print("\n💻 System Metrics:")
                system_metrics = metrics.get('system_metrics', {})
                cpu_usage = system_metrics.get('cpu_usage_percent', 0)
                memory_usage = system_metrics.get('memory_usage_percent', 0)
                print(f"   CPU Usage: {cpu_usage:.1f}%")
                print(f"   Memory Usage: {memory_usage:.1f}%")

            else:
                print(f"   ❌ Error fetching metrics: {response.status_code}")

        except Exception as e:
            print(f"   ❌ Request failed: {e}")

        print("\n🔗 Access Monitoring UIs:")
        print("   • MLflow: http://localhost:5000")
        print("   • Grafana: http://localhost:3000 (admin/admin)")
        print("   • Prometheus: http://localhost:9090")

    def run_complete_demo(self):
        """Run the complete MLOps demo"""
        print("🚀 MLOps Complete Demo")
        print("=" * 60)
        print(f"⏰ Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Check services
        if not self.check_services():
            sys.exit(1)

        # Wait for user input
        input("\n🎬 Press Enter to start the demo or Ctrl+C to exit...")

        try:
            # Run all demos
            self.demo_iris_prediction()
            input("\n⏸️  Press Enter to continue to next demo...")

            self.demo_house_price_prediction()
            input("\n⏸️  Press Enter to continue to next demo...")

            self.demo_sentiment_analysis()
            input("\n⏸️  Press Enter to continue to next demo...")

            self.demo_churn_prediction()
            input("\n⏸️  Press Enter to continue to next demo...")

            self.demo_image_classification()
            input("\n⏸️  Press Enter to continue to next demo...")

            self.demo_monitoring_metrics()

            print("\n" + "="*60)
            print("🎉 Demo Complete!")
            print("="*60)
            print("✅ Successfully demonstrated:")
            print("   • 5 different ML models")
            print("   • REST API predictions")
            print("   • Real-time monitoring")
            print("   • Complete MLOps pipeline")
            print("\n🔗 Next steps:")
            print("   • Explore MLflow UI: http://localhost:5000")
            print("   • View Grafana dashboards: http://localhost:3000")
            print("   • Check API documentation: http://localhost:8000/docs")

        except KeyboardInterrupt:
            print("\n\n⏹️  Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")

def main():
    """Main entry point"""
    demo = MLOpsDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()