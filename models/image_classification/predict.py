import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Any, Union

class ImageClassifier:
    def __init__(self, model_path: str = "artifacts/image_classifier.h5"):
        self.model = keras.models.load_model(model_path)
        self.class_names = ['red_dominant', 'green_dominant', 'blue_dominant']
        self.input_shape = (32, 32, 3)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a single image for prediction"""
        if image.shape != self.input_shape:
            # Resize if needed (simple approach)
            if len(image.shape) == 2:  # Grayscale
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[-1] == 1:  # Grayscale with channel dim
                image = np.repeat(image, 3, axis=-1)

            # Simple resize (you might want to use cv2.resize in practice)
            if image.shape[:2] != self.input_shape[:2]:
                print(f"Warning: Image shape {image.shape} doesn't match expected {self.input_shape}")

        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0

        return image

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict class for a single image"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        processed_image = self.preprocess_image(image[0])
        processed_image = np.expand_dims(processed_image, axis=0)

        predictions = self.model.predict(processed_image, verbose=0)
        probabilities = predictions[0]

        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])

        result = {
            'prediction': predicted_class,
            'prediction_id': int(predicted_class_idx),
            'confidence': confidence,
            'probabilities': {
                self.class_names[i]: float(prob)
                for i, prob in enumerate(probabilities)
            },
            'input_shape': processed_image.shape
        }

        return result

    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Predict classes for a batch of images"""
        return [self.predict(image) for image in images]

    def create_sample_image(self, class_type: str = 'red_dominant') -> np.ndarray:
        """Create a sample image for testing"""
        np.random.seed(42)
        image = np.random.rand(32, 32, 3)

        if class_type == 'red_dominant':
            image[:, :, 0] += 0.3  # Boost red
        elif class_type == 'green_dominant':
            image[:, :, 1] += 0.3  # Boost green
        elif class_type == 'blue_dominant':
            image[:, :, 2] += 0.3  # Boost blue

        return np.clip(image, 0, 1)

def main():
    classifier = ImageClassifier()

    # Create sample images for testing
    test_images = [
        classifier.create_sample_image('red_dominant'),
        classifier.create_sample_image('green_dominant'),
        classifier.create_sample_image('blue_dominant'),
    ]

    class_types = ['red_dominant', 'green_dominant', 'blue_dominant']

    for i, (image, expected_class) in enumerate(zip(test_images, class_types)):
        result = classifier.predict(image)
        print(f"Sample {i+1} (Expected: {expected_class}):")
        print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
        print(f"Probabilities: {result['probabilities']}")
        print(f"Input shape: {result['input_shape']}")
        print("-" * 50)

if __name__ == "__main__":
    main()