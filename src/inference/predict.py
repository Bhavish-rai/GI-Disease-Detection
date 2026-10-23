
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_efficientnetb0.keras"
CLASS_MAPPING_PATH = PROJECT_ROOT / "src" / "ml" / "class_mapping.json"

IMAGE_SIZE = (224, 224)


def load_class_mapping():
    """Load the class names and their corresponding model indices."""
    if not CLASS_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Class mapping not found: {CLASS_MAPPING_PATH}"
        )

    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as file:
        mapping = json.load(file)

    # Convert JSON indices to integers and order by model output index.
    return {
        int(index): class_name
        for class_name, index in mapping.items()
    }


def load_trained_model():
    """Load the previously trained EfficientNetB0 model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Make sure model training has completed successfully."
        )

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess_image(image_path):
    """Prepare an endoscopy image using the training image format."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize(IMAGE_SIZE)
            image_array = np.asarray(image, dtype=np.float32)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            f"Could not read a valid image: {image_path}"
        ) from exc

    # Keep pixel values in the 0-255 range.
    # EfficientNetB0 includes its own input rescaling.
    return np.expand_dims(image_array, axis=0)


def predict_image(image_path, model=None):
    """Predict the GI image class and return confidence scores."""
    if model is None:
        model = load_trained_model()

    class_mapping = load_class_mapping()
    image_array = preprocess_image(image_path)

    probabilities = model.predict(image_array, verbose=0)[0]

    if len(probabilities) != len(class_mapping):
        raise ValueError(
            "The number of model outputs does not match "
            "the number of classes in class_mapping.json."
        )

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_mapping[predicted_index]
    confidence = float(probabilities[predicted_index])

    top_indices = np.argsort(probabilities)[::-1][:3]

    return {
        "image": Path(image_path).name,
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2),
        "top_predictions": [
            {
                "class_name": class_mapping[int(index)],
                "confidence": round(float(probabilities[index]) * 100, 2),
            }
            for index in top_indices
        ],
        "notice": (
            "Research prototype only. This prediction is not "
            "a medical diagnosis."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict a gastrointestinal image class."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to an endoscopy image."
    )
    args = parser.parse_args()

    try:
        result = predict_image(args.image)
        print(json.dumps(result, indent=4))
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.exit(status=1, message=f"Error: {error}\n")


if __name__ == "__main__":
    main()
