
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from src.inference.predict import (
    PROJECT_ROOT,
    load_class_mapping,
    load_trained_model,
    preprocess_image,
)


OUTPUT_DIR = PROJECT_ROOT / "reports" / "gradcam"
LAST_CONV_LAYER = "top_conv"
BACKBONE_NAME = "efficientnetb0"


def generate_gradcam(image_path, model=None, output_path=None):
    """Generate a Grad-CAM heatmap for the predicted GI image class."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if model is None:
        model = load_trained_model()

    class_mapping = load_class_mapping()
    image_batch = preprocess_image(image_path)

    # Find the nested EfficientNetB0 backbone.
    try:
        backbone_index = next(
            i for i, layer in enumerate(model.layers)
            if layer.name == BACKBONE_NAME
        )
        backbone = model.layers[backbone_index]
        last_conv_layer = backbone.get_layer(LAST_CONV_LAYER)
    except (StopIteration, ValueError, AttributeError) as error:
        raise ValueError(
            "Could not locate EfficientNetB0/top_conv in the saved model."
        ) from error

    # Build a model using the backbone's own graph.
    # This avoids the Graph disconnected error caused by mixing
    # nested-backbone tensors with the outer model's input graph.
    feature_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[last_conv_layer.output, backbone.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, backbone_features = feature_model(
            image_batch, training=False
        )

        # Apply the trained classification head after the backbone.
        predictions = backbone_features
        for layer in model.layers[backbone_index + 1:]:
            predictions = layer(predictions, training=False)

        predicted_index_tensor = tf.argmax(
            predictions[0], output_type=tf.int32
        )
        class_score = tf.gather(
            predictions[0], predicted_index_tensor
        )

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients are unavailable. "
            "Check the saved model architecture."
        )

    # Average gradients to obtain the importance of each feature channel.
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    feature_maps = conv_outputs[0]

    heatmap = tf.reduce_sum(
        feature_maps * weights[0][tf.newaxis, tf.newaxis, :],
        axis=-1,
    )
    heatmap = tf.maximum(heatmap, 0)

    maximum = tf.reduce_max(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, maximum).numpy()

    predicted_index = int(predicted_index_tensor.numpy())
    predicted_class = class_mapping[predicted_index]
    confidence = float(predictions[0, predicted_index].numpy()) * 100

    # Load the original image for the visualization.
    with Image.open(image_path) as image:
        original_image = image.convert("RGB")
        original_array = np.asarray(original_image)

    # Resize the heatmap to the original image dimensions.
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        size=(original_array.shape[0], original_array.shape[1]),
    ).numpy().squeeze()

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{image_path.stem}_gradcam.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a side-by-side comparison.
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(original_array)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(original_array)
    axes[1].imshow(
        heatmap_resized,
        cmap="jet",
        alpha=0.4,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(
        f"Grad-CAM: {predicted_class}\n"
        f"Confidence: {confidence:.2f}%"
    )
    axes[1].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return {
        "image": image_path.name,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "gradcam_image": str(output_path),
        "notice": (
            "Research visualization only. Grad-CAM highlights model "
            "activations and is not a medical diagnosis."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Grad-CAM visualization for a GI image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to an endoscopy image.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for the heatmap PNG.",
    )
    args = parser.parse_args()

    try:
        result = generate_gradcam(
            args.image,
            output_path=args.output,
        )
        print(json.dumps(result, indent=4))
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        parser.exit(status=1, message=f"Error: {error}\n")


if __name__ == "__main__":
    main()
