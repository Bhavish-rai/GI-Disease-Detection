
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.ml.preprocess import BATCH_SIZE, build_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_efficientnetb0.keras"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
TEST_MANIFEST = PROJECT_ROOT / "reports" / "splits" / "test.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_JSON = REPORTS_DIR / "evaluation_metrics.json"
REPORT_CSV = REPORTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_PNG = REPORTS_DIR / "confusion_matrix.png"


def load_test_manifest():
    """Load the exact test split saved during training."""
    if not TEST_MANIFEST.exists():
        raise FileNotFoundError(
            f"Test manifest not found: {TEST_MANIFEST}\n"
            "Run the training pipeline first."
        )

    dataframe = pd.read_csv(TEST_MANIFEST)

    required_columns = {"filepath", "label", "class_name"}
    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"Test manifest is missing columns: {sorted(missing)}"
        )

    dataframe["filepath"] = dataframe["filepath"].apply(
        lambda path: str((PROJECT_ROOT / path).resolve())
    )

    missing_files = [
        path
        for path in dataframe["filepath"]
        if not Path(path).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} test images could not be found. "
            f"First missing file: {missing_files[0]}"
        )

    return dataframe


def save_confusion_matrix(y_true, y_pred, class_names):
    """Plot and save the confusion matrix."""
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    figure, axis = plt.subplots(figsize=(12, 10))

    image = axis.imshow(matrix, interpolation="nearest")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="Actual class",
        title="EfficientNetB0 - Test Confusion Matrix",
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = matrix.max() / 2 if matrix.size else 0

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=8,
            )

    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PNG, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return matrix


def evaluate():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {METADATA_PATH}"
        )

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    class_mapping = metadata["class_mapping"]
    class_names = [
        name
        for name, index in sorted(
            class_mapping.items(),
            key=lambda item: item[1],
        )
    ]

    test_df = load_test_manifest()

    if test_df.empty:
        raise ValueError("The test manifest contains no images.")

    if not set(test_df["label"].unique()).issubset(
        set(range(len(class_names)))
    ):
        raise ValueError("Test manifest contains an unknown class label.")

    print(f"Test images: {len(test_df)}")
    print(f"Classes: {len(class_names)}")
    print(f"Loading model: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    test_ds = build_dataset(
        test_df,
        training=False,
        batch_size=BATCH_SIZE,
    )

    print("\nEvaluating the held-out test set...")

    test_loss, test_accuracy = model.evaluate(
        test_ds,
        verbose=1,
    )

    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = test_df["label"].to_numpy(dtype=int)

    accuracy = float(accuracy_score(y_true, y_pred))

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )

    report_frame = pd.DataFrame(report_dict).transpose()
    report_frame.to_csv(REPORT_CSV, index=True)

    matrix = save_confusion_matrix(
        y_true,
        y_pred,
        class_names,
    )

    metrics = {
        "model_name": "EfficientNetB0",
        "test_image_count": int(len(test_df)),
        "class_count": int(len(class_names)),
        "test_loss": float(test_loss),
        "test_accuracy": accuracy,
        "test_accuracy_percent": round(accuracy * 100, 2),
        "macro_precision": float(
            report_dict["macro avg"]["precision"]
        ),
        "macro_recall": float(
            report_dict["macro avg"]["recall"]
        ),
        "macro_f1_score": float(
            report_dict["macro avg"]["f1-score"]
        ),
        "weighted_f1_score": float(
            report_dict["weighted avg"]["f1-score"]
        ),
        "class_mapping": class_mapping,
        "test_manifest": str(
            TEST_MANIFEST.relative_to(PROJECT_ROOT)
        ),
    }

    REPORT_JSON.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    print("\nCLASSIFICATION REPORT")
    print("=" * 70)
    print(report_text)

    print("\nTEST RESULTS")
    print("=" * 70)
    print(f"Test loss:       {test_loss:.4f}")
    print(f"Test accuracy:   {accuracy * 100:.2f}%")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1-score:  {metrics['macro_f1_score']:.4f}")
    print(f"Weighted F1:     {metrics['weighted_f1_score']:.4f}")

    print("\nSaved evaluation files:")
    print(f"Metrics:          {REPORT_JSON}")
    print(f"Classification:   {REPORT_CSV}")
    print(f"Confusion matrix: {CONFUSION_MATRIX_PNG}")

    print("\nEvaluation completed. The model was not retrained.")


if __name__ == "__main__":
    evaluate()
