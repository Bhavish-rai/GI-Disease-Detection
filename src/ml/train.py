
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from src.ml.data_loader import load_and_split_dataset
from src.ml.preprocess import BATCH_SIZE, IMAGE_SIZE, build_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SPLITS_DIR = REPORTS_DIR / "splits"

MODEL_PATH = MODELS_DIR / "best_efficientnetb0.keras"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
HISTORY_PATH = REPORTS_DIR / "training_history.csv"

SEED = 42
HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 5
FINE_TUNE_LAST_LAYERS = 20

LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5


def set_reproducibility():
    """Set random seeds for more reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def create_directories():
    """Create the required output directories."""
    for directory in (MODELS_DIR, REPORTS_DIR, SPLITS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_split_manifests(train_df, val_df, test_df):
    """Save split manifests without copying image files."""
    splits = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }

    for split_name, dataframe in splits.items():
        output = dataframe.copy()

        output["filepath"] = output["filepath"].apply(
            lambda path: str(
                Path(path).resolve().relative_to(PROJECT_ROOT)
            )
        )

        output.to_csv(
            SPLITS_DIR / f"{split_name}.csv",
            index=False,
        )

    print(f"Split manifests saved to: {SPLITS_DIR}")


def build_model(num_classes):
    """Build an EfficientNetB0 model with a pretrained backbone."""
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )

    backbone.trainable = False

    inputs = layers.Input(
        shape=(*IMAGE_SIZE, 3),
        name="image",
    )

    features = backbone(inputs, training=False)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dropout(0.3)(features)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(features)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="gi_efficientnetb0",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, backbone


def make_callbacks(checkpoint, append_history=False):
    """Create callbacks and reuse the same best-model checkpoint."""
    return [
        checkpoint,
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(
            str(HISTORY_PATH),
            append=append_history,
        ),
    ]


def save_metadata(class_mapping, train_df, val_df, test_df):
    """Save model configuration and dataset split information."""
    metadata = {
        "model_name": "EfficientNetB0",
        "input_size": list(IMAGE_SIZE),
        "num_classes": len(class_mapping),
        "class_mapping": class_mapping,
        "batch_size": BATCH_SIZE,
        "random_seed": SEED,
        "dataset_split_counts": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "best_model_path": str(
            MODEL_PATH.relative_to(PROJECT_ROOT)
        ),
        "training_history_path": str(
            HISTORY_PATH.relative_to(PROJECT_ROOT)
        ),
        "test_set_used_during_training": False,
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print(f"Model metadata saved to: {METADATA_PATH}")


def train(dry_run=False):
    set_reproducibility()
    create_directories()

    print("Loading dataset and creating splits...")

    train_df, val_df, test_df, class_mapping = (
        load_and_split_dataset(verify=False)
    )

    save_split_manifests(train_df, val_df, test_df)

    print("\nBuilding TensorFlow datasets...")

    train_ds = build_dataset(
        train_df,
        training=True,
        batch_size=BATCH_SIZE,
    )

    val_ds = build_dataset(
        val_df,
        training=False,
        batch_size=BATCH_SIZE,
    )

    print("\nBuilding EfficientNetB0 model...")

    model, _ = build_model(len(class_mapping))
    model.summary()

    if dry_run:
        print("\nDRY RUN PASSED.")
        print("Dataset, preprocessing pipeline and model build succeeded.")
        print("No training epochs were run.")
        return

    # Create this ONCE and reuse it in both training phases.
    # This preserves the best validation-loss value across phases.
    checkpoint = ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    print("\nPhase 1: Training the classification head...")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        callbacks=make_callbacks(
            checkpoint,
            append_history=False,
        ),
    )

    if not MODEL_PATH.exists():
        raise RuntimeError(
            "No model checkpoint was saved during Phase 1."
        )

    print("\nLoading the best Phase 1 checkpoint...")

    model = tf.keras.models.load_model(MODEL_PATH)
    backbone = model.get_layer("efficientnetb0")

    print("\nPhase 2: Fine-tuning the final backbone layers...")

    backbone.trainable = True

    for layer in backbone.layers[:-FINE_TUNE_LAST_LAYERS]:
        layer.trainable = False

    # Keep BatchNormalization layers frozen for stable fine-tuning.
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=FINE_TUNE_LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=make_callbacks(
            checkpoint,
            append_history=True,
        ),
    )

    # Reload the best checkpoint across both phases.
    best_model = tf.keras.models.load_model(MODEL_PATH)
    del best_model

    if HISTORY_PATH.exists():
        history_df = pd.read_csv(HISTORY_PATH)

        if "val_loss" in history_df.columns:
            best_val_loss = float(history_df["val_loss"].min())
            print(
                f"\nLowest logged validation loss: "
                f"{best_val_loss:.6f}"
            )

    save_metadata(
        class_mapping,
        train_df,
        val_df,
        test_df,
    )

    print("\nTraining completed successfully.")
    print(f"Best model: {MODEL_PATH}")
    print(f"Training history: {HISTORY_PATH}")
    print("The test set has not been evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the GI EfficientNetB0 classifier."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the pipeline without running training epochs.",
    )

    args = parser.parse_args()
    train(dry_run=args.dry_run)
