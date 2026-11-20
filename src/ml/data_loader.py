
from pathlib import Path
import json

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset" / "kvasir-dataset-v2"
CLASS_MAPPING_PATH = Path(__file__).resolve().parent / "class_mapping.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15


def discover_classes():
    """Return class directories in a deterministic order."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    classes = sorted(
        folder.name
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    if len(classes) != 8:
        raise ValueError(
            f"Expected 8 class folders, found {len(classes)}: {classes}"
        )

    return classes


def create_class_mapping(classes):
    """Create and save a stable class-name-to-index mapping."""
    mapping = {
        class_name: index
        for index, class_name in enumerate(classes)
    }

    CLASS_MAPPING_PATH.write_text(
        json.dumps(mapping, indent=4),
        encoding="utf-8",
    )

    return mapping


def collect_image_paths():
    """Collect image paths and integer labels from class folders."""
    classes = discover_classes()
    class_mapping = create_class_mapping(classes)

    records = []

    for class_name in classes:
        class_dir = DATASET_DIR / class_name

        image_paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            records.append(
                {
                    "filepath": str(image_path.resolve()),
                    "class_name": class_name,
                    "label": class_mapping[class_name],
                }
            )

    if not records:
        raise ValueError(f"No images found in {DATASET_DIR}")

    return pd.DataFrame(records), class_mapping


def verify_images(dataframe):
    """Check that images can be opened and decoded."""
    invalid_paths = []

    for image_path in dataframe["filepath"]:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception:
            invalid_paths.append(image_path)

    if invalid_paths:
        print(f"Invalid images found: {len(invalid_paths)}")

        for path in invalid_paths[:20]:
            print(f"  {path}")

        if len(invalid_paths) > 20:
            print("Only the first 20 invalid paths are shown.")

    valid_paths = set(dataframe["filepath"]) - set(invalid_paths)

    clean_dataframe = dataframe[
        dataframe["filepath"].isin(valid_paths)
    ].copy()

    if clean_dataframe.empty:
        raise ValueError("No valid images remain after verification.")

    return clean_dataframe.reset_index(drop=True)


def split_dataset(dataframe):
    """Create reproducible, stratified train/validation/test splits."""
    train_validation, test = train_test_split(
        dataframe,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataframe["label"],
    )

    validation_fraction_of_remaining = (
        VALIDATION_SIZE / (1.0 - TEST_SIZE)
    )

    train, validation = train_test_split(
        train_validation,
        test_size=validation_fraction_of_remaining,
        random_state=RANDOM_STATE,
        stratify=train_validation["label"],
    )

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def print_dataset_summary(dataframe, train, validation, test):
    """Print class counts and split sizes."""
    print("\nImages per class:")
    print(
        dataframe.groupby("class_name")
        .size()
        .sort_index()
        .to_string()
    )

    print("\nDataset split sizes:")
    print(f"Training:   {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Testing:    {len(test)}")
    print(f"Total:      {len(dataframe)}")

    print("\nTraining images per class:")
    print(train["class_name"].value_counts().sort_index().to_string())

    print("\nValidation images per class:")
    print(
        validation["class_name"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTesting images per class:")
    print(test["class_name"].value_counts().sort_index().to_string())


def load_and_split_dataset(verify=True):
    """Load image paths, optionally verify them, and create splits."""
    dataframe, class_mapping = collect_image_paths()

    if verify:
        dataframe = verify_images(dataframe)

    train, validation, test = split_dataset(dataframe)

    print_dataset_summary(dataframe, train, validation, test)

    return train, validation, test, class_mapping


if __name__ == "__main__":
    train_df, val_df, test_df, mapping = load_and_split_dataset()

    print("\nClass mapping:")
    print(json.dumps(mapping, indent=4))
