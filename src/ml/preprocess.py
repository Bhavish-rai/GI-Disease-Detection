
import tensorflow as tf


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE


def decode_and_resize(filepath, label):
    """Read an image, decode it as RGB and resize it."""
    image_bytes = tf.io.read_file(filepath)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    # Keep pixel values in the 0-255 range.
    # Keras EfficientNetB0 includes its own input rescaling.
    return image, tf.cast(label, tf.int32)


def build_dataset(dataframe, training=False, batch_size=BATCH_SIZE):
    """Build a TensorFlow dataset from a dataframe of paths and labels."""
    if dataframe.empty:
        raise ValueError("Cannot build a dataset from zero images.")

    paths = dataframe["filepath"].astype(str).to_numpy()
    labels = dataframe["label"].to_numpy(dtype="int32")

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=42,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        decode_and_resize,
        num_parallel_calls=AUTOTUNE,
    )

    if training:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomZoom(0.1),
            ],
            name="image_augmentation",
        )

        dataset = dataset.map(
            lambda image, label: (
                augmentation(image, training=True),
                label,
            ),
            num_parallel_calls=AUTOTUNE,
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset
