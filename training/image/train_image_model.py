"""Train the NewsXpose image classifier from local real/ and fake/ folders."""

from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(image_size: tuple[int, int]) -> Sequential:
    model = Sequential(
        [
            Input(shape=(image_size[0], image_size[1], 3)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dense(2, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(dataset_path: Path, output_path: Path, epochs: int) -> None:
    if not (dataset_path / "real").is_dir() or not (dataset_path / "fake").is_dir():
        raise FileNotFoundError(
            "The dataset directory must contain real/ and fake/ subdirectories."
        )

    image_size = (128, 128)
    batch_size = 64
    generator = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
    training = generator.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
    )
    validation = generator.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
    )

    model = build_model(image_size)
    model.fit(training, validation_data=validation, epochs=epochs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    output_path.write_bytes(converter.convert())
    print(f"Saved model to {output_path}")


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "image-model.tflite",
    )
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(arguments.dataset, arguments.output, arguments.epochs)
