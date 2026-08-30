"""Pick the hidden node count by experiment, as in the OCR chapter.

The chapter trains each candidate network on data.csv and scores it against a
held-out validation set. This version generates its own digits instead, so the
experiment runs with no dataset files present.
"""

from __future__ import annotations

from typing import Collection

import numpy as np
from numpy.typing import NDArray

from ocr import OCRNeuralNetwork


GRID_WIDTH = 20
SAMPLES_PER_DIGIT = 60
TRAINING_FRACTION = 0.75
TRAINING_EPOCHS = 6
TRIALS = 100

# How badly each copy of a digit is mangled. Without this the classes never
# overlap, every hidden node count scores 1.0, and the sweep says nothing.
SEGMENT_DROPOUT = 0.08
MAX_SHIFT = 2
NOISE_RATE = 0.06

# Each digit is drawn as a subset of seven bars, the way a calculator display
# does. Values are the (start, stop) row and column of each bar on the grid.
SEGMENTS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "top": ((2, 4), (5, 15)),
    "top_left": ((2, 11), (5, 7)),
    "top_right": ((2, 11), (13, 15)),
    "middle": ((9, 11), (5, 15)),
    "bottom_left": ((9, 18), (5, 7)),
    "bottom_right": ((9, 18), (13, 15)),
    "bottom": ((16, 18), (5, 15)),
}

DIGIT_SEGMENTS: dict[int, tuple[str, ...]] = {
    0: ("top", "top_left", "top_right", "bottom_left", "bottom_right", "bottom"),
    1: ("top_right", "bottom_right"),
    2: ("top", "top_right", "middle", "bottom_left", "bottom"),
    3: ("top", "top_right", "middle", "bottom_right", "bottom"),
    4: ("top_left", "top_right", "middle", "bottom_right"),
    5: ("top", "top_left", "middle", "bottom_right", "bottom"),
    6: ("top", "top_left", "middle", "bottom_left", "bottom_right", "bottom"),
    7: ("top", "top_right", "bottom_right"),
    8: (
        "top",
        "top_left",
        "top_right",
        "middle",
        "bottom_left",
        "bottom_right",
        "bottom",
    ),
    9: ("top", "top_left", "top_right", "middle", "bottom_right", "bottom"),
}


def digit_stencil(digit: int, *, skip: Collection[str] = ()) -> NDArray[np.float64]:
    """Draw one digit on a GRID_WIDTH x GRID_WIDTH grid, omitting `skip` bars."""
    grid = np.zeros((GRID_WIDTH, GRID_WIDTH), dtype=np.float64)

    for segment in DIGIT_SEGMENTS[digit]:
        if segment in skip:
            continue

        (row_start, row_stop), (col_start, col_stop) = SEGMENTS[segment]
        grid[row_start:row_stop, col_start:col_stop] = 1.0

    return grid


def noisy_digit(digit: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """One handwriting-like copy of a digit, flattened into a single row.

    Three things stand in for the variation in real handwriting: a bar may go
    missing, the whole digit slides around the grid, and stray pixels flip. The
    missing bars matter most, since they are what makes an 8 look like a 9.
    """
    segments = DIGIT_SEGMENTS[digit]
    skip = {name for name in segments if rng.random() < SEGMENT_DROPOUT}

    # Dropping every bar would leave a blank grid wearing a digit's label.
    if len(skip) == len(segments):
        skip.clear()

    grid = digit_stencil(digit, skip=skip)
    grid = np.roll(grid, rng.integers(-MAX_SHIFT, MAX_SHIFT + 1, size=2), axis=(0, 1))

    flipped = rng.random(grid.shape) < NOISE_RATE
    return np.where(flipped, 1.0 - grid, grid).reshape(-1)


def make_dataset(
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Build SAMPLES_PER_DIGIT noisy copies of each digit, one per row."""
    images = []
    labels = []

    for digit in range(10):
        for _ in range(SAMPLES_PER_DIGIT):
            images.append(noisy_digit(digit, rng))
            labels.append(digit)

    return np.asarray(images), np.asarray(labels)


def to_training_records(
    images: NDArray[np.float64],
    labels: NDArray[np.int_],
) -> list[dict[str, object]]:
    """Shape the arrays like the JSON the browser posts to the server."""
    return [
        {"y0": image, "label": int(label)}
        for image, label in zip(images, labels, strict=True)
    ]


def accuracy(
    network: OCRNeuralNetwork,
    images: NDArray[np.float64],
    labels: NDArray[np.int_],
) -> float:
    """Fraction of images the network labels correctly."""
    predictions = np.asarray([network.predict(image) for image in images])
    return float(np.mean(predictions == labels))


def average_accuracy(
    hidden_nodes: int,
    training_records: list[dict[str, object]],
    images: NDArray[np.float64],
    labels: NDArray[np.int_],
) -> float:
    """Score one configuration, averaged over TRIALS freshly trained networks.

    Weights start at random values, so a single run is too noisy to compare
    against another configuration.
    """
    total = 0.0

    for trial in range(TRIALS):
        network = OCRNeuralNetwork(hidden_nodes, use_file=False, seed=trial)
        network.train(training_records, epochs=TRAINING_EPOCHS)
        total += accuracy(network, images, labels)

    return total / TRIALS


def main() -> None:
    rng = np.random.default_rng(10)
    images, labels = make_dataset(rng)

    indices = rng.permutation(len(labels))
    split = int(len(labels) * TRAINING_FRACTION)
    training_indices, validation_indices = indices[:split], indices[split:]

    training_records = to_training_records(
        images[training_indices],
        labels[training_indices],
    )
    validation_images = images[validation_indices]
    validation_labels = labels[validation_indices]

    print("PERFORMANCE")
    print("-----------")

    for hidden_nodes in range(5, 50, 5):
        score = average_accuracy(
            hidden_nodes,
            training_records,
            validation_images,
            validation_labels,
        )
        print(f"{hidden_nodes} Hidden Nodes: {score:.4f}")


if __name__ == "__main__":
    main()
