from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class OCRNeuralNetwork:
    LEARNING_RATE = 0.1
    INPUT_SIZE = 400
    NUM_DIGITS = 10

    def __init__(
        self,
        num_hidden_nodes: int = 15,
        *,
        weights_path: str | Path = "nn.json",
        use_file: bool = True,
        seed: int = 42,
    ) -> None:
        self.num_hidden_nodes = num_hidden_nodes
        self.weights_path = Path(weights_path)
        self.use_file = use_file
        self.rng = np.random.default_rng(seed)

        if self.use_file and self.weights_path.exists():
            self._load()
        else:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        self.theta1 = self._rand_initialize_weights(
            self.INPUT_SIZE,
            self.num_hidden_nodes,
        )
        self.theta2 = self._rand_initialize_weights(
            self.num_hidden_nodes,
            self.NUM_DIGITS,
        )
        self.input_layer_bias = self._rand_initialize_weights(
            1,
            self.num_hidden_nodes,
        ).reshape(self.num_hidden_nodes)
        self.hidden_layer_bias = self._rand_initialize_weights(
            1,
            self.NUM_DIGITS,
        ).reshape(self.NUM_DIGITS)

    def _rand_initialize_weights(self, size_in: int, size_out: int) -> FloatArray:
        return self.rng.uniform(-0.06, 0.06, size=(size_out, size_in))

    @staticmethod
    def sigmoid(z: FloatArray) -> FloatArray:
        # Clipping prevents exp() overflow for unusually large values.
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def sigmoid_prime(cls, z: FloatArray) -> FloatArray:
        value = cls.sigmoid(z)
        return value * (1.0 - value)

    @classmethod
    def _validate_input(cls, values: Any) -> FloatArray:
        array = np.asarray(values, dtype=np.float64).reshape(-1)

        if array.shape != (cls.INPUT_SIZE,):
            raise ValueError(
                f"Expected {cls.INPUT_SIZE} input values, "
                f"received {array.size}."
            )

        return array

    @classmethod
    def _one_hot(cls, label: int) -> FloatArray:
        """The output layer we want for a label: 1.0 for it, 0.0 elsewhere."""
        target = np.zeros(cls.NUM_DIGITS, dtype=np.float64)
        target[label] = 1.0
        return target

    def _forward(self, input_values: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        hidden_sum = self.theta1 @ input_values + self.input_layer_bias
        hidden_output = self.sigmoid(hidden_sum)

        output_sum = self.theta2 @ hidden_output + self.hidden_layer_bias
        output = self.sigmoid(output_sum)

        return hidden_sum, hidden_output, output

    def train(
        self,
        training_data: Iterable[Mapping[str, Any]],
        *,
        epochs: int = 1,
    ) -> None:
        samples = list(training_data)

        if not samples:
            raise ValueError("No training samples were provided.")

        for _ in range(epochs):
            self.rng.shuffle(samples)

            for sample in samples:
                self._train_sample(sample)

    def _train_sample(self, sample: Mapping[str, Any]) -> None:
        """Feed one drawing forward, then nudge every weight toward its label."""
        input_values = self._validate_input(sample["y0"])
        label = int(sample["label"])

        if not 0 <= label < self.NUM_DIGITS:
            raise ValueError("Training label must be an integer from 0 through 9.")

        hidden_sum, hidden_output, output = self._forward(input_values)

        output_errors = self._one_hot(label) - output
        hidden_errors = (self.theta2.T @ output_errors) * self.sigmoid_prime(hidden_sum)

        self.theta1 += (
            self.LEARNING_RATE
            * np.outer(hidden_errors, input_values)
        )
        self.theta2 += (
            self.LEARNING_RATE
            * np.outer(output_errors, hidden_output)
        )
        self.input_layer_bias += self.LEARNING_RATE * hidden_errors
        self.hidden_layer_bias += self.LEARNING_RATE * output_errors

    def predict(self, test: Any) -> int:
        input_values = self._validate_input(test)
        _, _, output = self._forward(input_values)
        return int(np.argmax(output))

    def save(self) -> None:
        if not self.use_file:
            return

        payload = {
            "num_hidden_nodes": self.num_hidden_nodes,
            "theta1": self.theta1.tolist(),
            "theta2": self.theta2.tolist(),
            "input_layer_bias": self.input_layer_bias.tolist(),
            "hidden_layer_bias": self.hidden_layer_bias.tolist(),
        }

        self.weights_path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def _load(self) -> None:
        payload = json.loads(self.weights_path.read_text(encoding="utf-8"))

        self.num_hidden_nodes = int(payload["num_hidden_nodes"])
        self.theta1 = np.asarray(payload["theta1"], dtype=np.float64)
        self.theta2 = np.asarray(payload["theta2"], dtype=np.float64)
        self.input_layer_bias = np.asarray(
            payload["input_layer_bias"],
            dtype=np.float64,
        )
        self.hidden_layer_bias = np.asarray(
            payload["hidden_layer_bias"],
            dtype=np.float64,
        )
