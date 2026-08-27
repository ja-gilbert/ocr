"""HTTP server that exposes the OCR neural network to the browser client.

    python server.py        # serves on http://localhost:8000

A single endpoint accepts POSTed JSON; the operation is chosen by which flag
is present:

    {"train": true, "trainArray": [{"y0": [...400 values...], "label": 7}]}
    {"predict": true, "image": [...400 values...]}  ->  {"type": "test", "result": 7}
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ocr import OCRNeuralNetwork, load_training_data

HOST_NAME = "localhost"
PORT_NUMBER = 8000
# 15 hidden nodes was the sweet spot found by neural_network_design.py.
HIDDEN_NODE_COUNT = 15


class JSONHandler(BaseHTTPRequestHandler):
    """Handle the JSON POSTs sent by ocr.js."""

    server_version = "OCRDemo/1.0"

    def do_POST(self):
        response_code = 200
        response = None

        try:
            content_length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            content_length = 0
        if content_length <= 0:
            self._respond(400)
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
        except (ValueError, UnicodeDecodeError):
            self._respond(400)
            return
        if not isinstance(payload, dict):
            self._respond(400)
            return

        if payload.get("train"):
            try:
                with self.server.nn_lock:
                    self.server.nn.train(payload["trainArray"])
                    self.server.nn.save()
            except (KeyError, TypeError, ValueError):
                traceback.print_exc()
                response_code = 400
        elif payload.get("predict"):
            try:
                with self.server.nn_lock:
                    result = self.server.nn.predict(payload["image"])
                response = {"type": "test", "result": result}
            except Exception:
                traceback.print_exc()
                response_code = 500
        else:
            response_code = 400

        self._respond(response_code, response)

    def do_OPTIONS(self):
        """Answer the CORS preflight a JSON Content-Type header triggers."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _respond(self, response_code, response=None):
        body = b"" if response is None else json.dumps(response).encode("utf-8")
        self.send_response(response_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # So ocr.html can be opened straight off disk.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if body:
            self.wfile.write(body)


def build_network():
    """Load nn.json if it exists, otherwise train on all 5000 samples."""
    if os.path.isfile(OCRNeuralNetwork.NN_FILE_PATH):
        print(f"Loading weights from {OCRNeuralNetwork.NN_FILE_PATH}")
        return OCRNeuralNetwork(HIDDEN_NODE_COUNT)

    data_matrix, data_labels = load_training_data()
    print(f"No {OCRNeuralNetwork.NN_FILE_PATH} yet - training on "
          f"{len(data_matrix)} samples, this takes a moment...")
    return OCRNeuralNetwork(
        HIDDEN_NODE_COUNT, data_matrix, data_labels, range(len(data_matrix))
    )


def main():
    try:
        nn = build_network()
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    httpd = ThreadingHTTPServer((HOST_NAME, PORT_NUMBER), JSONHandler)
    # The handler runs on a worker thread per request; the network's weights
    # are shared mutable state, so serialise access to them.
    httpd.nn = nn
    httpd.nn_lock = threading.Lock()

    print(f"Serving on http://{HOST_NAME}:{PORT_NUMBER} - press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
