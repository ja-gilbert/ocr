import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from ocr import OCRNeuralNetwork


HOST = "127.0.0.1"
PORT = 8000

BASE_DIR = Path(__file__).parent

# ocr.js fetches relative URLs, so the page has to come from this server rather
# than from disk. Only these files are served, so a crafted path cannot reach
# anything else in the folder.
STATIC_FILES: dict[str, tuple[str, str]] = {
    "/": ("ocr.html", "text/html; charset=utf-8"),
    "/ocr.html": ("ocr.html", "text/html; charset=utf-8"),
    "/ocr.css": ("ocr.css", "text/css; charset=utf-8"),
    "/ocr.js": ("ocr.js", "text/javascript; charset=utf-8"),
}

# Weights load from nn.json if it exists, so training carries over restarts.
nn = OCRNeuralNetwork(15)


class OCRServer(BaseHTTPRequestHandler):

    def do_GET(self) -> None:
        entry = STATIC_FILES.get(self.path)

        if entry is None:
            self._send_json(404, {"error": f"Not found: {self.path}"})
            return

        filename, content_type = entry
        path = BASE_DIR / filename

        if not path.exists():
            self._send_json(404, {"error": f"Missing file: {filename}"})
            return

        self._send_bytes(200, content_type, path.read_bytes())

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return

        if self.path == "/api/train":
            self._train(payload)
        elif self.path == "/api/predict":
            self._predict(payload)
        else:
            self._send_json(404, {"error": f"Unknown endpoint: {self.path}"})

    def _train(self, payload: dict[str, Any]) -> None:
        samples = payload.get("trainArray")

        if not samples:
            self._send_json(400, {"error": "Expected a non-empty trainArray."})
            return

        try:
            nn.train(samples)
            nn.save()
        except (KeyError, TypeError, ValueError) as error:
            self._send_json(400, {"error": f"Could not train: {error}"})
            return

        self._send_json(
            200,
            {"message": f"Trained on {len(samples)} drawing(s), weights saved."},
        )

    def _predict(self, payload: dict[str, Any]) -> None:
        if "image" not in payload:
            self._send_json(400, {"error": "Expected an image."})
            return

        try:
            digit = nn.predict(payload["image"])
        except (TypeError, ValueError) as error:
            self._send_json(400, {"error": f"Could not read the drawing: {error}"})
            return

        self._send_json(200, {"result": digit})

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)

        if length == 0:
            raise ValueError("Expected a JSON body.")

        content = self.rfile.read(length)

        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Malformed JSON: {error}") from error

        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")

        return payload

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        # ocr.js parses JSON on every response, errors included.
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(status, "application/json", body)

    def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = HTTPServer((HOST, PORT), OCRServer)

    print(f"Open http://{HOST}:{PORT} in your browser. Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
