import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from ocr import OCRNeuralNetwork


HOST = "127.0.0.1"
PORT = 8000

# Create the neural network
nn = OCRNeuralNetwork(15)


class OCRServer(BaseHTTPRequestHandler):

    def do_POST(self):
        response_code = 200
        response = {}

        # Read the incoming JSON data
        content_length = int(self.headers["Content-Length"])
        content = self.rfile.read(content_length)

        # Convert bytes -> string -> Python dictionary
        payload = json.loads(content.decode("utf-8"))

        # TRAIN
        if payload.get("train"):
            nn.train(payload["trainArray"])
            nn.save()

        # PREDICT
        elif payload.get("predict"):
            try:
                response = {
                    "type": "test",
                    "result": nn.predict(payload["image"])
                }

            except Exception:
                response_code = 500

        # Neither train nor predict
        else:
            response_code = 400

        # Send response
        self.send_response(response_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Python 3 requires bytes when writing the response
        if response:
            self.wfile.write(
                json.dumps(response).encode("utf-8")
            )


# Start server
server = HTTPServer((HOST, PORT), OCRServer)

print(f"Server running at http://{HOST}:{PORT}")

server.serve_forever()