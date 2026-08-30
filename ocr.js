const CANVAS_WIDTH = 200;
const TRANSLATED_WIDTH = 20;
const PIXEL_WIDTH = CANVAS_WIDTH / TRANSLATED_WIDTH;

const canvas = document.querySelector("#canvas");
const ctx = canvas.getContext("2d");
const digitInput = document.querySelector("#digit");
const statusElement = document.querySelector("#status");
const trainButton = document.querySelector("#trainButton");
const testButton = document.querySelector("#testButton");
const resetButton = document.querySelector("#resetButton");

let isDrawing = false;
let data = new Array(TRANSLATED_WIDTH * TRANSLATED_WIDTH).fill(0);

function setStatus(message) {
  statusElement.textContent = message;
}

function drawGrid() {
  ctx.strokeStyle = "#1f4f78";
  ctx.lineWidth = 0.5;

  for (let x = PIXEL_WIDTH; x < CANVAS_WIDTH; x += PIXEL_WIDTH) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, CANVAS_WIDTH);
    ctx.stroke();
  }

  for (let y = PIXEL_WIDTH; y < CANVAS_WIDTH; y += PIXEL_WIDTH) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(CANVAS_WIDTH, y);
    ctx.stroke();
  }
}

function resetCanvas() {
  data = new Array(TRANSLATED_WIDTH * TRANSLATED_WIDTH).fill(0);
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_WIDTH);
  drawGrid();
  setStatus("Ready.");
}

function canvasCoordinates(event) {
  const rect = canvas.getBoundingClientRect();

  return {
    x: (event.clientX - rect.left) * (canvas.width / rect.width),
    y: (event.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function fillSquare(event) {
  const { x, y } = canvasCoordinates(event);
  const xPixel = Math.floor(x / PIXEL_WIDTH);
  const yPixel = Math.floor(y / PIXEL_WIDTH);

  if (
    xPixel < 0 ||
    xPixel >= TRANSLATED_WIDTH ||
    yPixel < 0 ||
    yPixel >= TRANSLATED_WIDTH
  ) {
    return;
  }

  // Row-major order: index = row * width + column.
  const index = yPixel * TRANSLATED_WIDTH + xPixel;
  data[index] = 1;

  ctx.fillStyle = "white";
  ctx.fillRect(
    xPixel * PIXEL_WIDTH,
    yPixel * PIXEL_WIDTH,
    PIXEL_WIDTH,
    PIXEL_WIDTH,
  );
}

async function sendJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const result = await response.json();

  if (!response.ok) {
    throw new Error(result.error ?? `HTTP ${response.status}`);
  }

  return result;
}

async function train() {
  const label = Number.parseInt(digitInput.value, 10);

  if (!Number.isInteger(label) || label < 0 || label > 9) {
    setStatus("Enter the correct digit from 0 through 9 first.");
    return;
  }

  if (!data.includes(1)) {
    setStatus("Draw a digit before training.");
    return;
  }

  try {
    setStatus("Training...");
    const result = await sendJson("/api/train", {
      trainArray: [{ y0: data, label }],
    });
    setStatus(result.message);
    resetCanvas();
    digitInput.value = "";
  } catch (error) {
    setStatus(`Training failed: ${error.message}`);
  }
}

async function test() {
  if (!data.includes(1)) {
    setStatus("Draw a digit before testing.");
    return;
  }

  try {
    setStatus("Predicting...");
    const result = await sendJson("/api/predict", { image: data });
    setStatus(`Prediction: ${result.result}`);
  } catch (error) {
    setStatus(`Prediction failed: ${error.message}`);
  }
}

canvas.addEventListener("pointerdown", (event) => {
  isDrawing = true;
  canvas.setPointerCapture(event.pointerId);
  fillSquare(event);
});

canvas.addEventListener("pointermove", (event) => {
  if (isDrawing) {
    fillSquare(event);
  }
});

canvas.addEventListener("pointerup", () => {
  isDrawing = false;
});

canvas.addEventListener("pointercancel", () => {
  isDrawing = false;
});

trainButton.addEventListener("click", train);
testButton.addEventListener("click", test);
resetButton.addEventListener("click", resetCanvas);

resetCanvas();
