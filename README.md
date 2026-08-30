# Optical Character Recognition

Optical Character Recognition (OCR for short) is a piece of software that can take images of
handwritten characters as input and interpret them into machine readable text.

> **This is a tutorial project.** Everything here follows the
> ["Optical Character Recognition (OCR)"](https://aosabook.org/en/500L/optical-character-recognition-ocr.html)
> chapter of *500 Lines or Less* by Marina Samuel - the network design, the training loop, the
> client/server split, and the hidden-node experiment all come from that chapter. I typed it out
> to learn how it works, and updated it to run on modern Python. See
> [Credits](#credits) and [Licences](#licences).

This project is an implementation of the OCR system described in that chapter. It recognises
handwritten **digits (0-9)** drawn in the browser, using a three-layer artificial neural network
written without any machine learning framework - just `numpy` and the backpropagation math done
by hand.

The design is the chapter's throughout. The code is not a copy of it, being a Python 3 port using
`numpy` 2 - the original's `BaseHTTPServer`, `xrange`, `print` statements and `numpy.mat` matrix
code are all gone. Where this version departs from the chapter, it says so.

## Artificial Neural Networks

*This section is adapted from the chapter's prose. Copyright (c) Marina Samuel,
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/), via [aosabook.org](https://aosabook.org).*

ANN for short, is a structure consisting of interconnected nodes that communicate with one
another. The structure and its functionality are inspired by neural networks found in a
biological brain.

Hebbian Theory explains how these networks can learn to identify patterns by physically
altering their structure and link strengths.

Similarly, a standard ANN has connections between nodes that have a weight which is updated as
the network learns. The nodes labelled "+1" are called biases. The leftmost blue column of nodes
are input nodes, the middle column contains hidden nodes, and the rightmost column contains
output nodes. There can be many columns of hidden nodes, known as hidden layers.

![Diagram of a three-layer artificial neural network: two input nodes and a bias feeding two hidden nodes, which feed a single output node](image.png)

*Figure 15.1, "An Artificial Neural Network", reproduced from
[the chapter](https://aosabook.org/en/500L/optical-character-recognition-ocr.html).
Copyright (c) Marina Samuel, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).*

## Network design

| Property | Value | Why |
|---|---|---|
| Input nodes | **400** | A 20x20 pixel grid, flattened. Each value is `1` (drawn) or `0` (blank). |
| Hidden nodes | **15** | Chosen empirically - see [Choosing the hidden layer size](#choosing-the-hidden-layer-size). |
| Output nodes | **10** | One per digit, 0-9. The highest-valued output wins. |
| Activation | **Sigmoid**, `1 / (1 + e^-z)` | Saturates toward 0 or 1, and is differentiable - a requirement for backpropagation. |
| Learning rate | **0.1** | Deliberately small: favours accuracy over convergence speed. |
| Weight init | Random in **[-0.06, 0.06]** | Small non-zero values, so nodes don't learn identical features. |

Weights live in two matrices - `theta1` (hidden x 400) and `theta2` (10 x hidden) - plus a bias
vector for each layer. A bias adds a constant to the node's linear input (`y = f(wx + b)`),
shifting the y-intercept and giving the node more flexibility.

Training is the standard four-step backpropagation loop:

1. Initialise weights to small random values.
2. **Forward propagate** - multiply by `theta1`, add the bias, apply sigmoid to get the hidden
   output `y1`; repeat with `theta2` to get the output layer `y2`.
3. **Backpropagate** the error - compare `y2` against the expected label, then push the error
   backward through the hidden layer using the sigmoid derivative.
4. **Update weights** - `weights += learning_rate * (error_matrix * previous_layer_output)`.

Each training request serialises the weights to `nn.json`, which is read back on startup, so
training progress survives a server restart. Delete the file to begin again from random weights.

## Architecture

The system is split across a browser client and a Python server:

```
  Browser                            Server (127.0.0.1:8000)
+------------------------+         +---------------------------------+
| ocr.html + ocr.css     |  GET /  | server.py                       |
|  200x200 canvas        | <------ |  OCRServer.do_GET  -> the page  |
|  Train / Test / Reset  |         |                                 |
|                        |  POST   |  OCRServer.do_POST              |
| ocr.js                 | ------> |   |- /api/train   -> nn.train() |
|  canvas -> 20x20       |  JSON   |   `- /api/predict -> nn.predict |
|  -> 400-item array     | <------ |                                 |
+------------------------+         | ocr.py                          |
                                   |  OCRNeuralNetwork (+ nn.json)   |
                                   +---------------------------------+
```

The server serves the page as well as the API, because `ocr.js` fetches relative URLs. Opening
`ocr.html` straight from disk does not work: the browser would resolve `/api/train` against
`file://` and the request would never reach Python.

| File | Role |
|---|---|
| `ocr.html` | The UI: a 200x200 `<canvas>`, a digit input box, Train / Test / Reset buttons, and a status line. |
| `ocr.css` | Styling for the page. |
| `ocr.js` | Client logic. Draws on the canvas, translates it to a 20x20 grid, and calls the API. |
| `server.py` | Serves the page and the two JSON endpoints, holding the one network instance. |
| `ocr.py` | `OCRNeuralNetwork` - feed forward, backpropagation, prediction, and JSON weight persistence. |
| `neural_network_design.py` | One-off experiment to pick the hidden node count. Not part of the running system. |
| `requirements.txt` | Pins `numpy`, the only dependency. |

### How drawing works

The canvas is 200x200 pixels, but the network only sees 20x20. `ocr.js` treats each network
pixel as a 10x10 block on screen (`PIXEL_WIDTH = CANVAS_WIDTH / TRANSLATED_WIDTH`), so strokes
are chunky enough to be visible while still reducing to 400 inputs. Pointer events map cursor
coordinates to a grid index in row-major order (`index = row * 20 + column`) and call
`fillSquare()`, which sets that cell to `1`.

Each click of **Train** posts the single digit currently on the canvas, then clears it. Because
the input grid is only 20x20, a thin stroke lights up very few of the 400 cells - drawing thick
and large gives the network more to work with.

## API

Two endpoints accept POSTed JSON. The operation is selected by the URL path.

**`POST /api/train`** - `y0` is the 400-item image, `label` is the digit it depicts:

```json
{ "trainArray": [ { "y0": [0, 1, 0], "label": 7 } ] }
```

```json
{ "message": "Trained on 1 drawing(s), weights saved." }
```

**`POST /api/predict`**:

```json
{ "image": [0, 1, 0] }
```

```json
{ "result": 7 }
```

Status codes: `200` on success, `400` for a malformed body, an empty `trainArray`, or an image
that is not 400 values, and `404` for an unknown path. Every response carries a JSON body,
errors included, since the client parses JSON unconditionally:

```json
{ "error": "Could not read the drawing: Expected 400 input values, received 3." }
```

## Choosing the hidden layer size

Too few hidden nodes and the network can't represent the problem; too many and you pay for
computation you don't need. `neural_network_design.py` settles it experimentally rather than by
guessing:

1. Generate 60 samples of each digit and split them 75/25 into training and validation sets.
2. For each hidden node count from 5 to 45, in steps of 5, train a fresh network.
3. Measure accuracy - the fraction of correctly classified validation samples - averaged over
   100 trials, since random weight initialisation makes any single run noisy.

The chapter trains one network and then loops its `test()` 100 times over that same network, so
every iteration classifies the same images with the same weights and returns an identical
number. Since the stated reason for averaging is that "each time an ANN is trained, its weights
may be slightly different", each trial here trains a fresh network with a new seed, which is
what makes the average mean anything.

Results from a run of `neural_network_design.py`:

| Hidden nodes | Accuracy |
|---|---|
| 5 | 0.5741 |
| 10 | 0.7039 |
| **15** | **0.7249** |
| 20 | 0.7266 |
| 30 | 0.7427 |
| 45 | 0.7413 |

15 is the sweet spot: a 13-point jump over 5 nodes, a couple more points on top of 10, then
diminishing returns that flatten out around 0.74. Where several counts perform similarly, prefer
the smallest - fewer computations for the same result. The absolute numbers sit below the
chapter's (which reports 0.8808 at 15 nodes) because these digits are generated rather than
handwritten, and are deliberately mangled; the shape of the curve is what the decision rests on.

## Training data

There is none to download. The chapter ships 5000 pre-labelled samples in `data.csv` and
`dataLabels.csv` and trains on them at startup; this version has no dataset files and no
dependency on them. The server starts from small random weights, and the only training data is
what you draw in the browser, so early predictions are essentially guesses.

Every **Train** click is one sample at a learning rate of 0.1 with a single pass over it, so one
drawing barely moves the weights. Expect to need a few dozen examples per digit. Cycle through
all ten rather than submitting twenty 0s in a row - training on a run of one label pushes the
network toward predicting that label for everything.

`neural_network_design.py` needs labelled data to measure accuracy, so it generates its own:
each digit is drawn as a subset of seven bars, the way a calculator display does, then each copy
is mangled by dropping the occasional bar, sliding the digit up to two pixels, and flipping
about 6% of its pixels. The dropped bars matter most, since they are what makes an 8 resemble a
9. Without that overlap every hidden node count scores 1.0 and the sweep says nothing.

## Running it

Set up a virtual environment and install the one dependency. On Windows:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The `py` launcher is only installed alongside a machine-wide Python. If `py` is not recognised,
point at the interpreter directly - a per-user install lands in
`%LOCALAPPDATA%\Programs\Python\Python314\python.exe`.

On macOS/Linux:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Then start the server:

```powershell
.\.venv\Scripts\python.exe server.py
```

Open **http://127.0.0.1:8000** in a browser - not the `ocr.html` file itself, or the API calls
will fail. Draw a digit, type what it is, and hit **Train** - or hit **Test** to see what the
network thinks it is. `Ctrl+C` stops the server.

To re-run the hidden node experiment (roughly two and a half minutes, since it trains 900
networks):

```powershell
.\.venv\Scripts\python.exe neural_network_design.py
```

Requires `numpy` only. The train/test split is done with `numpy`'s own random number
generator, so `scikit-learn` is not needed.

## Project status

**Working end to end** on Python 3.14 / numpy 2.5.2. Start the server, draw digits in the
browser, and train and test the network.

- [x] Implement `ocr.py`, `server.py` and `neural_network_design.py`
- [x] Port to **Python 3**. The reference implementation is Python 2: `BaseHTTPServer` is now
      `http.server`, `print` is a function, and `xrange` is `range`. Writing to `wfile` needs
      bytes rather than a `str`. `numpy.mat` was also removed in numpy 2.0, so the reference
      implementation's matrix code no longer runs.
- [x] Rewrite the client - `fetch` instead of synchronous `XMLHttpRequest`, pointer events
      instead of mouse events, and a status line instead of `alert()`
- [x] Serve the page from `server.py`, so the client's relative API calls resolve
- [x] Drop the dataset dependency - no `data.csv` or `dataLabels.csv` required
- [x] Add `ocr.css`

Possible next steps:

- [ ] Batch several drawings per request, as the chapter's `BATCH_SIZE` does
- [ ] Show the ten output activations, not just the winning digit, to see how close the call was
- [ ] Undo the last training sample, for when you mislabel one

## Credits

All of this comes from one tutorial: the chapter
["Optical Character Recognition (OCR)"](https://aosabook.org/en/500L/optical-character-recognition-ocr.html)
by **Marina Samuel**, in [*500 Lines or Less*](https://aosabook.org/en/500L/), part of
*The Architecture of Open Source Applications*. The reference implementation lives at
[aosabook/500lines/ocr](https://github.com/aosabook/500lines/tree/master/ocr).

Credit for the design belongs there. What is mine is the Python 3 / numpy 2 port, the rewritten
browser client, and the generated dataset that replaces the chapter's `data.csv`.

## Licences

Two licences apply, because *500 Lines or Less* releases its software and its written material
under different terms. [NOTICE](NOTICE) spells this out in full.

**Code** - MIT, Copyright (c) 2026 Jamie Gilbert. See [LICENSE](LICENSE). The chapter's own
example programs are MIT too, Copyright (c) Marina Samuel, and portions of this implementation
are derived from them.

**The diagram and one section of this README** - [CC BY
3.0](https://creativecommons.org/licenses/by/3.0/), Copyright (c) Marina Samuel, courtesy of
[aosabook.org](https://aosabook.org). `image.png` is Figure 15.1, "An Artificial Neural Network",
reproduced from the chapter, and the [Artificial Neural Networks](#artificial-neural-networks)
section is adapted from its prose. These are written material rather than software, so they are
not covered by the MIT licence above.
