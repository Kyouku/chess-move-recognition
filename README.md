# Chess Move Recognition

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12-green.svg)](https://opencv.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO-11-orange.svg)](https://github.com/ultralytics/ultralytics)
[![python-chess](https://img.shields.io/badge/python--chess-1.10%2B-lightgrey.svg)](https://python-chess.readthedocs.io/)
[![Code style: PEP 8](https://img.shields.io/badge/code%20style-PEP%208-blue.svg)](https://peps.python.org/pep-0008/)

> Real‑time chess move recognition from a single overhead camera. Detects the board, identifies pieces with YOLO, and converts what it sees into Standard Algebraic Notation (SAN) — frame by frame, move by move.

---

## Demo

> _A short demo GIF / screenshot belongs here._
> Drop an asset into `docs/assets/demo.gif` and reference it below.

```
docs/assets/demo.gif        ← annotated live capture
docs/assets/board_overlay.png ← stage 1 rectification result
docs/assets/piece_detect.png  ← stage 2 YOLO output
```

---

## Why this project

Recognising moves from a real chess game is a deceptively hard problem: the board appears in perspective, lighting changes between frames, hands occlude pieces mid‑move, and a single misclassified piece on a single frame can corrupt the whole game state. This project tackles all three issues with a **multi‑stage pipeline** where each stage solves one well‑defined problem and hands stable evidence to the next.

The result is a system that is robust enough to log a full game from a phone‑camera video, but also explicit enough to ablate every component and measure where the accuracy comes from.

---

## How it works

```
┌────────────────────────┐    ┌────────────────────────┐    ┌────────────────────────┐
│ Stage 1                │    │ Stage 2                │    │ Stage 3                │
│ Board Rectification    │───▶│ Piece Detection (YOLO) │───▶│ Move Tracking (SAN)    │
│                        │    │                        │    │                        │
│ • OpenCV calibration   │    │ • YOLO11 (s/m/l)       │    │ • Temporal filtering   │
│ • Homography caching   │    │ • ONNX runtime         │    │ • python-chess legality│
│ • Bird's-eye view      │    │ • Per-square assignment│    │ • Stable FEN/SAN log   │
└────────────────────────┘    └────────────────────────┘    └────────────────────────┘
        frame                      detections                      moves
```

| Stage | Purpose | Key implementation |
| ----- | ------- | ------------------ |
| **1 — Board Rectification** | Locate the chessboard in a perspective frame and warp it to a 640 × 640 bird's-eye view. Cache the homography across frames for stability. | `src/stage1/board_rectifier.py`, `src/common/homography_cache.py` |
| **2 — Piece Detection** | Run a YOLO11 model (custom-trained on the Roboflow dataset) on the rectified board, then assign detections to the 64 squares using IoU against square ROIs. | `src/stage2/piece_detection.py` |
| **3 — Move Tracking** | Apply a multi‑frame confirmation filter (`MOVE_FILTER_ALPHA`, `MOVE_MIN_CONFIRM_FRAMES`) and validate moves against the current chess position with `python-chess` to emit clean SAN. | `src/stage3/move_tracking.py` |

Two pipelines are built on top of these stages:

- **`multistage`** — full temporal tracker, robust to single‑frame errors.
- **`singleframe`** — naive FEN baseline used as ablation reference.

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/JannesWolarz/chess-move-recognition.git
cd chess-move-recognition
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the assets

| Asset | Source | Destination |
| ----- | ------ | ----------- |
| YOLO weights (`.onnx`) | See `models/scripts/training/` or release page | `models/` |
| Game videos & ground truth | [Google Drive](https://drive.google.com/drive/folders/10ZQ735rxp5QbD4C05zt-p7xGmJumA5yv?usp=sharing) | `data/videos/`, `data/gt/` |
| Pre-computed detections | bundled in repo | `data/detections/*.pkl` |
| Training dataset | [Roboflow](https://app.roboflow.com/chess-vision-ufw9m/automated-chess-move-recognition-lefwt/17) | optional |

By default the pipeline expects `models/yolo11s_best_640.onnx`. Override via `YOLO_PIECE_WEIGHTS` in `src/config.py`.

### 3. Run the live pipeline

```bash
# Multi‑stage (recommended)
python -m src.pipeline.multistage.live_multistage_main

# Single‑frame baseline
python -m src.pipeline.singleframe.live_singleframe_main
```

Switch between video file and live camera with `USE_VIDEO_FILE` in `src/config.py`.

### 4. Reproduce the evaluation (no GPU required)

The `.pkl` detection caches in `data/detections/` are pre-computed, so you can replay the full evaluation purely on CPU:

```bash
# Compare multistage vs single-frame on one game
python -m src.pipeline.comparison.compare_pipelines \
    --video data/videos/game1.mp4 \
    --detections data/detections/game1_detections.pkl \
    --gt data/gt/game1.json

# Full ablation across all games
python -m src.pipeline.comparison.run_ablation \
    --manifest data/eval_manifest.json \
    --out data/results/ablation.csv

# Hyper-parameter sensitivity
python -m src.pipeline.comparison.run_sensitivity \
    --manifest data/eval_manifest.json \
    --out data/results/sensitivity.csv
```

---

## Configuration

Everything lives in [`src/config.py`](src/config.py). The most-used knobs:

| Setting | Default | Purpose |
| ------- | ------- | ------- |
| `PIPELINE_MODE` | `"multistage"` | `multistage` (temporal tracker) or `singleframe` (baseline) |
| `USE_VIDEO_FILE` | `True` | `False` switches to a live camera at `CAMERA_INDEX` |
| `VIDEO_PATH` | `data/videos/game1.mp4` | Input video |
| `YOLO_PIECE_WEIGHTS` | `models/yolo11s_best_640.onnx` | Detection model |
| `YOLO_PIECE_CONF` | `0.15` | Detection confidence threshold |
| `MOVE_FILTER_ALPHA` | `0.6` | EMA factor for per-square stability |
| `MOVE_MIN_CONFIRM_FRAMES` | `3` | Frames a move must be stable before emission |
| `LOG_LEVEL` | `logging.DEBUG` | Logging verbosity |

---

## Results

> _Replace with the actual numbers from your latest `data/results/ablation.csv`._

| Pipeline | Move accuracy | False positives / game | Notes |
| -------- | ------------- | ---------------------- | ----- |
| `singleframe` baseline | … | … | FEN-only, no temporal info |
| `multistage` (default) | … | … | EMA filter + legality validation |
| `multistage` (no legality check) | … | … | Ablation: removes `python-chess` filter |

A full sensitivity sweep over `MOVE_FILTER_ALPHA`, `MOVE_MIN_CONFIRM_FRAMES`, and `YOLO_PIECE_CONF` is reproducible via `run_sensitivity.py`.

---

## Project structure

```
chess-move-recognition/
├── src/
│   ├── config.py                # Central configuration
│   ├── app_gui.py               # Headless-safe OpenCV window helpers
│   ├── common/                  # Shared: chess I/O, logging, homography cache, types
│   ├── stage1/                  # Board localisation & rectification
│   ├── stage2/                  # YOLO piece detection + per-square assignment
│   ├── stage3/                  # Temporal move tracking (FEN + SAN)
│   └── pipeline/
│       ├── live_base.py         # Threaded capture/detect/track scaffolding
│       ├── live_entrypoint.py   # Mode dispatcher
│       ├── multistage/          # Production pipeline entry point
│       ├── singleframe/         # Baseline pipeline entry point
│       └── comparison/          # Offline evaluation, ablation, failure analysis
├── models/
│   └── scripts/                 # Training, dataset generation, model comparison
├── data/
│   ├── detections/              # Pre-computed YOLO outputs (.pkl) for fast replay
│   ├── eval_manifest.json       # Evaluation manifest
│   └── ...                      # Videos, ground truth, results (gitignored)
├── failure_test/                # Captured edge-case games for regression testing
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## Tech stack

- **Python 3.10+**
- **OpenCV (contrib)** — board detection, homography, perspective warping
- **Ultralytics YOLO11** trained on a custom Roboflow dataset, exported to **ONNX** for portable inference via **onnxruntime**
- **PyTorch** for training and the optional `.pt` runtime
- **python-chess** for FEN/SAN handling and move legality
- **NumPy** throughout

---

## Roadmap

- [ ] Publish trained ONNX weights as a GitHub Release asset
- [ ] Add unit tests for `move_tracking` state transitions
- [ ] CI workflow (lint + smoke test) via GitHub Actions
- [ ] Web UI demo using the bundled `.pkl` detection caches

---

## License

[MIT](LICENSE) © Jannes Wolarz

---

## Acknowledgements

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) for the detection backbone
- [python-chess](https://github.com/niklasf/python-chess) for legality validation
- Training data hosted on [Roboflow](https://app.roboflow.com/chess-vision-ufw9m/automated-chess-move-recognition-lefwt)
