Chess Move Recognition (Live)

This repository provides a live chess move recognition application. It reads frames from a camera or video, rectifies
the chessboard, detects pieces with a YOLO model, and infers moves either with a temporal tracker (multi-stage) or a
simple single-frame baseline.

Overview

- BaseLivePipeline (src/pipeline/live_base.py)
    - Stage 1: Board rectifier (src/stage1/board_rectifier.py)
    - Stage 2: YOLO piece detector (src/stage2/piece_detection.py)
    - Manages threads/queues and optional GUI overlay (src/stage2/piece_overlay.py)
    - Exposes hooks for Stage 3 that derived pipelines implement
- MultiStagePipeline (src/pipeline/multistage/live_multistage_main.py)
    - Uses MoveTracker and MoveTrackerWorker for temporal filtering and tracking
- SingleFramePipeline (src/pipeline/singleframe/live_singleframe_main.py)
    - Uses SingleFrameBaseline for single-frame move inference
- Entrypoint (src/pipeline/live_entrypoint.py)
    - Chooses the pipeline via config.PIPELINE_MODE and starts the live run

Requirements

- Python 3.10+
- Packages: opencv-python, numpy, ultralytics, python-chess

Quick setup (Windows PowerShell)

1) Create and activate a virtual env
    - python -m venv .venv
    - .\.venv\Scripts\Activate.ps1
2) Install dependencies
    - pip install --upgrade pip
    - pip install opencv-python numpy ultralytics python-chess

Model weights

- Place your exported YOLO weights (e.g., ONNX) into the models/ folder.
- Adjust YOLO_PIECE_WEIGHTS in src/config.py if needed.
- BOARD_SIZE_PX and YOLO_PIECE_IMGSZ should match the model input size.

Configuration (src/config.py)

- Input source: CAMERA_INDEX, USE_VIDEO_FILE, VIDEO_PATH
- Frame size fallback: FRAME_WIDTH/FRAME_HEIGHT (actual size is probed at runtime)
- Board geometry: BOARD_SIZE_PX, BOARD_SQUARES, BOARD_MARGIN_SQUARES
- YOLO detector: YOLO_PIECE_WEIGHTS, YOLO_PIECE_IMGSZ, YOLO_PIECE_CONF, MIN_IOU
- Pipeline: PIPELINE_MODE = "multistage" or "singleframe"
- GUI toggle: GUI_ENABLED (True shows OpenCV windows)
- Logging and queue sizes

Run
From the project root:

- Set PYTHONPATH so that the src package is importable
    - PowerShell: $env:PYTHONPATH = "."
- Start the entrypoint
    - python -m src.pipeline.live_entrypoint

The entrypoint chooses the pipeline based on PIPELINE_MODE and the capture source based on USE_VIDEO_FILE/VIDEO_PATH or
CAMERA_INDEX. Press ESC to stop.

Choosing the pipeline

- Multi-stage (temporal tracking): set PIPELINE_MODE = "multistage"
- Single-frame baseline: set PIPELINE_MODE = "singleframe"

Outputs

- Live move logs and game summaries are written to data/detected_moves/ (paths configurable in src/config.py). These
  files are ignored by Git by default.

Troubleshooting

- Missing model weights: verify the file exists at YOLO_PIECE_WEIGHTS.
- No GUI window: set GUI_ENABLED=True and run in a desktop session; in headless mode OpenCV windows are disabled.
- Wrong camera: set CAMERA_INDEX or enable USE_VIDEO_FILE with VIDEO_PATH.
- Performance: set DETECTION_WORKERS (>1 spawns multiple YOLO models) and optionally OPENCV_NUM_THREADS if available.
