# Chess Move Recognition

This project provides a system for recognizing chess moves from video or live camera feeds. It uses a multi-stage
approach combining computer vision and temporal tracking to robustly detect and record moves in Standard Algebraic
Notation (SAN).

## Project Overview

The system operates in three main stages:

1. **Stage 1: Board Detection & Rectification** – Locates the chessboard and transforms it into a bird's-eye view.
2. **Stage 2: Piece Detection** – Uses YOLO to detect pieces on each square of the rectified board.
3. **Stage 3: Move Tracking** – Filters detections over time to identify moves and maintain the board state.

## Getting Started

### Prerequisites

- Python 3.10+
- OpenCV (with contrib modules recommended)
- PyTorch & Ultralytics (for YOLO piece detection)
- python-chess (for move validation and FEN handling)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd chess-move-recognition
   ```

2. Download the required data:
    - **Videos & Ground Truth**: Download the game videos and ground truth files
      from [Google Drive](https://drive.google.com/drive/folders/10ZQ735rxp5QbD4C05zt-p7xGmJumA5yv?usp=sharing). Place
      the videos in `data/videos/` and the ground truth files in `data/gt/`.
    - **Dataset**: The training dataset is available
      on [Roboflow](https://app.roboflow.com/chess-vision-ufw9m/automated-chess-move-recognition-lefwt/17).
    - **Pre-computed Detections**: Detection logs (`.pkl`) are already included in the repository (`data/detections/`)
      to enable fast testing of the evaluation scripts without a GPU.

3. Install the required dependencies:
   ```bash
   pip install -r src/requirements.txt
   ```

4. Ensure you have the required YOLO models in the `models/` directory. By default, the system looks for
   `yolo11s_best_640.onnx`.

### Running the Live Pipeline

You can run the live move recognition pipeline using the following command:

```bash
python -m src.pipeline.multistage.live_multistage_main
```

By default, this will use the configuration in `src/config.py`. You can also run the single-frame baseline:

```bash
python -m src.pipeline.singleframe.live_singleframe_main
```

## Configuration

The project is configured via `src/config.py`. Key settings include:

- `USE_VIDEO_FILE`: Set to `True` to use a video file as input, or `False` for a live camera.
- `VIDEO_PATH`: Path to the input video file (expected in `data/videos/`).
- `CAMERA_INDEX`: Index of the camera to use if `USE_VIDEO_FILE` is `False`.
- `YOLO_PIECE_WEIGHTS`: Path to the YOLO model weights (ONNX format recommended).
- `PIPELINE_MODE`: Choose between `"multistage"` (temporal tracker) or `"singleframe"` (FEN baseline).

## Evaluation and Development

The project includes tools for offline evaluation and comparison:

### Ground Truth Preparation

Use the interactive tool to annotate frame-to-move mappings:

```bash
python -m src.pipeline.comparison.prepare_ground_truth --video data/videos/game1.mp4 --pgn data/gt/game1.pgn --out data/gt/game1.json
```

### Pipeline Comparison

Compare the performance of the multistage pipeline against the single-frame baseline:

```bash
python -m src.pipeline.comparison.compare_pipelines --video data/videos/game1.mp4 --detections data/detections/game1_detections.pkl --gt data/gt/game1.json
```

### Ablation and Sensitivity Studies

- `python -m src.pipeline.comparison.run_ablation --manifest data/eval_manifest.json --out data/results/ablation.csv`:
  Runs ablation studies on pipeline components.
-
`python -m src.pipeline.comparison.run_sensitivity --manifest data/eval_manifest.json --out data/results/sensitivity.csv`:
Evaluates sensitivity to hyperparameters.

## Project Structure

- `data/`: Videos, ground truth files, results, and detection logs.
- `models/`: YOLO model weights and scripts for training and dataset generation.
- `src/`: Main source code.
    - `common/`: Shared utilities for chess logic, logging, and I/O.
    - `pipeline/`: Pipeline implementations (multistage, singleframe, and offline comparison).
    - `stage1/`: Board localization and rectification.
    - `stage2/`: YOLO-based piece detection.
    - `stage3/`: Move tracking and temporal filtering logic.
