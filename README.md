# Chess Move Recognition

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12-green.svg)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-11s-orange.svg)](https://github.com/ultralytics/ultralytics)
[![python-chess](https://img.shields.io/badge/python--chess-1.10%2B-lightgrey.svg)](https://python-chess.readthedocs.io/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.23%2B-blue.svg)](https://onnxruntime.ai/)
[![Bachelor Thesis](https://img.shields.io/badge/B.Sc.%20Thesis-2026-informational.svg)](#citation)

> **Automated chess move recognition from monocular over‑the‑board video.** Built around a multi‑stage CV pipeline — board rectification, YOLO11 piece detection, EMA‑smoothed temporal tracking with rule‑constrained move inference — and benchmarked head‑to‑head against a single‑frame FEN baseline that uses the same detector.

This is the open‑source companion to my Bachelor's thesis _"Automated Chess Move Recognition in Live Video using a Multi‑Stage Pipeline: A Comparison with a Single‑Image‑Based Approach"_ (Jannes Wolarz, 2026, B.Sc. Business Informatics & Data Science).

---

## Why this project

Reconstructing a full chess game from an overhead phone camera sounds easy — until hands occlude pieces mid‑move, glare flips a bishop into a knight, and a single misclassified square corrupts the rest of the game. Commercial e‑boards solve it with hardware sensors, but they are out of reach for amateur clubs and small tournaments.

This project tackles the same problem with a **single overhead camera and commodity CPU hardware**. It explicitly compares two pipeline designs that share the same detector and rectification front end, so the contribution of temporal modelling can be isolated cleanly:

- A **single‑frame baseline** that commits FEN plateaus once placement is stable.
- A **multi‑stage pipeline** that adds square‑level memory, EMA smoothing, multi‑frame confirmation, and rule‑constrained move inference via `python‑chess`.

The result is a system that reconstructs over‑the‑board games at near real‑time on a laptop CPU, with a clear, measurable accuracy gain attributable to each component.

---

## Architecture

```
┌────────────────────────┐    ┌────────────────────────┐    ┌────────────────────────┐
│ Stage 1                │    │ Stage 2                │    │ Stage 3                │
│ Board Rectification    │───▶│ Piece Detection        │───▶│ Move Tracking          │
│                        │    │                        │    │                        │
│ • OpenCV calibration   │    │ • YOLO11s @ 640×640    │    │ • EMA per‑square filter│
│ • Cached homography    │    │ • ONNX Runtime (CPU)   │    │ • N‑frame confirmation │
│ • Bird's‑eye 640×640   │    │ • IoU square assignment│    │ • Legal‑move tracking  │
└────────────────────────┘    └────────────────────────┘    └────────────────────────┘
        frame                      detections                     SAN + FEN log
                       (threaded: capture → detect → track)
```

| Stage | Purpose | Key implementation |
| ----- | ------- | ------------------ |
| **1 — Rectification** | Locate the chessboard in a perspective frame, warp it to a 640 × 640 bird's‑eye view, cache the homography for stability across frames. | `src/stage1/board_rectifier.py`, `src/common/homography_cache.py` |
| **2 — Piece Detection** | Run a YOLO11s ONNX model on the rectified board, then assign detections to the 64 squares using IoU against per‑square ROIs. | `src/stage2/piece_detection.py` |
| **3 — Move Tracking** | EMA smoothing of per‑square occupancy and label evidence, multi‑frame confirmation, candidate‑set memory across frames, and `python‑chess` legality filtering before emitting SAN. | `src/stage3/move_tracking.py` |

A capture / detect / track threading layer (`src/pipeline/live_base.py`) keeps the whole system near real‑time on commodity hardware.

---

## Results

All numbers are macro‑averaged over the five‑game over‑the‑board evaluation corpus (Game 1–5: 87, 33, 46, 78, 28 plies). Hyper‑parameters are fixed a priori and identical across games — none of the evaluation games were used for tuning.

### End‑to‑end pipeline comparison

| Pipeline | FEN accuracy | MRR | Move coverage | Avg. move delay | Committed‑step errors |
| -------- | -----------: | --: | ------------: | --------------: | --------------------: |
| Single‑frame baseline (FEN plateaus) | **8.9 %** | **40.4 %** | **64.5 %** | 101 frames (≈1.69 s) | 144 / 147 (98 %) |
| **Multi‑stage pipeline** | **58.4 %** | **58.4 %** | 59.3 % | **32 frames (≈0.53 s)** | **3 / 147 (2 %)** |

> The multi‑stage tracker raises FEN accuracy by **+49.5 pp** and cuts move‑detection delay to roughly a third, in exchange for a **5.2 pp** drop in coverage — a deliberate trade‑off: ambiguous moves are suppressed rather than guessed, protecting the integrity of the internal game state.

### Detector (YOLO11s, validation split)

| Input | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
| ----- | ------: | -----------: | --------: | -----: |
| 416   | 0.9944  | 0.9694       | 0.9965    | 0.9965 |
| 512   | 0.9948  | 0.9761       | 0.9977    | 0.9989 |
| **640** | **0.9948** | **0.9743** | **0.9961** | **0.9984** |

### Detector throughput (ONNX Runtime, Intel Core i7‑13800H CPU, 50 timed runs)

| Model | Input | Latency | FPS |
| ----- | ----: | ------: | --: |
| **YOLO11s** | 416 | 29.5 ms | 33.9 |
| **YOLO11s** | 512 | 43.6 ms | 23.0 |
| **YOLO11s** | **640** | **67.0 ms** | **14.9** |
| YOLO11m | 640 | 240.7 ms | 4.2 |
| YOLO11l | 640 | 209.0 ms | 4.8 |

YOLO11s @ 640 × 640 is the default — slightly slower than 416, but visibly more stable under reflections and occlusions while still maintaining a usable pipeline frame rate on CPU.

### Ablation — what actually drives the accuracy gains

| Variant | FEN acc. | MRR | Coverage | Move delay |
| ------- | -------: | --: | -------: | ---------: |
| Single‑frame baseline | 8.9 %  | 40.4 % | 64.5 % | 101 |
| **Multi‑stage (full)** | **58.4 %** | **58.4 %** | 59.3 % | **32** |
| − legality filtering | 58.4 % | 58.4 % | 59.3 % | 32 |
| − piece labels (occupancy only) | 46.7 % | 46.7 % | 46.9 % | 32 |
| − EMA smoothing | 58.4 % | 58.4 % | 59.3 % | 23 |
| − N‑frame confirmation | 45.2 % | 45.2 % | 46.3 % | 9 |
| − candidate‑set memory | 58.6 % | 58.6 % | 59.3 % | 32 |

**Take‑away:** multi‑frame confirmation and label‑conditioned disambiguation are the dominant contributors to correctness on this corpus; EMA primarily trades responsiveness against stability. Sensitivity sweeps for `MOVE_FILTER_ALPHA` and `MOVE_MIN_CONFIRM_FRAMES` are included in the comparison scripts.

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
| OTB game videos & ground‑truth PGN | [Google Drive](https://drive.google.com/drive/folders/10ZQ735rxp5QbD4C05zt-p7xGmJumA5yv?usp=sharing) | `data/videos/`, `data/gt/` |
| Pre‑computed detection caches | bundled in repo | `data/detections/*.pkl` |
| Rectified training dataset | [Roboflow](https://app.roboflow.com/chess-vision-ufw9m/automated-chess-move-recognition-lefwt/17) | optional |

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

The `.pkl` detection caches in `data/detections/` are pre‑computed, so the full evaluation can be replayed purely on CPU:

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

# Hyper‑parameter sensitivity
python -m src.pipeline.comparison.run_sensitivity \
    --manifest data/eval_manifest.json \
    --out data/results/sensitivity.csv
```

---

## Configuration

Everything lives in [`src/config.py`](src/config.py). The most‑used knobs:

| Setting | Default | Purpose |
| ------- | ------- | ------- |
| `PIPELINE_MODE` | `"multistage"` | `multistage` (temporal tracker) or `singleframe` (baseline) |
| `USE_VIDEO_FILE` | `True` | `False` switches to a live camera at `CAMERA_INDEX` |
| `VIDEO_PATH` | `data/videos/game1.mp4` | Input video |
| `YOLO_PIECE_WEIGHTS` | `models/yolo11s_best_640.onnx` | Detection model |
| `YOLO_PIECE_CONF` | `0.15` | Detection confidence threshold |
| `MOVE_FILTER_ALPHA` | `0.6` | EMA factor for per‑square stability (controls responsiveness) |
| `MOVE_MIN_CONFIRM_FRAMES` | `3` | Frames a move must persist before emission (controls reliability) |
| `FEN_MIN_STABLE_FRAMES` | `10` | Single‑frame baseline plateau length |
| `LOG_LEVEL` | `logging.DEBUG` | Logging verbosity |

---

## Project structure

```
chess-move-recognition/
├── src/
│   ├── config.py                # Central configuration (all hyper-parameters)
│   ├── app_gui.py               # Headless-safe OpenCV window helpers
│   ├── common/                  # Shared: chess I/O, logging, homography cache, types
│   ├── stage1/                  # Stage 1 — board localisation & rectification
│   ├── stage2/                  # Stage 2 — YOLO piece detection + per-square assignment
│   ├── stage3/                  # Stage 3 — temporal move tracking (FEN + SAN)
│   └── pipeline/
│       ├── live_base.py         # Threaded capture / detect / track scaffolding
│       ├── live_entrypoint.py   # Pipeline-mode dispatcher
│       ├── multistage/          # Multi-stage entry point
│       ├── singleframe/         # Baseline entry point
│       └── comparison/          # Offline evaluation, ablation, failure analysis
├── models/
│   └── scripts/                 # Training, dataset generation, model comparison
├── data/
│   ├── detections/              # Pre-computed YOLO outputs (.pkl) for fast replay
│   ├── eval_manifest.json       # Evaluation manifest (5 games)
│   └── ...                      # Videos, ground truth, results (gitignored)
├── failure_test/                # Captured edge-case games for regression testing
├── requirements.txt
├── pyproject.toml
├── CITATION.cff
└── LICENSE
```

---

## Tech stack

- **Python 3.10+**
- **OpenCV (contrib)** — chessboard corner detection, homography, perspective warping
- **Ultralytics YOLO11** trained on a custom rectified‑image dataset, exported to **ONNX** for portable CPU inference via **onnxruntime**
- **PyTorch** for training and the optional `.pt` runtime
- **python‑chess** for FEN/SAN handling and legal‑move generation
- **NumPy** throughout

Tested on Windows 11, Intel Core i7‑13800H. The detector trains on GPU but the full pipeline runs on commodity CPU.

---

## Hypotheses tested

The thesis evaluates three concrete hypotheses; all three are supported by the empirical results above.

| # | Claim | Result |
| - | ----- | ------ |
| **H1** | Given the same detector, the multi‑stage pipeline outperforms the single‑frame baseline on board‑state correctness (FEN accuracy). | ✅ 8.9 % → 58.4 % |
| **H2** | The multi‑stage pipeline yields better sequence‑level move correctness (MRR) and reduces illogical board transitions, despite a small drop in coverage. | ✅ 40.4 % → 58.4 % MRR; 144 → 3 committed errors |
| **H3** | The multi‑stage pipeline remains within the compute budget of consumer hardware while delivering these gains. | ✅ ≈14.9 detector FPS, ≈0.53 s mean move delay on i7‑13800H |

---

## Limitations

The published evaluation is in‑domain: same physical board, same camera, controlled club lighting, fixed tripod. External validity to different boards, piece sets, lighting setups, or moving cameras is not measured. The temporal layer is intentionally lightweight (no recurrent or transformer model) and the tracker does not currently support rollback after an incorrect commit — directions explicitly flagged for future work in the thesis.

---

## Roadmap

- [ ] Publish trained ONNX weights as a GitHub Release asset
- [ ] Unit tests for `move_tracking` state transitions
- [ ] CI workflow (lint + smoke test) via GitHub Actions
- [ ] Lightweight recurrent / transformer head over square‑level features
- [ ] Online homography refinement to relax the fixed‑camera assumption
- [ ] Confidence‑aware rollback / re‑synchronisation after incorrect commits

---

## Citation

If you use this code, the datasets, or the evaluation protocol in academic work, please cite the thesis:

```bibtex
@thesis{wolarz2026chess,
  author       = {Jannes Wolarz},
  title        = {Automated Chess Move Recognition in Live Video using a Multi-Stage Pipeline:
                  A Comparison with a Single-Image-Based Approach},
  type         = {Bachelor's Thesis},
  school       = {University of Europe for Applied Sciences},
  year         = {2026},
  month        = {January}
}
```

A machine‑readable [`CITATION.cff`](CITATION.cff) is provided so GitHub renders a "Cite this repository" widget on the project page.

---

## License

[MIT](LICENSE) © 2025 Jannes Wolarz

---

## Acknowledgements

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) for the detection backbone
- [python‑chess](https://github.com/niklasf/python-chess) for legality validation
- Training data hosted on [Roboflow](https://app.roboflow.com/chess-vision-ufw9m/automated-chess-move-recognition-lefwt)
- Supervised by Prof. Dr. Rand Kouatly and Prof. Dr. Daniel F. Heuermann
