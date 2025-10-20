# ♟️ Automated Chess Move Recognition in Live Video

**Bachelor Thesis — Jannes Wolarz**
University of Europe for Applied Sciences (UE Innovation Hub, Potsdam)
Supervisor: Prof. Dr. Rand Kouatly

---

## 🎯 Project Overview

This project implements a **camera-based, multi-stage pipeline** to recognize chess moves from real-world video
recordings.
It combines **classical computer vision** (OpenCV) with **deep learning** (YOLOv8) and **rule-based inference** to
detect and validate moves on a standard chessboard without specialized hardware.

---

## 🧠 Research Context

While end-to-end networks can classify chess positions, they often fail under real conditions (e.g., perspective,
lighting, occlusion).
This system proposes a **modular, interpretable approach**:

1. **Board Rectification** → perspective correction via homography
2. **Piece Detection** → YOLOv8 model fine-tuned on chess datasets
3. **Move Inference** → rule-based validation between consecutive frames

The pipeline is evaluated against a **single-frame baseline** to measure improvements in reliability and accuracy.

---

## 📁 Repository Structure

```
chess-move-recognition/
├─ data/
│  ├─ raw/              # Original video/images
│  ├─ calibration/      # Camera intrinsics + homography
│  ├─ processed/        # Rectified and annotated images
│  └─ labels/           # Ground-truth JSON / PGN
├─ models/
│  └─ yolov8n.pt          # pretrained YOLOv8 model (auto-downloaded if missing)
├─ results/               # outputs (JSON predictions)
├─ src/
│  ├─ calibration/      # Camera calibration + rectification
│  ├─ detection/        # YOLOv8 piece detection
│  ├─ inference/        # Rule-based move inference
│  ├─ baseline/         # Single-frame detection baseline
│  ├─ utils/            # Visualization, logging, I/O
│  └─ evaluation/       # Metrics and comparison scripts
├─ notebooks/           # Jupyter exploration and training logs
├─ requirements.txt     # Runtime dependencies
├─ dev-requirements.txt # Developer tools (linting, docs, tests)
├─ .pre-commit-config.yaml
├─ .gitignore
└─ README.md
```

---

## ⚙️ Installation

### 1. Clone and enter the repository

```bash
git clone https://github.com/<yourusername>/chess-move-recognition.git
cd chess-move-recognition
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# or
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r dev-requirements.txt  # (optional)
pre-commit install                   # enable auto-format checks
```

---

## 📸 Data Capture & Calibration

### Step 1 — Record calibration images

Capture ≥20 stills of a printed chessboard pattern from multiple angles.

### Step 2 — Run calibration

```bash
python src/calibration/calibrate_camera.py
```

This will output:

- `data/calibration/intrinsics.npz` (K, D matrices)
- Mean reprojection error for quality control

### Step 3 — Record real gameplay

Save videos in `data/raw/`, preferably static camera with good lighting.

---

## 🧩 Running the Pipeline

### Run full video inference

```bash
python src/inference/pipeline.py --video data/raw/game1.mp4
```

### Evaluate results

```bash
python src/evaluation/evaluate.py --pred results/game1_pred.json --truth data/labels/game1_truth.json
```

---

## 🧠 Model Training (YOLOv8)

Train or fine-tune your detector:

```bash
yolo train data=data/chess.yaml model=yolov8n.pt epochs=50 imgsz=640
```

Pretrained models and datasets (
e.g., [ChessReD](https://github.com/chessreid/chessred), [Roboflow Universe](https://universe.roboflow.com)) can be
linked via YAML configs in `data/`.

---

## 📊 Evaluation Metrics

- Piece classification accuracy
- Board cell localization accuracy
- Move detection precision/recall
- Frame-to-frame consistency

---

## 🧪 Development Tools

| Tool                               | Purpose                               |
|------------------------------------|---------------------------------------|
| `black`, `isort`, `flake8`, `mypy` | Code quality & consistency            |
| `pytest`, `pytest-cov`             | Testing & coverage                    |
| `jupyterlab`, `nbconvert`          | Experimentation & reporting           |
| `sphinx`, `pdoc`                   | Documentation generation              |
| `pre-commit`                       | Automatic style checks before commits |

---

## 📜 License

This project is developed for academic purposes as part of the **Bachelor Thesis at UE Innovation Hub (2025)**.
All rights reserved by the author unless otherwise stated.

---

## 📧 Contact

**Jannes Wolarz**
Email: your.email@ue-germany.de
GitHub: [github.com/cassyio](https://github.com/cassyio)
Supervisor: Prof. Dr. Rand Kouatly
Program: Business Informatics & Data Science (B.Sc.)
