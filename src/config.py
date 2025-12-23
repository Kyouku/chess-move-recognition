from __future__ import annotations

import logging
from pathlib import Path

# ===========================================================================
# MAIN SETTINGS
# ===========================================================================

# "multistage" for full temporal tracker
# "singleframe" for FEN-only baseline
PIPELINE_MODE: str = "multistage"

# If True, use VIDEO_PATH as input, otherwise use CAMERA_INDEX.
USE_VIDEO_FILE: bool = True
VIDEO_PATH: Path = Path("data/videos/game1.mp4")
CAMERA_INDEX: int = 1

# Logging level (logging.DEBUG, logging.INFO, etc.)
LOG_LEVEL: int = logging.DEBUG

# ===========================================================================
# PATHS
# ===========================================================================

# Resolve important directories relative to this file (src/) and project root.
_SRC_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = _SRC_DIR.parent

# Data and model locations
MODELS_DIR: Path = PROJECT_ROOT / "models"
DATA_DIR: Path = PROJECT_ROOT / "data"
VIDEOS_DIR: Path = DATA_DIR / "videos"

# Persisted artifacts live under data/
CALIBRATION_DIR: Path = DATA_DIR / "saved_h_cache"
GAME_MOVES_DIR: Path = DATA_DIR / "detected_moves"

# Update VIDEO_PATH to be absolute if it was relative to root
if not VIDEO_PATH.is_absolute() and (PROJECT_ROOT / VIDEO_PATH).exists():
    VIDEO_PATH = PROJECT_ROOT / VIDEO_PATH

# ===========================================================================
# BOARD GEOMETRY
# ===========================================================================

BOARD_SIZE_PX: int = 640
BOARD_SQUARES: int = 8
BOARD_MARGIN_SQUARES: float = 1.7

# ===========================================================================
# STAGE 1: CALIBRATION & RECTIFICATION
# ===========================================================================

CALIBRATION_MAX_FRAMES: int = 500
CALIBRATION_TARGET_LONG_EDGE: int = BOARD_SIZE_PX
AUTO_MIN_BOARD_AREA_RATIO: float = 0.08

USE_SAVED_HOMOGRAPHY: bool = False
HOMOGRAPHY_PATH: Path = CALIBRATION_DIR / "homography.npy"

# ===========================================================================
# STAGE 2: PIECE DETECTION (YOLO)
# ===========================================================================

YOLO_PIECE_WEIGHTS: Path = MODELS_DIR / f"yolo11s_best_{BOARD_SIZE_PX}.onnx"
YOLO_PIECE_IMGSZ: int = BOARD_SIZE_PX
YOLO_PIECE_CONF: float = 0.15
MIN_IOU: float = 0.15

DETECTION_WORKERS: int = 1
DETECTION_PROGRESS_EVERY: int = 50

# ===========================================================================
# STAGE 3: MOVE TRACKING (MULTISTAGE)
# ===========================================================================

MOVE_FILTER_ALPHA: float = 0.6
MOVE_FILTER_THRESHOLD: float = 0.6
MOVE_MIN_CONFIRM_FRAMES: int = 3
MOVE_DEBUG: bool = True

# ===========================================================================
# BASELINE (SINGLE-FRAME)
# ===========================================================================

START_MIN_CONFIRM_FRAMES: int = 4
FEN_MIN_STABLE_FRAMES: int = 10

# ===========================================================================
# LOGGING & PERSISTENCE
# ===========================================================================

# Logging for multistage move tracker
MOVES_LOG_PATH: Path = GAME_MOVES_DIR / "detected_moves.log"
GAME_MOVES_TXT_PATH: Path = GAME_MOVES_DIR / "game_moves.txt"

# Logging for single frame FEN baseline
FEN_LOG_PATH: Path = GAME_MOVES_DIR / "baseline_fens.txt"

# Default FPS used if video file header is missing
VIDEO_FPS: float = 30.0

# ===========================================================================
# TECHNICAL / UI SETTINGS
# ===========================================================================

DISPLAY_WINDOW_NAME: str = "Chess Recognition"
GUI_ENABLED: bool = True

# 0 lets OpenCV decide; values greater than 0 force a thread limit.
OPENCV_NUM_THREADS: int = 0

# Fallbacks for camera probing
FRAME_WIDTH: int = 3840
FRAME_HEIGHT: int = 2160

# ===========================================================================
# QUEUE SIZES
# ===========================================================================

FRAME_QUEUE_SIZE: int = 2
DETECTION_INPUT_QUEUE_SIZE: int = 1
DETECTION_OUTPUT_QUEUE_SIZE: int = 3
MOVE_IN_QUEUE_SIZE: int = 8
MOVE_OUT_QUEUE_SIZE: int = 64

# ===========================================================================
# RUNTIME STATE (Internal use)
# ===========================================================================

ACTUAL_FRAME_WIDTH: int = FRAME_WIDTH
ACTUAL_FRAME_HEIGHT: int = FRAME_HEIGHT


def set_actual_frame_size(width: int, height: int) -> None:
    """
    Update the probed source size so downstream stages can use
    ACTUAL_FRAME_WIDTH and ACTUAL_FRAME_HEIGHT.
    """
    global ACTUAL_FRAME_WIDTH, ACTUAL_FRAME_HEIGHT
    ACTUAL_FRAME_WIDTH = int(width)
    ACTUAL_FRAME_HEIGHT = int(height)
