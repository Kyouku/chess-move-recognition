from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

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


# Automatically create important directories if they don't exist.
def _ensure_core_directories() -> None:
    to_create = [
        MODELS_DIR,
        DATA_DIR,
        VIDEOS_DIR,
        CALIBRATION_DIR,
        GAME_MOVES_DIR,
    ]
    for p in to_create:
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Directory creation is best-effort; proceed even if it fails.
            pass


# Run on import so that using config guarantees directories exist.
_ensure_core_directories()

# ---------------------------------------------------------------------------
# Input source
# ---------------------------------------------------------------------------

# The live app automatically probes the actual source dimensions.
# FRAME_WIDTH and FRAME_HEIGHT are only used as fallbacks if probing fails.
CAMERA_INDEX: int = 1
FRAME_WIDTH: int = 3840  # fallback only
FRAME_HEIGHT: int = 2160  # fallback only

# Actual source size detected at runtime
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


# ---------------------------------------------------------------------------
# Board geometry
# ---------------------------------------------------------------------------

BOARD_SIZE_PX: int = 640
BOARD_SQUARES: int = 8
BOARD_MARGIN_SQUARES: float = 1.7

# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------

# If True, use VIDEO_PATH as input, otherwise use CAMERA_INDEX.
USE_VIDEO_FILE: bool = True
VIDEO_PATH: Path = VIDEOS_DIR / "game3.mp4"

# ---------------------------------------------------------------------------
# Calibration settings
# ---------------------------------------------------------------------------

CALIBRATION_MAX_FRAMES: int = 500
AUTO_MIN_BOARD_AREA_RATIO: float = 0.08
USE_SAVED_HOMOGRAPHY: bool = False
HOMOGRAPHY_PATH: Path = CALIBRATION_DIR / "homography.npy"
CALIBRATION_TARGET_LONG_EDGE: int = BOARD_SIZE_PX

# ---------------------------------------------------------------------------
# Detection and model settings (stage 2)
# ---------------------------------------------------------------------------

DETECTION_WORKERS: int = 1

YOLO_PIECE_WEIGHTS: Path = MODELS_DIR / f"yolo11s_best_{BOARD_SIZE_PX}.onnx"
YOLO_PIECE_IMGSZ: int = BOARD_SIZE_PX
YOLO_PIECE_CONF: float = 0.15
MIN_IOU: float = 0.15

# ---------------------------------------------------------------------------
# Logging paths
# ---------------------------------------------------------------------------

# Logging for multistage move tracker
MOVES_LOG_PATH: Path = GAME_MOVES_DIR / "detected_moves.log"
GAME_MOVES_TXT_PATH: Path = GAME_MOVES_DIR / "game_moves.txt"

# Logging for single frame FEN baseline
FEN_LOG_PATH: Path = GAME_MOVES_DIR / "baseline_fens.txt"

# ---------------------------------------------------------------------------
# Baseline FEN stability
# ---------------------------------------------------------------------------

START_MIN_CONFIRM_FRAMES: int = 4
FEN_MIN_STABLE_FRAMES: int = 10

# ---------------------------------------------------------------------------
# UI and OpenCV settings
# ---------------------------------------------------------------------------

DISPLAY_WINDOW_NAME: str = "Board"
GUI_ENABLED: bool = True

# 0 lets OpenCV decide; values greater than 0 force a thread limit.
OPENCV_NUM_THREADS: int = 0

# ---------------------------------------------------------------------------
# Move tracker settings (multistage pipeline)
# ---------------------------------------------------------------------------

MOVE_FILTER_ALPHA: float = 0.5
MOVE_FILTER_THRESHOLD: float = 0.60
MOVE_MIN_CONFIRM_FRAMES: int = 3
MOVE_DEBUG: bool = True

# ---------------------------------------------------------------------------
# Queue sizes
# ---------------------------------------------------------------------------

FRAME_QUEUE_SIZE: int = 2
DETECTION_INPUT_QUEUE_SIZE: int = 1
DETECTION_OUTPUT_QUEUE_SIZE: int = 3
MOVE_IN_QUEUE_SIZE: int = 8
MOVE_OUT_QUEUE_SIZE: int = 64

# ---------------------------------------------------------------------------
# Default pipeline mode
# ---------------------------------------------------------------------------

# "multistage" for full temporal tracker
# "singleframe" for FEN only baseline
PIPELINE_MODE: str = "multistage"
