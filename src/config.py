from __future__ import annotations

from pathlib import Path

# Paths
# Resolve important directories relative to this file (src/) and project root.
_SRC_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = _SRC_DIR.parent
MODELS_DIR: Path = PROJECT_ROOT / "models"
DATA_DIR: Path = PROJECT_ROOT / "data"
VIDEOS_DIR: Path = DATA_DIR / "videos"
CALIBRATION_DIR: Path = _SRC_DIR / "pipeline" / "calibration"
GAME_MOVES_DIR: Path = _SRC_DIR / "pipeline" / "moves_log"

# Input source
# The live app automatically probes the actual source dimensions.
# FRAME_WIDTH/HEIGHT are only used as fallbacks if probing fails.
CAMERA_INDEX: int = 1
FRAME_WIDTH: int = 3840  # fallback only
FRAME_HEIGHT: int = 2160  # fallback only

# Actual source size detected at runtime
ACTUAL_FRAME_WIDTH: int = FRAME_WIDTH
ACTUAL_FRAME_HEIGHT: int = FRAME_HEIGHT


def set_actual_frame_size(width: int, height: int) -> None:
    """
    Update the probed source size.

    Downstream stages can read ACTUAL_FRAME_WIDTH and ACTUAL_FRAME_HEIGHT
    to know the real input geometry.
    """
    global ACTUAL_FRAME_WIDTH, ACTUAL_FRAME_HEIGHT
    ACTUAL_FRAME_WIDTH = int(width)
    ACTUAL_FRAME_HEIGHT = int(height)


# Board geometry
BOARD_SIZE_PX: int = 640
BOARD_SQUARES: int = 8
BOARD_MARGIN_SQUARES: float = 1.7

# Source selection
USE_VIDEO_FILE: bool = True
VIDEO_PATH: Path = VIDEOS_DIR / "game2.mp4"

# Calibration
CALIBRATION_MAX_FRAMES: int = 500
AUTO_MIN_BOARD_AREA_RATIO: float = 0.08
USE_SAVED_HOMOGRAPHY: bool = True
HOMOGRAPHY_PATH: Path = CALIBRATION_DIR / "homography.npy"
CALIBRATION_TARGET_LONG_EDGE: int = BOARD_SIZE_PX

# Parallel detection workers
DETECTION_WORKERS: int = 1

# Logging for multistage move tracker
MOVES_LOG_PATH: Path = GAME_MOVES_DIR / "moves.log"
GAME_MOVES_TXT_PATH: Path = GAME_MOVES_DIR / "game_moves.txt"

# Logging for single frame FEN baseline
FEN_LOG_PATH: Path = GAME_MOVES_DIR / "baseline_fens.txt"
START_MIN_CONFIRM_FRAMES: int = 4
FEN_MIN_STABLE_FRAMES: int = 10

# YOLO based piece detector for stage 2
YOLO_PIECE_WEIGHTS: Path = MODELS_DIR / f"yolo11s_best_{BOARD_SIZE_PX}.onnx"
YOLO_PIECE_IMGSZ: int = BOARD_SIZE_PX
YOLO_PIECE_CONF: float = 0.15
MIN_IOU: float = 0.15

# UI
DISPLAY_WINDOW_NAME: str = "Board"
GUI_ENABLED: bool = True
OPENCV_NUM_THREADS: int = 0  # 0 lets OpenCV decide, >0 forces a limit

# Move tracker settings (used by multistage pipeline)
MOVE_FILTER_ALPHA: float = 0.5
MOVE_FILTER_THRESHOLD: float = 0.60
MOVE_MIN_CONFIRM_FRAMES: int = 3
MOVE_DEBUG: bool = True

# Queue sizes
FRAME_QUEUE_SIZE: int = 2
DETECTION_INPUT_QUEUE_SIZE: int = 1
DETECTION_OUTPUT_QUEUE_SIZE: int = 3
MOVE_IN_QUEUE_SIZE: int = 8
MOVE_OUT_QUEUE_SIZE: int = 64

# Which live pipeline should run by default:
# "multistage" for full temporal tracker
# "singleframe" for FEN only baseline
PIPELINE_MODE: str = "singleframe"
