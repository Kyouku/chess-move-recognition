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
GAME_MOVES_DIR: Path = _SRC_DIR / "pipeline" / "game_moves"

# Input source
# Note: The live app now automatically probes the actual source dimensions.
# FRAME_WIDTH/HEIGHT are only used as a fallback if probing fails.
CAMERA_INDEX: int = 1
FRAME_WIDTH: int = 3840  # fallback only; actual capture size is probed at runtime
FRAME_HEIGHT: int = 2160  # fallback only; actual capture size is probed at runtime

# Actual source size detected at runtime (exposed for downstream users)
ACTUAL_FRAME_WIDTH: int = FRAME_WIDTH
ACTUAL_FRAME_HEIGHT: int = FRAME_HEIGHT


def set_actual_frame_size(width: int, height: int) -> None:
    """
    Update the probed source size. Downstream stages can read
    config.ACTUAL_FRAME_WIDTH/HEIGHT to know the real input geometry.
    """
    global ACTUAL_FRAME_WIDTH, ACTUAL_FRAME_HEIGHT
    ACTUAL_FRAME_WIDTH = int(width)
    ACTUAL_FRAME_HEIGHT = int(height)


# Board geometry
BOARD_SIZE_PX: int = 640
BOARD_SQUARES: int = 8
BOARD_MARGIN_SQUARES: float = 1.7

USE_VIDEO_FILE: bool = False
# Default sample video resolved under project data/videos
VIDEO_PATH: Path = VIDEOS_DIR / "test.mp4"

# Calibration
CALIBRATION_MAX_FRAMES: int = 500
AUTO_MIN_BOARD_AREA_RATIO: float = 0.08
# Whether to load a previously saved homography instead of calibrating
USE_SAVED_HOMOGRAPHY: bool = True
# Where to save/load the homography matrix (NumPy .npy file)
HOMOGRAPHY_PATH: Path = CALIBRATION_DIR / "homography.npy"
# Preprocessing size for Stage1 (auto/manual calibration and rectification)
# The camera can capture at 4 K, but corner detection works more reliably and
# faster on a moderately sized preprocessed image. The rectifier will use this
# long-edge size consistently both during calibration and runtime.
CALIBRATION_TARGET_LONG_EDGE: int = BOARD_SIZE_PX

# Parallel detection workers
DETECTION_WORKERS: int = 1

# Logging (resolve under src/pipeline/game_moves)
MOVES_LOG_PATH: Path = GAME_MOVES_DIR / "moves.log"
GAME_MOVES_TXT_PATH: Path = GAME_MOVES_DIR / "game_moves.txt"

# YOLO based piece detector for stage2
# Resolve models directory relative to the project root
YOLO_PIECE_WEIGHTS: Path = MODELS_DIR / f"yolo11m_best_{BOARD_SIZE_PX}.onnx"
YOLO_PIECE_IMGSZ: int = BOARD_SIZE_PX
YOLO_PIECE_CONF: float = 0.5
MIN_IOU: float = 0.15

# UI
DISPLAY_WINDOW_NAME: str = "Board"
GUI_ENABLED: bool = True
OPENCV_NUM_THREADS: int = 0  # 0 = OpenCV decides; >0 to force a limit

# Move tracker settings
MOVE_FILTER_ALPHA: float = 0.3
MOVE_FILTER_THRESHOLD: float = 0.65
MOVE_MIN_CONFIRM_FRAMES: int = 2
MOVE_DEBUG: bool = True

# Queue sizes
FRAME_QUEUE_SIZE: int = 3
DETECTION_INPUT_QUEUE_SIZE: int = 1
DETECTION_OUTPUT_QUEUE_SIZE: int = 3
MOVE_IN_QUEUE_SIZE: int = 8
MOVE_OUT_QUEUE_SIZE: int = 64

# Which live pipeline should run?
# "multistage" -> MoveTracker with temporal filter
# "singleframe" -> SingleFrameBaseline without history
PIPELINE_MODE: str = "singleframe"
