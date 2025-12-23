from __future__ import annotations

"""
Dataclasses and helpers used only in the offline comparison pipeline.

These types and functions are shared between the single frame and multistage
offline runners that are used for evaluation on recorded videos.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Any, Dict, Optional
import json
import inspect

from src.pipeline.comparison.detection_log import load_detections
from src.common.app_logging import get_logger

_log = get_logger(__name__)


@dataclass
class PipelineResult:
    """
    Unified result type returned by offline pipeline runners.

    This dataclass is used only in the offline comparison pipeline.

    All lists are aligned to the same input stream of frames.

    - frame_fens:
        Length equals the number of input frames.
        Element i is the model board representation (FEN) derived
        directly from detections for frame i. Only the placement
        field is meaningful for comparison with ground truth.
    - moves_uci:
        Detected moves in UCI notation, in chronological order.
    - moves_san:
        Detected moves in SAN notation, in the same order as moves_uci.
        May contain empty strings if a pipeline does not compute SAN.
    - move_frames:
        For each detected move in moves_uci, the frame index at which the
        move was committed by the pipeline.
    """

    frame_fens: List[str]
    moves_uci: List[str]
    moves_san: List[str]
    move_frames: List[int]

    @property
    def fens(self) -> List[str]:
        """Backwards compatible alias for frame_fens."""
        return self.frame_fens


# -------------------------------------------------------------
# Shared helpers for offline comparison utilities only
# -------------------------------------------------------------


def is_game3(name: str) -> bool:
    """
    Best-effort identification of "game3" naming variants.
    """
    n = name.strip().lower().replace(" ", "").replace("-", "_")
    return n in {"game3", "game_3", "3"}


def load_homography_matrix(path: Path) -> List[List[float]]:
    """
    Load a saved 3x3 homography matrix from .json, .npy, or .txt/.csv.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    suf = path.suffix.lower()

    if suf == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        h_mat = obj.get("H") or obj.get("homography") or obj.get("matrix")
        if not isinstance(h_mat, list) or len(h_mat) != 3:
            raise ValueError(f"Invalid homography JSON in {path}")
        return [[float(x) for x in row] for row in h_mat]

    if suf == ".npy":
        import numpy as np
        arr = np.load(str(path))
        if arr.shape != (3, 3):
            raise ValueError(f"Invalid homography npy in {path} (shape {arr.shape})")
        return [[float(x) for x in row] for row in arr.tolist()]

    # Fallback: parse simple text
    txt = path.read_text(encoding="utf-8").strip().splitlines()
    rows: List[List[float]] = []
    for line in txt:
        line = line.strip()
        if not line:
            continue
        parts = [p for p in line.replace(",", " ").split(" ") if p]
        if parts:
            rows.append([float(p) for p in parts])

    if len(rows) != 3 or any(len(r) != 3 for r in rows):
        raise ValueError(f"Invalid homography text in {path}")
    return rows


def attach_homography_best_effort(det_log: Any, homography_path: Path) -> None:
    """
    Attach homography matrix to the det_log object metadata.
    """
    try:
        h_mat = load_homography_matrix(homography_path)
    except Exception as e:
        _log.warning("Failed to read homography file %s: %s", homography_path, e)
        return

    for attr in ("H", "homography", "homography_matrix"):
        try:
            setattr(det_log, attr, h_mat)
            return
        except Exception:
            pass


def load_detections_with_optional_homography(det_path: Path, homography_path: Optional[Path]) -> Any:
    """
    Load detections and optionally inject or attach a homography override.
    """
    if homography_path is None:
        return load_detections(str(det_path))

    sig = inspect.signature(load_detections)
    kwargs: Dict[str, Any] = {}
    candidate_keys = (
        "homography_path",
        "override_homography_path",
        "calibration_path",
        "saved_homography_path",
        "homography",
    )

    for key in candidate_keys:
        if key in sig.parameters:
            kwargs[key] = str(homography_path)
            break

    if not kwargs and any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        kwargs["homography_path"] = str(homography_path)

    det_log = load_detections(str(det_path), **kwargs) if kwargs else load_detections(str(det_path))

    if not kwargs:
        attach_homography_best_effort(det_log, homography_path)

    return det_log


def compute_sampling_step(
        video_fps: float | None,
        detector_fps: float | None,
) -> int:
    """
    Compute an integer frame step for approximating live sampling when
    running on pre recorded frames in the offline comparison pipeline.

    If fps values are missing or invalid, returns 1 (process every frame).
    """
    try:
        if video_fps is None or detector_fps is None or float(detector_fps) <= 0.0:
            return 1
        ratio = float(video_fps) / float(detector_fps)
        step = max(1, int(round(ratio)))
        return step
    except Exception:
        return 1


def should_process_frame(
        frame_idx: int,
        video_fps: float | None,
        detector_fps: float | None,
) -> bool:
    """
    Decide whether to process a given frame index under the computed
    sampling step for the offline comparison pipeline.

    Returns True for frames that should be processed.
    """
    step = compute_sampling_step(video_fps, detector_fps)
    return (frame_idx % step) == 0


def iter_time_based_should_process(
        video_fps: float | None,
        detector_fps: float | None,
        total_frames: int,
):
    """
    Yield (index, should_process) for each frame, using a time based model
    that approximates a live system with a detector that needs
    1 / detector_fps seconds per processed frame and a queue size of 1.

    If parameters are missing or invalid or detector_fps >= video_fps, the
    iterator yields should_process=True for all frames (process every frame).
    """
    try:
        vf = float(video_fps) if video_fps is not None else None
        df = float(detector_fps) if detector_fps is not None else None
    except Exception:
        vf = None
        df = None

    use_live = (
            vf is not None
            and vf > 0.0
            and df is not None
            and 0.0 < df < vf
    )

    if not use_live:
        for idx in range(total_frames):
            yield idx, True
        return

    dt_video = 1.0 / vf  # type: ignore[arg-type]
    dt_det = 1.0 / df  # type: ignore[arg-type]
    next_free_time = 0.0
    for idx in range(total_frames):
        t = idx * dt_video
        if t >= next_free_time:
            next_free_time = t + dt_det
            yield idx, True
        else:
            yield idx, False
