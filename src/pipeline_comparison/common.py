from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class PipelineResult:
    """
    Unified result type returned by offline pipeline runners.

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


# -------------------------------------------------------------
# Shared helpers for offline pipeline utilities
# -------------------------------------------------------------


def compute_sampling_step(
        video_fps: float | None,
        detector_fps: float | None,
) -> int:
    """
    Compute an integer frame step for approximating live sampling when
    running on pre-recorded frames.

    If fps values are missing or invalid, returns 1 (process every frame).
    """
    try:
        if video_fps is None or detector_fps is None or float(detector_fps) <= 0.0:
            return 1
        ratio = float(video_fps) / float(detector_fps)
        step = max(1, int(round(ratio)))
        return step
    except Exception:
        # Be robust against any bad inputs
        return 1


def should_process_frame(
        frame_idx: int,
        video_fps: float | None,
        detector_fps: float | None,
) -> bool:
    """
    Decide whether to process a given frame index under the computed
    sampling step. Returns True for frames that should be processed.
    """
    step = compute_sampling_step(video_fps, detector_fps)
    return (frame_idx % step) == 0


def iter_time_based_should_process(
        video_fps: float | None,
        detector_fps: float | None,
        total_frames: int,
):
    """
    Yield (index, should_process) for each frame, using a time-based model
    that approximates a live system with a detector that needs
    1 / detector_fps seconds per processed frame and a queue size of 1.

    If parameters are missing/invalid or detector_fps >= video_fps, the
    iterator yields should_process=True for all frames (process every frame).
    """
    # Validate parameters
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

    # Time-based gating
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
