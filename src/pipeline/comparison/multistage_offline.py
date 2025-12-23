from __future__ import annotations

"""
Offline runner for the multistage MoveTracker in the comparison pipeline.

Takes a DetectionState sequence, runs the tracker with optional time based
sampling, and returns a PipelineResult for evaluation.
"""

from typing import List

from src import config
from src.common.app_logging import get_logger
from src.common.types import DetectionState, MoveInfo
from src.pipeline.comparison.common import (
    PipelineResult,
    iter_time_based_should_process,
)
from src.stage3.move_tracking import MoveTracker

_log = get_logger(__name__)


def run_multistage(
        states: List[DetectionState],
        *,
        video_fps: float | None = None,
        detector_fps: float | None = None,
) -> PipelineResult:
    """
    Run the multistage MoveTracker on a list of DetectionState objects.

    If video_fps and detector_fps are provided and detector_fps < video_fps,
    a live style sampling is simulated where only frames that the detector
    could realistically process are fed into the tracker, similar to a
    queue of size 1.
    """
    tracker = MoveTracker(
        alpha=config.MOVE_FILTER_ALPHA,
        occ_threshold=config.MOVE_FILTER_THRESHOLD,
        min_confirm_frames=config.MOVE_MIN_CONFIRM_FRAMES,
        debug=config.MOVE_DEBUG,
    )

    frame_fens: List[str] = []
    moves_uci: List[str] = []
    moves_san: List[str] = []
    move_frames: List[int] = []

    total_frames = len(states)

    # Unified iteration using time based gating helper
    for idx, should_process in iter_time_based_should_process(
            video_fps, detector_fps, total_frames,
    ):
        state = states[idx]
        _log.debug(
            "[MULTISTAGE OFFLINE] processing frame %d/%d",
            idx + 1,
            total_frames,
        )

        info: MoveInfo | None = None
        if should_process:
            info = tracker.update_from_detection_state(state)

        # Board FEN is defined for every original frame index
        frame_fens.append(tracker.board.fen())

        if info is None:
            continue

        moves_uci.append(info.move.uci())
        moves_san.append(info.san)
        move_frames.append(idx)

        _log.debug(
            "[MULTISTAGE OFFLINE] frame %d move %s (%s) | FEN: %s",
            idx,
            info.san,
            info.move.uci(),
            getattr(info, "fen_after", ""),
        )

    return PipelineResult(
        frame_fens=frame_fens,
        moves_uci=moves_uci,
        moves_san=moves_san,
        move_frames=move_frames,
    )
