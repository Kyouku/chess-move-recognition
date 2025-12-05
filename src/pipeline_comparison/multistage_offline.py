from __future__ import annotations

from typing import List

from src import config
from src.app_logging import get_logger
from src.pipeline_comparison.common import PipelineResult
from src.stage3.move_tracking import MoveTracker
from src.types import DetectionState, MoveInfo

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
        alpha=float(getattr(config, "MOVE_FILTER_ALPHA", 0.6)),
        occ_threshold=float(getattr(config, "MOVE_FILTER_THRESHOLD", 0.6)),
        min_confirm_frames=int(
            getattr(config, "MOVE_MIN_CONFIRM_FRAMES", 2)
        ),
        debug=bool(getattr(config, "MOVE_DEBUG", False)),
    )

    frame_fens: List[str] = []
    moves_uci: List[str] = []
    moves_san: List[str] = []
    move_frames: List[int] = []

    total_frames = len(states)
    use_live_sampling = (
            video_fps is not None
            and video_fps > 0.0
            and detector_fps is not None
            and detector_fps > 0.0
            and detector_fps < video_fps
    )

    if not use_live_sampling:
        # Original behavior: process every frame
        for idx, state in enumerate(states):
            _log.debug(
                "[MULTISTAGE OFFLINE] processing frame %d/%d",
                idx + 1,
                total_frames,
            )
            info: MoveInfo | None = tracker.update_from_detection_state(state)
            frame_fens.append(tracker.board.fen())
            if info is None:
                continue

            moves_uci.append(info.move.uci())
            moves_san.append(info.san)
            move_frames.append(idx)

            _log.info(
                "[MULTISTAGE OFFLINE] frame %d move %s (%s) | FEN: %s",
                idx,
                info.san,
                info.move.uci(),
                getattr(info, "fen_after", ""),
            )
    else:
        # Live style sampling approximation
        dt_video = 1.0 / float(video_fps)
        dt_det = 1.0 / float(detector_fps)
        next_free_time = 0.0

        for idx, state in enumerate(states):
            _log.debug(
                "[MULTISTAGE OFFLINE] processing frame %d/%d",
                idx + 1,
                total_frames,
            )
            t = idx * dt_video
            info: MoveInfo | None = None

            # Only process this frame if the "detector" would be free
            if t >= next_free_time:
                next_free_time = t + dt_det
                info = tracker.update_from_detection_state(state)

            # Board FEN is defined for every original frame index
            frame_fens.append(tracker.board.fen())

            if info is None:
                continue

            moves_uci.append(info.move.uci())
            moves_san.append(info.san)
            move_frames.append(idx)

            _log.info(
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
