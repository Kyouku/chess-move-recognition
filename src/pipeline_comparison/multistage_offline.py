from __future__ import annotations

from typing import List

from src import config
from src.app_logging import get_logger
from src.pipeline_comparison.common import PipelineResult
from src.stage3.move_tracking import MoveTracker
from src.types import DetectionState, MoveInfo

_log = get_logger(__name__)


def run_multistage(states: List[DetectionState]) -> PipelineResult:
    """
    Run the multistage MoveTracker on a list of DetectionState objects.

    Uses the same MoveTracker that is used in the live pipeline.
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
    for idx, state in enumerate(states):
        _log.debug("[MULTISTAGE OFFLINE] processing frame %d/%d", idx + 1, total_frames)
        info: MoveInfo | None = tracker.update_from_detection_state(state)
        frame_fens.append(tracker.board.fen())
        if info is None:
            continue

        moves_uci.append(info.move.uci())
        moves_san.append(info.san)
        move_frames.append(idx)

        _log.info(
            "[MULTISTAGE OFFLINE] frame %d move %s (%s)",
            idx,
            info.san,
            info.move.uci(),
        )

    return PipelineResult(
        frame_fens=frame_fens,
        moves_uci=moves_uci,
        moves_san=moves_san,
        move_frames=move_frames,
    )
