from __future__ import annotations

"""
Live single frame baseline pipeline.

Reuses Stage 1 and 2 from BaseLivePipeline and exports stable FEN snapshots
once the initial position has been detected reliably. No move inference or
game state tracking is performed.
"""

from typing import Dict, Optional

import chess

from src import config
from src.common.app_logging import get_logger
from src.common.chess_io import append_fen_log
from src.common.types import DetectionState
from src.pipeline.fen_utils import (
    detection_state_to_fen,
    detection_state_to_placement,
)
from src.pipeline.live_base import (
    BaseLivePipeline,
    get_capture_source,
    CaptureSource,
)

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Single frame live baseline: FEN only
# ---------------------------------------------------------------------------


class SingleFramePipeline(BaseLivePipeline):
    """
    Live app that reuses Stage 1 and 2 from BaseLivePipeline and outputs
    FEN snapshots for stable board frames.

    No move inference or game state tracking is performed here.
    """

    def __init__(
            self,
            source: CaptureSource,
            *,
            width: Optional[int] = None,
            height: Optional[int] = None,
    ) -> None:
        super().__init__(
            source=source,
            width=width,
            height=height,
            board_size_px=int(getattr(config, "BOARD_SIZE_PX", 640)),
            window_name=str(getattr(config, "DISPLAY_WINDOW_NAME", "Board")),
        )

        # Used to wait until the standard start position is stable
        self._start_occ: Dict[str, bool] = self._board_to_occupancy(chess.Board())
        self._start_seen_frames: int = 0
        self._initialized: bool = False

        # Number of frames the start position must be seen to accept it
        self._min_start_frames: int = int(
            getattr(config, "START_MIN_CONFIRM_FRAMES", 4),
        )

        # Number of identical placements before they are treated as stable
        self._min_stable_frames: int = int(
            getattr(config, "FEN_MIN_STABLE_FRAMES", 10),
        )

        # Stability tracking for placements
        self._last_seen_placement: Optional[str] = None
        self._same_seen_frames: int = 0

        # Last stable placement that has already been logged
        self._committed_placement: Optional[str] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _board_to_occupancy(board: chess.Board) -> Dict[str, bool]:
        """
        Convert a python chess Board into an occupancy mask over algebraic squares.
        """
        occupancy: Dict[str, bool] = {}
        for sq in chess.SQUARES:
            occupancy[chess.square_name(sq)] = board.piece_at(sq) is not None
        return occupancy

    def _update_start_initialization(self, state: DetectionState) -> None:
        """
        Wait until the start position is detected stably before logging FENs.
        """
        if self._initialized:
            return

        curr_occ = state.occupancy
        same = all(
            bool(curr_occ.get(sq, False)) == want
            for sq, want in self._start_occ.items()
        )
        if same:
            self._start_seen_frames += 1
            if self._start_seen_frames >= self._min_start_frames:
                self._initialized = True
                _log.info("Baseline: start position detected and initialized")
        else:
            self._start_seen_frames = 0

    # ------------------------------------------------------------------
    # Hook from BaseLivePipeline
    # ------------------------------------------------------------------

    def handle_detection_state(self, state: DetectionState) -> None:
        """
        Called by BaseLivePipeline for each detected board state.

        Only performs stability filtering and FEN export.
        """
        self._update_start_initialization(state)
        if not self._initialized:
            return

        placement = detection_state_to_placement(state)

        # Stability tracking over placements
        if placement == self._last_seen_placement:
            self._same_seen_frames += 1
        else:
            self._last_seen_placement = placement
            self._same_seen_frames = 1

        # Not yet stable
        if self._same_seen_frames < self._min_stable_frames:
            return

        # Already logged this plateau
        if placement == self._committed_placement:
            return

        # New stable plateau
        fen = detection_state_to_fen(state)
        self._committed_placement = placement

        _log.info("Baseline stable FEN: %s", fen)

        # Optional: write FEN snapshots using centralized helper (non fatal)
        append_fen_log(fen)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_live(
        source: CaptureSource,
        width: Optional[int] = None,
        height: Optional[int] = None,
) -> None:
    """
    Convenience function to start the live single frame baseline.
    """
    if width is None:
        width = int(getattr(config, "FRAME_WIDTH", 1280))
    if height is None:
        height = int(getattr(config, "FRAME_HEIGHT", 720))

    pipeline = SingleFramePipeline(
        source=source,
        width=width,
        height=height,
    )
    pipeline.run()


def main() -> None:
    source = get_capture_source()
    run_live(source)


if __name__ == "__main__":
    main()
