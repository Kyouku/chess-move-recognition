from __future__ import annotations

import logging
import queue
from typing import Optional, Tuple, Dict, List

from src import config
from src.common.app_logging import get_logger, set_log_level
from src.common.chess_io import write_moves_txt, append_move_log
from src.common.types import MoveInfo, DetectionState
from src.pipeline.live_base import (
    BaseLivePipeline,
    get_capture_source,
    CaptureSource,
)
from src.stage3.move_tracking import MoveTracker, MoveTrackerWorker

_log = get_logger(__name__)


class MultiStagePipeline(BaseLivePipeline):
    """
    Full multi stage live pipeline.

    Responsibilities:
      - Stage 1 and 2 orchestration are handled by BaseLivePipeline
        (capture, rectification, piece detection, GUI).
      - Stage 3 is handled here via MoveTracker and a dedicated
        MoveTrackerWorker thread that consumes occupancy and labels
        and emits MoveInfo objects.
    """

    def __init__(
            self,
            source: CaptureSource,
            *,
            width: Optional[int] = None,
            height: Optional[int] = None,
            board_size_px: Optional[int] = None,
    ) -> None:
        if board_size_px is None:
            board_size_px = config.BOARD_SIZE_PX

        super().__init__(
            source=source,
            width=width,
            height=height,
            board_size_px=board_size_px,
            window_name=config.DISPLAY_WINDOW_NAME,
        )

        # Stage 3: MoveTracker with temporal filter and its own worker
        self.move_tracker = MoveTracker(
            alpha=config.MOVE_FILTER_ALPHA,
            occ_threshold=config.MOVE_FILTER_THRESHOLD,
            min_confirm_frames=config.MOVE_MIN_CONFIRM_FRAMES,
            debug=config.MOVE_DEBUG,
        )

        # Queues between Stage 2 and Stage 3
        self.move_in_queue: "queue.Queue[Tuple[Dict[str, bool], Dict[str, Optional[str]]]]" = (
            queue.Queue(maxsize=config.MOVE_IN_QUEUE_SIZE)
        )
        self.move_out_queue: "queue.Queue[MoveInfo]" = queue.Queue(
            maxsize=config.MOVE_OUT_QUEUE_SIZE,
        )

        self.move_worker = MoveTrackerWorker(
            tracker=self.move_tracker,
            input_queue=self.move_in_queue,
            output_queue=self.move_out_queue,
            stop_event=self.stop_event,
            name="MoveTrackerWorker-1",
        )

        # Collected SAN moves for summary or PGN export
        self.moves: List[str] = []

    # ------------------------------------------------------------------
    # BaseLivePipeline hooks
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """
        Called after Stage 1 and 2 threads have been started.

        Starts the Stage 3 MoveTracker worker.
        """
        self.move_worker.start()

    def handle_detection_state(self, state: DetectionState) -> None:
        """
        Called once for every DetectionState drained from the Stage 2 queue.

        Only occupancy and labels are needed by MoveTracker, so we enqueue
        a compact tuple for the MoveTrackerWorker.
        """
        payload: Tuple[Dict[str, bool], Dict[str, Optional[str]]] = (
            state.occupancy,
            state.pieces,
        )
        try:
            self.move_in_queue.put_nowait(payload)
        except queue.Full:
            # Worker is busy, drop current state rather than blocking
            _log.debug("Move input queue full, dropping detection state")

    def after_detection_batch(self) -> None:
        """
        Called once per frame after all DetectionStates for that frame
        have been processed.

        Here we drain all MoveInfo objects currently available from
        the move worker output queue.
        """
        while True:
            try:
                info: MoveInfo = self.move_out_queue.get_nowait()
            except queue.Empty:
                break

            self._on_move_detected(info)

    def on_stop(self) -> None:
        """
        Called during shutdown after Stage 1 and 2 have been stopped.

        Joins the MoveTracker worker and writes remaining moves to disk
        if there are any pending.
        """
        try:
            self.move_worker.join(timeout=1.0)
        except RuntimeError:
            pass

        if self.moves:
            try:
                write_moves_txt(self.moves)
            except (OSError, ValueError) as exc:
                _log.warning(
                    "Failed to write game moves on shutdown: %s",
                    exc,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_move_detected(self, info: MoveInfo) -> None:
        """
        Handle a single confirmed MoveInfo emitted by the MoveTracker.
        """
        msg_lines = [
            "==================================================",
            f"[MAIN] Move detected: {info.san} ({info.move.uci()})",
            f"[MAIN] FEN after: {info.fen_after}",
            "[MAIN] Board:",
            str(self.move_tracker.board),
            "==================================================",
        ]
        _log.info("\n".join(msg_lines))

        self.moves.append(info.san)

        # Optional file logging (non fatal on errors)
        append_move_log(info)

        # Reset only on checkmate, draws are ignored intentionally
        if getattr(info, "is_checkmate", False):
            _log.info(
                "[MAIN] Checkmate detected. Saving game and resetting to initial position...",
            )
            # Save current game moves to PGN or text
            try:
                write_moves_txt(self.moves)
            except (OSError, ValueError) as exc:
                _log.warning(
                    "Failed to write game moves on game over: %s",
                    exc,
                )

            # Clear current move list
            self.moves.clear()

            # Reset MoveTracker to the initial position and wait for a new game
            self.move_tracker.reset_to_start()

            # Drain any pending outputs from the previous game to avoid duplicates
            try:
                while True:
                    self.move_out_queue.get_nowait()
            except queue.Empty:
                pass


# ----------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------


def run_live(
        source: CaptureSource,
        width: int = None,
        height: int = None,
        board_size_px: Optional[int] = None,
) -> None:
    """
    Convenience function to start the live multi stage pipeline.
    """
    if width is None:
        width = config.FRAME_WIDTH
    if height is None:
        height = config.FRAME_HEIGHT

    pipeline = MultiStagePipeline(
        source=source,
        width=width,
        height=height,
        board_size_px=board_size_px,
    )
    pipeline.run()


def main() -> None:
    """
    Minimal CLI entry point for manual testing.

    Currently just sets a fixed log level and runs with the
    configured capture source.
    """
    set_log_level(logging.DEBUG)
    src = get_capture_source()
    run_live(src)


if __name__ == "__main__":
    main()
