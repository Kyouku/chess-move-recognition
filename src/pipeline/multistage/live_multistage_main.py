from __future__ import annotations

import logging
import queue
from typing import Optional, Tuple, Dict, List

from src import config
from src.app_logging import get_logger, set_log_level
from src.chess_io import write_moves_txt, append_move_log
from src.pipeline.live_base import (
    BaseLivePipeline,
    get_capture_source,
    CaptureSource,
)
from src.stage3.move_tracking import MoveTracker, MoveTrackerWorker
from src.types import MoveInfo, DetectionState

_log = get_logger(__name__)

"""
Move logging is handled by src.chess_io.append_move_log for reuse across modules.
"""


class MultiStagePipeline(BaseLivePipeline):
    """
    Full multi stage pipeline with temporal filtering and MoveTracker.
    Stage 1 + 2 live handling comes from BaseLivePipeline.
    """

    def __init__(
            self,
            source: CaptureSource,
            *,
            width: Optional[int] = None,
            height: Optional[int] = None,
            board_size_px: int = config.BOARD_SIZE_PX,
    ) -> None:
        super().__init__(
            source=source,
            width=width,
            height=height,
            board_size_px=board_size_px,
            window_name=config.DISPLAY_WINDOW_NAME,
        )

        # Stage 3: MoveTracker with temporal filter and its own worker
        self.move_tracker = MoveTracker(
            alpha=float(config.MOVE_FILTER_ALPHA),
            occ_threshold=float(config.MOVE_FILTER_THRESHOLD),
            min_confirm_frames=int(config.MOVE_MIN_CONFIRM_FRAMES),
            debug=bool(config.MOVE_DEBUG),
        )

        self.move_in_queue: "queue.Queue[Tuple[Dict[str, bool], Dict[str, Optional[str]]]]" = queue.Queue(
            maxsize=config.MOVE_IN_QUEUE_SIZE
        )
        self.move_out_queue: "queue.Queue[MoveInfo]" = queue.Queue(
            maxsize=config.MOVE_OUT_QUEUE_SIZE
        )

        self.move_worker = MoveTrackerWorker(
            tracker=self.move_tracker,
            input_queue=self.move_in_queue,
            output_queue=self.move_out_queue,
            stop_event=self.stop_event,
            name="MoveTrackerWorker-1",
        )

        # Collected SAN moves for summary / PGN export
        self.moves: List[str] = []

    # ------------------------------------------------------------------
    # BaseLivePipeline hooks
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        # Start Stage 3 worker after Stage 1 + 2 are running
        self.move_worker.start()

    def handle_detection_state(self, state: DetectionState) -> None:
        # Only occupancy and labels are needed by MoveTracker
        try:
            self.move_in_queue.put_nowait((state.occupancy, state.pieces))
        except queue.Full:
            # Drop if worker is busy
            pass

    def after_detection_batch(self) -> None:
        # Drain finished moves from the move worker
        while True:
            try:
                info: MoveInfo = self.move_out_queue.get_nowait()
            except queue.Empty:
                break

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

            # Optional file logging (non-fatal on errors)
            append_move_log(info)

            # Reset ONLY on checkmate; ignore draws
            if getattr(info, "is_checkmate", False):
                _log.info(
                    "[MAIN] Checkmate detected. Saving game and resetting to initial position...",
                )
                # Save current game moves to PGN/text
                try:
                    write_moves_txt(self.moves)
                except (OSError, ValueError) as e:
                    _log.warning("Failed to write game moves on game over: %s", e)

                # Clear current move list
                self.moves.clear()

                # Reset the move tracker to wait for initial position again
                # Reset MoveTracker to initial position
                self.move_tracker.reset_to_start()

                # Drain any pending outputs from the previous game to avoid duplicates
                try:
                    while True:
                        self.move_out_queue.get_nowait()
                except queue.Empty:
                    pass

    def on_stop(self) -> None:
        # Join move worker and write remaining moves, if any
        try:
            self.move_worker.join(timeout=1.0)
        except RuntimeError:
            pass

        if self.moves:
            try:
                write_moves_txt(self.moves)
            except (OSError, ValueError) as e:
                _log.warning("Failed to write game moves on shutdown: %s", e)


def run_live(
        source: CaptureSource,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        board_size_px: int = config.BOARD_SIZE_PX,
) -> None:
    pipeline = MultiStagePipeline(
        source=source,
        width=width,
        height=height,
        board_size_px=board_size_px,
    )
    pipeline.run()


def main() -> None:
    set_log_level(logging.DEBUG)
    src = get_capture_source()
    run_live(src)


if __name__ == "__main__":
    main()
