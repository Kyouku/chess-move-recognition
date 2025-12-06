from __future__ import annotations

"""
Offline single frame baseline for the comparison pipeline.

This module takes DetectionState sequences from the single frame baseline,
reconstructs moves from FEN snapshots, and returns a unified PipelineResult
for evaluation.
"""

from typing import List, Optional

import chess

from src import config
from src.common.app_logging import get_logger
from src.common.types import DetectionState
from src.pipeline.comparison.common import (
    PipelineResult,
    iter_time_based_should_process,
)
from src.pipeline.fen_utils import (
    detection_state_to_fen,
    detection_state_to_placement,
)

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Baseline runner: reconstruct moves from FEN snapshots
# ---------------------------------------------------------------------------


class _BaselineRunner:
    """
    Offline move reconstruction based on FEN snapshots of the single frame baseline.

    Logic:
      - Wait until the standard initial position is detected stably.
      - Keep an internal python chess Board as reference.
      - For each processed frame:
          * Build a placement string from detections.
          * If placement equals the reference board, no move happened.
          * Otherwise, search all legal moves of the reference board and check
            which moves lead to the detected placement.
          * If exactly one move matches, propose that move.
          * Use debouncing over several frames before confirming a move.
          * If no unique move exists, mark this as a missing move region and
            skip the FEN, keeping the reference board unchanged so that later
            correct FENs can still produce valid moves.

    The runner also keeps a logical move index that counts both confirmed
    moves and missing move regions. This allows numbering moves as

      (move 7)

    even if only two moves were actually reconstructed and five are missing
    between them.
    """

    def __init__(self) -> None:
        start_board = chess.Board()
        self._start_placement: str = start_board.board_fen()
        self._start_seen_frames: int = 0
        self._initialized: bool = False

        # Frames required to accept the start position
        self._min_start_frames: int = int(
            getattr(config, "START_MIN_CONFIRM_FRAMES", 4),
        )

        # Debouncing for proposed moves
        self._pending_uci: Optional[str] = None
        self._pending_count: int = 0
        self._confirm_frames: int = int(
            getattr(config, "MOVE_MIN_CONFIRM_FRAMES", 2),
        )

        # Current reference board (updated only on confirmed moves)
        self._board: Optional[chess.Board] = None

        # Missing move statistics
        self.missing_frames: int = 0
        self.missing_segments: int = 0
        self.missing_moves: int = 0
        self._in_missing_segment: bool = False

        # Logical move index including missing moves
        # First real move after the initial position has index 1.
        self._next_ply_index: int = 1

    @property
    def initialized(self) -> bool:
        """Return True once the initial position has been locked."""
        return self._initialized

    def log_summary(self) -> None:
        """
        Log a short summary about initialization and missing move statistics.

        Intended to be called once at the end of run_baseline.
        """
        if not self._initialized:
            _log.info(
                "[BASELINE OFFLINE] initial position was never locked, "
                "no moves reconstructed",
            )
            return

        _log.info(
            "[BASELINE OFFLINE] missing move segments=%d, frames=%d, moves=%d",
            self.missing_segments,
            self.missing_frames,
            self.missing_moves,
        )

    def _unique_legal_move_to(self, target_placement: str) -> Optional[chess.Move]:
        """
        Return the unique legal move that transforms the current reference
        board into the target placement, or None if there is none or more than one.
        """
        if self._board is None:
            return None

        board = self._board
        candidates: list[chess.Move] = []

        for move in board.legal_moves:
            tmp = board.copy()
            tmp.push(move)
            if tmp.board_fen() == target_placement:
                candidates.append(move)
                if len(candidates) > 1:
                    # Ambiguous transformation, treat as not reconstructable
                    return None

        if len(candidates) == 1:
            return candidates[0]
        return None

    def process_state(
            self,
            state: DetectionState,
            frame_idx: int,
    ) -> Optional[tuple[str, Optional[str], Optional[str], int, int]]:
        """
        Process a DetectionState.

        Returns (uci, san, fen_after, frame_idx, ply_index) once a move
        is confirmed. Returns None if no move can be confirmed.

        san is left as None since reconstruction is driven only by FEN
        snapshots. fen_after is the FEN derived from detections.
        ply_index is the logical move index including missing moves.
        """
        placement = detection_state_to_placement(state)

        # 1) Start gating on the standard initial placement
        if not self._initialized:
            if placement == self._start_placement:
                self._start_seen_frames += 1
                if self._start_seen_frames >= self._min_start_frames:
                    self._initialized = True
                    self._board = chess.Board()
                    self._pending_uci = None
                    self._pending_count = 0
                    self._in_missing_segment = False
                    self.missing_frames = 0
                    self.missing_segments = 0
                    self.missing_moves = 0
                    self._next_ply_index = 1
                    _log.info(
                        "[BASELINE OFFLINE] Initial position locked at frame %d",
                        frame_idx,
                    )
            else:
                if self._start_seen_frames > 0:
                    self._start_seen_frames = 0
            return None

        # From here on, a reference board must exist.
        assert self._board is not None

        # 2) No change in placement relative to reference board: no move
        if placement == self._board.board_fen():
            self._pending_uci = None
            self._pending_count = 0
            self._in_missing_segment = False
            return None

        # 3) Try to find a unique legal move from the reference board
        move = self._unique_legal_move_to(placement)
        if move is None:
            # Board changed but cannot be explained by a single legal move
            self._pending_uci = None
            self._pending_count = 0

            self.missing_frames += 1
            if not self._in_missing_segment:
                # New missing segment started
                self.missing_segments += 1
                self.missing_moves += 1
                self._next_ply_index += 1
                self._in_missing_segment = True

            _log.debug(
                "[BASELINE OFFLINE] cannot find unique move for frame %d",
                frame_idx,
            )
            return None

        # 4) We have a candidate move, no longer in a missing segment
        self._in_missing_segment = False
        uci = move.uci()

        # Debouncing over frames
        if self._pending_uci == uci:
            self._pending_count += 1
        else:
            self._pending_uci = uci
            self._pending_count = 1

        _log.debug(
            "[BASELINE OFFLINE] candidate %s (%d/%d) at frame %d",
            self._pending_uci,
            self._pending_count,
            self._confirm_frames,
            frame_idx,
        )

        if self._pending_count < self._confirm_frames:
            return None

        # 5) Confirmed move
        self._pending_uci = None
        self._pending_count = 0

        # Apply move on the reference board
        self._board.push(move)

        san: Optional[str] = None
        fen_after: Optional[str] = detection_state_to_fen(state)

        # Logical move index including missing moves
        ply_index = self._next_ply_index
        self._next_ply_index += 1

        return uci, san, fen_after, frame_idx, ply_index


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------


def run_baseline(
        states: List[DetectionState],
        *,
        video_fps: float | None = None,
        detector_fps: float | None = None,
) -> PipelineResult:
    """
    Offline single frame baseline with FEN based move reconstruction.

    Per frame FENs and reconstructed moves are collected in the returned
    PipelineResult. Detailed logging of selected FENs is handled by the
    comparison_results driver.

    If video_fps and detector_fps are provided and detector_fps < video_fps,
    a time based live sampling approximation is used. The detector is modeled
    as a worker that needs 1 / detector_fps seconds per frame and only accepts
    a new frame once it is free again. This simulates a queue of size 1 and
    is closer to real live processing.
    """
    runner = _BaselineRunner()

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

        # FEN snapshot for every frame
        fen_for_frame = detection_state_to_fen(state)
        frame_fens.append(fen_for_frame)

        _log.debug(
            "[BASELINE OFFLINE] processing frame %d/%d",
            idx + 1,
            total_frames,
        )

        info = runner.process_state(state, idx) if should_process else None
        if info is None:
            continue

        uci, san, _fen_after, frame_idx, _ply_index = info
        moves_uci.append(uci)
        moves_san.append(san or "")
        move_frames.append(frame_idx)

    runner.log_summary()

    return PipelineResult(
        frame_fens=frame_fens,
        moves_uci=moves_uci,
        moves_san=moves_san,
        move_frames=move_frames,
    )
