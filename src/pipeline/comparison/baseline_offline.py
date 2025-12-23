from __future__ import annotations

"""
Offline single frame baseline for the comparison pipeline.

This module takes DetectionState sequences from the single frame baseline,
reconstructs moves from FEN snapshots, and returns a unified PipelineResult
for evaluation.
"""

from typing import List, Optional

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

import chess


def _board_to_occupancy(board: chess.Board) -> dict[str, bool]:
    """
    Occupancy of a python-chess board as {square_name: bool}.
    """
    occ: dict[str, bool] = {}
    for sq in chess.SQUARES:
        name = chess.square_name(sq)
        occ[name] = board.piece_at(sq) is not None
    return occ


def _normalize_occupancy(occ: dict[str, bool]) -> dict[str, bool]:
    """
    Return a normalized occupancy map for all 64 squares.
    Missing keys are treated as empty squares.
    """
    norm: dict[str, bool] = {}
    for sq in chess.SQUARES:
        name = chess.square_name(sq)
        norm[name] = bool(occ.get(name, False))
    return norm


# ---------------------------------------------------------------------------
# Baseline runner: reconstruct moves from FEN snapshots
# ---------------------------------------------------------------------------


class _BaselineRunner:
    """
    Offline move reconstruction for the single-frame baseline.

    Design:

      - The detector is strictly single frame (unchanged).
      - For evaluation, we use the same idea as the live baseline:
        we only react to *stable* FEN plateaus, not to every noisy frame.
      - When a new stable plateau appears, we try to explain the change
        from the current reference board by exactly one legal move.
      - Move matching uses *occupancy* equality, which is much more
        robust to piece type noise than full FEN string equality.
      - If no unique legal move is found, we count a missing move and
        still sync the internal board to the detected FEN, so errors
        can accumulate, which is what your methods section describes.
    """

    def __init__(self) -> None:
        start_board = chess.Board()
        self._start_occ: dict[str, bool] = _board_to_occupancy(start_board)
        self._start_seen_frames: int = 0
        self._initialized: bool = False

        # Frames required to accept the start position
        self._min_start_frames: int = config.START_MIN_CONFIRM_FRAMES

        # Frames with identical placement before we treat it as stable
        self._min_stable_frames: int = config.FEN_MIN_STABLE_FRAMES

        # Current reference board (updated on confirmed or forced moves)
        self._board: Optional[chess.Board] = None

        # Stability tracking for placements
        self._last_seen_placement: Optional[str] = None
        self._same_seen_frames: int = 0

        # Last committed stable placement
        self._committed_placement: Optional[str] = None

        # Missing move statistics
        self.missing_frames: int = 0
        self.missing_segments: int = 0
        self.missing_moves: int = 0
        self._in_missing_segment: bool = False

        # Logical move index including missing moves
        self._next_ply_index: int = 1

    @property
    def initialized(self) -> bool:
        return self._initialized

    def log_summary(self) -> None:
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

    def _unique_legal_move_to(self, target_occ_raw: dict[str, bool]) -> Optional[chess.Move]:
        """
        Return the unique legal move that transforms the current reference
        board into the target occupancy mask, or None if there is none or
        more than one.

        Uses occupancy only, not full board_fen equality, so piece swaps
        in the detector output do not immediately kill an otherwise valid move.
        """
        if self._board is None:
            return None

        board = self._board
        target_occ = _normalize_occupancy(target_occ_raw)
        candidates: list[chess.Move] = []

        for move in board.legal_moves:
            tmp = board.copy()
            tmp.push(move)
            tmp_occ = _board_to_occupancy(tmp)

            if tmp_occ == target_occ:
                candidates.append(move)
                if len(candidates) > 1:
                    # Ambiguous transformation, treat as not reconstructable
                    return None

        if len(candidates) == 1:
            return candidates[0]
        return None

    def _mark_missing_plateau(self) -> None:
        """
        Update missing statistics for a stable plateau that cannot be
        explained by a single legal move.
        """
        self.missing_frames += 1
        if not self._in_missing_segment:
            self._in_missing_segment = True
            self.missing_segments += 1
            self.missing_moves += 1
            self._next_ply_index += 1

    def process_state(
            self,
            state: DetectionState,
            frame_idx: int,
    ) -> Optional[tuple[str, Optional[str], Optional[str], int, int]]:
        """
        Process a DetectionState for a potentially stable frame.

        Returns (uci, san, fen_after, frame_idx, ply_index) once a move
        is emitted for a *new stable plateau*. Returns None otherwise.

        san is left as None since reconstruction is driven only by FEN
        snapshots. fen_after is the FEN derived from detections.
        ply_index is the logical move index including missing moves.
        """
        placement = detection_state_to_placement(state)
        fen_after = detection_state_to_fen(state)
        curr_occ = _normalize_occupancy(state.occupancy)

        # 1) Detect the start position based on occupancy only
        if not self._initialized:
            same_start = all(
                curr_occ[sq] == want for sq, want in self._start_occ.items()
            )
            if same_start:
                self._start_seen_frames += 1
                if self._start_seen_frames >= self._min_start_frames:
                    self._initialized = True
                    self._board = chess.Board()
                    self._last_seen_placement = self._board.board_fen()
                    self._same_seen_frames = 0
                    self._committed_placement = None
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

        assert self._board is not None

        # 2) Stability tracking on placement, like in the live baseline
        if placement == self._last_seen_placement:
            self._same_seen_frames += 1
        else:
            self._last_seen_placement = placement
            self._same_seen_frames = 1

        # Not yet stable
        if self._same_seen_frames < self._min_stable_frames:
            return None

        # Plateau already committed
        if placement == self._committed_placement:
            return None

        # New stable plateau
        self._committed_placement = placement

        # 3) Check occupancy difference relative to current reference board
        ref_occ = _board_to_occupancy(self._board)

        if ref_occ == curr_occ:
            # Stable plateau, but no change in occupancy -> no move
            self._in_missing_segment = False
            return None

        # 4) Try to find a unique legal move that explains this plateau
        move = self._unique_legal_move_to(curr_occ)
        if move is None:
            # Cannot explain plateau by a single legal move
            self._mark_missing_plateau()
            _log.debug(
                "[BASELINE OFFLINE] cannot find unique move for plateau at frame %d",
                frame_idx,
            )

            # Still follow the detector so that future moves reflect its state
            try:
                self._board = chess.Board(fen_after)
            except ValueError:
                # Broken FEN, keep old board and hope it recovers later
                _log.debug(
                    "[BASELINE OFFLINE] invalid FEN at frame %d: %s",
                    frame_idx,
                    fen_after,
                )
            self._last_seen_placement = self._board.board_fen()
            return None

        # 5) Confirmed move between stable plateaus
        self._in_missing_segment = False
        self._board.push(move)
        self._last_seen_placement = self._board.board_fen()

        uci = move.uci()
        san: Optional[str] = None
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
