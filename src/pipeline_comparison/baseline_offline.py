from __future__ import annotations

from typing import Dict, List, Optional

import chess

from src import config
from src.app_logging import get_logger
from src.pipeline_comparison.common import PipelineResult
from src.stage3.baseline_single_frame import SingleFrameBaseline
from src.types import DetectionState

_log = get_logger(__name__)


class _BaselineRunner:
    """
    Offline version of SingleFramePipeline.handle_detection_state.

    Uses SingleFrameBaseline for move proposals and python chess only
    for SAN and FEN output.
    """

    def __init__(self) -> None:
        self.baseline = SingleFrameBaseline()
        self.board = chess.Board()
        self.moves: List[str] = []

        self._start_occ: Dict[str, bool] = self._board_to_occupancy(
            chess.Board()
        )
        self._start_seen_frames: int = 0
        self._initialized: bool = False
        self._min_confirm_frames: int = int(
            getattr(config, "MOVE_MIN_CONFIRM_FRAMES", 2)
        )

    @staticmethod
    def _board_to_occupancy(board: chess.Board) -> Dict[str, bool]:
        occ: Dict[str, bool] = {}
        for sq in chess.SQUARES:
            occ[chess.square_name(sq)] = board.piece_at(sq) is not None
        return occ

    def process_state(
            self,
            state: DetectionState,
            frame_idx: int,
    ) -> Optional[tuple[str, Optional[str], Optional[str], int]]:
        """
        Process a single DetectionState.

        Returns tuple (uci, san, fen_after, frame_idx) if a move is detected,
        otherwise None.
        """
        if not self._initialized:
            curr_occ = state.occupancy
            is_match = all(
                self._start_occ[sq] == bool(curr_occ.get(sq, False))
                for sq in self._start_occ
            )
            if is_match:
                self._start_seen_frames += 1
                if self._start_seen_frames >= self._min_confirm_frames:
                    self._initialized = True
                    self._start_seen_frames = 0
                    self.baseline.reset(
                        initial_occ=self._start_occ,
                        initial_pieces=state.pieces,
                    )
                    self.board = chess.Board()
                    self.moves.clear()
                    _log.info(
                        "[BASELINE OFFLINE] Initial position locked at frame %d",
                        frame_idx,
                    )
                    return None
            else:
                if self._start_seen_frames > 0:
                    self._start_seen_frames = 0
            return None

        curr_occ = state.occupancy
        curr_pieces = state.pieces
        proposed_uci = self.baseline.update_state(curr_occ, curr_pieces)

        if proposed_uci is None:
            return None

        # Accept immediately, no debouncing in offline baseline
        uci = proposed_uci
        self.baseline.commit(curr_occ, curr_pieces)

        san: Optional[str] = None
        fen_after: Optional[str] = None
        try:
            move_obj = chess.Move.from_uci(uci)
            if move_obj in self.board.legal_moves:
                san = self.board.san(move_obj)
                self.board.push(move_obj)
                fen_after = self.board.fen()
                self.moves.append(san)
        except (ValueError, chess.IllegalMoveError) as exc:
            # Invalid UCI or illegal move for current board; ignore gracefully.
            _log.debug("[BASELINE OFFLINE] Ignoring invalid move %s: %s", uci, exc)

        if san is not None:
            _log.info(
                "[BASELINE OFFLINE] frame %d move %s (%s)",
                frame_idx,
                san,
                uci,
            )
        else:
            _log.info(
                "[BASELINE OFFLINE] frame %d move (UCI) %s",
                frame_idx,
                uci,
            )

        return uci, san, fen_after, frame_idx


def run_baseline(states: List[DetectionState]) -> PipelineResult:
    """
    Run the single frame baseline on a list of DetectionState objects.

    Returns per frame model FEN plus the detected move sequence.
    """
    runner = _BaselineRunner()

    frame_fens: List[str] = []
    moves_uci: List[str] = []
    moves_san: List[str] = []
    move_frames: List[int] = []

    total_frames = len(states)
    for idx, state in enumerate(states):
        _log.debug("[BASELINE OFFLINE] processing frame %d/%d", idx + 1, total_frames)
        info = runner.process_state(state, idx)
        frame_fens.append(runner.board.fen())
        if info is None:
            continue

        uci, san, fen_after, frame_idx = info
        moves_uci.append(uci)
        moves_san.append(san or "")
        move_frames.append(frame_idx)

    return PipelineResult(
        frame_fens=frame_fens,
        moves_uci=moves_uci,
        moves_san=moves_san,
        move_frames=move_frames,
    )
