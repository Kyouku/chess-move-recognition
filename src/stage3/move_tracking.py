from __future__ import annotations

import queue
import re
import threading
from typing import Dict, List, Optional, Iterable, Tuple

import chess

from src.app_logging import get_logger
from src.types import PieceDetection, MoveInfo, DetectionState

_log = get_logger(__name__)


def _evidence_from_occupancy(occ: Dict[str, bool]) -> Dict[str, float]:
    """
    Map occupancy bools to evidence floats.
    True -> 1.0, False -> 0.0.
    """
    ev: Dict[str, float] = {}
    for sq in chess.SQUARES:
        name = chess.square_name(sq)
        ev[name] = 1.0 if occ.get(name, False) else 0.0
    return ev


class BoardStateFilter:
    """
    Simple temporal filter for board occupancy per square.

    For each of the 64 squares we keep an occupancy probability p in [0, 1].
    Each new frame provides evidence e in [0, 1] for that square being occupied.

        p_new = (1 - alpha) * p_old + alpha * e

    Then we threshold p to get a binary occupancy map.
    """

    def __init__(self, alpha: float = 0.6, occ_threshold: float = 0.6) -> None:
        self.alpha = alpha
        self.occ_threshold = occ_threshold

        self._p_occ: List[float] = [0.0] * 64
        self._prev_binary: List[bool] = [False] * 64
        self._current_binary: List[bool] = [False] * 64

    def reset(self) -> None:
        """
        Reset filter state.
        """
        self._p_occ = [0.0] * 64
        self._prev_binary = [False] * 64
        self._current_binary = [False] * 64

    def update_from_detections(
            self,
            detections: Iterable[PieceDetection],
    ) -> Tuple[Dict[str, bool], Dict[str, bool]]:
        """
        Update filter from piece detections.

        Each detection on a square counts as full evidence 1.0 for occupied.
        """
        evidence: Dict[str, float] = {
            chess.square_name(sq): 0.0 for sq in chess.SQUARES
        }

        for det in detections:
            s = det.square
            if s in evidence:
                evidence[s] = 1.0

        return self._update(evidence)

    def update_from_occupancy(
            self,
            occupancy: Dict[str, bool],
    ) -> Tuple[Dict[str, bool], Dict[str, bool]]:
        """
        Convenience wrapper for occupancy only evidence.
        """
        ev = _evidence_from_occupancy(occupancy)
        return self._update(ev)

    def _update(
            self,
            evidence: Dict[str, float],
    ) -> Tuple[Dict[str, bool], Dict[str, bool]]:
        """
        Low level update with evidence in [0, 1] per square.
        """
        self._prev_binary = self._current_binary[:]

        for sq in chess.SQUARES:
            name = chess.square_name(sq)
            e = float(evidence.get(name, 0.0))
            e = max(0.0, min(1.0, e))
            p_old = self._p_occ[sq]
            p_new = (1.0 - self.alpha) * p_old + self.alpha * e
            self._p_occ[sq] = p_new

        self._current_binary = [p >= self.occ_threshold for p in self._p_occ]

        prev = {chess.square_name(sq): self._prev_binary[sq] for sq in chess.SQUARES}
        curr = {chess.square_name(sq): self._current_binary[sq] for sq in chess.SQUARES}
        return prev, curr


def _board_to_occupancy(board: chess.Board) -> Dict[str, bool]:
    occ: Dict[str, bool] = {}
    for sq in chess.SQUARES:
        name = chess.square_name(sq)
        occ[name] = board.piece_at(sq) is not None
    return occ


def _sorted_squares() -> List[str]:
    return [chess.square_name(sq) for sq in chess.SQUARES]


class MoveTracker:
    """
    Keeps track of the chess game state using python chess and
    derives moves from temporally filtered occupancy maps.

    The dictionaries passed in (occupancy, pieces) use the same
    algebraic square names as python chess, for example "e4".
    """

    def __init__(
            self,
            alpha: float = 0.6,
            occ_threshold: float = 0.6,
            min_confirm_frames: int = 2,
            debug: bool = False,
    ) -> None:
        self._start_board = chess.Board()
        self._start_occ = _board_to_occupancy(self._start_board)

        self.board = chess.Board()

        self.filter = BoardStateFilter(alpha=alpha, occ_threshold=occ_threshold)
        self.min_confirm_frames = min_confirm_frames

        self._pending_candidates: Optional[List[chess.Move]] = None
        self._pending_frames: int = 0

        self._waiting_for_start: bool = True
        self._initialized: bool = False
        self._start_seen_frames: int = 0

        self._debug: bool = debug

        self._frame_index: int = 0
        self._last_occ_for_debug: Optional[Dict[str, bool]] = None
        self._last_pieces_for_debug: Optional[Dict[str, Optional[str]]] = None

    @property
    def is_initialized(self) -> bool:
        """
        True once the tracker has locked on the initial position
        and started tracking moves.
        """
        return self._initialized

    def _debug_print_board(self, context: str) -> None:
        if not self._debug:
            return
        _log.debug("[MoveTracker] %s board:\n%s", context, self.board)
        _log.debug("[MoveTracker] FEN: %s", self.board.fen())

    def _debug_dump_state(
            self,
            tag: str,
            frame_idx: int,
            occ: Dict[str, bool],
            pieces: Dict[str, Optional[str]],
    ) -> None:
        if not self._debug:
            return

        _log.debug("[MoveTracker] %s - frame %d", tag, frame_idx)
        for name in _sorted_squares():
            o = occ.get(name, False)
            lbl = pieces.get(name)
            lbl_str = "." if lbl is None else str(lbl)
            _log.debug("  %s: occ=%d label=%s", name, int(bool(o)), lbl_str)
        _log.debug("[MoveTracker] end of state dump")

    @staticmethod
    def _normalize_label(label: Optional[str]) -> Optional[str]:
        if label is None:
            return None
        lower = label.strip().lower()
        lower = re.sub(r"\s+", "", lower)
        return lower

    @staticmethod
    def _piece_code(piece: chess.Piece) -> str:
        color = "w" if piece.color == chess.WHITE else "b"
        kind = {
            chess.PAWN: "p",
            chess.KNIGHT: "n",
            chess.BISHOP: "b",
            chess.ROOK: "r",
            chess.QUEEN: "q",
            chess.KING: "k",
        }[piece.piece_type]
        return color + kind

    def _label_matches_piece(
            self,
            label: Optional[str],
            piece: chess.Piece,
    ) -> bool:
        """
        Check if a YOLO style label matches a chess.Piece.

        Supported formats include for example:
          "white-pawn", "black-queen", "wp", "bq".
        """
        lower = self._normalize_label(label)
        if not lower:
            return False

        code = self._piece_code(piece)
        if lower == code:
            return True

        stripped = re.sub(r"[^a-z]", "", lower)

        color_ok = (
                ("white" in stripped and piece.color == chess.WHITE)
                or ("black" in stripped and piece.color == chess.BLACK)
                or (stripped.startswith("w") and piece.color == chess.WHITE)
                or (stripped.startswith("b") and piece.color == chess.BLACK)
        )

        kind = piece.piece_type
        if kind == chess.PAWN:
            type_ok = "pawn" in stripped or stripped.endswith("p")
        elif kind == chess.KNIGHT:
            type_ok = "knight" in stripped or stripped.endswith("n")
        elif kind == chess.BISHOP:
            type_ok = "bishop" in stripped or stripped.endswith("b")
        elif kind == chess.ROOK:
            type_ok = "rook" in stripped or stripped.endswith("r")
        elif kind == chess.QUEEN:
            type_ok = "queen" in stripped or stripped.endswith("q")
        elif kind == chess.KING:
            type_ok = "king" in stripped or stripped.endswith("k")
        else:
            type_ok = False

        return color_ok and type_ok

    def _position_matches(
            self,
            board: chess.Board,
            occ: Dict[str, bool],
            pieces: Dict[str, Optional[str]],
    ) -> bool:
        """
        Check whether a board state matches the observed occupancy and piece labels.

        Occupancy must match exactly.
        Labels are only checked where board actually has a piece and a label is present.
        """
        for sq in chess.SQUARES:
            name = chess.square_name(sq)
            board_piece = board.piece_at(sq)

            occ_observed = occ.get(name, False)

            if board_piece is None:
                if occ_observed:
                    return False
                continue

            if not occ_observed:
                return False

            label = pieces.get(name)
            if label and not self._label_matches_piece(label, board_piece):
                return False

        return True

    def _candidate_moves_for_target_occupancy(
            self,
            target_occ: Dict[str, bool],
    ) -> List[chess.Move]:
        """
        Search among all legal moves for those that produce the observed
        occupancy pattern only.
        """
        candidates: List[chess.Move] = []

        for move in self.board.legal_moves:
            tmp = self.board.copy()
            tmp.push(move)
            occ_after = _board_to_occupancy(tmp)
            if occ_after == target_occ:
                candidates.append(move)

        if self._debug:
            _log.debug(
                "[MoveTracker] occupancy only candidate search: found %d moves",
                len(candidates),
            )

        return candidates

    def _candidate_moves_for_target_state(
            self,
            target_occ: Dict[str, bool],
            target_pieces: Dict[str, Optional[str]],
    ) -> List[chess.Move]:
        """
        Search among all legal moves for those that produce the observed
        board state (occupancy plus piece labels).
        """
        candidates: List[chess.Move] = []

        for move in self.board.legal_moves:
            tmp = self.board.copy()
            tmp.push(move)

            if not self._position_matches(tmp, target_occ, target_pieces):
                continue

            candidates.append(move)

        if self._debug:
            _log.debug(
                "[MoveTracker] full state candidate search: found %d moves",
                len(candidates),
            )

        return candidates

    def _confirm_or_store_candidates(
            self,
            candidates: List[chess.Move],
    ) -> Optional[chess.Move]:
        """
        Confirmation logic with hysteresis over multiple frames.
        """
        if not candidates:
            if self._debug and self._pending_candidates:
                _log.debug("[MoveTracker] candidates dropped, clearing pending state")
            self._pending_candidates = None
            self._pending_frames = 0
            return None

        new_set = frozenset(m.uci() for m in candidates)
        old_set = (
            frozenset(m.uci() for m in self._pending_candidates)
            if self._pending_candidates
            else None
        )

        if old_set is not None and new_set == old_set:
            self._pending_frames += 1
        else:
            self._pending_candidates = candidates
            self._pending_frames = 1

        if self._debug:
            _log.debug(
                "[MoveTracker] pending candidates %s frames=%d",
                sorted(new_set),
                self._pending_frames,
            )

        if len(candidates) == 1 and self._pending_frames >= self.min_confirm_frames:
            move = candidates[0]
            if self._debug:
                _log.debug(
                    "[MoveTracker] confirming move %s after %d frames",
                    move.uci(),
                    self._pending_frames,
                )
            self._pending_candidates = None
            self._pending_frames = 0
            return move

        return None

    def _handle_start_detection(
            self,
            curr_occ: Dict[str, bool],
    ) -> bool:
        """
        Handle auto initialization based on the filtered occupancy.

        Returns True if we are still in the waiting phase or just finished
        initialization and should not search for moves in this frame.
        """
        if self._initialized or not self._waiting_for_start:
            return False

        if curr_occ == self._start_occ:
            self._start_seen_frames += 1
            if self._debug:
                _log.debug(
                    "[MoveTracker] start position match frame %d/%d",
                    self._start_seen_frames,
                    self.min_confirm_frames,
                )
        else:
            if self._start_seen_frames > 0 and self._debug:
                diff = [
                    sq
                    for sq in self._start_occ
                    if self._start_occ[sq] != curr_occ.get(sq, False)
                ]
                _log.debug(
                    "[MoveTracker] start position mismatch, resetting counter, "
                    "differing squares: %s",
                    ", ".join(diff[:8]),
                )
            self._start_seen_frames = 0

        if self._start_seen_frames >= self.min_confirm_frames:
            if self._debug:
                _log.debug("[MoveTracker] start position locked, tracker initialized")
            self._initialized = True
            self._waiting_for_start = False

            self.filter.reset()
            self._pending_candidates = None
            self._pending_frames = 0

            self.board = chess.Board()
            self._debug_print_board("initial")

        return True

    def _finalize_move(
            self,
            move: chess.Move,
            curr_occ: Dict[str, bool],
            pieces: Optional[Dict[str, Optional[str]]] = None,
    ) -> MoveInfo:
        """
        Common tail logic after a move has been selected/confirmed.

        - Compute SAN
        - Push move to board and compute FEN
        - Debug logging and board dump
        - Update last debug state caches
        - Return MoveInfo
        """
        san = self.board.san(move)
        self.board.push(move)
        fen_after = self.board.fen()
        if self._debug:
            _log.debug(
                "[MoveTracker] committed move %s (%s), new FEN: %s",
                move.uci(),
                san,
                fen_after,
            )
            self._debug_print_board("after move")

        self._last_occ_for_debug = dict(curr_occ)
        self._last_pieces_for_debug = (
            dict(pieces) if pieces is not None else None
        )
        # Check game status after the move
        is_mate = self.board.is_checkmate()
        is_over = self.board.is_game_over(claim_draw=True)
        result = self.board.result(claim_draw=True) if is_over else None

        return MoveInfo(
            move=move,
            san=san,
            fen_after=fen_after,
            is_checkmate=is_mate,
            is_game_over=is_over,
            result=result,
        )

    def reset_to_start(self) -> None:
        """
        Reset tracker state so the pipeline waits for the initial position
        again. This clears temporal filters, pending candidates and returns the
        internal board to the initial setup.
        """
        if self._debug:
            _log.debug("[MoveTracker] reset_to_start() called")

        # Reset chess position and temporal filter
        self.board = chess.Board()
        self.filter.reset()

        # Clear tracking state
        self._pending_candidates = None
        self._pending_frames = 0
        self._frame_index = 0
        self._last_occ_for_debug = None
        self._last_pieces_for_debug = None

        # Return to waiting for start position
        self._waiting_for_start = True
        self._initialized = False
        self._start_seen_frames = 0

        self._debug_print_board("reset-to-start initial")

    def update_from_detections(
            self,
            detections: Iterable[PieceDetection],
    ) -> Optional[MoveInfo]:
        """
        Legacy API when you have piece level detections as PieceDetection.
        This only uses occupancy internally.
        """
        self._frame_index += 1
        prev_occ, curr_occ = self.filter.update_from_detections(detections)

        if self._handle_start_detection(curr_occ):
            return None

        if prev_occ == curr_occ and not self._pending_candidates:
            if self._debug:
                _log.debug(
                    "[MoveTracker] no change in filtered occupancy, "
                    "no pending candidates, skipping move search",
                )
            return None

        candidates = self._candidate_moves_for_target_occupancy(curr_occ)

        if self._debug and candidates:
            if self._last_occ_for_debug is not None:
                self._debug_dump_state(
                    "previous filtered state (detections)",
                    self._frame_index - 1,
                    self._last_occ_for_debug,
                    {},
                )
            self._debug_dump_state(
                "current filtered state (detections)",
                self._frame_index,
                curr_occ,
                {},
            )

        move = self._confirm_or_store_candidates(candidates)
        if move is None:
            self._last_occ_for_debug = dict(curr_occ)
            self._last_pieces_for_debug = None
            return None

        return self._finalize_move(move, curr_occ, None)

    def update_from_occupancy(
            self,
            occupancy: Dict[str, bool],
    ) -> Optional[MoveInfo]:
        """
        Legacy convenience wrapper for occupancy only detections.
        """
        self._frame_index += 1
        prev_occ, curr_occ = self.filter.update_from_occupancy(occupancy)

        if self._handle_start_detection(curr_occ):
            return None

        if prev_occ == curr_occ and not self._pending_candidates:
            if self._debug:
                _log.debug(
                    "[MoveTracker] no change in filtered occupancy, "
                    "no pending candidates, skipping move search",
                )
            return None

        candidates = self._candidate_moves_for_target_occupancy(curr_occ)

        if self._debug and candidates:
            if self._last_occ_for_debug is not None:
                self._debug_dump_state(
                    "previous filtered state (occupancy)",
                    self._frame_index - 1,
                    self._last_occ_for_debug,
                    {},
                )
            self._debug_dump_state(
                "current filtered state (occupancy)",
                self._frame_index,
                curr_occ,
                {},
            )

        move = self._confirm_or_store_candidates(candidates)
        if move is None:
            self._last_occ_for_debug = dict(curr_occ)
            self._last_pieces_for_debug = None
            return None

        return self._finalize_move(move, curr_occ, None)

    def update_from_state(
            self,
            occupancy: Dict[str, bool],
            pieces: Dict[str, Optional[str]],
    ) -> Optional[MoveInfo]:
        """
        Main realtime API.

        You pass in:
          occupancy: square -> bool
          pieces:    square -> YOLO label string or None

        Implementation detail:
          First search based on occupancy only.
          If there are multiple candidate moves, try to use labels
          as a soft filter. If labels do not help, fall back to
          pure occupancy candidates.
        """
        self._frame_index += 1
        prev_occ, curr_occ = self.filter.update_from_occupancy(occupancy)

        if self._debug:
            changed = [sq for sq in curr_occ if curr_occ[sq] != prev_occ.get(sq, False)]
            if changed:
                _log.debug(
                    "[MoveTracker] filtered occupancy changed at %d squares: %s",
                    len(changed),
                    ", ".join(changed[:16]),
                )
            else:
                _log.debug("[MoveTracker] filtered occupancy unchanged")

        if self._handle_start_detection(curr_occ):
            return None

        if prev_occ == curr_occ and not self._pending_candidates:
            if self._debug:
                _log.debug(
                    "[MoveTracker] initialized, no change in filtered "
                    "occupancy and no pending candidates, skipping",
                )
            self._last_occ_for_debug = dict(curr_occ)
            self._last_pieces_for_debug = dict(pieces)
            return None

        candidates = self._candidate_moves_for_target_occupancy(curr_occ)

        if len(candidates) > 1:
            # Use move-local label evidence to disambiguate.
            # Instead of demanding a full-board label match (too strict
            # and brittle with occasional mislabels), we only look at the
            # destination square label. If it exists and matches the piece
            # that would be on that square after the move, we consider it
            # supporting evidence for that candidate.
            label_filtered: List[chess.Move] = []
            for move in candidates:
                try:
                    tmp = self.board.copy()
                    tmp.push(move)
                    to_piece = tmp.piece_at(move.to_square)
                    to_sq_name = chess.square_name(move.to_square)
                    to_label = pieces.get(to_sq_name)
                    # If a label is present on the destination square, it must match.
                    # If there is no label, we don't reject the move (remain agnostic).
                    if to_label is None or (
                            to_piece is not None and self._label_matches_piece(to_label, to_piece)
                    ):
                        label_filtered.append(move)
                except (ValueError, AssertionError):
                    # Be conservative: if anything goes wrong with temp push,
                    # do not use label filtering for this move.
                    pass

            if label_filtered:
                if self._debug:
                    _log.debug(
                        "[MoveTracker] move-local label filter reduced candidates "
                        "from %d to %d",
                        len(candidates),
                        len(label_filtered),
                    )
                candidates = label_filtered
            else:
                if self._debug:
                    _log.debug(
                        "[MoveTracker] move-local label filter rejected all occupancy "
                        "candidates, keeping occupancy only result",
                    )

        if self._debug and candidates:
            if self._last_occ_for_debug is not None and self._last_pieces_for_debug:
                self._debug_dump_state(
                    "previous filtered state (state API)",
                    self._frame_index - 1,
                    self._last_occ_for_debug,
                    self._last_pieces_for_debug,
                )
            self._debug_dump_state(
                "current filtered state (state API)",
                self._frame_index,
                curr_occ,
                pieces,
            )

        move = self._confirm_or_store_candidates(candidates)
        if move is None:
            self._last_occ_for_debug = dict(curr_occ)
            self._last_pieces_for_debug = dict(pieces)
            return None

        return self._finalize_move(move, curr_occ, pieces)

    def update_from_detection_state(self, state: DetectionState) -> Optional[MoveInfo]:
        """
        Convenience wrapper to accept a DetectionState container produced by
        Stage 2 and delegate to the main update_from_state API.

        Keeps separation of concerns: Stage 3 consumes only logical state
        (occupancy + labels) and does not depend on raw boxes/confidences.
        """
        return self.update_from_state(state.occupancy, state.pieces)


class MoveTrackerWorker(threading.Thread):
    """
    Optional worker that runs MoveTracker in a background thread.

    detection thread puts (occupancy, pieces) states into input_queue.
    This worker consumes them and runs update_from_state.
    Confirmed moves are pushed into output_queue as MoveInfo.
    """

    def __init__(
            self,
            tracker: MoveTracker,
            input_queue: "queue.Queue[Tuple[Dict[str, bool], Dict[str, Optional[str]]]]",
            output_queue: "queue.Queue[MoveInfo]",
            stop_event: threading.Event,
            name: str = "MoveTrackerWorker",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.tracker = tracker
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event

    def run(self) -> None:
        _log.info("%s started", self.name)
        while not self.stop_event.is_set():
            try:
                occupancy, pieces = self.input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                info = self.tracker.update_from_state(occupancy, pieces)
                if info is not None:
                    try:
                        self.output_queue.put_nowait(info)
                    except queue.Full:
                        try:
                            _ = self.output_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.output_queue.put_nowait(info)
                        except queue.Full:
                            pass
            except (ValueError, RuntimeError, KeyError, AssertionError) as exc:
                _log.error("%s error: %s", self.name, exc, exc_info=True)

        _log.info("%s stopped", self.name)
