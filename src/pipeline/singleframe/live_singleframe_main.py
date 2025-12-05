from __future__ import annotations

from typing import List, Tuple, Dict, Optional

import chess

from src import config
from src.app_logging import get_logger
from src.pipeline.live_base import (
    BaseLivePipeline,
    get_capture_source,
    CaptureSource,
)
from src.stage3.baseline_single_frame import SingleFrameBaseline
from src.types import PieceDetection, DetectionState

_log = get_logger(__name__)


def _state_to_fen(state: DetectionState) -> str:
    """
    Build a FEN string directly from detections (occupancy + labels),
    without any legality checks. Unknown or missing labels are treated
    as empty squares. Non-position fields are filled with placeholders.
    """
    files = ["a", "b", "c", "d", "e", "f", "g", "h"]

    def code_to_fen_char(code: Optional[str]) -> Optional[str]:
        if not code or len(code) < 2:
            return None
        c0 = code[0].lower()
        c1 = code[1].lower()
        if c0 not in ("w", "b"):
            return None
        if c1 not in ("p", "n", "b", "r", "q", "k"):
            return None
        return c1.upper() if c0 == "w" else c1

    rows: List[str] = []
    for rank_idx in range(7, -1, -1):
        run = 0
        row_chars: List[str] = []
        for file_idx in range(8):
            sq = f"{files[file_idx]}{rank_idx + 1}"
            occ = bool(state.occupancy.get(sq, False))
            letter: Optional[str] = None
            if occ:
                letter = code_to_fen_char(state.pieces.get(sq))
            if letter is None:
                run += 1
            else:
                if run > 0:
                    row_chars.append(str(run))
                    run = 0
                row_chars.append(letter)
        if run > 0:
            row_chars.append(str(run))
        rows.append("".join(row_chars))

    placement = "/".join(rows)
    return f"{placement} w - - 0 1"


def _append_move_log(uci: str, san: Optional[str], fen_after: Optional[str]) -> None:
    """Append a single move to the CSV-like log file as uci;san;fen. Errors are non-fatal."""
    try:
        # Ensure directory exists
        try:
            config.MOVES_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        san_str = san or ""
        fen_str = fen_after or ""
        with open(config.MOVES_LOG_PATH, "a", encoding="utf8") as f:
            f.write(f"{uci};{san_str};{fen_str}\n")
    except OSError as e:
        _log.warning(
            "Could not append move to log file %s: %s",
            config.MOVES_LOG_PATH,
            e,
        )


def _parse_piece_code(code: str) -> Optional[Tuple[str, str]]:
    """
    Map compact code like "wP"/"bN" to (color, piece_type) expected by PieceDetection.
    Returns None if the code cannot be parsed.
    """
    if not code or len(code) < 2:
        return None
    color_char = code[0].lower()
    piece_char = code[1].lower()
    if color_char not in ("w", "b"):
        return None
    if piece_char not in ("p", "n", "b", "r", "q", "k"):
        return None
    color = "white" if color_char == "w" else "black"
    return color, piece_char


def convert_to_piece_detections(
        detections_raw: Tuple[
            Dict[str, bool],
            Dict[str, Optional[str]],
            Dict[str, Optional[Tuple[float, float, float, float]]],
            Dict[str, Optional[float]],
        ]
) -> List[PieceDetection]:
    """
    Convert PieceDetector.detect() outputs to a list of PieceDetection objects
    (one per occupied square with a known piece code).
    """
    occupancy, pieces, _boxes, confs = detections_raw
    out: List[PieceDetection] = []
    for sq, occ in occupancy.items():
        if not occ:
            continue
        code = pieces.get(sq)
        if not code:
            continue
        parsed = _parse_piece_code(code)
        if not parsed:
            continue
        color, piece_type = parsed
        score = float(confs.get(sq) or 0.0)
        out.append(
            PieceDetection(
                square=sq,
                color=color,
                piece_type=piece_type,
                score=score,
            )
        )
    return out


def convert_state_to_piece_detections(state: DetectionState) -> List[PieceDetection]:
    """Convert a DetectionState to a list of PieceDetection items."""
    return convert_to_piece_detections(
        (
            state.occupancy,
            state.pieces,
            state.boxes,
            state.confidences,
        )
    )


class SingleFramePipeline(BaseLivePipeline):
    """
    Live app that reuses Stage 1 + 2 from BaseLivePipeline and performs
    single frame move inference in the main thread via SingleFrameBaseline.
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

        self.baseline = SingleFrameBaseline()

        # Note: python-chess is only used for deriving the start position occupancy.
        # No legality checks, SAN/FEN generation or game state tracking during runtime.

        # Auto initialization: wait until the start position is stably detected
        self._start_occ: Dict[str, bool] = self._board_to_occupancy(chess.Board())
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

    # ------------------------------------------------------------------
    # BaseLivePipeline hooks
    # ------------------------------------------------------------------

    def handle_detection_state(self, state: DetectionState) -> None:
        """
        Startup gating (always on) + single-frame move inference.

        Wait until the starting position occupancy is observed for
        MOVE_MIN_CONFIRM_FRAMES consecutive frames before attempting to infer
        moves. This mirrors the multistage pipeline's startup behavior. Once
        initialized, run inference on each state via SingleFrameBaseline.
        """
        # Always-on startup gating like multistage pipeline (allowed to use python-chess)
        if not self._initialized:
            curr_occ = state.occupancy
            is_match = all(self._start_occ[sq] == bool(curr_occ.get(sq, False)) for sq in self._start_occ)
            if is_match:
                self._start_seen_frames += 1
                if self._start_seen_frames >= self._min_confirm_frames:
                    self._initialized = True
                    self._start_seen_frames = 0
                    # Initialize baseline reference at start position, including labels
                    # so that a capture as the very first move can be detected.
                    self.baseline.reset(initial_occ=self._start_occ, initial_pieces=state.pieces)
                    _log.info("[BASELINE] Initial position locked. Move detection enabled.")
                    # Skip inference this frame (consistent with multistage behavior)
                    return
            else:
                if self._start_seen_frames > 0:
                    self._start_seen_frames = 0
            # Still waiting for stable start position
            return

        # Move inference once initialized (no python-chess for detection). Use occupancy/labels diff only.
        curr_occ = state.occupancy
        curr_pieces = state.pieces
        proposed_uci = self.baseline.update_state(curr_occ, curr_pieces)

        if proposed_uci is None:
            # No candidate this frame
            return

        # Immediately accept and commit the proposed move (no debouncing)
        uci = proposed_uci

        # Commit baseline state to the current occupancy/labels
        self.baseline.commit(curr_occ, curr_pieces)

        # No legality checks or SAN here. Derive FEN from detections directly.
        san: Optional[str] = None
        fen_after: Optional[str] = _state_to_fen(state)

        # Log to console
        if san is not None:
            msg_lines = [
                "==================================================",
                f"[BASELINE] Move detected: {san} ({uci})",
                f"[BASELINE] FEN after: {fen_after}",
                "==================================================",
            ]
        else:
            msg_lines = [
                "==================================================",
                f"[BASELINE] Move detected (UCI): {uci}",
                f"[BASELINE] FEN after: {fen_after}",
                "==================================================",
            ]
        _log.info("\n".join(msg_lines))

        # Append to log file (uci;san;fen)
        _append_move_log(uci, san, fen_after)

        # No game-over handling via python-chess in single-frame baseline mode.

    def on_stop(self) -> None:
        # Nothing to persist here in single-frame baseline mode
        return None


def main() -> None:
    src = get_capture_source()
    pipeline = SingleFramePipeline(source=src)
    pipeline.run()


if __name__ == "__main__":
    main()
