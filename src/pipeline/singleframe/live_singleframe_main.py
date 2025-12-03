from __future__ import annotations

from typing import List, Tuple, Dict, Optional

import chess

from src import config
from src.pipeline.live_base import (
    BaseLivePipeline,
    get_capture_source,
    CaptureSource,
)
from src.stage3.baseline_single_frame import SingleFrameBaseline
from src.types import PieceDetection, DetectionState


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
            window_name="Baseline",
        )

        self.baseline = SingleFrameBaseline()

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
        Gating and single frame move inference:
          first wait until we have seen the initial position for N frames
          once initialized, run SingleFrameBaseline on each latest state
        """
        curr_occ = state.occupancy

        # 1) Wait for the start position to be seen for N consecutive frames
        if not self._initialized:
            is_match = all(
                self._start_occ[sq] == bool(curr_occ.get(sq, False))
                for sq in self._start_occ
            )
            if is_match:
                self._start_seen_frames += 1
                if self._start_seen_frames >= self._min_confirm_frames:
                    self._initialized = True
                    self.baseline.reset()
            else:
                if self._start_seen_frames > 0:
                    # reset the counter on mismatch
                    self._start_seen_frames = 0
            return

        # 2) Only infer moves after initialization
        dets = convert_state_to_piece_detections(state)
        move_info = self.baseline.update(dets)
        if move_info is not None:
            # Could be logged or emitted if needed; keep minimal behavior
            pass


def main() -> None:
    src = get_capture_source()
    pipeline = SingleFramePipeline(source=src)
    pipeline.run()


if __name__ == "__main__":
    main()
