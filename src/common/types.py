from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import chess


@dataclass
class DetectionState:
    occupancy: Dict[str, bool]
    pieces: Dict[str, Optional[str]]
    boxes: Dict[str, Optional[Tuple[float, float, float, float]]]
    confidences: Dict[str, Optional[float]]


@dataclass
class PieceDetection:
    """
    Generic detection for a single square.

    square: algebraic square name, for example "e4"
    color: "white" or "black"
    piece_type: one of "p", "n", "b", "r", "q", "k" (lowercase)
    score: confidence in [0, 1]
    """

    square: str
    color: str
    piece_type: str
    score: float


@dataclass
class MoveInfo:
    """
    Information about an inferred move.
    """

    move: chess.Move
    san: str
    fen_after: str
    # Optional end-of-game metadata (added for live pipeline control)
    is_checkmate: bool = False
    is_game_over: bool = False
    result: Optional[str] = None
