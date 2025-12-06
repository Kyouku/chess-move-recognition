from __future__ import annotations

"""
Shared dataclasses for all pipelines.

These types are used as the common interface between single frame and
multi stage pipelines as well as live and offline processing steps.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import chess


@dataclass
class DetectionState:
    """
    Snapshot of detector output for one frame.

    Shared between all pipelines to represent board occupancy, piece identity
    and bounding boxes in a pipeline agnostic way.
    """

    occupancy: Dict[str, bool]
    pieces: Dict[str, Optional[str]]
    boxes: Dict[str, Optional[Tuple[float, float, float, float]]]
    confidences: Dict[str, Optional[float]]


@dataclass
class PieceDetection:
    """
    Generic detection for a single square.

    Shared across pipelines so that detectors can report their findings
    in a consistent format.

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

    Used by all pipelines to pass move metadata to logging, evaluation
    and live control logic.
    """

    move: chess.Move
    san: str
    fen_after: str
    # Optional end-of-game metadata (used in live pipeline control)
    is_checkmate: bool = False
    is_game_over: bool = False
    result: Optional[str] = None
