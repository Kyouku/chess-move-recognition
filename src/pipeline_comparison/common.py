from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class PipelineResult:
    """
    Unified result type returned by offline pipeline runners.

    - frame_fens: model FEN after processing each frame (same length as input)
    - moves_uci: detected moves in UCI notation, in order
    - moves_san: detected moves in SAN notation, in order (may be empty strings)
    - move_frames: frame indices at which the corresponding moves were committed
    """

    frame_fens: List[str]
    moves_uci: List[str]
    moves_san: List[str]
    move_frames: List[int]
