from __future__ import annotations

from typing import Optional, List

from src.common.types import DetectionState


def placement_from_fen(fen: str) -> str:
    """
    Return only the placement field (first field) of a FEN string.
    Safe for empty strings.
    """
    return fen.split()[0] if fen else ""


def detection_state_to_fen(state: DetectionState) -> str:
    """
    Build a FEN string directly from detections (occupancy and labels),
    without any legality checks.

    Unknown or missing labels are treated as empty squares.
    Non positional fields are filled with placeholders: "w - - 0 1".
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
    # Side to move, castling rights, en passant, and clocks are placeholders
    return f"{placement} w - - 0 1"


def detection_state_to_placement(state: DetectionState) -> str:
    """
    Return only the FEN placement field (first field) built from detections.
    """
    fen = detection_state_to_fen(state)
    return placement_from_fen(fen)
