from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def _square_name(rank_idx: int, file_idx: int) -> str:
    """
    Map (rank_idx, file_idx) to a square like "a1".

    Same convention as in PieceDetector:
      rank_idx: 0 top .. squares-1 bottom  -> a..h
      file_idx: 0 left .. squares-1 right  -> 1..8
    """
    file_char = chr(ord("a") + rank_idx)
    rank_char = str(file_idx + 1)
    return f"{file_char}{rank_char}"


def _compute_square_boxes(
        img_w: int,
        img_h: int,
        squares: int,
        margin_squares: float,
) -> Dict[str, np.ndarray]:
    """
    Precompute pixel regions for all squares in the board image.
    Must match the geometry in PieceDetector.
    """
    m = float(margin_squares)

    if m > 0.0:
        step_x = img_w / (squares + 2.0 * m)
        step_y = img_h / (squares + 2.0 * m)
        off_x = step_x * m
        off_y = step_y * m
    else:
        step_x = img_w / squares
        step_y = img_h / squares
        off_x = 0.0
        off_y = 0.0

    boxes: Dict[str, np.ndarray] = {}
    for rank_idx in range(squares):
        for file_idx in range(squares):
            x0 = off_x + file_idx * step_x
            x1 = off_x + (file_idx + 1) * step_x
            y0 = off_y + rank_idx * step_y
            y1 = off_y + (rank_idx + 1) * step_y

            sq = _square_name(rank_idx, file_idx)
            boxes[sq] = np.array([x0, y0, x1, y1], dtype=np.float32)

    return boxes


# BGR colors per piece code
_PIECE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "wP": (255, 255, 0),
    "wN": (255, 0, 255),
    "wB": (0, 255, 255),
    "wR": (0, 128, 255),
    "wQ": (0, 0, 255),
    "wK": (0, 255, 0),
    "bP": (128, 128, 255),
    "bN": (255, 0, 128),
    "bB": (255, 128, 0),
    "bR": (128, 0, 255),
    "bQ": (0, 0, 128),
    "bK": (0, 128, 0),
}


def _piece_code_from_label(label: str) -> str:
    """
    Try to derive a compact code like "wP" from a YOLO label.
    Accepts formats like "wP", "bK", "white-pawn", etc.
    Simple heuristic is enough for debug overlays.
    """
    if not label:
        return ""

    s = label.strip()

    # Direct codes like "wP"
    if len(s) >= 2 and s[0] in ("w", "b") and s[1].upper() in "PNBRQK":
        return s[0] + s[1].upper()

    lower = s.lower()
    is_white = "white" in lower or lower.startswith("w")
    is_black = "black" in lower or lower.startswith("b")

    piece_char = "P"
    if "knight" in lower or " n" in lower:
        piece_char = "N"
    if "bishop" in lower or " b" in lower:
        piece_char = "B"
    if "rook" in lower or " r" in lower:
        piece_char = "R"
    if "queen" in lower or " q" in lower:
        piece_char = "Q"
    if "king" in lower or " k" in lower:
        piece_char = "K"

    color_char = "w" if is_white and not is_black else "b"
    return color_char + piece_char


def draw_piece_overlay(
        img: np.ndarray,
        pieces: Dict[str, Optional[str]],
        *,
        squares: int = 8,
        margin_squares: float = 0.0,
        raw_boxes: Optional[Dict[str, Optional[Tuple[float, float, float, float]]]] = None,
        confidences: Optional[Dict[str, Optional[float]]] = None,
        draw_grid: bool = True,
        debug_overlays: bool = True,
) -> np.ndarray:
    """
    Draw a colored field overlay per piece and optional debug overlays:

      - board grid
      - YOLO label text with confidence
      - original YOLO boxes as thin white rectangles
    """
    if img is None or img.size == 0:
        raise ValueError("draw_piece_overlay got empty image")

    out = img.copy()
    h, w = out.shape[:2]

    square_boxes = _compute_square_boxes(w, h, squares, margin_squares)

    # optional board grid
    if debug_overlays and draw_grid:
        for cell in square_boxes.values():
            x0, y0, x1, y1 = cell.astype(int)
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 1)

    # prepare semi transparent field overlays
    overlay = out.copy()
    alpha = 0.30

    for sq, label in pieces.items():
        if not label:
            continue
        cell = square_boxes.get(sq)
        if cell is None:
            continue

        x0, y0, x1, y1 = cell.astype(int)

        code = _piece_code_from_label(label)
        color = _PIECE_COLORS.get(code, (200, 200, 200))

        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness=-1)

    cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0, out)

    if not debug_overlays:
        return out

    # Labels and confidence text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    text_thickness = 1

    for sq, label in pieces.items():
        if not label:
            continue
        cell = square_boxes.get(sq)
        if cell is None:
            continue

        x0, y0, x1, y1 = cell.astype(int)

        conf_val = None
        if confidences is not None:
            conf_val = confidences.get(sq)

        if conf_val is not None:
            text = f"{label} {conf_val:.2f}"
        else:
            text = label

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, text_thickness)

        text_x = int(x0 + (x1 - x0) * 0.05)
        text_y = int(y0 + th + 4)

        bg_x0 = text_x - 2
        bg_y0 = text_y - th - 2
        bg_x1 = text_x + tw + 2
        bg_y1 = text_y + 2
        cv2.rectangle(out, (bg_x0, bg_y0), (bg_x1, bg_y1), (0, 0, 0), thickness=-1)

        cv2.putText(
            out,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            lineType=cv2.LINE_AA,
        )

    # Original YOLO boxes as thin white rectangles
    if raw_boxes is not None:
        for sq, box in raw_boxes.items():
            if not box:
                continue
            x1, y1, x2, y2 = box
            cv2.rectangle(
                out,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 255, 255),
                thickness=1,
            )

    return out
