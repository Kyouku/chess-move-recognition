from __future__ import annotations

import argparse
import bisect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess
import cv2

from src import config
from src.common.homography_cache import apply_saved_homography, save_homography_from_pipeline
from src.pipeline.comparison.baseline_offline import run_baseline
from src.pipeline.comparison.detection_log import load_detections
from src.pipeline.comparison.metrics import load_ground_truth
from src.pipeline.comparison.multistage_offline import run_multistage
from src.pipeline.fen_utils import placement_from_fen
from src.stage1.board_rectifier import LivePipeline

try:
    from src.common.app_logging import get_logger
except Exception:  # pragma: no cover
    from src.common.io_utils import get_logger  # type: ignore

log = get_logger(__name__)

# ------------------------------------------------------------
# Failure categories wanted for Chapter 6
# ------------------------------------------------------------
FAILURE_TAGS_DEFAULT = [
    "glare",
    "hand_occlusion",
    "phantom_border",
    "persistent_omission_hallucination",
    "ambiguous_legal_move",
]


@dataclass(frozen=True)
class FailureFrame:
    pipeline: str
    tags: Tuple[str, ...]
    reason: str  # move_wrong_committed | fen_wrong_after_committed_move
    frame: int  # predicted commit frame
    ply: Optional[int]  # aligned GT ply
    gt_frame: Optional[int]
    gt_move_uci: Optional[str]
    pred_move_uci: Optional[str]
    gt_placement: Optional[str]
    pred_placement: Optional[str]


# Will be set in main via CLI, default from config
_MARGIN_SQUARES: float = float(getattr(config, "BOARD_MARGIN_SQUARES", 0.0))


# ------------------------------------------------------------
# Game3 homography override helpers
# ------------------------------------------------------------
def _is_game3(name: str) -> bool:
    """
    Best-effort identification of "game3" naming variants.
    Adjust if your manifest uses a different identifier.
    """
    n = name.strip().lower().replace(" ", "").replace("-", "_")
    return n in {"game3", "game_3", "3"}


def _load_homography_matrix(path: Path) -> List[List[float]]:
    """
    Load a saved 3x3 homography matrix.

    Supported formats:
      - .json with {"H": [[...],[...],[...]]} or {"homography": [[...],[...],[...]]}
      - .npy containing a 3x3 array
      - .txt/.csv containing 3 rows of 3 numbers (space or comma separated)

    Returns:
      3x3 list of floats.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    suf = path.suffix.lower()

    if suf == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        H = obj.get("H", None) or obj.get("homography", None) or obj.get("matrix", None)
        if not isinstance(H, list) or len(H) != 3:
            raise ValueError(
                f"Invalid homography JSON in {path} (expected 3x3 list under key H/homography/matrix)."
            )
        if any((not isinstance(row, list) or len(row) != 3) for row in H):
            raise ValueError(f"Invalid homography JSON in {path} (expected 3x3 list).")
        return [[float(x) for x in row] for row in H]

    if suf == ".npy":
        import numpy as np  # type: ignore

        arr = np.load(str(path))
        if getattr(arr, "shape", None) != (3, 3):
            raise ValueError(
                f"Invalid homography npy in {path} (expected shape (3,3), got {getattr(arr, 'shape', None)})."
            )
        return [[float(x) for x in row] for row in arr.tolist()]

    # Fallback: parse simple text
    txt = path.read_text(encoding="utf-8").strip().splitlines()
    rows: List[List[float]] = []
    for line in txt:
        line = line.strip()
        if not line:
            continue
        parts = [p for p in line.replace(",", " ").split(" ") if p]
        rows.append([float(p) for p in parts])

    if len(rows) != 3 or any(len(r) != 3 for r in rows):
        raise ValueError(f"Invalid homography text in {path} (expected 3 lines of 3 numbers).")
    return rows


class _WarpRectifier:
    """
    Minimal rectifier that uses a fixed homography and cv2.warpPerspective.
    Provides process_frame() and last_warped_board.
    """

    def __init__(self, H: Any, out_size: int):
        import numpy as np  # type: ignore

        self.H = np.array(H, dtype=np.float32)
        self.out_size = int(out_size)
        self.last_warped_board: Any = None

    def process_frame(self, frame_bgr: Any) -> None:
        self.last_warped_board = cv2.warpPerspective(
            frame_bgr,
            self.H,
            (self.out_size, self.out_size),
        )


def _choose_H_or_invH(cap: cv2.VideoCapture, H: Any, out_size: int) -> Any:
    """
    Some saved homographies may be stored in the opposite direction.
    Try H and inv(H) on the first frame and keep the one that yields
    higher contrast in the warped image (simple heuristic).
    """
    import numpy as np  # type: ignore

    try:
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    except Exception:
        pos = 0.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
    ok, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(pos))

    H_np = np.array(H, dtype=np.float32)

    if not ok or frame is None:
        return H_np

    def score(Hcand: Any) -> float:
        warped = cv2.warpPerspective(frame, Hcand, (out_size, out_size))
        if warped is None:
            return -1.0
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _m, s = cv2.meanStdDev(gray)
        return float(s[0][0])

    s1 = score(H_np)
    try:
        H_inv = np.linalg.inv(H_np)
        s2 = score(H_inv)
    except Exception:
        H_inv = None
        s2 = -1.0

    if H_inv is not None and s2 > s1:
        return H_inv
    return H_np


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def _safe_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    return [str(v) for v in list(x)]


def _safe_list_int(x: Any) -> List[int]:
    if x is None:
        return []
    if isinstance(x, list):
        return [int(v) for v in x]
    return [int(v) for v in list(x)]


def _safe_frame_fens(res: Any) -> List[str]:
    if hasattr(res, "frame_fens"):
        return list(getattr(res, "frame_fens"))
    if hasattr(res, "fens"):
        return list(getattr(res, "fens"))
    return []


def _safe_move_frames(res: Any) -> List[int]:
    if hasattr(res, "move_frames"):
        return list(getattr(res, "move_frames"))
    return []


def _export_img(img_bgr: Any, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), img_bgr))


def _unique_preserve_order(items: Sequence[FailureFrame]) -> List[FailureFrame]:
    seen: set[tuple[str, int, str, Tuple[str, ...]]] = set()
    out: List[FailureFrame] = []
    for it in items:
        key = (it.pipeline, int(it.frame), it.reason, it.tags)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# ------------------------------------------------------------
# Box parsing + label drawing
# ------------------------------------------------------------
def _parse_box_to_xyxy(
        box: Any,
        w: int,
        h: int,
) -> Optional[Tuple[int, int, int, int]]:
    x1 = y1 = x2 = y2 = None

    if isinstance(box, dict):
        if "xyxy" in box and isinstance(box["xyxy"], (list, tuple)) and len(box["xyxy"]) == 4:
            x1, y1, x2, y2 = box["xyxy"]
        elif "bbox" in box and isinstance(box["bbox"], (list, tuple)) and len(box["bbox"]) == 4:
            x1, y1, x2, y2 = box["bbox"]
        else:
            if all(k in box for k in ("x1", "y1", "x2", "y2")):
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            elif all(k in box for k in ("xmin", "ymin", "xmax", "ymax")):
                x1, y1, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]

    elif isinstance(box, (list, tuple)) and len(box) == 4:
        x1, y1, x2, y2 = box

    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None

    try:
        fx1, fy1, fx2, fy2 = float(x1), float(y1), float(x2), float(y2)
    except (TypeError, ValueError):
        return None

    # normalized coords
    if max(abs(fx1), abs(fy1), abs(fx2), abs(fy2)) <= 1.5:
        fx1 *= w
        fx2 *= w
        fy1 *= h
        fy2 *= h

    ix1 = int(round(min(fx1, fx2)))
    iy1 = int(round(min(fy1, fy2)))
    ix2 = int(round(max(fx1, fx2)))
    iy2 = int(round(max(fy1, fy2)))

    ix1 = max(0, min(ix1, w - 1))
    iy1 = max(0, min(iy1, h - 1))
    ix2 = max(0, min(ix2, w - 1))
    iy2 = max(0, min(iy2, h - 1))

    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def _draw_label_box(
        img: Any,
        xyxy: Tuple[int, int, int, int],
        text: str,
        color: Tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    if not text:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 4

    tx1 = x1
    ty1 = max(0, y1 - th - 2 * pad)
    tx2 = min(img.shape[1] - 1, x1 + tw + 2 * pad)
    ty2 = min(img.shape[0] - 1, y1)

    cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (0, 0, 0), -1)
    cv2.putText(img, text, (x1 + pad, y1 - pad), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


# ------------------------------------------------------------
# Square drawing with margin support
# Orientation (as specified):
# a1 is top-left, h1 is bottom-left.
# That means:
#   x axis = rank 1..8 left->right
#   y axis = file a..h top->bottom
# Plus: margin squares around the 8x8 board are present in the rectified image.
# ------------------------------------------------------------
def _board_grid_params(w: int, h: int) -> Tuple[float, float, float, float]:
    m = float(_MARGIN_SQUARES)
    total_x = 8.0 + 2.0 * m
    total_y = 8.0 + 2.0 * m
    sq_w = float(w) / total_x
    sq_h = float(h) / total_y
    x0 = m * sq_w
    y0 = m * sq_h
    return x0, y0, sq_w, sq_h


def _square_to_rect(sq: str, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(sq, str) or len(sq) != 2:
        return None

    file_c = sq[0].lower()
    rank_c = sq[1]
    if file_c < "a" or file_c > "h":
        return None
    if rank_c < "1" or rank_c > "8":
        return None

    file_idx = ord(file_c) - ord("a")  # 0..7 maps to y
    rank_idx = int(rank_c) - 1  # 0..7 maps to x

    x0, y0, sq_w, sq_h = _board_grid_params(w, h)

    x1f = x0 + rank_idx * sq_w
    y1f = y0 + file_idx * sq_h
    x2f = x0 + (rank_idx + 1) * sq_w
    y2f = y0 + (file_idx + 1) * sq_h

    x1 = int(round(x1f))
    y1 = int(round(y1f))
    x2 = int(round(x2f))
    y2 = int(round(y2f))

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _draw_square_outline(img_bgr: Any, sq: str, color_bgr: Tuple[int, int, int], thickness: int = 3) -> None:
    h, w = img_bgr.shape[:2]
    r = _square_to_rect(sq, w, h)
    if r is None:
        return
    x1, y1, x2, y2 = r
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, thickness)


def _square_center(sq: str, w: int, h: int) -> Optional[Tuple[int, int]]:
    r = _square_to_rect(sq, w, h)
    if r is None:
        return None
    x1, y1, x2, y2 = r
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def _draw_move_arrow(img_bgr: Any, from_sq: str, to_sq: str, color_bgr: Tuple[int, int, int]) -> None:
    h, w = img_bgr.shape[:2]
    c1 = _square_center(from_sq, w, h)
    c2 = _square_center(to_sq, w, h)
    if c1 is None or c2 is None:
        return
    cv2.arrowedLine(img_bgr, c1, c2, color_bgr, 3, tipLength=0.25)


def _uci_from_to(uci: Optional[str]) -> Optional[Tuple[str, str]]:
    if not uci:
        return None
    try:
        m = chess.Move.from_uci(str(uci))
        return (chess.square_name(m.from_square), chess.square_name(m.to_square))
    except Exception:
        return None


# ------------------------------------------------------------
# Overlay: draw ONLY affected pieces, plus squares for GT and committed move
# Colors:
#   GT move: blue (255,0,0)
#   committed pred move: red (0,0,255)
#   FEN diffs: yellow (0,255,255)
# ------------------------------------------------------------
def _overlay_affected(
        img_bgr: Any,
        state: Any,
        *,
        only_squares: Optional[set[str]] = None,
        gt_move_uci: Optional[str] = None,
        pred_move_uci: Optional[str] = None,
        diff_squares: Optional[set[str]] = None,
) -> None:
    if img_bgr is None:
        return

    only_squares = set(only_squares or [])
    diff_squares = set(diff_squares or [])

    gt_from_to = _uci_from_to(gt_move_uci)
    pred_from_to = _uci_from_to(pred_move_uci)

    gt_squares: set[str] = set()
    pred_squares: set[str] = set()

    if gt_from_to is not None:
        gt_squares |= {gt_from_to[0], gt_from_to[1]}
        _draw_square_outline(img_bgr, gt_from_to[0], (255, 0, 0), 3)
        _draw_square_outline(img_bgr, gt_from_to[1], (255, 0, 0), 3)
        _draw_move_arrow(img_bgr, gt_from_to[0], gt_from_to[1], (255, 0, 0))

    if pred_from_to is not None:
        pred_squares |= {pred_from_to[0], pred_from_to[1]}
        _draw_square_outline(img_bgr, pred_from_to[0], (0, 0, 255), 3)
        _draw_square_outline(img_bgr, pred_from_to[1], (0, 0, 255), 3)
        _draw_move_arrow(img_bgr, pred_from_to[0], pred_from_to[1], (0, 0, 255))

    for sq in sorted(diff_squares):
        _draw_square_outline(img_bgr, sq, (0, 255, 255), 3)

    if state is None:
        return

    boxes = getattr(state, "boxes", None) or {}
    pieces = getattr(state, "pieces", None) or {}
    if not isinstance(boxes, dict) or not isinstance(pieces, dict):
        return

    h, w = img_bgr.shape[:2]

    for sq, box in boxes.items():
        sq_name = str(sq)

        if only_squares and sq_name not in only_squares:
            continue

        xyxy = _parse_box_to_xyxy(box, w, h)
        if xyxy is None:
            continue

        label = pieces.get(sq_name)
        if label is None:
            continue

        text = str(label).strip()
        if not text:
            continue

        if sq_name in pred_squares:
            color = (0, 0, 255)
        elif sq_name in gt_squares:
            color = (255, 0, 0)
        elif sq_name in diff_squares:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        _draw_label_box(img_bgr, xyxy, text, color=color)


# ------------------------------------------------------------
# Alignment: only evaluate committed moves
# ------------------------------------------------------------
def _sorted_gt_events(gt: Any) -> List[Tuple[int, int]]:
    items: List[Tuple[int, int]] = []
    for ply, fr in getattr(gt, "frame_for_ply", {}).items():
        if ply is None or fr is None:
            continue
        try:
            ply_i = int(ply)
            fr_i = int(fr)
        except Exception:
            continue
        if ply_i <= 0 or fr_i < 0:
            continue
        items.append((fr_i, ply_i))
    items.sort(key=lambda x: x[0])
    return items


def _align_pred_moves_to_gt(
        pred_move_frames: List[int],
        gt_events: List[Tuple[int, int]],
        *,
        max_frame_diff: int,
) -> List[Tuple[int, int, int]]:
    if not pred_move_frames or not gt_events:
        return []

    gt_frames = [fr for fr, _ply in gt_events]
    gt_used: set[int] = set()
    aligned: List[Tuple[int, int, int]] = []

    for pred_idx, fr_pred in sorted(enumerate(pred_move_frames), key=lambda x: x[1]):
        if fr_pred is None:
            continue

        pos = bisect.bisect_left(gt_frames, fr_pred)

        candidates: List[Tuple[int, int, int, int]] = []  # (abs_diff, ply, gt_frame, gt_index)
        for j in (pos - 2, pos - 1, pos, pos + 1, pos + 2):
            if 0 <= j < len(gt_events):
                if j in gt_used:
                    continue
                gt_frame, ply = gt_events[j]
                d = abs(int(gt_frame) - int(fr_pred))
                if d <= max_frame_diff:
                    candidates.append((d, int(ply), int(gt_frame), int(j)))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0])
        _d, ply_best, gt_frame_best, gt_index = candidates[0]
        gt_used.add(gt_index)
        aligned.append((int(pred_idx), int(ply_best), int(gt_frame_best)))

    aligned.sort(key=lambda x: x[0])
    return aligned


# ------------------------------------------------------------
# Failure classification heuristics
# ------------------------------------------------------------
def _piece_count(state: Any) -> int:
    pieces = getattr(state, "pieces", None) or {}
    if not isinstance(pieces, dict):
        return 0
    return sum(1 for _sq, v in pieces.items() if v is not None and str(v).strip() != "")


def _median_int(vals: List[int]) -> int:
    if not vals:
        return 0
    s = sorted(vals)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return int(s[mid])
    return int(round(0.5 * (s[mid - 1] + s[mid])))


def _detect_hand_occlusion(det_log: Any, frame_idx: int, *, window: int = 30) -> bool:
    if det_log is None or not hasattr(det_log, "detections"):
        return False
    dets = det_log.detections
    if not isinstance(dets, list) or not dets:
        return False

    n = len(dets)
    if not (0 <= frame_idx < n):
        return False

    cur = _piece_count(dets[frame_idx])
    left = max(0, frame_idx - window)
    right = min(n - 1, frame_idx + window)

    neigh: List[int] = []
    for i in range(left, right + 1):
        if i == frame_idx:
            continue
        neigh.append(_piece_count(dets[i]))

    med = _median_int(neigh)
    return (cur < 18) or (med > 0 and (med - cur) >= 6)


def _detect_glare(img_bgr: Any) -> bool:
    if img_bgr is None:
        return False
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return False
    sat = float((gray >= 245).mean())
    return bool(sat >= 0.01)


def _is_border_square(sq: str) -> bool:
    if len(sq) != 2:
        return False
    f, r = sq[0], sq[1]
    return (f in ("a", "h")) or (r in ("1", "8"))


def _detect_phantom_border(state: Any, img_bgr: Any, highlight_squares: set[str]) -> bool:
    if any(_is_border_square(s) for s in highlight_squares):
        return True

    if state is None or img_bgr is None:
        return False

    boxes = getattr(state, "boxes", None) or {}
    if not isinstance(boxes, dict):
        return False

    h, w = img_bgr.shape[:2]
    margin_px = max(6, int(round(0.01 * min(w, h))))

    for _sq, box in boxes.items():
        xyxy = _parse_box_to_xyxy(box, w, h)
        if xyxy is None:
            continue
        x1, y1, x2, y2 = xyxy
        if x1 <= margin_px or y1 <= margin_px or (w - 1 - x2) <= margin_px or (h - 1 - y2) <= margin_px:
            return True
    return False


def _gt_prev_fen(gt: Any, ply: int) -> str:
    if ply <= 1:
        return chess.STARTING_FEN
    try:
        return str(gt.fens_after_ply[ply - 2])
    except Exception:
        return chess.STARTING_FEN


def _is_ambiguous_legal_move(gt: Any, ply: int, pred_move_uci: Optional[str]) -> bool:
    if not pred_move_uci:
        return False
    try:
        b = chess.Board(_gt_prev_fen(gt, ply))
        m = chess.Move.from_uci(str(pred_move_uci))
        return m in b.legal_moves
    except Exception:
        return False


def _persistent_mismatch_after_commit(
        frame_fens: List[str],
        gt: Any,
        ply: int,
        frame_idx: int,
        *,
        window: int = 45,
        threshold: float = 0.8,
) -> bool:
    if ply <= 0:
        return False
    if not frame_fens:
        return False
    if not (0 <= frame_idx < len(frame_fens)):
        return False

    try:
        gt_fen = str(gt.fens_after_ply[ply - 1])
        gt_place = placement_from_fen(gt_fen)
    except Exception:
        return False

    end = min(len(frame_fens), frame_idx + max(1, window))
    total = 0
    wrong = 0
    for i in range(frame_idx, end):
        try:
            pred_place = placement_from_fen(frame_fens[i])
        except Exception:
            continue
        total += 1
        if pred_place != gt_place:
            wrong += 1

    if total == 0:
        return False
    return (wrong / total) >= threshold


def _squares_from_uci(uci: Optional[str]) -> set[str]:
    if not uci:
        return set()
    try:
        m = chess.Move.from_uci(str(uci))
        return {chess.square_name(m.from_square), chess.square_name(m.to_square)}
    except Exception:
        return set()


def _fen_diff_squares(pred_fen: str, gt_fen: str) -> set[str]:
    try:
        b_pred = chess.Board(pred_fen)
        b_gt = chess.Board(gt_fen)
    except Exception:
        return set()

    diffs: set[str] = set()
    for sq in chess.SQUARES:
        p1 = b_pred.piece_at(sq)
        p2 = b_gt.piece_at(sq)
        if (p1 is None) != (p2 is None):
            diffs.add(chess.square_name(sq))
        elif p1 is not None and p2 is not None and p1.symbol() != p2.symbol():
            diffs.add(chess.square_name(sq))
    return diffs


def _classify_failure_tags(
        *,
        img_bgr: Any,
        det_log: Any,
        state: Any,
        highlight_squares: set[str],
        frame_fens: List[str],
        gt: Any,
        ply: int,
        frame_idx: int,
        pred_move_uci: Optional[str],
) -> Tuple[str, ...]:
    tags: List[str] = []

    if _detect_hand_occlusion(det_log, frame_idx):
        tags.append("hand_occlusion")

    if _detect_glare(img_bgr):
        tags.append("glare")

    if _detect_phantom_border(state, img_bgr, highlight_squares):
        tags.append("phantom_border")

    if _persistent_mismatch_after_commit(frame_fens, gt, ply, frame_idx):
        tags.append("persistent_omission_hallucination")

    if _is_ambiguous_legal_move(gt, ply, pred_move_uci):
        tags.append("ambiguous_legal_move")

    if not tags:
        tags.append("other")

    out: List[str] = []
    seen: set[str] = set()
    for t in tags:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return tuple(out)


# ------------------------------------------------------------
# Collect only committed-move failures and tag them
# ------------------------------------------------------------
def _collect_committed_move_failures(
        *,
        pipeline: str,
        frame_fens: List[str],
        pred_moves: List[str],
        pred_move_frames: List[int],
        det_log: Any,
        cap: cv2.VideoCapture,
        rectifier: Any,
        gt: Any,
        max_frame_diff: int,
) -> List[FailureFrame]:
    failures: List[FailureFrame] = []

    gt_events = _sorted_gt_events(gt)
    aligned = _align_pred_moves_to_gt(pred_move_frames, gt_events, max_frame_diff=max_frame_diff)

    for pred_idx, ply, gt_frame in aligned:
        if not (0 <= pred_idx < len(pred_moves) and 0 <= pred_idx < len(pred_move_frames)):
            continue

        fr_pred = int(pred_move_frames[pred_idx])
        pred_move = str(pred_moves[pred_idx]) if pred_moves[pred_idx] is not None else None

        try:
            gt_move = str(gt.moves_uci[ply - 1])
        except Exception:
            gt_move = None

        warped = _get_rectified_board_at_frame(cap, rectifier, fr_pred)
        if warped is None:
            continue

        state = (
            det_log.detections[fr_pred]
            if (hasattr(det_log, "detections") and 0 <= fr_pred < len(det_log.detections))
            else None
        )

        highlight: set[str] = set()
        highlight |= _squares_from_uci(gt_move)
        highlight |= _squares_from_uci(pred_move)

        tags = _classify_failure_tags(
            img_bgr=warped,
            det_log=det_log,
            state=state,
            highlight_squares=highlight,
            frame_fens=frame_fens,
            gt=gt,
            ply=ply,
            frame_idx=fr_pred,
            pred_move_uci=pred_move,
        )

        if gt_move is not None and pred_move is not None and pred_move != gt_move:
            failures.append(
                FailureFrame(
                    pipeline=pipeline,
                    tags=tags,
                    reason="move_wrong_committed",
                    frame=fr_pred,
                    ply=ply,
                    gt_frame=gt_frame,
                    gt_move_uci=gt_move,
                    pred_move_uci=pred_move,
                    gt_placement=None,
                    pred_placement=None,
                )
            )

        if 0 <= fr_pred < len(frame_fens):
            try:
                pred_place = placement_from_fen(frame_fens[fr_pred])
                gt_place = placement_from_fen(gt.fens_after_ply[ply - 1])
            except Exception:
                pred_place = None
                gt_place = None

            if pred_place is not None and gt_place is not None and pred_place != gt_place:
                failures.append(
                    FailureFrame(
                        pipeline=pipeline,
                        tags=tags,
                        reason="fen_wrong_after_committed_move",
                        frame=fr_pred,
                        ply=ply,
                        gt_frame=gt_frame,
                        gt_move_uci=gt_move,
                        pred_move_uci=pred_move,
                        gt_placement=str(gt_place),
                        pred_placement=str(pred_place),
                    )
                )

    return _unique_preserve_order(failures)


# ------------------------------------------------------------
# Rectifier helpers
# ------------------------------------------------------------
def _init_rectifier_for_video(
        cap: cv2.VideoCapture,
        *,
        is_game3: bool,
        homography_override_path: Optional[Path],
) -> Any:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(getattr(config, "FRAME_WIDTH", 1280))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(getattr(config, "FRAME_HEIGHT", 720))
    board_size_px = int(getattr(config, "BOARD_SIZE_PX", 640))

    # Game3: if a homography override is provided, force fixed warp and never calibrate
    if is_game3 and homography_override_path is not None:
        H_list = _load_homography_matrix(homography_override_path)
        H_best = _choose_H_or_invH(cap, H_list, board_size_px)
        log.info("Game3: using fixed homography warp from %s", str(homography_override_path))
        return _WarpRectifier(H_best, board_size_px)

    # Otherwise use the normal LivePipeline path (cache or calibration)
    pipeline = LivePipeline(
        frame_width=width,
        frame_height=height,
        board_size_px=board_size_px,
        margin_squares=float(getattr(config, "BOARD_MARGIN_SQUARES", 1.7)),
        input_target_long_edge=int(getattr(config, "CALIBRATION_TARGET_LONG_EDGE", max(width, height))),
        min_board_area_ratio=float(getattr(config, "AUTO_MIN_BOARD_AREA_RATIO", 0.0)),
        display=False,
    )

    ok = False
    try:
        ok = bool(apply_saved_homography(pipeline))
    except Exception:
        ok = False

    if not ok:
        log.info("Calibrating for this video")
        ok = pipeline.calibrate_from_capture(
            cap,
            max_frames=int(getattr(config, "CALIBRATION_MAX_FRAMES", 200)),
        )
        if not ok:
            raise RuntimeError("Calibration failed, cannot export rectified frames")
        try:
            save_homography_from_pipeline(pipeline)
        except Exception:
            pass

    return pipeline


def _get_rectified_board_at_frame(
        cap: cv2.VideoCapture,
        rectifier: Any,
        frame_idx: int,
) -> Optional[Any]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None

    rectifier.process_frame(frame)
    warped = getattr(rectifier, "last_warped_board", None)
    if warped is None:
        return None
    return warped


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    global _MARGIN_SQUARES

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)

    ap.add_argument("--pipelines", nargs="+", default=["multistage"], choices=["baseline", "multistage"])

    ap.add_argument("--tags", nargs="+", default=FAILURE_TAGS_DEFAULT)
    ap.add_argument("--max_per_tag", type=int, default=2)
    ap.add_argument("--align_max_frame_diff", type=int, default=180)

    ap.add_argument(
        "--margin_squares",
        type=float,
        default=float(getattr(config, "BOARD_MARGIN_SQUARES", 0.0)),
        help="Margin squares around the 8x8 grid in the rectified board image (must match rectifier).",
    )

    ap.add_argument(
        "--game3_homography",
        type=str,
        default=None,
        help="Path to saved homography for game3 (.json, .npy, or txt). If omitted, game3 uses cache or calibration.",
    )

    args = ap.parse_args()

    _MARGIN_SQUARES = float(args.margin_squares)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    games = manifest.get("games", [])
    if not games:
        raise ValueError("Manifest contains no games.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted_tags = [str(t) for t in args.tags]

    for g in games:
        name = str(g["name"])
        video_path = str(g["video"])

        det_log = load_detections(g["detections"])
        gt = load_ground_truth(g["gt"])

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        is_g3 = _is_game3(name)

        homography_override: Optional[Path] = None
        if is_g3:
            if args.game3_homography:
                homography_override = Path(args.game3_homography)
            else:
                for k in ("homography", "homography_path", "saved_homography", "saved_homography_path"):
                    if k in g and g[k]:
                        homography_override = Path(str(g[k]))
                        break

        log.info(
            "[%s] Init rectifier (is_game3=%s, override=%s)",
            name,
            str(is_g3),
            str(homography_override) if homography_override is not None else "None",
        )

        rectifier = _init_rectifier_for_video(
            cap,
            is_game3=is_g3,
            homography_override_path=homography_override,
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)

        per_game_meta: Dict[str, Any] = {"game": name, "video": video_path, "exports": []}

        def export_for_pipeline(pipeline_name: str, res: Any) -> None:
            frame_fens = _safe_frame_fens(res)
            pred_moves = _safe_list_str(getattr(res, "moves_uci", []))
            pred_move_frames = _safe_list_int(_safe_move_frames(res))

            failures_all = _collect_committed_move_failures(
                pipeline=pipeline_name,
                frame_fens=frame_fens,
                pred_moves=pred_moves,
                pred_move_frames=pred_move_frames,
                det_log=det_log,
                cap=cap,
                rectifier=rectifier,
                gt=gt,
                max_frame_diff=int(args.align_max_frame_diff),
            )

            chosen: List[FailureFrame] = []
            used_count: Dict[str, int] = {t: 0 for t in wanted_tags}

            for ff in failures_all:
                for tag in ff.tags:
                    if tag not in used_count:
                        continue
                    if used_count[tag] >= int(args.max_per_tag):
                        continue
                    chosen.append(ff)
                    used_count[tag] += 1
                    break

            for i, ff in enumerate(chosen):
                fr = int(ff.frame)

                warped = _get_rectified_board_at_frame(cap, rectifier, fr)
                if warped is None:
                    log.warning("[%s] Could not rectify frame %d, skipping export", name, fr)
                    continue

                state = (
                    det_log.detections[fr]
                    if (hasattr(det_log, "detections") and 0 <= fr < len(det_log.detections))
                    else None
                )

                gt_sq = _squares_from_uci(ff.gt_move_uci)
                pred_sq = _squares_from_uci(ff.pred_move_uci)

                diff_sq: set[str] = set()
                if ff.reason == "fen_wrong_after_committed_move" and ff.ply is not None and 0 <= fr < len(frame_fens):
                    try:
                        gt_fen_full = gt.fens_after_ply[int(ff.ply) - 1]
                        pred_fen_full = frame_fens[fr]
                        diff_sq = _fen_diff_squares(pred_fen_full, gt_fen_full)
                    except Exception:
                        diff_sq = set()

                if ff.reason == "move_wrong_committed":
                    only = gt_sq | pred_sq
                elif ff.reason == "fen_wrong_after_committed_move":
                    only = diff_sq
                else:
                    only = gt_sq | pred_sq | diff_sq

                _overlay_affected(
                    warped,
                    state,
                    only_squares=only,
                    gt_move_uci=ff.gt_move_uci,
                    pred_move_uci=ff.pred_move_uci,
                    diff_squares=diff_sq,
                )

                main_tag = ff.tags[0] if ff.tags else "other"
                out_path = out_dir / name / f"{pipeline_name}_{main_tag}_{ff.reason}_{i:02d}_frame_{fr}.png"
                ok = _export_img(warped, out_path)
                if ok:
                    per_game_meta["exports"].append({**asdict(ff), "path": str(out_path)})

        try:
            if "baseline" in args.pipelines:
                log.info("[%s] Running baseline for failure export", name)
                base = run_baseline(
                    det_log.detections,
                    video_fps=getattr(det_log, "video_fps", None),
                    detector_fps=getattr(det_log, "detector_fps", None),
                )
                export_for_pipeline("baseline", base)

            if "multistage" in args.pipelines:
                log.info("[%s] Running multistage for failure export", name)
                ms = run_multistage(
                    det_log.detections,
                    video_fps=getattr(det_log, "video_fps", None),
                    detector_fps=getattr(det_log, "detector_fps", None),
                )
                export_for_pipeline("multistage", ms)
        finally:
            cap.release()

        meta_path = out_dir / name / "failure_frames_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(per_game_meta, indent=2), encoding="utf-8")

        log.info("[%s] Exported rectified tagged failure frames to: %s", name, out_dir / name)


if __name__ == "__main__":
    main()
