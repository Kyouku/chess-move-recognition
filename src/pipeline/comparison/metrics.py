from __future__ import annotations

"""
Ground truth handling and evaluation metrics for the offline comparison pipeline.
"""

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import chess
import chess.pgn

from src.pipeline.fen_utils import placement_from_fen


@dataclass
class GroundTruth:
    """
    Ground truth representation for a single game.

    - pgn:
        Original PGN text.
    - moves_uci:
        Mainline moves in UCI notation.
    - fens_after_ply:
        FEN strings for the position after each ply.
    - frame_for_ply:
        Optional mapping from ply index (1 based) to frame index.
    """

    pgn: str
    moves_uci: List[str]
    fens_after_ply: List[str]
    frame_for_ply: Dict[int, int]


def _moves_and_fens_from_pgn(pgn_text: str) -> Tuple[List[str], List[str]]:
    game_io = io.StringIO(pgn_text)
    game = chess.pgn.read_game(game_io)
    if game is None:
        raise ValueError("Could not parse PGN text")

    board = game.board()
    moves_uci: List[str] = []
    fens_after: List[str] = []

    for move in game.mainline_moves():
        moves_uci.append(move.uci())
        board.push(move)
        fens_after.append(board.fen())

    return moves_uci, fens_after


def load_ground_truth(path: str | Path) -> GroundTruth:
    """
    Load ground truth from a JSON file.

    Expected format:
      {
        "pgn": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6",
        "frame_for_ply": {
          "1": 42,
          "2": 78,
          "3": 115
        }
      }

    frame_for_ply is optional and may be empty.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf8"))

    pgn = str(data["pgn"])
    moves_uci, fens_after = _moves_and_fens_from_pgn(pgn)

    raw_map = data.get("frame_for_ply", {})
    frame_for_ply: Dict[int, int] = {}
    for k, v in raw_map.items():
        try:
            ply_idx = int(k)
            frame_for_ply[ply_idx] = int(v)
        except (TypeError, ValueError):
            continue

    return GroundTruth(
        pgn=pgn,
        moves_uci=moves_uci,
        fens_after_ply=fens_after,
        frame_for_ply=frame_for_ply,
    )


# -------------------------------------------------------------
# FEN metrics at annotated frames or intervals
# -------------------------------------------------------------


def fen_frame_counts(frame_fens: List[str], gt: GroundTruth) -> Tuple[int, int]:
    """
    Compare predicted vs GT placement at the exact annotated frame for each ply.

    Returns (correct, total).
    """
    if not gt.frame_for_ply:
        return 0, 0

    correct = 0
    total = 0

    for ply_idx, frame_idx in sorted(gt.frame_for_ply.items()):
        if ply_idx <= 0 or ply_idx > len(gt.fens_after_ply):
            continue
        if frame_idx < 0 or frame_idx >= len(frame_fens):
            continue

        pred_fen = frame_fens[frame_idx]
        gt_fen = gt.fens_after_ply[ply_idx - 1]

        pred_placement = placement_from_fen(pred_fen)
        gt_placement = placement_from_fen(gt_fen)

        total += 1
        if pred_placement == gt_placement:
            correct += 1

    return correct, total


def fen_frame_accuracy(frame_fens: List[str], gt: GroundTruth) -> float:
    correct, total = fen_frame_counts(frame_fens, gt)
    if total == 0:
        return 0.0
    return correct / float(total)


def fen_at_ply_accuracy(frame_fens: List[str], gt: GroundTruth) -> float:
    """Alias for fen_frame_accuracy used by comparison scripts."""
    return fen_frame_accuracy(frame_fens, gt)


def fen_interval_counts(frame_fens: List[str], gt: GroundTruth) -> Tuple[int, int]:
    """
    Interval based FEN accuracy.

    For each annotated ply k with frame_for_ply[k] = start,
    define end as (frame_for_ply[next_k] - 1) or last frame.

    A ply counts as correct if the GT placement appears at least once
    anywhere in [start, end] in the predicted frame_fens.
    """
    if not gt.frame_for_ply or not frame_fens:
        return 0, 0

    last_frame_idx = len(frame_fens) - 1
    items = sorted(gt.frame_for_ply.items())

    correct = 0
    total = 0

    for i, (ply_idx, start_frame) in enumerate(items):
        if ply_idx <= 0 or ply_idx > len(gt.fens_after_ply):
            continue
        if start_frame < 0 or start_frame > last_frame_idx:
            continue

        if i + 1 < len(items):
            _, next_start = items[i + 1]
            end_frame = min(next_start - 1, last_frame_idx)
        else:
            end_frame = last_frame_idx

        if end_frame < start_frame:
            continue

        target_placement = placement_from_fen(gt.fens_after_ply[ply_idx - 1])
        total += 1

        for f_idx in range(start_frame, end_frame + 1):
            if placement_from_fen(frame_fens[f_idx]) == target_placement:
                correct += 1
                break

    return correct, total


def fen_interval_accuracy(frame_fens: List[str], gt: GroundTruth) -> float:
    correct, total = fen_interval_counts(frame_fens, gt)
    if total == 0:
        return 0.0
    return correct / float(total)


# -------------------------------------------------------------
# Move metrics
# -------------------------------------------------------------


def move_accuracy_counts(pred_moves: List[str], gt_moves: List[str]) -> Tuple[int, int]:
    if not gt_moves:
        return 0, 0
    correct = 0
    for i, gt_move in enumerate(gt_moves):
        if i < len(pred_moves) and pred_moves[i] == gt_move:
            correct += 1
    return correct, len(gt_moves)


def move_reconstruction_rate(pred_moves: List[str], gt_moves: List[str]) -> float:
    """
    In this project, "MRR" is used as ply-aligned move reconstruction accuracy.
    """
    correct, total = move_accuracy_counts(pred_moves, gt_moves)
    if total == 0:
        return 0.0
    return correct / float(total)


def move_coverage(pred_moves: List[str], gt_moves: List[str]) -> float:
    """
    coverage = min(len(pred), len(gt)) / len(gt)
    """
    if not gt_moves:
        return 0.0
    return min(len(pred_moves), len(gt_moves)) / float(len(gt_moves))


def move_detection_delays(*args):
    """
    Compatibility helper.

    Supported calls:
      A) move_detection_delays(pred_moves, pred_move_frames, gt) -> list[int] (delays in frames)
      B) move_detection_delays(gt_moves, gt_frames_by_ply, pred_moves, pred_frames, video_fps) -> list[float] (seconds)

    For B), gt_frames_by_ply can be:
      - a list[int] aligned to plies (0-based), or
      - a dict[int,int] frame_for_ply (1-based ply index).
    """
    if len(args) == 3:
        pred_moves, pred_move_frames, gt = args
        if not isinstance(gt, GroundTruth):
            raise TypeError("Third argument must be GroundTruth for 3-arg move_detection_delays")

        if not gt.frame_for_ply:
            return []

        delays: List[int] = []
        for ply_idx, gt_frame in sorted(gt.frame_for_ply.items()):
            if ply_idx <= 0 or ply_idx > len(gt.moves_uci):
                continue
            if ply_idx - 1 >= len(pred_moves):
                continue
            if ply_idx - 1 >= len(pred_move_frames):
                continue

            gt_move = gt.moves_uci[ply_idx - 1]
            pred_move = pred_moves[ply_idx - 1]
            if pred_move != gt_move:
                continue

            pred_frame = int(pred_move_frames[ply_idx - 1])
            gt_frame_i = int(gt_frame)
            if pred_frame < gt_frame_i:
                continue
            delays.append(pred_frame - gt_frame_i)

        return delays

    if len(args) == 5:
        gt_moves, gt_frames_by_ply, pred_moves, pred_frames, video_fps = args
        fps = float(video_fps) if video_fps and float(video_fps) > 0 else None
        if fps is None:
            return []

        # normalize gt frame lookup for 0-based index
        def gt_frame_for_index(i0: int) -> int | None:
            if isinstance(gt_frames_by_ply, dict):
                return gt_frames_by_ply.get(i0 + 1)
            if isinstance(gt_frames_by_ply, list):
                if 0 <= i0 < len(gt_frames_by_ply):
                    return int(gt_frames_by_ply[i0])
                return None
            return None

        delays_s: List[float] = []
        for i0 in range(min(len(gt_moves), len(pred_moves), len(pred_frames))):
            if pred_moves[i0] != gt_moves[i0]:
                continue
            gf = gt_frame_for_index(i0)
            if gf is None:
                continue
            pf = int(pred_frames[i0])
            if pf < int(gf):
                continue
            delays_s.append((pf - int(gf)) / fps)

        return delays_s

    raise TypeError("Unsupported move_detection_delays signature")


def mean_move_delay_seconds(
        pred_moves: List[str],
        pred_move_frames: List[int],
        gt: GroundTruth,
        video_fps: float | None,
) -> Tuple[float | None, float | None, int]:
    """
    Returns (mean_delay_s, median_delay_s, count) for correct, ply-aligned moves
    at annotated plies. If no delays, returns (None, None, 0).
    """
    delays_frames = move_detection_delays(pred_moves, pred_move_frames, gt)
    if not delays_frames:
        return None, None, 0

    if video_fps is None or float(video_fps) <= 0:
        return None, None, len(delays_frames)

    fps = float(video_fps)
    delays_s = sorted([float(d) / fps for d in delays_frames])
    n = len(delays_s)
    mean_s = sum(delays_s) / float(n)

    if n % 2 == 1:
        median_s = delays_s[n // 2]
    else:
        median_s = 0.5 * (delays_s[n // 2 - 1] + delays_s[n // 2])

    return mean_s, median_s, n


# -------------------------------------------------------------
# Whole game board level metrics (piece placement)
# -------------------------------------------------------------

# Piece symbols used in FEN placements. Everything else is treated as empty.
_PIECE_CHARS = set("prnbqkPRNBQK")


def _is_piece_symbol(sym: str | None) -> bool:
    return isinstance(sym, str) and sym in _PIECE_CHARS


def _fen_to_square_map_safe(fen: str) -> Dict[str, str | None] | None:
    """
    Convert a full FEN string into a mapping {square_name: piece_symbol_or_None}.
    Returns None if the FEN cannot be parsed.
    """
    try:
        board = chess.Board(fen)
    except Exception:
        return None

    mapping: Dict[str, str | None] = {}
    for sq in chess.SQUARES:
        name = chess.square_name(sq)
        piece = board.piece_at(sq)
        mapping[name] = piece.symbol() if piece is not None else None
    return mapping


def fen_game_counts(frame_fens: List[str], gt: GroundTruth) -> Tuple[int, int, int]:
    """
    Count TP/FP/FN for piece placement over the whole game, restricted to
    annotated ply intervals defined by gt.frame_for_ply.

    Counting per square:
      GT empty, pred empty         -> ignored
      GT piece X, pred piece X     -> TP
      GT empty, pred piece Y       -> FP
      GT piece X, pred empty       -> FN
      GT piece X, pred piece Y!=X  -> FP + FN
    """
    if not gt.frame_for_ply:
        return 0, 0, 0
    if not frame_fens:
        return 0, 0, 0

    last_frame_idx = len(frame_fens) - 1
    items = sorted(gt.frame_for_ply.items())  # (ply_idx, start_frame)

    tp = fp = fn = 0

    for i, (ply_idx, start_frame) in enumerate(items):
        if ply_idx <= 0 or ply_idx > len(gt.fens_after_ply):
            continue
        if start_frame < 0 or start_frame > last_frame_idx:
            continue

        if i + 1 < len(items):
            _, next_start = items[i + 1]
            end_frame = min(next_start - 1, last_frame_idx)
        else:
            end_frame = last_frame_idx

        if end_frame < start_frame:
            continue

        gt_map = _fen_to_square_map_safe(gt.fens_after_ply[ply_idx - 1])
        if gt_map is None:
            continue

        squares = list(gt_map.keys())

        for frame_idx in range(start_frame, end_frame + 1):
            pred_map = _fen_to_square_map_safe(frame_fens[frame_idx])
            if pred_map is None:
                continue

            for sq in squares:
                gt_piece = gt_map.get(sq)
                pred_piece = pred_map.get(sq)

                has_gt = _is_piece_symbol(gt_piece)
                has_pred = _is_piece_symbol(pred_piece)

                if has_gt and has_pred and pred_piece == gt_piece:
                    tp += 1
                elif has_pred and not has_gt:
                    fp += 1
                elif has_gt and not has_pred:
                    fn += 1
                elif has_gt and has_pred and pred_piece != gt_piece:
                    fp += 1
                    fn += 1

    return tp, fp, fn


def fen_game_precision_recall_f1(frame_fens: List[str], gt: GroundTruth) -> Tuple[float, float, float]:
    tp, fp, fn = fen_game_counts(frame_fens, gt)

    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def fen_game_iou(frame_fens: List[str], gt: GroundTruth) -> float:
    tp, fp, fn = fen_game_counts(frame_fens, gt)
    denom = tp + fp + fn
    return tp / float(denom) if denom > 0 else 0.0


# -------------------------------------------------------------
# Move coverage counts (needed by compare_pipelines.py)
# -------------------------------------------------------------

def move_coverage_counts(pred_moves: List[str], gt_moves: List[str]) -> Tuple[int, int]:
    """
    emitted = number of predicted moves
    total_gt = number of ground truth moves
    """
    return len(pred_moves), len(gt_moves)
