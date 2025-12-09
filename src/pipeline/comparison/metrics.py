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

    The frame map is optional and can be empty.
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


def fen_frame_counts(
        frame_fens: List[str],
        gt: GroundTruth,
) -> Tuple[int, int]:
    """
    Count how many annotated frames have a correct FEN placement at the
    exact ground truth annotation frame.

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


def fen_frame_accuracy(
        frame_fens: List[str],
        gt: GroundTruth,
) -> float:
    """
    Strict FEN based accuracy at annotated frames.

    For each ground truth ply that has a frame annotation, compare the
    placement portion of

      - the predicted FEN at that frame index, and
      - the ground truth FEN after that ply.

    The result is the fraction of annotated plies where the placement strings
    are identical.
    """
    correct, total = fen_frame_counts(frame_fens, gt)
    if total == 0:
        return 0.0
    return correct / float(total)


def fen_interval_counts(
        frame_fens: List[str],
        gt: GroundTruth,
) -> Tuple[int, int]:
    """
    Interval based FEN accuracy.

    For each ground truth ply k that has a frame annotation, consider the
    interval

      [frame_for_ply[k], frame_for_ply[k_next] - 1]

    where k_next is the next ply with an annotation. For the last annotated
    ply, the interval extends to the end of the video.

    Count a ply as correct if the target placement (ground truth FEN after
    that ply) appears at least once in this interval in the predicted FEN
    sequence.

    Returns (correct, total).
    """
    if not gt.frame_for_ply:
        return 0, 0

    if not frame_fens:
        return 0, 0

    last_frame_idx = len(frame_fens) - 1
    items = sorted(gt.frame_for_ply.items())  # list of (ply_idx, start_frame)

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
            pred_placement = placement_from_fen(frame_fens[f_idx])
            if pred_placement == target_placement:
                correct += 1
                break

    return correct, total


def fen_interval_accuracy(
        frame_fens: List[str],
        gt: GroundTruth,
) -> float:
    """
    Fraction of plies where the correct FEN placement is seen at least once
    in the ground truth interval for that ply.

    This metric is robust to delayed confirmations in multistage pipelines.
    """
    correct, total = fen_interval_counts(frame_fens, gt)
    if total == 0:
        return 0.0
    return correct / float(total)


# -------------------------------------------------------------
# Whole game board level metrics
# -------------------------------------------------------------


# Piece symbols used in FEN placements. Everything else is treated as empty.
_PIECE_CHARS = set("prnbqkPRNBQK")


def _is_piece_symbol(sym: str | None) -> bool:
    """
    Return True if sym looks like a piece symbol from a FEN placement,
    False for empty or missing squares.
    """
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


def fen_game_counts(
        frame_fens: List[str],
        gt: GroundTruth,
) -> Tuple[int, int, int]:
    """
    Count true positives, false positives, and false negatives at the level of
    single pieces on squares, aggregated over the whole game.

    Evaluation is restricted to frames that lie inside the ground truth
    intervals defined by gt.frame_for_ply, similar to fen_interval_counts:

      - For each annotated ply k, we take the ground truth FEN after ply k.
      - For all frames in the interval assigned to this ply, we compare the
        predicted FEN to this ground truth FEN.
      - At each square, we compare occupancy and piece identity.

    Counting rules (per square):

      GT empty, pred empty         -> ignored
      GT piece X, pred piece X     -> TP
      GT empty, pred piece Y       -> FP
      GT piece X, pred empty       -> FN
      GT piece X, pred piece Y!=X  -> FP + FN

    Returns (tp, fp, fn).
    """
    if not gt.frame_for_ply:
        return 0, 0, 0
    if not frame_fens:
        return 0, 0, 0

    last_frame_idx = len(frame_fens) - 1
    items = sorted(gt.frame_for_ply.items())  # list of (ply_idx, start_frame)

    tp = 0
    fp = 0
    fn = 0

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
                # both empty -> ignore

    return tp, fp, fn


def fen_game_precision_recall_f1(
        frame_fens: List[str],
        gt: GroundTruth,
) -> Tuple[float, float, float]:
    """
    Whole game piece placement precision, recall, and F1.

    These are micro averaged over all frames and squares inside the annotated
    ground truth intervals.
    """
    tp, fp, fn = fen_game_counts(frame_fens, gt)

    if tp + fp > 0:
        precision = tp / float(tp + fp)
    else:
        precision = 0.0

    if tp + fn > 0:
        recall = tp / float(tp + fn)
    else:
        recall = 0.0

    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def fen_game_iou(
        frame_fens: List[str],
        gt: GroundTruth,
) -> float:
    """
    Whole game Intersection over Union for piece placements.

    This uses the same TP, FP, FN as fen_game_precision_recall_f1:

      IoU = TP / (TP + FP + FN)

    Values are aggregated over all annotated intervals.
    """
    tp, fp, fn = fen_game_counts(frame_fens, gt)
    denom = tp + fp + fn
    if denom == 0:
        return 0.0
    return tp / float(denom)


# -------------------------------------------------------------
# Move based metrics
# -------------------------------------------------------------


def move_accuracy_counts(
        pred_moves: List[str],
        gt_moves: List[str],
) -> Tuple[int, int]:
    """
    Exact move accuracy at the sequence level.

    Returns (correct, total) where total is the number of ground truth plies.
    """
    if not gt_moves:
        return 0, 0

    correct = 0
    for i, gt_move in enumerate(gt_moves):
        if i < len(pred_moves) and pred_moves[i] == gt_move:
            correct += 1
    return correct, len(gt_moves)


def move_accuracy(pred_moves: List[str], gt_moves: List[str]) -> float:
    """
    Exact move accuracy at the sequence level.

    Counts how many ground truth plies are predicted correctly at the
    correct position.
    """
    correct, total = move_accuracy_counts(pred_moves, gt_moves)
    if total == 0:
        return 0.0
    return correct / float(total)


def move_reconstruction_rate(
        pred_moves: List[str],
        gt_moves: List[str],
) -> float:
    """
    Move Reconstruction Rate (MRR) as used in the thesis:

      fraction of moves in a game that are reconstructed correctly,
      counted over the entire sequence.

    This is identical to sequence level move accuracy.
    """
    return move_accuracy(pred_moves, gt_moves)


def move_detection_delays(
        pred_moves: List[str],
        pred_move_frames: List[int],
        gt: GroundTruth,
) -> List[int]:
    """
    For each ground truth ply that

      - has a frame annotation,
      - has a corresponding predicted move at the same index, and
      - that predicted move matches the ground truth UCI,

    compute

        delay = frame_predicted - frame_ground_truth

    Returns a list of non negative integer delays in frames.
    """
    if not gt.frame_for_ply:
        return []

    delays: List[int] = []

    for ply_idx, gt_frame in sorted(gt.frame_for_ply.items()):
        # Check valid ply
        if ply_idx <= 0 or ply_idx > len(gt.moves_uci):
            continue

        gt_move = gt.moves_uci[ply_idx - 1]

        # There must be a predicted move at this ply position
        if ply_idx - 1 >= len(pred_moves):
            continue

        pred_move = pred_moves[ply_idx - 1]

        # Only consider moves that are correct at the correct index
        if pred_move != gt_move:
            continue

        # There must be a frame index for this predicted move
        if ply_idx - 1 >= len(pred_move_frames):
            continue

        pred_frame = int(pred_move_frames[ply_idx - 1])
        gt_frame = int(gt_frame)

        # Skip clearly inconsistent negative delays
        if pred_frame < gt_frame:
            continue

        delays.append(pred_frame - gt_frame)

    return delays


# -------------------------------------------------------------
# Move coverage
# -------------------------------------------------------------


def move_coverage_counts(
        pred_moves: List[str],
        gt_moves: List[str],
) -> Tuple[int, int]:
    """
    Counts for move coverage:

      emitted = number of moves emitted by the pipeline
      total_gt = number of ground truth moves
    """
    emitted = len(pred_moves)
    total_gt = len(gt_moves)
    return emitted, total_gt


def move_coverage(
        pred_moves: List[str],
        gt_moves: List[str],
) -> float:
    """
    Move coverage for a single game.

      coverage = min(number of moves emitted, number of ground truth moves)
                 / number of ground truth moves

    This yields 1.0 if the pipeline emits at least as many moves
    as the ground truth and less than 1.0 if some moves are missing.
    """
    emitted, total_gt = move_coverage_counts(pred_moves, gt_moves)
    if total_gt == 0:
        return 0.0
    covered = min(emitted, total_gt)
    return covered / float(total_gt)
