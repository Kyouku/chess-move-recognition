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
# FEN metrics
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
    For each ground truth ply that has both a frame annotation and a
    matching predicted move, compute

      delay = frame_predicted - frame_ground_truth

    Returns a list of integer delays in frames.
    """
    if not gt.frame_for_ply:
        return []

    delays: List[int] = []
    pred_cursor = -1

    for ply_idx in sorted(gt.frame_for_ply.keys()):
        gt_frame = gt.frame_for_ply[ply_idx]
        if ply_idx <= 0 or ply_idx > len(gt.moves_uci):
            continue
        gt_move = gt.moves_uci[ply_idx - 1]

        # Find the next occurrence of this move at or after pred_cursor + 1
        try:
            pos = pred_moves.index(gt_move, pred_cursor + 1)
        except ValueError:
            continue

        if pos < 0 or pos >= len(pred_move_frames):
            continue

        pred_frame = pred_move_frames[pos]
        delays.append(int(pred_frame) - int(gt_frame))
        pred_cursor = pos

    return delays
