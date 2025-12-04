from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import chess
import chess.pgn


@dataclass
class GroundTruth:
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


def move_accuracy(pred_moves: List[str], gt_moves: List[str]) -> float:
    """
    Exact move accuracy at the sequence level.

    Counts how many ground truth plies are predicted correctly at the
    correct position.
    """
    if not gt_moves:
        return 0.0

    correct = 0
    for i, gt in enumerate(gt_moves):
        if i < len(pred_moves) and pred_moves[i] == gt:
            correct += 1
    return correct / float(len(gt_moves))


def move_reconstruction_rate(
        pred_moves: List[str],
        gt_moves: List[str],
) -> float:
    """
    Move Reconstruction Rate (MRR) as defined in the thesis:

      fraction of moves in a game that are reconstructed correctly,
      counted over the entire sequence.

    This is identical to sequence level move accuracy.
    """
    return move_accuracy(pred_moves, gt_moves)


def fen_frame_accuracy(
        pred_frame_fens: List[str],
        gt: GroundTruth,
) -> float:
    """
    Compare predicted model FEN per frame against ground truth FEN for
    frames that are annotated in gt.frame_for_ply.
    """
    if not gt.frame_for_ply:
        return 0.0

    total = 0
    correct = 0

    for ply_idx, frame_idx in gt.frame_for_ply.items():
        if frame_idx < 0 or frame_idx >= len(pred_frame_fens):
            continue
        if ply_idx <= 0 or ply_idx > len(gt.fens_after_ply):
            continue

        total += 1
        gt_fen = gt.fens_after_ply[ply_idx - 1]
        pred_fen = pred_frame_fens[frame_idx]
        if pred_fen == gt_fen:
            correct += 1

    if total == 0:
        return 0.0
    return correct / float(total)


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
    for ply_idx, gt_frame in gt.frame_for_ply.items():
        if ply_idx <= 0 or ply_idx > len(gt.moves_uci):
            continue
        gt_move = gt.moves_uci[ply_idx - 1]

        try:
            pos = pred_moves.index(gt_move)
        except ValueError:
            continue

        if pos < 0 or pos >= len(pred_move_frames):
            continue

        pred_frame = pred_move_frames[pos]
        delays.append(int(pred_frame) - int(gt_frame))

    return delays
