from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import chess
import chess.pgn
import cv2

from src.app_logging import get_logger

_log = get_logger(__name__)

"""
python -m src.pipeline_comparison.prepare_ground_truth --video data/videos/game1.mp4 --pgn data/gt/game1.pgn --out data/gt/game1.json
"""


def _load_pgn_moves(pgn_path: Path) -> Tuple[str, List[str]]:
    """
    Load PGN text and return it together with the main line moves in SAN.
    """
    text = pgn_path.read_text(encoding="utf8")
    game_io = io.StringIO(text)
    game = chess.pgn.read_game(game_io)
    if game is None:
        raise ValueError(f"Could not parse PGN at {pgn_path}")

    board = game.board()
    san_moves: List[str] = []
    for move in game.mainline_moves():
        san = board.san(move)
        san_moves.append(san)
        board.push(move)

    return text, san_moves


def _draw_overlay(
        frame,
        frame_idx: int,
        num_frames: int,
        ply_idx: int,
        san_moves: List[str],
        frame_for_ply: Dict[int, int],
) -> None:
    """
    Draw a readable text overlay with current frame, move and instructions.
    """

    h, w = frame.shape[:2]

    # Scale text for readability based on frame height
    base_scale = max(h / 720.0, 1.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(1.4, 0.9 * base_scale)
    thickness = max(2, int(round(2 * base_scale)))
    line_gap = int(round(12 * base_scale))
    left_pad = int(round(16 * base_scale))
    top_pad = int(round(16 * base_scale))
    right_pad = int(round(16 * base_scale))
    bottom_pad = int(round(16 * base_scale))

    curr_move = san_moves[ply_idx] if 0 <= ply_idx < len(san_moves) else "<done>"
    total_plies = len(san_moves)
    marked_count = len(frame_for_ply)

    lines: List[str] = [
        f"Frame {frame_idx + 1} / {num_frames}  |  Ply {ply_idx + 1} / {total_plies}: {curr_move}",
        "Controls:",
        "j: next frame    k: previous frame    J: +10    K: -10    O: +100    I: -100    L: +300    H: -300",
        "m: mark ply at current frame    s: skip ply",
        "u: undo last mark    q or ESC: quit",
        f"Marked plies: {marked_count}",
    ]

    # Compute panel size
    text_sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
    text_w = max((ts[0] for ts in text_sizes), default=0)
    text_hs = [ts[1] for ts in text_sizes]
    total_text_h = sum(text_hs) + line_gap * (len(lines) - 1)

    panel_x0 = 10
    panel_y0 = 10
    panel_x1 = panel_x0 + left_pad + text_w + right_pad
    panel_y1 = panel_y0 + top_pad + total_text_h + bottom_pad

    # Draw semi-transparent background panel for readability
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x0, panel_y0),
        (panel_x1, panel_y1),
        (0, 0, 0),
        thickness=-1,
    )
    alpha = 0.55
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Render text lines on top of the panel
    x = panel_x0 + left_pad
    y = panel_y0 + top_pad
    for i, (text, ts) in enumerate(zip(lines, text_sizes)):
        # Draw shadow for contrast
        cv2.putText(frame, text, (x + 2, y + ts[1] + 2), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y + ts[1]), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += ts[1] + line_gap


def _get_frame(cap: cv2.VideoCapture, idx: int) -> Tuple[bool, any]:
    """
    Seek to frame idx and return (ok, frame).
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return False, None
    return True, frame


def annotate_game(
        video_path: Path,
        pgn_path: Path,
        out_json: Path,
) -> None:
    """
    Interactive tool to map plies in a PGN game to video frame indices.

    Output format:

      {
        "pgn": "<full pgn text>",
        "frame_for_ply": {
          "1": 42,
          "2": 78,
          ...
        }
      }
    """
    pgn_text, san_moves = _load_pgn_moves(pgn_path)
    total_plies = len(san_moves)
    _log.info(
        "Loaded PGN from %s with %d plies", pgn_path, total_plies
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if num_frames <= 0:
        _log.warning("Video reports no frame count, controls may be limited")

    frame_idx = 0
    ply_idx = 0
    frame_for_ply: Dict[int, int] = {}

    window_name = "GT annotation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Try to open in full screen for better visibility. Fall back silently.
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except cv2.error:
        pass

    while True:
        ok, frame = _get_frame(cap, frame_idx)
        if not ok:
            _log.info("Could not read frame %d, stopping", frame_idx)
            break

        _draw_overlay(
            frame,
            frame_idx,
            num_frames,
            ply_idx,
            san_moves,
            frame_for_ply,
        )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(0) & 0xFF

        if key in (27, ord("q")):  # ESC or q
            _log.info("User requested quit")
            break

        # One frame forward / back
        elif key == ord("j"):  # next frame
            frame_idx = min(frame_idx + 1, max(num_frames - 1, 0))
        elif key == ord("k"):  # previous frame
            frame_idx = max(frame_idx - 1, 0)

        # Jump 10 frames
        elif key == ord("J"):  # forward 10
            frame_idx = min(frame_idx + 10, max(num_frames - 1, 0))
        elif key == ord("K"):  # backward 10
            frame_idx = max(frame_idx - 10, 0)

        # Jump 100 frames
        elif key == ord("O"):  # forward 100
            frame_idx = min(frame_idx + 100, max(num_frames - 1, 0))
        elif key == ord("I"):  # backward 100
            frame_idx = max(frame_idx - 100, 0)

        # Jump 300 frames
        elif key == ord("L"):  # forward 300
            frame_idx = min(frame_idx + 300, max(num_frames - 1, 0))
        elif key == ord("H"):  # backward 300
            frame_idx = max(frame_idx - 300, 0)

        # Mark current frame as stable board after current ply
        elif key == ord("m"):
            if ply_idx >= total_plies:
                _log.info("All plies already processed, nothing to mark")
            else:
                ply_num = ply_idx + 1
                frame_for_ply[ply_num] = frame_idx
                _log.info(
                    "Marked ply %d (%s) at frame %d",
                    ply_num,
                    san_moves[ply_idx],
                    frame_idx,
                )
                ply_idx += 1
                # Optionally jump a bit forward for next ply
                frame_idx = min(frame_idx + 5, max(num_frames - 1, 0))

        # Skip current ply without marking
        elif key == ord("s"):
            if ply_idx < total_plies:
                _log.info(
                    "Skipped ply %d (%s)",
                    ply_idx + 1,
                    san_moves[ply_idx],
                )
                ply_idx += 1

        # Undo last mark
        elif key == ord("u"):
            if ply_idx <= 0:
                _log.info("Nothing to undo")
            else:
                last_ply = ply_idx
                if last_ply in frame_for_ply:
                    last_frame = frame_for_ply.pop(last_ply)
                    frame_idx = last_frame
                    _log.info(
                        "Removed mark for ply %d and returned to frame %d",
                        last_ply,
                        last_frame,
                    )
                ply_idx -= 1

        # Any other key: ignore and redraw
        else:
            continue

        # If we went beyond the last ply, keep ply_idx at total_plies
        if ply_idx > total_plies:
            ply_idx = total_plies

    cap.release()
    cv2.destroyAllWindows()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pgn": pgn_text,
        "frame_for_ply": {str(k): int(v) for k, v in frame_for_ply.items()},
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf8")
    _log.info("Saved ground truth JSON to %s", out_json)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive tool to annotate ply-to-frame mapping "
                    "for tournament videos.",
    )
    p.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the tournament video file",
    )
    p.add_argument(
        "--pgn",
        type=Path,
        required=True,
        help="Path to a PGN file for this game",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON file for ground truth",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    annotate_game(
        video_path=args.video,
        pgn_path=args.pgn,
        out_json=args.out,
    )


if __name__ == "__main__":
    main()
