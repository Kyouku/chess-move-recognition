from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import chess
import chess.pgn

from src import config
from .app_logging import get_logger
from .types import MoveInfo

_log = get_logger(__name__)

PathLike = Union[Path, str]


def ensure_parent_dir(path: Path) -> None:
    """
    Ensure the parent directory of the given file path exists.

    Errors are logged as warnings and are non-fatal.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _log.warning("Could not ensure parent dir for %s: %s", path, exc)


def _normalize_path(path_like: Optional[PathLike]) -> Optional[Path]:
    """Convert a Path or string to Path, keep None as None."""
    if path_like is None:
        return None
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _safe_append_line(
        path_like: Optional[PathLike],
        line: str,
        description: str,
) -> None:
    """
    Append a single line to a text file.

    - path_like: destination path or None. If None, this is skipped.
    - line: line without trailing newline.
    - description: short label for logging, for example "move" or "FEN".
    """
    path = _normalize_path(path_like)
    if path is None:
        _log.debug(
            "Skipping append of %s because no log path is configured",
            description,
        )
        return

    try:
        ensure_parent_dir(path)
        with path.open("a", encoding="utf8") as f:
            f.write(line + "\n")
    except OSError as e:
        _log.warning("Could not append %s to log file %s: %s", description, path, e)


def write_moves_txt(moves: List[str], out_path: Optional[Path] = None) -> None:
    """
    Write the finished game in PGN format.

    - moves: list of SAN strings in order.
    - out_path: optional override for the output file path. Defaults to
      config.GAME_MOVES_TXT_PATH.
    """
    if not moves:
        _log.info("write_moves_txt called with empty move list, skipping")
        return

    target_path_like: PathLike = out_path if out_path is not None else config.GAME_MOVES_TXT_PATH
    out_path = _normalize_path(target_path_like)
    if out_path is None:
        _log.warning("No GAME_MOVES_TXT_PATH configured, cannot write PGN")
        return

    ensure_parent_dir(out_path)

    board = chess.Board()
    game = chess.pgn.Game()

    # Basic headers
    now = datetime.now()
    game.headers["Event"] = "Live Capture"
    game.headers["Site"] = "chess-live"
    game.headers["Date"] = now.strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    game.headers["White"] = "White"
    game.headers["Black"] = "Black"

    node = game

    try:
        for san in moves:
            move = board.parse_san(san)
            node = node.add_variation(move)
            board.push(move)
    except ValueError as exc:
        # If anything goes wrong, fall back to writing the SAN list only
        _log.warning(
            "Failed to build PGN from SAN list, writing SAN-only text instead: %s",
            exc,
        )
        try:
            ensure_parent_dir(out_path)
            with out_path.open("w", encoding="utf8") as f:
                f.write(" ".join(moves) + "\n")
            _log.info("Moves written to %s (SAN only)", out_path)
        except OSError as e:
            _log.warning("Could not write detected_moves to %s: %s", out_path, e)
        return

    # Result policy: only declare a result on checkmate, otherwise use "*"
    # This intentionally ignores draws (stalemate, repetition, 50-move rule)
    # and keeps the PGN result as "*" unless there is a checkmate.
    result = board.result() if board.is_checkmate() else "*"
    game.headers["Result"] = result

    exporter = chess.pgn.StringExporter(
        headers=True,
        variations=False,
        comments=False,
    )
    pgn_str = game.accept(exporter).strip()

    try:
        with out_path.open("w", encoding="utf8") as f:
            f.write(pgn_str + "\n")
        _log.info("PGN written to %s (%d moves)", out_path, len(moves))
    except OSError as e:
        _log.warning("Could not write PGN to %s: %s", out_path, e)


def append_move_log(info: MoveInfo, out_path: Optional[Path] = None) -> None:
    """
    Append a single move to a CSV-like log file as "uci;san;fen".

    - out_path: optional override for destination file, defaults to
      config.MOVES_LOG_PATH.
    - Errors are non fatal and will only be logged as warnings.
    """
    target = out_path if out_path is not None else config.MOVES_LOG_PATH

    move_obj = getattr(info, "move", None)
    uci = move_obj.uci() if move_obj is not None else ""
    san = getattr(info, "san", "") or ""
    fen_after = getattr(info, "fen_after", "") or ""

    line = f"{uci};{san};{fen_after}"
    _safe_append_line(target, line, "move")


def append_fen_log(fen: str, out_path: Optional[Path] = None) -> None:
    """
    Append a FEN snapshot to a text log (one FEN per line).

    - If out_path is None, uses config.FEN_LOG_PATH. If that is None, this is a no-op.
    - Ensures parent directory exists. Errors are logged as warnings and are non fatal.
    """
    target = out_path if out_path is not None else config.FEN_LOG_PATH
    _safe_append_line(target, fen, "FEN")
