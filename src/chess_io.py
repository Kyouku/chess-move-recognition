from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import chess
import chess.pgn

from . import config
from .app_logging import get_logger
from .io_utils import ensure_parent_dir
from .types import MoveInfo

_log = get_logger(__name__)


def write_moves_txt(moves: List[str], out_path: Optional[Path] = None) -> None:
    """
    Write the finished game in PGN format and include a Lichess link.

    moves: list of SAN strings in order.
    out_path: optional override for the output file path. Defaults to
              config.GAME_MOVES_TXT_PATH.
    """
    if not moves:
        return

    out_path = Path(out_path) if out_path is not None else config.GAME_MOVES_TXT_PATH
    # Ensure destination directory exists (non-fatal)
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
    uci_moves: List[str] = []

    try:
        for san in moves:
            move = board.parse_san(san)
            node = node.add_variation(move)
            board.push(move)
            uci_moves.append(move.uci())
    except ValueError as exc:
        # If anything goes wrong, fall back to writing the SAN list only
        _log.warning(
            "Failed to build PGN from SAN list, writing SAN-only text instead: %s",
            exc,
        )
        try:
            with open(out_path, "w", encoding="utf8") as f:
                f.write(" ".join(moves) + "\n")
            _log.info("Moves written to %s (SAN only)", out_path)
        except OSError as e:
            _log.warning("Could not write moves to %s: %s", out_path, e)
        return

    # Result policy: only declare a result on checkmate; otherwise use "*"
    # This intentionally ignores draws (stalemate, repetition, 50-move) and
    # keeps the PGN result as "*" unless there is a checkmate.
    result = board.result() if board.is_checkmate() else "*"
    game.headers["Result"] = result

    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    pgn_str = game.accept(exporter)

    # Compose output content: PGN then link
    out_lines = [
        pgn_str.strip()
    ]

    try:
        with open(out_path, "w", encoding="utf8") as f:
            f.write("\n".join(out_lines) + "\n")
        _log.info("PGN written to %s", out_path)
    except OSError as e:
        _log.warning("Could not write PGN to %s: %s", out_path, e)


def append_move_log(info: MoveInfo, out_path: Optional[Path] = None) -> None:
    """
    Append a single move to a CSV-like log file as "uci;san;fen".

    - out_path: optional override for destination file; defaults to
      config.MOVES_LOG_PATH.
    - Errors are non-fatal and will only be logged as warnings.
    """
    path = Path(out_path) if out_path is not None else config.MOVES_LOG_PATH
    try:
        ensure_parent_dir(path)
        uci = getattr(info.move, "uci", lambda: "")()
        san = getattr(info, "san", None) or ""
        fen_after = getattr(info, "fen_after", None) or ""
        with open(path, "a", encoding="utf8") as f:
            f.write(f"{uci};{san};{fen_after}\n")
    except OSError as e:
        _log.warning("Could not append move to log file %s: %s", path, e)


def append_fen_log(fen: str, out_path: Optional[Path] = None) -> None:
    """
    Append a FEN snapshot to a text log (one FEN per line).

    - If out_path is None, uses config.FEN_LOG_PATH. If that is None, this is a no-op.
    - Ensures parent directory exists. Errors are logged as warnings and are non-fatal.
    """
    # Resolve destination path
    if out_path is not None:
        path: Optional[Path] = Path(out_path)
    else:
        path = getattr(config, "FEN_LOG_PATH", None)

    if path is None:
        return

    try:
        ensure_parent_dir(path)
        with open(path, "a", encoding="utf8") as f:
            f.write(f"{fen}\n")
    except OSError as e:
        _log.warning("Could not append FEN to log file %s: %s", path, e)
