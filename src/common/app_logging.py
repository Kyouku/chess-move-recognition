from __future__ import annotations

import logging
from typing import Optional, Union

_DEFAULT_LEVEL = logging.INFO

LevelLike = Union[int, str]


def _create_root_logger() -> logging.Logger:
    """
    Create or return the root logger for the application namespace "chess_live".
    """
    logger = logging.getLogger("chess_live")
    if logger.handlers:
        # Already initialized in this process
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        "%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(_DEFAULT_LEVEL)
    logger.propagate = False
    return logger


_root_logger = _create_root_logger()


def set_log_level(level: LevelLike) -> None:
    """
    Set global log level, for example "DEBUG", "INFO", "WARNING" or an int.
    """
    if isinstance(level, str):
        name = level.upper()
        numeric = logging.getLevelName(name)
        if isinstance(numeric, int):
            numeric_level = numeric
        else:
            _root_logger.warning(
                "Unknown log level name %r, falling back to INFO",
                level,
            )
            numeric_level = logging.INFO
    else:
        numeric_level = int(level)

    _root_logger.setLevel(numeric_level)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a module specific logger below the root "chess_live" logger.

    If name is None or empty, the root logger is returned.
    """
    if not name:
        return _root_logger

    # Allow passing fully qualified names that already start with "chess_live."
    prefix = "chess_live."
    if name.startswith(prefix):
        name = name[len(prefix):]

    return _root_logger.getChild(name)
