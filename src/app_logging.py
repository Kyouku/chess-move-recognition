from __future__ import annotations

import logging
from typing import Optional

_DEFAULT_LEVEL = logging.INFO


def _create_root_logger() -> logging.Logger:
    logger = logging.getLogger("chess_live")
    if logger.handlers:
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


def set_log_level(level: int | str) -> None:
    """
    Set global log level, for example "DEBUG", "INFO", "WARNING".
    """
    if isinstance(level, str):
        name = level.upper()
        lvl = getattr(logging, name, logging.INFO)
    else:
        lvl = int(level)
    _root_logger.setLevel(lvl)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get module specific logger below the root "chess_live" logger.
    """
    if not name:
        return _root_logger
    return _root_logger.getChild(name)
