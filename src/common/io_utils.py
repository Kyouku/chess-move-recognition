from __future__ import annotations

from pathlib import Path

from .app_logging import get_logger

_log = get_logger(__name__)


def ensure_parent_dir(path: Path) -> None:
    """
    Ensure the parent directory of the given file path exists.

    Errors are logged as warnings and are non-fatal.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _log.warning("Could not ensure parent dir for %s: %s", path, exc)
