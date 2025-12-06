from __future__ import annotations

"""
Shared helpers to load and save a board homography based on config.

Used by both the offline detection recorder and the live pipelines.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np

from src import config
from src.common.app_logging import get_logger

_log = get_logger(__name__)


def _get_homography_path() -> Optional[Path]:
    """
    Resolve the homography path from config.

    Returns a Path or None if homography caching is not configured.
    """
    use_saved = bool(getattr(config, "USE_SAVED_HOMOGRAPHY", False))
    if not use_saved:
        return None

    h_path = getattr(config, "HOMOGRAPHY_PATH", None)
    if not h_path:
        return None

    return Path(h_path)


def load_homography_from_disk() -> Optional[np.ndarray]:
    """
    Load a homography matrix from disk based on config.

    Returns a (3, 3) float32 numpy array if successful, otherwise None.
    """
    path = _get_homography_path()
    if path is None:
        return None

    if not path.exists():
        _log.info(
            "Configured to use saved homography, but file not found at %s",
            path,
        )
        return None

    try:
        homography = np.load(str(path))
    except Exception as exc:
        _log.warning(
            "Failed to load saved homography from %s: %s",
            path,
            exc,
        )
        return None

    if (
            isinstance(homography, np.ndarray)
            and homography.shape == (3, 3)
            and np.isfinite(homography).all()
    ):
        return homography.astype(np.float32)

    _log.warning(
        "Saved homography at %s is invalid; shape=%s",
        path,
        getattr(homography, "shape", None),
    )
    return None


def apply_saved_homography(pipeline: Any) -> bool:
    """
    Load homography from disk and apply it to a pipeline.

    The pipeline is expected to have attributes similar to LivePipeline:

      - _H_board
      - _is_calibrated

    Returns True if a valid homography was loaded and applied, otherwise False.
    """
    homography = load_homography_from_disk()
    if homography is None:
        return False

    try:
        setattr(pipeline, "_H_board", homography)
        setattr(pipeline, "_is_calibrated", True)
        path = _get_homography_path()
        _log.info("Loaded saved homography from %s", path)
        return True
    except Exception as exc:
        _log.warning(
            "Failed to apply loaded homography to pipeline: %s",
            exc,
        )
        return False


def save_homography_from_pipeline(pipeline: Any) -> None:
    """
    Save homography from a calibrated pipeline to disk based on config.

    Reads pipeline._H_board and writes it to HOMOGRAPHY_PATH if
    USE_SAVED_HOMOGRAPHY is enabled. This is best effort and never raises.
    """
    path = _get_homography_path()
    if path is None:
        return

    homography = getattr(pipeline, "_H_board", None)
    if not (isinstance(homography, np.ndarray) and homography.shape == (3, 3)):
        return

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        np.save(str(path), homography)
        try:
            homography_chk = np.load(str(path))
            if (
                    isinstance(homography_chk, np.ndarray)
                    and homography_chk.shape == (3, 3)
            ):
                _log.info("Saved homography to %s", path)
            else:
                _log.warning(
                    "Homography file written at %s but contents invalid (shape=%s)",
                    path,
                    getattr(homography_chk, "shape", None),
                )
        except Exception as exc:
            _log.warning(
                "Homography save verification failed for %s: %s",
                path,
                exc,
            )
    except Exception as exc:
        _log.warning("Failed to save homography to %s: %s", path, exc)
