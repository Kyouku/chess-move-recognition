from __future__ import annotations

from typing import Callable, Any

import cv2

try:
    from src.pipeline.multistage import config as _config
except ImportError:
    # Ensure _config is defined even if import fails to satisfy linters and type checkers
    _config = None  # type: ignore[assignment]
    _DEFAULT_ENABLED = True
else:
    _DEFAULT_ENABLED = bool(getattr(_config, "GUI_ENABLED", True))

_GUI_ENABLED: bool = _DEFAULT_ENABLED


def enable_gui(enabled: bool) -> None:
    """
    Set global flag whether OpenCV windows should be used.
    """
    global _GUI_ENABLED
    _GUI_ENABLED = bool(enabled)


def is_enabled() -> bool:
    return _GUI_ENABLED


def create_window(name: str, x: int = 50, y: int = 50) -> None:
    if not _GUI_ENABLED:
        return
    try:
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(name, x, y)
    except cv2.error:
        # headless or no display
        pass


def show_image(name: str, img: Any) -> None:
    if not _GUI_ENABLED:
        return
    cv2.imshow(name, img)


def wait_key(delay_ms: int = 1) -> int:
    """
    Returns an 8 bit key code or -1 when GUI is disabled.
    """
    if not _GUI_ENABLED:
        return -1
    return cv2.waitKey(delay_ms) & 0xFF


def set_mouse_callback(
        name: str,
        callback: Callable[[int, int, int, int, Any], None],
) -> None:
    if not _GUI_ENABLED:
        return
    cv2.setMouseCallback(name, callback)


def destroy_window(name: str) -> None:
    if not _GUI_ENABLED:
        return
    try:
        cv2.destroyWindow(name)
    except cv2.error:
        pass


def destroy_all_windows() -> None:
    if not _GUI_ENABLED:
        return
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
