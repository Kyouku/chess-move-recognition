from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from src.app_gui import (
    is_enabled,
    create_window,
    destroy_window,
    set_mouse_callback,
    show_image,
    wait_key,
)
from src.common.app_logging import get_logger

_log = get_logger(__name__)

# -------------------------------------------------------------
# Helpers for automatic calibration
# -------------------------------------------------------------


PATTERN_SIZE: Tuple[int, int] = (7, 7)  # inner corners
CORNER_SUBPIX_WIN = (5, 5)
CORNER_SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    60,
    1e-4,
)


def _resize_to_fixed_long_edge(
        bgr: np.ndarray,
        target_long: int,
) -> Tuple[np.ndarray, float]:
    """
    Resize image so that its longer side equals target_long pixels.
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("Input image is empty in _resize_to_fixed_long_edge")
    h, w = bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge == 0:
        raise ValueError("Invalid image with zero dimension.")
    if long_edge == target_long:
        return bgr, 1.0
    scale = float(target_long) / float(long_edge)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(bgr, (new_w, new_h), interpolation=interp), scale


def _px_step_and_offset(
        board_size_px: int,
        margin_squares: float,
) -> Tuple[float, float]:
    """
    Pixel step (size of one square) and offset for a canonical board image
    with margins measured in squares.
    """
    step = board_size_px / (8.0 + 2.0 * margin_squares)
    offset = step * margin_squares
    return step, offset


def _s_margin_matrix(board_size_px: int, margin_squares: float) -> np.ndarray:
    """
    Scale translate transform that places the board into the output canvas
    with the desired margins.
    """
    step, offset = _px_step_and_offset(board_size_px, margin_squares)
    s_mat = np.array(
        [
            [step, 0.0, offset],
            [0.0, step, offset],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return s_mat


def _try_find_corners(
        gray: np.ndarray,
        pattern_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Detect inner chessboard corners.

    Returns a float32 array of corners or None if detection fails.
    """
    corners: Optional[np.ndarray] = None

    # Flags for corner detection
    pattern_flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FILTER_QUADS
    )

    # Attempt modern SB method first
    if hasattr(cv2, "findChessboardCornersSB"):
        sb_res = cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
        if isinstance(sb_res, tuple):
            ok, crn = sb_res
            if ok and crn is not None:
                corners = np.asarray(crn, dtype=np.float32)
        elif sb_res is not None:
            corners = np.asarray(sb_res, dtype=np.float32)

    # Classic fallback variants
    if corners is None:
        ok, crn = cv2.findChessboardCorners(gray, pattern_size, pattern_flags)
        if ok and crn is not None:
            corners = np.asarray(crn, dtype=np.float32)

    if corners is None:
        try:
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            ok, crn = cv2.findChessboardCorners(gray_blur, pattern_size, pattern_flags)
            if ok and crn is not None:
                corners = np.asarray(crn, dtype=np.float32)
        except cv2.error:
            pass

    if corners is None:
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)
            ok, crn = cv2.findChessboardCorners(gray_eq, pattern_size, pattern_flags)
            if ok and crn is not None:
                corners = np.asarray(crn, dtype=np.float32)
        except cv2.error:
            pass

    return corners


def _auto_homography_from_frame_pre(
        frame_pre: np.ndarray,
        pattern_size: Tuple[int, int],
        board_size_px: int,
        margin_squares: float,
        min_board_area_ratio: float = 0.0,
) -> Optional[np.ndarray]:
    """
    Compute homography from a pre resized frame to the canonical board image
    using inner corners and a virtual board coordinate system.
    Includes a sanity check on board area to reject spurious detections.
    """
    gray = cv2.cvtColor(frame_pre, cv2.COLOR_BGR2GRAY)

    corners = _try_find_corners(gray, pattern_size)
    if corners is None:
        # No corners found
        return None

    # Reject boards that are too small in the frame
    img_h, img_w = gray.shape[:2]
    img_area = float(img_h * img_w)
    x_box, y_box, w_box, h_box = cv2.boundingRect(corners)
    board_area = float(w_box * h_box)
    area_ratio = board_area / img_area

    if area_ratio < float(min_board_area_ratio):
        _log.debug(
            "[Stage1] Rejecting candidate: board_area_ratio=%.4f < min=%.4f",
            area_ratio,
            float(min_board_area_ratio),
        )
        return None

    try:
        cv2.cornerSubPix(
            gray,
            corners,
            winSize=CORNER_SUBPIX_WIN,
            zeroZone=(-1, -1),
            criteria=CORNER_SUBPIX_CRITERIA,
        )
    except cv2.error:
        pass

    imgp = corners.reshape(-1, 2).astype(np.float32)
    nx, ny = pattern_size

    # Object points in board coordinate system (unit = one square)
    # Inner corners from (1, 1) to (7, 7)
    objp = np.array(
        [[ix + 1, iy + 1] for iy in range(ny) for ix in range(nx)],
        dtype=np.float32,
    )

    h_board, _ = cv2.findHomography(imgp, objp, method=cv2.RANSAC)
    if h_board is None or not np.isfinite(h_board).all():
        return None

    margin_mat = _s_margin_matrix(board_size_px, margin_squares)
    h_total = (margin_mat @ h_board).astype(np.float32)
    return h_total


def _compute_board_outer_corners(
        board_size_px: int,
        margin_squares: float,
) -> np.ndarray:
    """
    Compute canonical coordinates of the outer board corners in the
    output image with margins.
    Order: TL, TR, BR, BL.
    """
    step, offset = _px_step_and_offset(board_size_px, margin_squares)
    tl = (offset, offset)
    tr = (offset + 8 * step, offset)
    br = (offset + 8 * step, offset + 8 * step)
    bl = (offset, offset + 8 * step)
    return np.array([tl, tr, br, bl], dtype=np.float32)


class LivePipeline:
    """
    Stage 1 pipeline for realtime use:
      - automatic calibration via chessboard corner detection
      - fallback to manual corner selection if auto fails
      - board rectification each frame

    All parameters are passed through the constructor, there is no
    direct access to a global config module.
    """

    def __init__(
            self,
            frame_width: int,
            frame_height: int,
            board_size_px: int = 1024,
            margin_squares: float = 1.7,
            display: bool = True,
            input_target_long_edge: Optional[int] = None,
            min_board_area_ratio: float = 0.0,
    ) -> None:
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.board_size_px = int(board_size_px)
        self.margin_squares = float(margin_squares)
        self.display = display

        self._frame_idx = 0
        self._is_calibrated = False

        self._H_board: Optional[np.ndarray] = None

        # Target long edge for preprocessing
        default_long = max(self.frame_width, self.frame_height)
        self._target_long_edge = int(
            input_target_long_edge if input_target_long_edge is not None else default_long,
        )

        self._min_board_area_ratio = float(min_board_area_ratio)

        # Last warped board for downstream detectors
        self._last_warped_board: Optional[np.ndarray] = None

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def last_warped_board(self) -> Optional[np.ndarray]:
        return self._last_warped_board

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resize_to_frame_size(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize raw capture frame to frame_width x frame_height.
        Calibration and runtime see the same geometry.
        """
        target_w, target_h = self.frame_width, self.frame_height
        h, w = frame.shape[:2]
        if w == target_w and h == target_h:
            return frame
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # Calibration: auto then manual
    # ------------------------------------------------------------------

    def calibrate_from_capture(
            self,
            cap: cv2.VideoCapture,
            max_frames: int = 300,
    ) -> bool:
        """
        Perform calibration from a short sequence.

        If automatic calibration succeeds, no manual window is shown.
        Only if automatic calibration fails, a manual calibration
        window is displayed to let the user select the four board corners.
        """
        win_name = "Calibration"

        auto_limit = max_frames
        last_pre_frame: Optional[np.ndarray] = None

        _log.info("Starting automatic calibration (silent).")
        _log.debug(
            "[Stage1] Auto calibration params: target_long_edge=%d, board_size_px=%d, min_board_area_ratio=%.3f",
            self._target_long_edge,
            self.board_size_px,
            self._min_board_area_ratio,
        )

        for i in range(auto_limit):
            ret, frame = cap.read()
            if not ret:
                break

            frame = self._resize_to_frame_size(frame)

            frame_pre, _ = _resize_to_fixed_long_edge(
                frame,
                self._target_long_edge,
            )
            last_pre_frame = frame_pre

            h_total = _auto_homography_from_frame_pre(
                frame_pre,
                PATTERN_SIZE,
                self.board_size_px,
                self.margin_squares,
                self._min_board_area_ratio,
            )

            # No GUI during automatic calibration; run silently
            if h_total is None:
                continue

            # Use this homography
            self._H_board = h_total.astype(np.float32)
            self._is_calibrated = True

            _log.info("Calibration: auto homography set on frame %d", i + 1)
            _log.debug(
                "H_board=%s board_size_px=%d target_long_edge=%d margin_squares=%.3f",
                self._H_board,
                self.board_size_px,
                self._target_long_edge,
                self.margin_squares,
            )
            return True

        # Manual fallback only when GUI is active
        if not is_enabled():
            _log.warning(
                "Auto calibration did not find a board and GUI is disabled, "
                "manual calibration is not possible.",
            )
            return False

        _log.info(
            "Auto calibration did not find a board, switching to manual corner selection.",
        )
        if last_pre_frame is None:
            _log.warning("No frame available for manual calibration.")
            return False

        # Create the window only for manual calibration
        create_window(win_name, 50, 50)

        manual_ok = self._manual_calibration(
            last_pre_frame,
            win_name,
            self.margin_squares,
        )

        destroy_window(win_name)

        self._is_calibrated = manual_ok
        if manual_ok:
            _log.info("Manual calibration successful.")
        else:
            _log.warning("Manual calibration failed.")

        return manual_ok

    def _manual_calibration(
            self,
            frame_pre: np.ndarray,
            win_name: str,
            margin_squares: float,
    ) -> bool:
        """
        Manual calibration.

        User clicks 4 corners of the board on the pre resized frame:
          1 top left
          2 top right
          3 bottom right
          4 bottom left
        """
        if not is_enabled():
            _log.warning("Manual calibration requested but GUI is disabled.")
            return False

        points: list[tuple[int, int]] = []

        def on_mouse(event, x_pos, y_pos, _mouse_flags, _param) -> None:
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x_pos, y_pos))

        set_mouse_callback(win_name, on_mouse)

        while True:
            display = frame_pre.copy()
            cv2.putText(
                display,
                "Click TL, TR, BR, BL. ENTER when done, ESC to cancel.",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            for idx, (px, py) in enumerate(points):
                cv2.circle(display, (px, py), 6, (0, 255, 0), -1)
                cv2.putText(
                    display,
                    str(idx + 1),
                    (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            show_image(win_name, display)
            key = wait_key(20)
            if key == 27:
                _log.info("Manual calibration cancelled.")
                return False
            if key in (13, 32) and len(points) == 4:
                break

        if len(points) != 4:
            _log.warning("Need exactly 4 points for calibration.")
            return False

        src = np.array(points, dtype=np.float32)
        dst = _compute_board_outer_corners(
            self.board_size_px,
            margin_squares,
        )

        h_total, _ = cv2.findHomography(src, dst, method=cv2.RANSAC)
        if h_total is None or not np.isfinite(h_total).all():
            _log.warning("Failed to compute homography from manual points.")
            return False

        self._H_board = h_total.astype(np.float32)

        _log.info("Calibration: manual homography set.")
        _log.debug(
            "H_board=%s board_size_px=%d target_long_edge=%d margin_squares=%.3f",
            self._H_board,
            self.board_size_px,
            self._target_long_edge,
            self.margin_squares,
        )

        return True

    # ------------------------------------------------------------------
    # Per frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Main call from live loop.

        Rectifies the board on every frame and stores it for detectors.
        Returns only the warped board view for display if calibrated,
        otherwise the raw frame.
        """
        self._frame_idx += 1

        if self._frame_idx == 1:
            _log.debug(
                "[Stage1] First frame: input shape=%s target_long_edge=%d "
                "board_size_px=%d margin_squares=%.3f",
                frame_bgr.shape,
                self._target_long_edge,
                self.board_size_px,
                self.margin_squares,
            )

        if not self._is_calibrated or self._H_board is None:
            return frame_bgr

        warped_board = self._rectify_board(frame_bgr)
        return warped_board

    def _rectify_board(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Warp the full camera frame to a fixed board_size_px x board_size_px image.
        """
        frame_pre, _ = _resize_to_fixed_long_edge(
            frame_bgr,
            self._target_long_edge,
        )
        warped = cv2.warpPerspective(
            frame_pre,
            self._H_board,
            (self.board_size_px, self.board_size_px),
            flags=cv2.INTER_LINEAR,
        )

        self._last_warped_board = warped
        return warped
