from __future__ import annotations

"""
Shared live base for multistage and baseline pipelines.

Handles capture, Stage 1 rectification, Stage 2 detection and GUI.
Stage 3 logic is delegated to subclasses via handle_detection_state.
"""

import queue
import threading
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Dict, List

import cv2
import numpy as np

from src import config
from src.app_gui import show_image, wait_key, destroy_all_windows, enable_gui
from src.common.app_logging import get_logger
from src.common.homography_cache import (
    apply_saved_homography,
    save_homography_from_pipeline,
)
from src.common.types import DetectionState
from src.stage1.board_rectifier import LivePipeline
from src.stage2.piece_detection import PieceDetector
from src.stage2.piece_overlay import draw_piece_overlay

_log = get_logger(__name__)

CaptureSource = Union[int, str]


class FrameReader(threading.Thread):
    """
    Read frames from camera or video file in a background thread.

    The queue only keeps the most recent frame so that slow downstream
    processing does not create a large backlog.
    """

    def __init__(
            self,
            source: CaptureSource,
            width: int,
            height: int,
            frame_queue: "queue.Queue[np.ndarray]",
            stop_event: threading.Event,
            name: str = "FrameReader",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.source = source
        self.width = width
        self.height = height
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self._cap: Optional[cv2.VideoCapture] = None

    def run(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            _log.error("Could not open source: %s", self.source)
            return

        # Hint desired capture properties to reduce downstream resizing cost
        try:
            if self.width > 0:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            if self.height > 0:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
            # Reduce internal buffering if backend supports it
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass

        # Log both the requested and the actual negotiated capture size
        try:
            act_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            act_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except cv2.error:
            act_w, act_h = -1, -1
        _log.info(
            "Capture started from %s (requested %dx%d, got %dx%d)",
            self.source,
            int(self.width),
            int(self.height),
            act_w,
            act_h,
        )

        while not self.stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                frame = cv2.resize(
                    frame,
                    (self.width, self.height),
                    interpolation=cv2.INTER_AREA,
                )

            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    _ = self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put_nowait(frame)

        if self._cap is not None:
            self._cap.release()
        _log.info("FrameReader stopped")


class DetectionWorker(threading.Thread):
    """
    Stage 2 worker thread.

    Consumes rectified boards, runs the piece detector and produces a
    DetectionState per board.
    """

    def __init__(
            self,
            detector: PieceDetector,
            input_queue: "queue.Queue[np.ndarray]",
            output_queue: "queue.Queue[DetectionState]",
            stop_event: threading.Event,
            name: str = "DetectionWorker",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.detector = detector
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event

    def run(self) -> None:
        _log.info("%s started", self.name)
        try:
            while not self.stop_event.is_set():
                try:
                    warped_board = self.input_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                try:
                    occupancy, pieces, boxes, confs = self.detector.detect(warped_board)
                    state = DetectionState(
                        occupancy=occupancy,
                        pieces=pieces,
                        boxes=boxes,
                        confidences=confs,
                    )

                    try:
                        self.output_queue.put_nowait(state)
                    except queue.Full:
                        try:
                            _ = self.output_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.output_queue.put_nowait(state)
                except (cv2.error, ValueError, RuntimeError) as exc:
                    _log.error("%s detect() error: %s", self.name, exc, exc_info=True)
        except (RuntimeError, cv2.error, ValueError) as exc:
            _log.error("%s loop error: %s", self.name, exc, exc_info=True)

        _log.info("%s stopped", self.name)


def get_capture_source() -> CaptureSource:
    if config.USE_VIDEO_FILE:
        return str(config.VIDEO_PATH)
    return int(config.CAMERA_INDEX)


def _calibrate_pipeline(
        pipeline: LivePipeline,
        source: CaptureSource,
        desired_width: Optional[int] = None,
        desired_height: Optional[int] = None,
) -> bool:
    """
    Calibrate Stage 1 once before starting the live threads.

    For camera sources this requests the same resolution as used at runtime.
    """
    # First try to load a previously saved homography and skip re calibration
    try:
        if apply_saved_homography(pipeline):
            return True
    except Exception:
        pass

    temp_cap = cv2.VideoCapture(source)
    if not temp_cap.isOpened():
        _log.error("Could not open source for calibration: %s", source)
        return False

    try:
        if isinstance(source, int):
            if desired_width and desired_width > 0:
                temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(desired_width))
            if desired_height and desired_height > 0:
                temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(desired_height))
            temp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except cv2.error:
        pass

    _log.info("Calibrating from source...")
    ok = pipeline.calibrate_from_capture(
        temp_cap,
        max_frames=config.CALIBRATION_MAX_FRAMES,
    )
    temp_cap.release()
    if not ok:
        _log.error("Calibration failed, aborting")
        return False
    _log.info("Calibration successful")

    try:
        save_homography_from_pipeline(pipeline)
    except Exception:
        pass

    return True


def _probe_source_size(
        source: CaptureSource,
        fallback_w: int,
        fallback_h: int,
) -> Tuple[int, int]:
    """
    Open the source once and return the actual frame size reported by OpenCV.

    If probing fails, return the provided fallback size.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        _log.warning(
            "Could not open source %s for size probing, falling back to %dx%d",
            source,
            fallback_w,
            fallback_h,
        )
        return int(fallback_w), int(fallback_h)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if w <= 0 or h <= 0:
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
    cap.release()

    if w <= 0 or h <= 0:
        _log.warning(
            "Source size probing returned invalid %dx%d, using fallback %dx%d",
            w,
            h,
            fallback_w,
            fallback_h,
        )
        return int(fallback_w), int(fallback_h)

    return w, h


class BaseLivePipeline(ABC):
    """
    Base class that orchestrates the live chess pipeline.

    Separation of concerns:

      - Frame acquisition (FrameReader) in its own thread
      - Rectification (Stage 1) by LivePipeline
      - Piece detection (Stage 2) by DetectionWorker threads
      - Stage 3 logic in subclasses via handle_detection_state
    """

    def __init__(
            self,
            source: CaptureSource,
            *,
            width: Optional[int] = None,
            height: Optional[int] = None,
            board_size_px: int = 640,
            window_name: str = "Live",
    ) -> None:
        self.source = source
        if width is None:
            width = config.FRAME_WIDTH
        if height is None:
            height = config.FRAME_HEIGHT

        # Decide capture size strategy based on source type
        if isinstance(source, int):
            self.width = int(width)
            self.height = int(height)
        else:
            probed_w, probed_h = _probe_source_size(source, width, height)
            self.width = int(probed_w)
            self.height = int(probed_h)

        # Expose the actual source size via config for any downstream users
        config.set_actual_frame_size(self.width, self.height)
        config.FRAME_WIDTH = self.width
        config.FRAME_HEIGHT = self.height

        self.board_size_px = int(board_size_px)
        self.window_name = window_name

        self.stop_event = threading.Event()

        # Queues for frames and detection states
        self.frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(
            maxsize=config.FRAME_QUEUE_SIZE,
        )
        self.det_input_queue: "queue.Queue[np.ndarray]" = queue.Queue(
            maxsize=config.DETECTION_INPUT_QUEUE_SIZE,
        )
        self.det_output_queue: "queue.Queue[DetectionState]" = queue.Queue(
            maxsize=config.DETECTION_OUTPUT_QUEUE_SIZE,
        )

        # Stage 1: board rectifier
        self.pipeline = LivePipeline(
            frame_width=self.width,
            frame_height=self.height,
            board_size_px=self.board_size_px,
            margin_squares=config.BOARD_MARGIN_SQUARES,
            input_target_long_edge=config.CALIBRATION_TARGET_LONG_EDGE,
            min_board_area_ratio=config.AUTO_MIN_BOARD_AREA_RATIO,
            display=config.GUI_ENABLED,
        )

        # Stage 2: detector(s)
        self.detector = PieceDetector(
            weights=config.YOLO_PIECE_WEIGHTS,
            squares=config.BOARD_SQUARES,
            imgsz=config.YOLO_PIECE_IMGSZ,
            margin_squares=config.BOARD_MARGIN_SQUARES,
            conf_threshold=config.YOLO_PIECE_CONF,
            min_iou=config.MIN_IOU,
        )

        self.reader_thread = FrameReader(
            source=self.source,
            width=self.width,
            height=self.height,
            frame_queue=self.frame_queue,
            stop_event=self.stop_event,
        )

        self.det_workers: List[DetectionWorker] = []

        num_workers = config.DETECTION_WORKERS
        if num_workers <= 1:
            worker = DetectionWorker(
                detector=self.detector,
                input_queue=self.det_input_queue,
                output_queue=self.det_output_queue,
                stop_event=self.stop_event,
                name="DetectionWorker-1",
            )
            self.det_workers.append(worker)
        else:
            for i in range(num_workers):
                det_i = PieceDetector(
                    weights=config.YOLO_PIECE_WEIGHTS,
                    squares=config.BOARD_SQUARES,
                    imgsz=config.YOLO_PIECE_IMGSZ,
                    margin_squares=config.BOARD_MARGIN_SQUARES,
                    conf_threshold=config.YOLO_PIECE_CONF,
                    min_iou=config.MIN_IOU,
                )
                worker = DetectionWorker(
                    detector=det_i,
                    input_queue=self.det_input_queue,
                    output_queue=self.det_output_queue,
                    stop_event=self.stop_event,
                    name=f"DetectionWorker-{i + 1}",
                )
                self.det_workers.append(worker)

        # Latest detection info for overlay
        self.latest_state: Optional[DetectionState] = None
        self.latest_pieces: Optional[Dict[str, Optional[str]]] = None
        self.latest_boxes: Optional[
            Dict[str, Optional[Tuple[float, float, float, float]]]
        ] = None
        self.latest_confs: Optional[Dict[str, Optional[float]]] = None

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def handle_detection_state(self, state: DetectionState) -> None:
        """
        Called for every DetectionState drained from the Stage 2 queue.

        Subclasses implement their Stage 3 logic here.
        """
        raise NotImplementedError

    def after_detection_batch(self) -> None:
        """
        Optional hook that runs once per frame after all DetectionStates
        have been drained and handle_detection_state has been called.
        """
        return None

    def on_start(self) -> None:
        """
        Hook called after Stage 1 and Stage 2 threads have been started,
        but before entering the main loop.
        """
        return None

    def on_stop(self) -> None:
        """
        Hook called during shutdown after Stage 1 and Stage 2 threads have
        been requested to stop and joined.
        """
        return None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_opencv() -> None:
        cv2.setUseOptimized(True)
        threads = config.OPENCV_NUM_THREADS
        if threads > 0:
            cv2.setNumThreads(threads)

    @staticmethod
    def _configure_gui() -> None:
        enable_gui(config.GUI_ENABLED)

    # ------------------------------------------------------------------
    # Main run and stop logic
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._configure_opencv()
        self._configure_gui()

        if not self._calibrate_stage1():
            return

        self._start_core_threads()

        self.on_start()

        _log.info("%s started. Press ESC to stop.", self.__class__.__name__)

        try:
            while True:
                frame, finished = self._next_frame()
                if finished:
                    break
                if frame is None:
                    continue

                warped_or_raw = self._rectify_or_passthrough(frame)
                self._try_enqueue_for_detection()
                self._drain_detection_output()
                self.after_detection_batch()

                display_frame = self._compose_display_frame(warped_or_raw, frame)
                if self._handle_gui(display_frame):
                    break
        finally:
            self.stop()

    def stop(self) -> None:
        self.stop_event.set()
        self._stop_core_threads()
        self.on_stop()
        destroy_all_windows()
        _log.info("%s stopped cleanly", self.__class__.__name__)

    # ------------------------------------------------------------------
    # Private helpers, orchestration building blocks
    # ------------------------------------------------------------------

    def _calibrate_stage1(self) -> bool:
        return _calibrate_pipeline(
            self.pipeline,
            self.source,
            desired_width=self.width,
            desired_height=self.height,
        )

    def _start_core_threads(self) -> None:
        self.reader_thread.start()
        for worker in self.det_workers:
            worker.start()

    def _stop_core_threads(self) -> None:
        try:
            self.reader_thread.join(timeout=1.0)
        except RuntimeError:
            pass
        for worker in self.det_workers:
            try:
                worker.join(timeout=1.0)
            except RuntimeError:
                pass

    def _next_frame(self) -> Tuple[Optional[np.ndarray], bool]:
        """
        Attempt to get the next frame.

        Returns tuple (frame, finished):
          - frame: image array or None if none is currently available
          - finished: True if the reader thread has ended and no more
            frames will come
        """
        try:
            return self.frame_queue.get(timeout=1.0), False
        except queue.Empty:
            if not self.reader_thread.is_alive():
                _log.info("FrameReader finished")
                return None, True
            return None, False

    def _rectify_or_passthrough(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run Stage 1 rectification and return warped board if available.

        If rectification is not ready, this returns None and callers can
        fall back to the original frame.
        """
        return self.pipeline.process_frame(frame)

    def _try_enqueue_for_detection(self) -> None:
        warped_board = self.pipeline.last_warped_board
        if warped_board is None:
            return
        try:
            self.det_input_queue.put_nowait(warped_board)
        except queue.Full:
            pass

    def _drain_detection_output(self) -> None:
        while True:
            try:
                det_state = self.det_output_queue.get_nowait()
            except queue.Empty:
                break

            self.latest_state = det_state
            self.latest_pieces = det_state.pieces
            self.latest_boxes = det_state.boxes
            self.latest_confs = det_state.confidences

            self.handle_detection_state(det_state)

    def _compose_display_frame(
            self,
            warped_or_raw: Optional[np.ndarray],
            raw_frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        display_frame = warped_or_raw if warped_or_raw is not None else raw_frame
        if (
                display_frame is not None
                and self.latest_pieces is not None
                and self.pipeline.is_calibrated
        ):
            display_frame = draw_piece_overlay(
                display_frame,
                self.latest_pieces,
                squares=config.BOARD_SQUARES,
                margin_squares=config.BOARD_MARGIN_SQUARES,
                raw_boxes=self.latest_boxes,
                confidences=self.latest_confs,
            )
        return display_frame

    def _handle_gui(self, display_frame: Optional[np.ndarray]) -> bool:
        """
        Render the current frame if GUI is enabled.

        Returns True if the user pressed ESC and requested exit.
        """
        if display_frame is not None:
            show_image(self.window_name, display_frame)
        key = wait_key(1)
        return key == 27  # ESC
