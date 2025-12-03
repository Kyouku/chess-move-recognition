from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List

import cv2
import numpy as np

from src import config
from src.app_gui import show_image, wait_key, destroy_all_windows, enable_gui
from src.app_logging import get_logger
from src.stage1.board_rectifier import LivePipeline
from src.stage2.piece_detection import PieceDetector
from src.stage2.piece_overlay import draw_piece_overlay
from src.types import DetectionState

_log = get_logger(__name__)

CaptureSource = Union[int, str]


class FrameReader(threading.Thread):
    """
    Read frames from camera or video file on a background thread.
    Keeps only the most recent frame to avoid backlog.
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
            # Reduce internal buffering if backend supports it (lower latency)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            # Some backends may not support these properties
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

            # Resize to the expected geometry so downstream sees consistent size
            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                frame = cv2.resize(
                    frame,
                    (self.width, self.height),
                    interpolation=cv2.INTER_AREA,
                )

            # Keep only the latest frame in the queue
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
    DetectionState (occupancy, labels, raw boxes, confidences) per board.
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

                    # Keep only the newest result in the output
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
    """Return camera index or video path based on config."""
    if getattr(config, "USE_VIDEO_FILE", False):
        return str(getattr(config, "VIDEO_PATH"))
    return int(getattr(config, "CAMERA_INDEX", 0))


def _calibrate_pipeline(
        pipeline: LivePipeline,
        source: CaptureSource,
        desired_width: Optional[int] = None,
        desired_height: Optional[int] = None,
) -> bool:
    """Calibrate stage 1 once before starting the live threads.

    For camera sources we request the desired resolution to match runtime.
    """
    # If configured, try to load a previously saved homography and skip calibration
    try:
        use_saved = bool(getattr(config, "USE_SAVED_HOMOGRAPHY", False))
        if use_saved:
            h_path = getattr(config, "HOMOGRAPHY_PATH", None)
            if h_path is not None:
                try:
                    h_file = Path(h_path)
                    if h_file.exists():
                        H = np.load(str(h_file))
                        if isinstance(H, np.ndarray) and H.shape == (3, 3) and np.isfinite(H).all():
                            pipeline._H_board = H.astype(np.float32)  # type: ignore[attr-defined]
                            pipeline._is_calibrated = True  # type: ignore[attr-defined]
                            _log.info("Loaded saved homography from %s", h_file)
                            return True
                        else:
                            _log.warning("Saved homography at %s is invalid; falling back to calibration.", h_file)
                    else:
                        _log.info("Configured to use saved homography, but file not found at %s; calibrating.", h_file)
                except Exception as exc:  # noqa: BLE001 - log and continue to calibrate
                    _log.warning("Failed to load saved homography: %s; calibrating instead.", exc)
    except Exception:
        # If config attributes are missing or any error occurs, just proceed with calibration
        pass

    temp_cap = cv2.VideoCapture(source)
    if not temp_cap.isOpened():
        _log.error("Could not open source for calibration: %s", source)
        return False

    # Try to request the same size as we will use at runtime (camera only)
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
        max_frames=getattr(config, "CALIBRATION_MAX_FRAMES", 200),
    )
    temp_cap.release()
    if not ok:
        _log.error("Calibration failed, aborting")
        return False
    _log.info("Calibration successful")

    # Always save homography after a successful calibration so it can be reused
    try:
        h_path = getattr(config, "HOMOGRAPHY_PATH", None)
        H = getattr(pipeline, "_H_board", None)
        if h_path is not None and isinstance(H, np.ndarray) and H.shape == (3, 3):
            h_file = Path(h_path)
            # Ensure directory exists
            try:
                h_file.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            try:
                np.save(str(h_file), H)
                # Verify immediately
                try:
                    H_chk = np.load(str(h_file))
                    if isinstance(H_chk, np.ndarray) and H_chk.shape == (3, 3):
                        _log.info("Saved homography to %s", h_file)
                    else:
                        _log.warning("Homography file written at %s but contents invalid (shape=%s)", h_file,
                                     getattr(H_chk, 'shape', None))
                except Exception as exc:
                    _log.warning("Homography save verification failed for %s: %s", h_file, exc)
            except Exception as exc:  # noqa: BLE001
                _log.warning("Failed to save homography to %s: %s", h_file, exc)
    except Exception:
        # Do not fail the run if saving failed
        pass
    return True


def _probe_source_size(
        source: CaptureSource, fallback_w: int, fallback_h: int
) -> Tuple[int, int]:
    """Open the source once and return the actual frame size reported by OpenCV.

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

    # Read the reported size from the capture
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Some backends report 0 until a frame is read; try to grab one frame
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
    Base class that wires Stage 1 (rectification) and Stage 2 (piece detection)
    into a threaded live app. Subclasses provide Stage 3 logic (move inference).
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
            width = int(getattr(config, "FRAME_WIDTH", 1280))
        if height is None:
            height = int(getattr(config, "FRAME_HEIGHT", 720))

        # Decide capture size strategy based on source type
        if isinstance(source, int):
            self.width = int(width)
            self.height = int(height)
        else:
            probed_w, probed_h = _probe_source_size(source, width, height)
            self.width = int(probed_w)
            self.height = int(probed_h)

        # Expose the actual source size via config for any downstream users
        if hasattr(config, "set_actual_frame_size"):
            try:
                config.set_actual_frame_size(self.width, self.height)
            except (AttributeError, TypeError):
                pass
        if hasattr(config, "FRAME_WIDTH"):
            config.FRAME_WIDTH = self.width
        if hasattr(config, "FRAME_HEIGHT"):
            config.FRAME_HEIGHT = self.height

        self.board_size_px = int(board_size_px)
        self.window_name = window_name

        self.stop_event = threading.Event()

        # Queues for frames and detection states
        self.frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(
            maxsize=getattr(config, "FRAME_QUEUE_SIZE", 3)
        )
        self.det_input_queue: "queue.Queue[np.ndarray]" = queue.Queue(
            maxsize=getattr(config, "DETECTION_INPUT_QUEUE_SIZE", 1)
        )
        self.det_output_queue: "queue.Queue[DetectionState]" = queue.Queue(
            maxsize=getattr(config, "DETECTION_OUTPUT_QUEUE_SIZE", 3)
        )

        # Stage 1: board rectifier
        self.pipeline = LivePipeline(
            frame_width=self.width,
            frame_height=self.height,
            board_size_px=self.board_size_px,
            margin_squares=float(getattr(config, "BOARD_MARGIN_SQUARES", 1.7)),
            input_target_long_edge=int(
                getattr(
                    config,
                    "CALIBRATION_TARGET_LONG_EDGE",
                    max(self.width, self.height),
                )
            ),
            min_board_area_ratio=float(
                getattr(config, "AUTO_MIN_BOARD_AREA_RATIO", 0.0)
            ),
            display=bool(getattr(config, "GUI_ENABLED", True)),
        )

        # Stage 2: detector(s)
        self.detector = PieceDetector(
            weights=getattr(config, "YOLO_PIECE_WEIGHTS"),
            squares=int(getattr(config, "BOARD_SQUARES", 8)),
            imgsz=int(getattr(config, "YOLO_PIECE_IMGSZ", 640)),
            margin_squares=float(getattr(config, "BOARD_MARGIN_SQUARES", 1.7)),
            conf_threshold=float(getattr(config, "YOLO_PIECE_CONF", 0.5)),
            min_iou=float(getattr(config, "MIN_IOU", 0.15)),
        )

        self.reader_thread = FrameReader(
            source=self.source,
            width=self.width,
            height=self.height,
            frame_queue=self.frame_queue,
            stop_event=self.stop_event,
        )

        self.det_workers: List[DetectionWorker] = []

        num_workers = int(getattr(config, "DETECTION_WORKERS", 1))
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
                    weights=getattr(config, "YOLO_PIECE_WEIGHTS"),
                    squares=int(getattr(config, "BOARD_SQUARES", 8)),
                    imgsz=int(getattr(config, "YOLO_PIECE_IMGSZ", 640)),
                    margin_squares=float(getattr(config, "BOARD_MARGIN_SQUARES", 1.7)),
                    conf_threshold=float(getattr(config, "YOLO_PIECE_CONF", 0.5)),
                    min_iou=float(getattr(config, "MIN_IOU", 0.15)),
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
        Implement Stage 3 logic (move inference etc.) here.
        """
        raise NotImplementedError

    def after_detection_batch(self) -> None:
        """
        Optional hook that runs once per frame after all DetectionStates
        have been drained and handle_detection_state has been called on each.
        """
        return None

    def on_start(self) -> None:
        """
        Hook called after all Stage 1+2 threads have been started,
        but before entering the main loop. Subclasses can start their
        own worker threads here.
        """
        return None

    def on_stop(self) -> None:
        """
        Hook called during shutdown after Stage 1+2 threads have been
        requested to stop and joined. Subclasses can clean up additional
        resources here.
        """
        return None

    # ------------------------------------------------------------------
    # Main run/stop logic
    # ------------------------------------------------------------------

    def run(self) -> None:
        cv2.setUseOptimized(True)
        # Allow users to cap OpenCV threads via config if provided
        try:
            if hasattr(config, "OPENCV_NUM_THREADS"):
                threads = int(getattr(config, "OPENCV_NUM_THREADS") or 0)
                if threads > 0:
                    cv2.setNumThreads(threads)
        except (AttributeError, TypeError, cv2.error):
            pass

        # Respect GUI toggle from config via app_gui helper
        enable_gui(bool(getattr(config, "GUI_ENABLED", True)))

        # Calibrate Stage 1 once before starting the threads
        if not _calibrate_pipeline(
                self.pipeline,
                self.source,
                desired_width=self.width,
                desired_height=self.height,
        ):
            return

        # Start Stage 1 + 2 threads
        self.reader_thread.start()
        for worker in self.det_workers:
            worker.start()

        # Give subclasses a chance to start their own threads
        self.on_start()

        _log.info("%s started. Press ESC to stop.", self.__class__.__name__)

        try:
            while True:
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    if not self.reader_thread.is_alive():
                        _log.info("FrameReader finished")
                        break
                    continue

                # Stage 1: rectify board
                warped_or_raw = self.pipeline.process_frame(frame)

                # Only enqueue when we have a rectified board
                warped_board = self.pipeline.last_warped_board
                if warped_board is not None:
                    try:
                        self.det_input_queue.put_nowait(warped_board)
                    except queue.Full:
                        # Drop if worker is busy
                        pass

                # Drain all finished detection results
                while True:
                    try:
                        det_state = self.det_output_queue.get_nowait()
                    except queue.Empty:
                        break

                    self.latest_state = det_state
                    self.latest_pieces = det_state.pieces
                    self.latest_boxes = det_state.boxes
                    self.latest_confs = det_state.confidences

                    # Stage 3 hook
                    self.handle_detection_state(det_state)

                # After batch hook for Stage 3
                self.after_detection_batch()

                # Display: board (warped) or raw frame
                display_frame = warped_or_raw if warped_or_raw is not None else frame

                # Overlay only when we have labels
                if (
                        display_frame is not None
                        and self.latest_pieces is not None
                        and self.pipeline.is_calibrated
                ):
                    display_frame = draw_piece_overlay(
                        display_frame,
                        self.latest_pieces,
                        squares=int(getattr(config, "BOARD_SQUARES", 8)),
                        margin_squares=float(getattr(config, "BOARD_MARGIN_SQUARES", 1.7)),
                        raw_boxes=self.latest_boxes,
                        confidences=self.latest_confs,
                    )

                if display_frame is not None:
                    # GUI mode shows, in console mode show_image is a NOP
                    show_image(self.window_name, display_frame)

                # Short delay for GUI, but no FPS cap for capture
                key = wait_key(1)
                if key == 27:  # ESC
                    break
        finally:
            self.stop()

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.reader_thread.join(timeout=1.0)
        except RuntimeError:
            pass
        for worker in self.det_workers:
            try:
                worker.join(timeout=1.0)
            except RuntimeError:
                pass
        # Allow subclasses to clean up their own resources
        self.on_stop()
        destroy_all_windows()
        _log.info("%s stopped cleanly", self.__class__.__name__)
