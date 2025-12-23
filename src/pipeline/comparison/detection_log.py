from __future__ import annotations

"""
Utilities for recording and loading DetectionState logs for offline comparison.

record_detections runs Stage 1 and Stage 2 on a video once and stores
DetectionState objects together with basic metadata. load_detections
wraps that data in a DetectionLog dataclass.
"""

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2

from src import config
from src.common.app_logging import get_logger
from src.common.homography_cache import (
    apply_saved_homography,
    save_homography_from_pipeline,
)
from src.common.chess_io import ensure_parent_dir
from src.common.types import DetectionState
from src.stage1.board_rectifier import LivePipeline
from src.stage2.piece_detection import PieceDetector

_log = get_logger(__name__)


@dataclass
class DetectionLog:
    """
    Recorded detection log for a single video.

    detections:
        List of DetectionState objects, one per processed frame.
    video_fps:
        Nominal video frames per second if known.
    detector_fps:
        Effective processing speed during recording.
    """

    detections: List[DetectionState]
    video_fps: Optional[float]
    detector_fps: Optional[float]


def record_detections(
        video_path: str | Path,
        out_path: str | Path,
        *,
        max_frames: int | None = None,
) -> None:
    """
    Run Stage 1 (rectification) and Stage 2 (piece detection) on a video once
    and save a list of DetectionState objects to disk.

    The output file is a pickle that contains a dictionary:

      {
        "video_path": str,
        "detections": List[DetectionState],
        "video_fps": float,
        "detector_fps": float
      }
    """
    video_path = Path(video_path)
    out_path = Path(out_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.FRAME_WIDTH
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.FRAME_HEIGHT
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Nominal video FPS with fallback
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    if raw_fps and raw_fps > 0:
        video_fps = float(raw_fps)
    else:
        video_fps = config.VIDEO_FPS
        _log.warning(
            "CAP_PROP_FPS not available, falling back to VIDEO_FPS=%.2f",
            video_fps,
        )

    pipeline = LivePipeline(
        frame_width=width,
        frame_height=height,
        board_size_px=config.BOARD_SIZE_PX,
        margin_squares=config.BOARD_MARGIN_SQUARES,
        input_target_long_edge=config.CALIBRATION_TARGET_LONG_EDGE,
        min_board_area_ratio=config.AUTO_MIN_BOARD_AREA_RATIO,
        display=False,
    )

    detector = PieceDetector(
        weights=config.YOLO_PIECE_WEIGHTS,
        squares=config.BOARD_SQUARES,
        imgsz=config.YOLO_PIECE_IMGSZ,
        margin_squares=config.BOARD_MARGIN_SQUARES,
        conf_threshold=config.YOLO_PIECE_CONF,
        min_iou=config.MIN_IOU,
    )

    # Try to use a saved homography if configured, otherwise calibrate
    try:
        ok = apply_saved_homography(pipeline)
    except Exception:
        ok = False

    if not ok:
        _log.info("Calibrating LivePipeline from video %s", video_path)
        ok = pipeline.calibrate_from_capture(
            cap,
            max_frames=config.CALIBRATION_MAX_FRAMES,
        )
        if not ok:
            cap.release()
            raise RuntimeError(f"Calibration failed for video {video_path}")

        try:
            save_homography_from_pipeline(pipeline)
        except Exception:
            # Do not fail the recording if saving failed
            pass

    # Rewind to start for detection pass
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    limit_txt = f"{max_frames} frames" if max_frames is not None else "all frames"
    total_txt = f"/~{total_frames} total" if total_frames > 0 else ""
    _log.info("Starting detection pass over %s%s", limit_txt, total_txt)

    detections: List[DetectionState] = []
    frame_idx = 0
    t_start = time.perf_counter()
    report_interval = config.DETECTION_PROGRESS_EVERY

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1

        pipeline.process_frame(frame)
        warped_board = pipeline.last_warped_board

        if warped_board is None:
            detections.append(
                DetectionState(
                    occupancy={},
                    pieces={},
                    boxes={},
                    confidences={},
                ),
            )
            continue

        occ, pieces, boxes, confs = detector.detect(warped_board)
        detections.append(
            DetectionState(
                occupancy=occ,
                pieces=pieces,
                boxes=boxes,
                confidences=confs,
            ),
        )

        if frame_idx % max(1, report_interval) == 0:
            now = time.perf_counter()
            elapsed = now - t_start
            fps = frame_idx / elapsed if elapsed > 0 else 0.0
            if total_frames > 0:
                pct = 100.0 * min(frame_idx, total_frames) / total_frames
                _log.info(
                    "Processed %d/%d frames (%.1f%%) at ~%.2f FPS",
                    frame_idx,
                    total_frames,
                    pct,
                    fps,
                )
            else:
                _log.info(
                    "Processed %d frames at ~%.2f FPS",
                    frame_idx,
                    fps,
                )

    cap.release()
    elapsed_total = max(1e-6, time.perf_counter() - t_start)
    fps_final = frame_idx / elapsed_total
    _log.info(
        "Recorded %d detection states from %s (%.2f FPS)",
        len(detections),
        video_path,
        fps_final,
    )

    ensure_parent_dir(out_path)
    payload = {
        "video_path": str(video_path),
        "detections": detections,
        "video_fps": video_fps,
        "detector_fps": fps_final,
    }
    data = pickle.dumps(payload)
    with out_path.open("wb") as f:
        f.write(data)
    _log.info("Saved detection log to %s", out_path)


def load_detections(path: str | Path) -> DetectionLog:
    """
    Load a detection log created by record_detections.

    Supports both the new format with metadata and the old
    format that was just a plain list of DetectionState objects.
    """
    path = Path(path)
    with path.open("rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "detections" in payload:
        detections = payload["detections"]
        if not isinstance(detections, list):
            raise ValueError(
                f"Unexpected 'detections' entry type in {path}: {type(detections)}",
            )

        video_fps_raw = payload.get("video_fps")
        det_fps_raw = payload.get("detector_fps")

        try:
            video_fps = float(video_fps_raw) if video_fps_raw is not None else None
        except (TypeError, ValueError):
            video_fps = None

        try:
            detector_fps = float(det_fps_raw) if det_fps_raw is not None else None
        except (TypeError, ValueError):
            detector_fps = None

        return DetectionLog(
            detections=detections,
            video_fps=video_fps,
            detector_fps=detector_fps,
        )

    if isinstance(payload, list):
        return DetectionLog(
            detections=payload,
            video_fps=None,
            detector_fps=None,
        )

    raise ValueError(f"Unexpected content in {path}: {type(payload)}")
