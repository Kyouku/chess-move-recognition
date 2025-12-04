from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import List

import cv2

from src import config
from src.app_logging import get_logger
from src.stage1.board_rectifier import LivePipeline
from src.stage2.piece_detection import PieceDetector
from src.types import DetectionState

_log = get_logger(__name__)


def _ensure_parent_dir(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # Best-effort: log and continue; subsequent file I/O may still fail.
        _log.warning("Could not ensure parent dir for %s: %s", path, exc)


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
      }
    """
    video_path = Path(video_path)
    out_path = Path(out_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(
        getattr(config, "FRAME_WIDTH", 1280)
    )
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(
        getattr(config, "FRAME_HEIGHT", 720)
    )
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    pipeline = LivePipeline(
        frame_width=width,
        frame_height=height,
        board_size_px=int(getattr(config, "BOARD_SIZE_PX", 640)),
        margin_squares=float(getattr(config, "BOARD_MARGIN_SQUARES", 1.7)),
        input_target_long_edge=int(
            getattr(
                config,
                "CALIBRATION_TARGET_LONG_EDGE",
                max(width, height),
            )
        ),
        min_board_area_ratio=float(
            getattr(config, "AUTO_MIN_BOARD_AREA_RATIO", 0.0)
        ),
        display=False,
    )

    detector = PieceDetector(
        weights=getattr(config, "YOLO_PIECE_WEIGHTS"),
        squares=int(getattr(config, "BOARD_SQUARES", 8)),
        imgsz=int(getattr(config, "YOLO_PIECE_IMGSZ", 640)),
        margin_squares=float(getattr(config, "BOARD_MARGIN_SQUARES", 1.7)),
        conf_threshold=float(getattr(config, "YOLO_PIECE_CONF", 0.5)),
        min_iou=float(getattr(config, "MIN_IOU", 0.15)),
    )

    _log.info("Calibrating LivePipeline from video %s", video_path)
    ok = pipeline.calibrate_from_capture(
        cap,
        max_frames=int(getattr(config, "CALIBRATION_MAX_FRAMES", 200)),
    )
    if not ok:
        cap.release()
        raise RuntimeError(f"Calibration failed for video {video_path}")

    # Rewind to start for detection pass
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Log planned pass length
    limit_txt = (
        f"{max_frames} frames" if max_frames is not None else "all frames"
    )
    total_txt = f"/~{total_frames} total" if total_frames > 0 else ""
    _log.info(
        "Starting detection pass over %s%s", limit_txt, total_txt
    )

    detections: List[DetectionState] = []
    frame_idx = 0
    t_start = time.perf_counter()
    last_report = t_start
    report_interval = int(getattr(config, "DETECTION_PROGRESS_EVERY", 50))

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1

        # Process frame to update pipeline state; output is accessed via
        # pipeline.last_warped_board.
        pipeline.process_frame(frame)
        warped_board = pipeline.last_warped_board

        if warped_board is None:
            detections.append(
                DetectionState(
                    occupancy={},
                    pieces={},
                    boxes={},
                    confidences={},
                )
            )
            continue

        occ, pieces, boxes, confs = detector.detect(warped_board)
        detections.append(
            DetectionState(
                occupancy=occ,
                pieces=pieces,
                boxes=boxes,
                confidences=confs,
            )
        )

        # Periodic progress report (INFO so it's visible by default)
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
    # Final progress log with FPS
    elapsed_total = max(1e-6, time.perf_counter() - t_start)
    fps_final = frame_idx / elapsed_total
    _log.info(
        "Recorded %d detection states from %s (%.2f FPS)",
        len(detections),
        video_path,
        fps_final,
    )

    _ensure_parent_dir(out_path)
    payload = {
        "video_path": str(video_path),
        "detections": detections,
    }
    data = pickle.dumps(payload)
    with out_path.open("wb") as f:
        f.write(data)
    _log.info("Saved detection log to %s", out_path)


def load_detections(path: str | Path) -> List[DetectionState]:
    """
    Load a detection log created by record_detections.
    """
    path = Path(path)
    with path.open("rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "detections" in payload:
        return payload["detections"]

    # Backward compatible: plain list
    if isinstance(payload, list):
        return payload

    raise ValueError(f"Unexpected content in {path}: {type(payload)}")
