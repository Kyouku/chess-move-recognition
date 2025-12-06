from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

import cv2
import numpy as np
from ultralytics import YOLO

from src.common.app_logging import get_logger
from src.common.types import DetectionState
from .piece_overlay import _compute_square_boxes, _piece_code_from_label

_log = get_logger(__name__)


def _square_name(rank_idx: int, file_idx: int) -> str:
    """
    Map (rank_idx, file_idx) to a square like "a1".

    Convention (naming preserved for backward compatibility):
      - rank_idx: 0..squares-1 maps across files 'a'.. (columns, left to right)
      - file_idx: 0..squares-1 maps across ranks '1'.. (rows, bottom to top)
    """
    file_char = chr(ord("a") + rank_idx)
    rank_char = str(file_idx + 1)
    return f"{file_char}{rank_char}"


class PieceDetector:
    """
    YOLO based piece detector operating on a warped board image.

    Each YOLO box is assigned to the board square with the highest IoU
    (intersection over union) overlap. The original YOLO boxes per
    occupied square are stored as well so an overlay can draw them.
    """

    def __init__(
            self,
            weights: Union[str, Path],
            squares: int = 8,
            imgsz: int = 640,
            margin_squares: float = 0.0,
            conf_threshold: float = 0.5,
            min_iou: float = 0.15,
            # How many consecutive frames without a detection before we clear a square.
            # This makes occupancy "sticky" and avoids flicker.
            miss_clear_frames: int = 3,
    ) -> None:
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found at {weights_path}")

        _log.info("Loading YOLO model from %s", weights_path)
        self._yolo_model = YOLO(str(weights_path), task="detect", verbose=False)

        self.squares = int(squares)
        self.imgsz = int(imgsz)
        self.margin_squares = float(margin_squares)
        self.conf_threshold = float(conf_threshold)
        self.min_iou = float(min_iou)
        self._miss_clear_frames = int(max(1, miss_clear_frames))

        # Board geometry in YOLO input space
        self._square_boxes: Dict[str, np.ndarray] = {}
        self._cell_height: float = 0.0
        self._img_w: float = float(self.imgsz)
        self._img_h: float = float(self.imgsz)
        self._step_x: float = 0.0
        self._step_y: float = 0.0
        self._off_x: float = 0.0
        self._off_y: float = 0.0
        # Precompute board geometry and related caches
        self._squares_list: List[str] = []
        self._square_items: List[Tuple[str, np.ndarray]] = []
        self._init_board_geometry()

        # Cache class-id to compact label mapping once (avoids per-frame string ops)
        # ultralytics stores mapping in model.names: Dict[int, str]
        try:
            # Some YOLO versions expose .names as list or dict; normalize to dict[int, str]
            if isinstance(self._yolo_model.names, dict):
                names_map = {int(k): str(v) for k, v in self._yolo_model.names.items()}
            else:
                names_map = {int(i): str(n) for i, n in enumerate(list(self._yolo_model.names))}
        except (AttributeError, TypeError, ValueError):
            names_map = {}

        self._class_code_map: Dict[int, str] = {}
        for cid, name in names_map.items():
            code = _piece_code_from_label(name) or name
            self._class_code_map[int(cid)] = str(code)

        # Persisted per-square state to implement confidence-aware, sticky occupancy.
        # Initialized on first detect() call.
        self._mem_pieces: Optional[Dict[str, Optional[str]]] = None
        self._mem_occupancy: Optional[Dict[str, bool]] = None
        self._mem_boxes: Optional[Dict[str, Optional[Tuple[float, float, float, float]]]] = None
        self._mem_conf: Optional[Dict[str, Optional[float]]] = None
        self._miss_counts: Optional[Dict[str, int]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
            self,
            board_bgr: np.ndarray,
    ) -> Tuple[
        Dict[str, bool],
        Dict[str, Optional[str]],
        Dict[str, Optional[Tuple[float, float, float, float]]],
        Dict[str, Optional[float]],
    ]:
        """
        Run YOLO on the warped board and map detections to squares.

        Returns:
          occupancy: Dict[square_name, bool]
          pieces:    Dict[square_name, Optional[str]]
          boxes:     Dict[square_name, Optional[Tuple[x1, y1, x2, y2]]]
          confs:     Dict[square_name, Optional[float]]
        """
        if board_bgr is None or board_bgr.size == 0:
            raise ValueError("Empty board image in PieceDetector.detect")

        if board_bgr.shape[0] != self.imgsz or board_bgr.shape[1] != self.imgsz:
            # Choose interpolation based on scaling direction (AREA for shrink, LINEAR for enlarge)
            h0, w0 = board_bgr.shape[:2]
            target_area = float(self.imgsz * self.imgsz)
            src_area = float(w0 * h0)
            interp = cv2.INTER_AREA if target_area < src_area else cv2.INTER_LINEAR
            board_bgr = cv2.resize(board_bgr, (self.imgsz, self.imgsz), interpolation=interp)

        # Initialize raw per-frame accumulation (best per square in this frame)
        raw_pieces: Dict[str, Optional[str]] = dict.fromkeys(self._squares_list, None)
        raw_occupancy: Dict[str, bool] = dict.fromkeys(self._squares_list, False)
        best_conf_per_square: Dict[str, float] = {sq: -1.0 for sq in self._squares_list}
        raw_boxes: Dict[str, Optional[Tuple[float, float, float, float]]] = dict.fromkeys(
            self._squares_list, None
        )
        raw_conf: Dict[str, Optional[float]] = dict.fromkeys(self._squares_list, None)

        # YOLO inference
        results = self._yolo_model.predict(
            source=board_bgr,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            verbose=False,
            augment=False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            # No detections this frame; use memory with miss handling below
            pass

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            # process highest confidence first, like your notebook code picking best match
            order = np.argsort(-confs)

            for idx in order:
                x1, y1, x2, y2 = xyxy[idx]
                conf = float(confs[idx])

                # Cut tall boxes from the top in pixel units,
                # similar to:
                # if box_y4 - box_y1 > 60:
                #     y1 += 40
                x1_adj, y1_adj, x2_adj, y2_adj = self._adjust_tall_box(x1, y1, x2, y2)

                best_sq, best_iou = self._assign_box_to_square_iou(
                    x1_adj, y1_adj, x2_adj, y2_adj
                )
                if best_sq is None or best_iou < self.min_iou:
                    continue

                # keep only highest confidence detection per square
                if conf <= best_conf_per_square[best_sq]:
                    continue

                # Map class id to compact code like "wP", "bN" using precomputed map
                cls_name = self._class_code_map.get(int(cls_ids[idx]), str(int(cls_ids[idx])))
                best_conf_per_square[best_sq] = conf
                raw_pieces[best_sq] = cls_name
                raw_occupancy[best_sq] = True
                raw_boxes[best_sq] = (float(x1), float(y1), float(x2), float(y2))
                raw_conf[best_sq] = conf

        # Initialize memory on first run
        if self._mem_pieces is None:
            self._mem_pieces = dict.fromkeys(self._squares_list, None)
            self._mem_occupancy = dict.fromkeys(self._squares_list, False)
            self._mem_boxes = dict.fromkeys(self._squares_list, None)
            self._mem_conf = dict.fromkeys(self._squares_list, None)
            self._miss_counts = {sq: 0 for sq in self._squares_list}

        assert self._mem_pieces is not None
        assert self._mem_occupancy is not None
        assert self._mem_boxes is not None
        assert self._mem_conf is not None
        assert self._miss_counts is not None

        # Merge raw detections with memory using confidence-aware sticky logic
        merged_pieces: Dict[str, Optional[str]] = {}
        merged_occupancy: Dict[str, bool] = {}
        merged_boxes: Dict[str, Optional[Tuple[float, float, float, float]]] = {}
        merged_conf: Dict[str, Optional[float]] = {}

        for sq in self._squares_list:
            prev_occ = bool(self._mem_occupancy.get(sq, False))
            prev_conf = self._mem_conf.get(sq)
            prev_label = self._mem_pieces.get(sq)
            prev_box = self._mem_boxes.get(sq)

            has_det = bool(raw_occupancy.get(sq, False))
            new_conf = raw_conf.get(sq)
            new_label = raw_pieces.get(sq)
            new_box = raw_boxes.get(sq)

            if has_det:
                # Reset miss counter
                self._miss_counts[sq] = 0
                # Update only if we previously had no piece or the new confidence is higher
                if (prev_conf is None) or (new_conf is not None and new_conf >= prev_conf):
                    merged_occupancy[sq] = True
                    merged_pieces[sq] = new_label
                    merged_boxes[sq] = new_box
                    merged_conf[sq] = new_conf
                else:
                    # Keep previous stronger state
                    merged_occupancy[sq] = prev_occ
                    merged_pieces[sq] = prev_label
                    merged_boxes[sq] = prev_box
                    merged_conf[sq] = prev_conf
            else:
                # No detection on this square for this frame: increase miss count
                self._miss_counts[sq] = self._miss_counts.get(sq, 0) + 1
                if prev_occ and self._miss_counts[sq] < self._miss_clear_frames:
                    # Keep previous state until we are confident the piece moved away
                    merged_occupancy[sq] = True
                    merged_pieces[sq] = prev_label
                    merged_boxes[sq] = prev_box
                    merged_conf[sq] = prev_conf
                else:
                    # Consider square empty (either previously empty or misses exceeded)
                    merged_occupancy[sq] = False
                    merged_pieces[sq] = None
                    merged_boxes[sq] = None
                    merged_conf[sq] = None

        # Commit merged state back to memory for next frame
        self._mem_pieces = dict(merged_pieces)
        self._mem_occupancy = dict(merged_occupancy)
        self._mem_boxes = dict(merged_boxes)
        self._mem_conf = dict(merged_conf)

        return merged_occupancy, merged_pieces, merged_boxes, merged_conf

    def detect_state(self, board_bgr: np.ndarray) -> DetectionState:
        """
        Run detection and return a unified DetectionState container.
        """
        occ, pieces, boxes, confs = self.detect(board_bgr)
        return DetectionState(
            occupancy=occ,
            pieces=pieces,
            boxes=boxes,
            confidences=confs,
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _init_board_geometry(self) -> None:
        """
        Precompute pixel regions for all squares in YOLO input space.
        """
        img_w = float(self.imgsz)
        img_h = float(self.imgsz)
        m = self.margin_squares

        if m > 0.0:
            step_x = img_w / (self.squares + 2.0 * m)
            step_y = img_h / (self.squares + 2.0 * m)
            off_x = step_x * m
            off_y = step_y * m
        else:
            step_x = img_w / self.squares
            step_y = img_h / self.squares
            off_x = 0.0
            off_y = 0.0

        self._cell_height = float(step_y)
        self._img_w = img_w
        self._img_h = img_h
        self._step_x = step_x
        self._step_y = step_y
        self._off_x = off_x
        self._off_y = off_y

        self._square_boxes = _compute_square_boxes(
            int(img_w), int(img_h), self.squares, m
        )
        # Keep a cached list of square keys and items for faster iterations
        self._squares_list = list(self._square_boxes.keys())
        self._square_items = list(self._square_boxes.items())

    def _adjust_tall_box(
            self,
            x1: float,
            y1: float,
            x2: float,
            y2: float,
    ) -> Tuple[float, float, float, float]:
        """
        Mimic the notebook heuristic:

          if box_height > 60:
              y1 += 40

        but scaled with the board size so it still works if imgsz is not 640.
        """
        h_box = max(1.0, y2 - y1)

        # Scale the 60 and 40 from your notebook from a 640 board to current imgsz
        scale = float(self.imgsz) / 640.0
        tall_thr = 60.0 * scale
        crop_px = 40.0 * scale

        if h_box > tall_thr:
            new_y1 = y1 + crop_px
            # clamp to image
            new_y1 = min(new_y1, self._img_h - 1.0)
            return x1, new_y1, x2, y2

        return x1, y1, x2, y2

    # ------------------------------------------------------------------
    # Box to square mapping
    # ------------------------------------------------------------------

    def _assign_box_to_square_iou(
            self,
            x1: float,
            y1: float,
            x2: float,
            y2: float,
    ) -> Tuple[Optional[str], float]:
        """
        Pure IoU assignment to all squares, like calculate_iou(box, square).
        """
        box = np.array([x1, y1, x2, y2], dtype=np.float32)

        best_sq: Optional[str] = None
        best_iou: float = 0.0

        for sq, cell in self._square_items:
            iou = self._iou(box, cell)
            if iou > best_iou:
                best_iou = iou
                best_sq = sq

        return best_sq, best_iou

    @staticmethod
    def _iou(box: np.ndarray, cell: np.ndarray) -> float:
        """
        Intersection over union for two axis aligned boxes in [x1, y1, x2, y2] format.

        This is equivalent to Polygon IoU for rectangles,
        just without the shapely dependency.
        """
        x1 = max(float(box[0]), float(cell[0]))
        y1 = max(float(box[1]), float(cell[1]))
        x2 = min(float(box[2]), float(cell[2]))
        y2 = min(float(box[3]), float(cell[3]))

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        if inter_area <= 0.0:
            return 0.0

        box_area = max(0.0, (float(box[2]) - float(box[0]))) * max(
            0.0, (float(box[3]) - float(box[1]))
        )
        cell_area = max(0.0, (float(cell[2]) - float(cell[0]))) * max(
            0.0, (float(cell[3]) - float(cell[1]))
        )

        if box_area <= 0.0 or cell_area <= 0.0:
            return 0.0

        union = box_area + cell_area - inter_area
        if union <= 0.0:
            return 0.0

        return inter_area / union
