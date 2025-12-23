from __future__ import annotations

import argparse
import csv
import inspect
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src import config
from src.pipeline.comparison.common import PipelineResult, iter_time_based_should_process
from src.pipeline.comparison.detection_log import load_detections
from src.pipeline.comparison.metrics import (
    fen_at_ply_accuracy,
    fen_interval_accuracy,
    load_ground_truth,
    move_coverage,
    move_detection_delays,
    move_reconstruction_rate,
)
from src.stage3.move_tracking import MoveTracker

try:
    from src.common.app_logging import get_logger
except Exception:  # pragma: no cover
    from src.common.io_utils import get_logger  # type: ignore

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Homography override only for game3
# ---------------------------------------------------------------------------

def _is_game3(name: str) -> bool:
    """
    Best-effort identification of "game3" naming variants.
    Adjust if your manifest uses a different identifier.
    """
    n = name.strip().lower().replace(" ", "").replace("-", "_")
    return n in {"game3", "game_3", "3"}


def _load_homography_matrix(path: Path) -> List[List[float]]:
    """
    Load a saved 3x3 homography matrix.

    Supported formats:
      - .json with {"H": [[...],[...],[...]]} or {"homography": [[...],[...],[...]]}
      - .npy containing a 3x3 array
      - .txt/.csv containing 3 rows of 3 numbers (space or comma separated)

    Returns:
      3x3 list of floats.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    suf = path.suffix.lower()

    if suf == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        H = obj.get("H", None) or obj.get("homography", None) or obj.get("matrix", None)
        if not isinstance(H, list) or len(H) != 3:
            raise ValueError(
                f"Invalid homography JSON in {path} (expected 3x3 list under key H/homography/matrix)."
            )
        if any((not isinstance(row, list) or len(row) != 3) for row in H):
            raise ValueError(f"Invalid homography JSON in {path} (expected 3x3 list).")
        return [[float(x) for x in row] for row in H]

    if suf == ".npy":
        try:
            import numpy as np  # type: ignore
        except Exception as e:
            raise RuntimeError("Cannot load .npy homography because numpy is not available.") from e
        arr = np.load(str(path))
        if getattr(arr, "shape", None) != (3, 3):
            raise ValueError(
                f"Invalid homography npy in {path} (expected shape (3,3), got {getattr(arr, 'shape', None)})."
            )
        return [[float(x) for x in row] for row in arr.tolist()]

    # Fallback: parse simple text
    txt = path.read_text(encoding="utf-8").strip().splitlines()
    rows: List[List[float]] = []
    for line in txt:
        line = line.strip()
        if not line:
            continue
        parts = [p for p in line.replace(",", " ").split(" ") if p]
        rows.append([float(p) for p in parts])

    if len(rows) != 3 or any(len(r) != 3 for r in rows):
        raise ValueError(f"Invalid homography text in {path} (expected 3 lines of 3 numbers).")
    return rows


def _attach_homography_best_effort(det_log: Any, homography_path: Path) -> None:
    """
    If load_detections cannot accept a homography override directly, attach the matrix
    to the returned det_log object as metadata. This is harmless if unused downstream.
    """
    try:
        H = _load_homography_matrix(homography_path)
    except Exception as e:
        log.warning("Failed to read homography file %s: %s", str(homography_path), str(e))
        return

    for attr in ("H", "homography", "homography_matrix"):
        try:
            setattr(det_log, attr, H)
            return
        except Exception:
            pass

    log.warning(
        "Homography override provided, but detection log object has no known homography attribute. "
        "If your detections are already rectified into board squares, you must regenerate the detection "
        "log for game3 using the saved homography at recording time."
    )


def _load_detections_with_optional_homography(det_path: str, homography_path: Path | None) -> Any:
    """
    Try to pass a homography override into load_detections if supported.
    Otherwise load normally and attach metadata best-effort.
    """
    if homography_path is None:
        return load_detections(det_path)

    sig = inspect.signature(load_detections)
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    kwargs: Dict[str, Any] = {}
    candidate_keys = (
        "homography_path",
        "override_homography_path",
        "calibration_path",
        "saved_homography_path",
        "homography",
    )

    for key in candidate_keys:
        if key in sig.parameters:
            kwargs[key] = str(homography_path)
            break

    if not kwargs and has_varkw:
        kwargs["homography_path"] = str(homography_path)

    det_log = load_detections(det_path, **kwargs) if kwargs else load_detections(det_path)

    if not kwargs:
        _attach_homography_best_effort(det_log, homography_path)

    return det_log


# ---------------------------------------------------------------------------
# Sensitivity grid runner
# ---------------------------------------------------------------------------

def _run_multistage(
        states: List[Any],
        *,
        video_fps: float | None,
        detector_fps: float | None,
        alpha: float,
        occ_threshold: float,
        min_confirm_frames: int,
        start_confirm_frames: int,
        debug: bool,
) -> PipelineResult:
    tracker = MoveTracker(
        alpha=float(alpha),
        occ_threshold=float(occ_threshold),
        min_confirm_frames=int(min_confirm_frames),
        start_confirm_frames=int(start_confirm_frames),
        debug=bool(debug),
    )

    frame_fens: List[str] = []
    moves_uci: List[str] = []
    moves_san: List[str] = []
    move_frames: List[int] = []

    total_frames = len(states)
    for idx, should_process in iter_time_based_should_process(video_fps, detector_fps, total_frames):
        state = states[idx]
        info = tracker.update_from_detection_state(state) if should_process else None
        frame_fens.append(tracker.board.fen())
        if info is None:
            continue
        moves_uci.append(info.move.uci())
        moves_san.append(info.san)
        move_frames.append(idx)

    return PipelineResult(
        frame_fens=frame_fens,
        moves_uci=moves_uci,
        moves_san=moves_san,
        move_frames=move_frames,
    )


def _mean_median_delay_seconds(
        delays_frames: List[int],
        video_fps: float | None,
) -> Tuple[float | None, float | None]:
    if not delays_frames:
        return None, None
    if not video_fps or video_fps <= 0:
        return None, None
    delays_s = [d / float(video_fps) for d in delays_frames]
    return float(sum(delays_s) / len(delays_s)), float(statistics.median(delays_s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--confirm_frames", nargs="+", type=int, default=[1, 2, 3, 4, 5])

    # Homography override only for game3
    ap.add_argument(
        "--game3_homography",
        type=str,
        default=None,
        help="Path to saved homography (json, npy, or txt). Applied only when game name resolves to game3.",
    )

    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    games = manifest.get("games", [])
    if not games:
        raise ValueError("Manifest contains no games.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    occ_threshold = config.MOVE_FILTER_THRESHOLD
    debug = config.MOVE_DEBUG

    # Keep start locking constant, otherwise the grid mixes 2 effects.
    start_confirm_frames = config.MOVE_MIN_CONFIRM_FRAMES

    rows: List[Dict[str, Any]] = []

    for g in games:
        name = str(g["name"])

        # Only for game3: determine homography path, CLI takes precedence over manifest fields
        homography_override: Path | None = None
        if _is_game3(name):
            if args.game3_homography:
                homography_override = Path(args.game3_homography)
            else:
                for k in ("homography", "homography_path", "saved_homography", "saved_homography_path"):
                    if k in g and g[k]:
                        homography_override = Path(str(g[k]))
                        break

        det_log = _load_detections_with_optional_homography(str(g["detections"]), homography_override)
        gt = load_ground_truth(str(g["gt"]))

        if homography_override is not None:
            log.info("Game %s: using homography override: %s", name, str(homography_override))

        for alpha in args.alphas:
            for k in args.confirm_frames:
                res = _run_multistage(
                    det_log.detections,
                    video_fps=getattr(det_log, "video_fps", None),
                    detector_fps=getattr(det_log, "detector_fps", None),
                    alpha=float(alpha),
                    occ_threshold=occ_threshold,
                    min_confirm_frames=int(k),
                    start_confirm_frames=int(start_confirm_frames),
                    debug=debug,
                )

                delays_frames = move_detection_delays(res.moves_uci, res.move_frames, gt)
                mean_delay_s, median_delay_s = _mean_median_delay_seconds(
                    delays_frames,
                    getattr(det_log, "video_fps", None),
                )

                rows.append(
                    {
                        "game": name,
                        "alpha": float(alpha),
                        "min_confirm_frames": int(k),
                        "start_confirm_frames": int(start_confirm_frames),
                        "fen_at_ply_acc": fen_at_ply_accuracy(res.frame_fens, gt),
                        "fen_interval_acc": fen_interval_accuracy(res.frame_fens, gt),
                        "mrr": move_reconstruction_rate(res.moves_uci, gt.moves_uci),
                        "move_coverage": move_coverage(res.moves_uci, gt.moves_uci),
                        "num_gt_moves": len(gt.moves_uci),
                        "num_pred_moves": len(res.moves_uci),
                        "move_delay_n": len(delays_frames),
                        "mean_move_delay_s": mean_delay_s,
                        "median_move_delay_s": median_delay_s,
                    }
                )

        log.info("Finished sensitivity grid for %s", name)

    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    log.info("Wrote: %s", out_path)


if __name__ == "__main__":
    main()
