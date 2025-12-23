from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src import config
from src.common.app_logging import get_logger
from src.pipeline.comparison.common import (
    PipelineResult,
    iter_time_based_should_process,
    is_game3,
    load_detections_with_optional_homography,
)
from src.pipeline.comparison.metrics import (
    fen_at_ply_accuracy,
    fen_interval_accuracy,
    load_ground_truth,
    move_coverage,
    move_detection_delays,
    move_reconstruction_rate,
)
from src.stage3.move_tracking import MoveTracker

log = get_logger(__name__)




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
        if is_game3(name):
            if args.game3_homography:
                homography_override = Path(args.game3_homography)
            else:
                for k in ("homography", "homography_path", "saved_homography", "saved_homography_path"):
                    if k in g and g[k]:
                        homography_override = Path(str(g[k]))
                        break

        det_log = load_detections_with_optional_homography(Path(g["detections"]), homography_override)
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
