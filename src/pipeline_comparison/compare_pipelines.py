from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import List

from src.app_logging import get_logger, set_log_level
from src.pipeline_comparison.baseline_offline import run_baseline
from src.pipeline_comparison.common import PipelineResult
from src.pipeline_comparison.detection_log import (
    DetectionLog,
    record_detections,
    load_detections,
)
from src.pipeline_comparison.metrics import (
    GroundTruth,
    load_ground_truth,
    move_detection_delays,
    move_reconstruction_rate,
)
from src.pipeline_comparison.multistage_offline import run_multistage

_log = get_logger(__name__)

"""
python -m src.pipeline_comparison.compare_pipelines \
  --video data/videos/game1.mp4 \
  --detections data/logs/testgame.pkl \
  --gt data/gt/testgame.json
"""


def _report_for_pipeline(name: str, result: PipelineResult, gt: GroundTruth) -> None:
    """
    Log metrics for a single pipeline result against provided ground truth.
    Pure reporting: does not mutate inputs or global state.
    """
    mrr = move_reconstruction_rate(result.moves_uci, gt.moves_uci)
    _log.info("[%s] Move Reconstruction Rate (MRR) = %.3f", name, mrr)

    if gt.frame_for_ply:
        delays = move_detection_delays(result.moves_uci, result.move_frames, gt)
        if delays:
            _log.info(
                "[%s] mean detection delay = %.2f frames (n = %d)",
                name,
                mean(delays),
                len(delays),
            )


def run_comparison(
        detections_path: Path,
        gt_path: Path | None = None,
) -> None:
    """
    Orchestrate offline comparison between the single-frame baseline and the
    multistage tracker using a pre recorded detection log. Produces logs only.

    If the detection log contains FPS metadata, a live style sampling is
    simulated where only frames that the detector could realistically process
    are fed into the pipelines, similar to the behavior of the threaded
    live app with a small queue.
    """
    det_log: DetectionLog = load_detections(detections_path)
    states = det_log.detections
    video_fps = det_log.video_fps
    detector_fps = det_log.detector_fps

    _log.info("Loaded detection log: %s", detections_path)
    _log.debug("Detection states loaded: %d", len(states))

    if video_fps is not None and detector_fps is not None:
        _log.info(
            "Detection log metadata: video_fps=%.2f, detector_fps=%.2f",
            video_fps,
            detector_fps,
        )
    else:
        _log.info(
            "Detection log has no FPS metadata. Comparison will use all frames "
            "without live style sampling."
        )

    _log.info("Running single frame baseline on %d frames", len(states))
    _log.debug("Starting baseline pass")
    baseline = run_baseline(
        states,
        video_fps=video_fps,
        detector_fps=detector_fps,
    )
    _log.debug("Finished baseline pass")

    _log.info("Running multistage tracker on %d frames", len(states))
    _log.debug("Starting multistage pass")
    multistage = run_multistage(
        states,
        video_fps=video_fps,
        detector_fps=detector_fps,
    )
    _log.debug("Finished multistage pass")

    _log.info("Baseline detected %d moves", len(baseline.moves_uci))
    _log.info("Multistage detected %d moves", len(multistage.moves_uci))

    if gt_path is None:
        _log.info(
            "No ground truth file provided. Only move counts were reported."
        )
        return

    _log.debug("Loading ground truth from %s", gt_path)
    gt: GroundTruth = load_ground_truth(gt_path)
    _log.info("Ground truth has %d moves", len(gt.moves_uci))
    _log.debug("Evaluating metrics against ground truth")

    _report_for_pipeline("baseline", baseline, gt)
    _report_for_pipeline("multistage", multistage, gt)


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare single frame baseline and multistage pipeline "
            "on a recorded detection log"
        ),
    )
    p.add_argument(
        "--video",
        type=Path,
        help=(
            "Optional video file. If given and the detection log does not "
            "exist yet it will be created first."
        ),
    )
    p.add_argument(
        "--detections",
        type=Path,
        required=True,
        help="Path to detection log pickle file",
    )
    p.add_argument(
        "--gt",
        type=Path,
        help="Optional JSON file with ground truth PGN and frame map",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on frames when recording detections",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING)",
    )
    p.add_argument(
        "--force-record",
        action="store_true",
        help="Recompute detection log even if it already exists",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    # Allow enabling DEBUG output to see per frame progress
    try:
        set_log_level(args.log_level)
    except Exception:
        # Fall back silently if invalid value
        pass

    if args.force_record or not args.detections.exists():
        if args.video is None:
            raise SystemExit(
                "Detection log does not exist and no video was provided."
            )
        record_detections(
            video_path=args.video,
            out_path=args.detections,
            max_frames=args.max_frames,
        )

    run_comparison(
        detections_path=args.detections,
        gt_path=args.gt,
    )


if __name__ == "__main__":
    main()
