from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import List, Optional, Dict

from src.common.app_logging import get_logger, set_log_level
from src.pipeline.comparison.baseline_offline import run_baseline
from src.pipeline.comparison.common import PipelineResult
from src.pipeline.comparison.detection_log import (
    DetectionLog,
    record_detections,
    load_detections,
)
from src.pipeline.comparison.metrics import (
    GroundTruth,
    fen_interval_accuracy,
    fen_interval_counts,
    load_ground_truth,
    move_accuracy_counts,
    move_detection_delays,
    move_reconstruction_rate,
)
from src.pipeline.comparison.multistage_offline import run_multistage
from src.pipeline.fen_utils import placement_from_fen

_log = get_logger(__name__)


"""
Example usage:

python -m src.comparison_results.compare_pipelines \
  --video data/videos/game1.mp4 \
  --detections data/detections/game1.pkl \
  --gt data/gt/game1.json
"""


def _log_moves_with_fens(name: str, result: PipelineResult) -> None:
    """
    Log all committed detected_moves for a pipeline together with the FEN at the
    committing frame.
    """
    for idx, (uci, frame_idx) in enumerate(
            zip(result.moves_uci, result.move_frames),
            start=1,
    ):
        fen = ""
        if 0 <= frame_idx < len(result.frame_fens):
            fen = result.frame_fens[frame_idx]
        _log.info(
            "[%s] move %d at frame %d: %s | FEN: %s",
            name,
            idx,
            frame_idx,
            uci,
            fen,
        )


def _log_baseline_fens(
        result: PipelineResult,
        gt: Optional[GroundTruth],
) -> None:
    """
    Log FENs from the baseline that either

      1) correspond to a committed legal move, or
      2) match a legal ground truth position (placement identical).

    If the FEN placement stays the same over consecutive frames,
    it is only printed once for the baseline.
    """
    frames_with_moves = set(result.move_frames)

    placement_to_plys: Dict[str, List[int]] = {}
    if gt is not None:
        for ply_idx, fen in enumerate(gt.fens_after_ply, start=1):
            placement = placement_from_fen(fen)
            placement_to_plys.setdefault(placement, []).append(ply_idx)

    last_printed_placement: Optional[str] = None

    for frame_idx, fen in enumerate(result.frame_fens):
        placement = placement_from_fen(fen)
        has_move = frame_idx in frames_with_moves
        matching_plys = placement_to_plys.get(placement, [])
        matches_gt = bool(matching_plys)

        if not has_move and not matches_gt:
            continue

        # Avoid duplicate detections when the placement does not change
        # across consecutive frames in the baseline.
        if placement == last_printed_placement:
            continue

        moves_for_frame: List[str] = []
        for pos, f_idx in enumerate(result.move_frames):
            if f_idx == frame_idx and pos < len(result.moves_uci):
                moves_for_frame.append(result.moves_uci[pos])

        move_txt = ""
        if moves_for_frame:
            move_txt = " detected_moves " + ", ".join(moves_for_frame) + " |"

        reason_parts: List[str] = []
        if has_move:
            reason_parts.append("legal move")
        if matches_gt:
            if matching_plys:
                reason_parts.append(
                    "matches ground truth after ply "
                    + ", ".join(str(p) for p in matching_plys),
                )
            else:
                reason_parts.append("matches ground truth")

        reason_txt = "; ".join(reason_parts) if reason_parts else ""
        suffix = f" ({reason_txt})" if reason_txt else ""

        _log.info(
            "[baseline] frame %d%s FEN: %s%s",
            frame_idx,
            move_txt,
            fen,
            suffix,
        )

        last_printed_placement = placement


def _report_for_pipeline(name: str, result: PipelineResult, gt: GroundTruth) -> None:
    """
    Log metrics for a single pipeline result against provided ground truth.

    Pure reporting: does not mutate inputs or global state.
    """
    # FEN accuracy based on intervals between annotated plies
    correct_f, total_f = fen_interval_counts(result.frame_fens, gt)
    acc_f = fen_interval_accuracy(result.frame_fens, gt)
    _log.info(
        "[%s] FEN frame accuracy = %.3f (%d/%d)",
        name,
        acc_f,
        correct_f,
        total_f,
    )

    # Only pipelines that emit detected_moves are evaluated with MRR and delays
    if result.moves_uci:
        correct_m, total_m = move_accuracy_counts(result.moves_uci, gt.moves_uci)
        mrr = move_reconstruction_rate(result.moves_uci, gt.moves_uci)
        _log.info(
            "[%s] Move Reconstruction Rate (MRR) = %.3f (%d/%d)",
            name,
            mrr,
            correct_m,
            total_m,
        )

        if gt.frame_for_ply:
            delays = move_detection_delays(result.moves_uci, result.move_frames, gt)
            if delays:
                _log.info(
                    "[%s] mean detection delay = %.2f frames (n = %d)",
                    name,
                    mean(delays),
                    len(delays),
                )
    else:
        _log.info(
            "[%s] no detected_moves emitted by this pipeline, MRR and delays are not applicable",
            name,
        )


def run_comparison(
        detections_path: Path,
        gt_path: Path | None = None,
) -> None:
    """
    Orchestrate offline comparison_results between the single frame baseline and the
    multistage tracker using a recorded detection log.

    Both pipelines are run once, their results are stored, and afterwards
    selected FENs are logged according to the criteria described above.
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
            "Detection log has no FPS metadata. All frames will be used "
            "without live style sampling.",
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

    _log.info("Baseline emitted %d detected_moves", len(baseline.moves_uci))
    _log.info("Multistage emitted %d detected_moves", len(multistage.moves_uci))

    gt: Optional[GroundTruth] = None
    if gt_path is None:
        _log.info(
            "No ground truth file provided. Only raw counts and FEN subsets will be reported.",
        )
    else:
        _log.debug("Loading ground truth from %s", gt_path)
        gt = load_ground_truth(gt_path)
        _log.info("Ground truth has %d detected_moves", len(gt.moves_uci))
        _log.debug("Evaluating metrics against ground truth")

        _report_for_pipeline("baseline", baseline, gt)
        _report_for_pipeline("multistage", multistage, gt)

    # Second phase: log selected FENs from both pipelines
    _log.info(
        "Logging baseline FENs with legal detected_moves or matching ground truth positions",
    )
    _log_baseline_fens(baseline, gt)

    _log.info("Logging multistage detected_moves and their FENs")
    _log_moves_with_fens("multistage", multistage)


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare single frame baseline and multistage pipeline "
            "on a recorded detection log."
        ),
    )
    p.add_argument(
        "--video",
        type=Path,
        help=(
            "Optional video file. If given and the detection log does not "
            "exist yet, it will be created first."
        ),
    )
    p.add_argument(
        "--detections",
        type=Path,
        required=True,
        help="Path to detection log pickle file.",
    )
    p.add_argument(
        "--gt",
        type=Path,
        help="Optional JSON file with ground truth PGN and frame map.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on frames when recording detections.",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level, for example DEBUG, INFO, WARNING.",
    )
    p.add_argument(
        "--force-record",
        action="store_true",
        help="Recompute detection log even if it already exists.",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        set_log_level(args.log_level)
    except Exception:
        pass

    if args.force_record or not args.detections.exists():
        if args.video is None:
            raise SystemExit(
                "Detection log does not exist and no video was provided.",
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
