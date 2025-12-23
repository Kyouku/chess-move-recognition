from __future__ import annotations

import argparse
import csv
import inspect
import json
import statistics
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src import config
from src.pipeline.comparison.baseline_offline import run_baseline
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
    n = name.strip().lower().replace(" ", "").replace("-", "_")
    return n in {"game3", "game_3", "3"}


def _load_homography_matrix(path: Path) -> List[List[float]]:
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
        "If your detections are already rectified into board squares, regenerate the detection log "
        "for game3 using the saved homography at recording time."
    )


def _load_detections_with_optional_homography(det_path: Path, homography_path: Path | None) -> Any:
    if homography_path is None:
        return load_detections(str(det_path))

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

    det_log = load_detections(str(det_path), **kwargs) if kwargs else load_detections(str(det_path))

    if not kwargs:
        _attach_homography_best_effort(det_log, homography_path)

    return det_log


# ---------------------------------------------------------------------------
# Ablation helper: disable candidate-set memory
# ---------------------------------------------------------------------------

def _disable_candidate_set_memory(tracker: MoveTracker) -> None:
    """
    Disable storing multi-candidate sets across frames, but keep N-frame confirmation
    for a unique candidate. No grace misses.
    """

    def _unique_only(self: MoveTracker, candidates: List[Any]):  # type: ignore[override]
        self._pending_grace_misses = 0  # type: ignore[attr-defined]

        if not candidates or len(candidates) != 1:
            self._pending_candidates = None  # type: ignore[attr-defined]
            self._pending_frames = 0  # type: ignore[attr-defined]
            return None

        move = candidates[0]
        new_uci = move.uci()

        old_uci = None
        pend = getattr(self, "_pending_candidates", None)
        if isinstance(pend, list) and len(pend) == 1:
            try:
                old_uci = pend[0].uci()
            except Exception:
                old_uci = None

        if old_uci == new_uci:
            self._pending_frames += 1  # type: ignore[attr-defined]
        else:
            self._pending_candidates = [move]  # type: ignore[attr-defined]
            self._pending_frames = 1  # type: ignore[attr-defined]

        if self._pending_frames >= self.min_confirm_frames:
            self._pending_candidates = None  # type: ignore[attr-defined]
            self._pending_frames = 0  # type: ignore[attr-defined]
            return move

        return None

    tracker._max_grace_misses = 0  # type: ignore[attr-defined]
    tracker._confirm_or_store_candidates = types.MethodType(_unique_only, tracker)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Frame index helpers (used only when GT frame indices exceed len(states))
# ---------------------------------------------------------------------------

def _state_video_frame_idx(state: Any, fallback: int) -> int:
    for key in ("frame_idx", "frame_index", "frame", "video_frame", "source_frame_idx"):
        v = getattr(state, key, None)
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
    return fallback


def _run_multistage_variant(
        states: List[Any],
        *,
        video_fps: float | None,
        detector_fps: float | None,
        alpha: float,
        occ_threshold: float,
        min_confirm_frames: int,
        start_confirm_frames: int,
        debug: bool,
        use_labels: bool,
        use_legal_moves: bool,
        pending_mode: str,
        frame_index_mode: str,
) -> Tuple[PipelineResult, Dict[str, Any]]:
    """
    frame_index_mode:
      - "state": align to list index, same behavior as compare_pipelines
      - "video": align to state.frame_idx style indices (only when GT exceeds len(states))
    """
    tracker = MoveTracker(
        alpha=float(alpha),
        occ_threshold=float(occ_threshold),
        min_confirm_frames=int(min_confirm_frames),
        start_confirm_frames=int(start_confirm_frames),
        use_legal_moves=bool(use_legal_moves),
        debug=bool(debug),
    )

    if pending_mode == "no_candidate_set_memory":
        _disable_candidate_set_memory(tracker)

    frame_fens: List[str] = []
    moves_uci: List[str] = []
    moves_san: List[str] = []
    move_frames: List[int] = []

    if not states:
        return PipelineResult([], [], [], []), {"pending_mode": pending_mode, "frame_index_mode": frame_index_mode}

    # Default path: behave like compare_pipelines (index by state list index)
    if frame_index_mode == "state":
        total_frames = len(states)
        for idx, should_process in iter_time_based_should_process(video_fps, detector_fps, total_frames):
            state = states[idx]

            info = None
            if should_process:
                if use_labels:
                    info = tracker.update_from_detection_state(state)
                else:
                    info = tracker.update_from_state(state.occupancy, {})

            frame_fens.append(tracker.board.fen())

            if info is not None:
                moves_uci.append(info.move.uci())
                moves_san.append(info.san)
                move_frames.append(idx)

        dbg: Dict[str, Any] = {}
        try:
            dbg = tracker.get_debug_counters()  # type: ignore[attr-defined]
        except Exception:
            dbg = {}
        dbg["pending_mode"] = pending_mode
        dbg["frame_index_mode"] = frame_index_mode
        dbg["frame_fens_len"] = len(frame_fens)
        return PipelineResult(frame_fens, moves_uci, moves_san, move_frames), dbg

    # Video-index mode: only used when GT frame indices do not fit into len(states).
    probe_n = min(50, len(states))
    vids = [_state_video_frame_idx(states[i], i) for i in range(probe_n)]
    sampled = any((vids[i + 1] - vids[i]) > 1 for i in range(len(vids) - 1))

    last_vid = -1

    def emit_gap_until(target_vid: int) -> None:
        nonlocal last_vid
        if target_vid <= last_vid:
            return
        for _ in range(last_vid + 1, target_vid):
            frame_fens.append(tracker.board.fen())
        last_vid = target_vid - 1

    if sampled:
        # States already look sampled, do not skip again.
        for i, state in enumerate(states):
            vid = _state_video_frame_idx(state, i)
            if vid <= last_vid:
                vid = last_vid + 1

            emit_gap_until(vid)

            if use_labels:
                info = tracker.update_from_detection_state(state)
            else:
                info = tracker.update_from_state(state.occupancy, {})

            frame_fens.append(tracker.board.fen())
            last_vid = vid

            if info is not None:
                moves_uci.append(info.move.uci())
                moves_san.append(info.san)
                move_frames.append(vid)
    else:
        # States look per-frame but their indices do not match GT, so we densify by vid.
        total_frames = len(states)
        for idx, should_process in iter_time_based_should_process(video_fps, detector_fps, total_frames):
            state = states[idx]
            vid = _state_video_frame_idx(state, idx)
            if vid <= last_vid:
                vid = last_vid + 1

            emit_gap_until(vid)

            info = None
            if should_process:
                if use_labels:
                    info = tracker.update_from_detection_state(state)
                else:
                    info = tracker.update_from_state(state.occupancy, {})

            frame_fens.append(tracker.board.fen())
            last_vid = vid

            if info is not None:
                moves_uci.append(info.move.uci())
                moves_san.append(info.san)
                move_frames.append(vid)

    dbg2: Dict[str, Any] = {}
    try:
        dbg2 = tracker.get_debug_counters()  # type: ignore[attr-defined]
    except Exception:
        dbg2 = {}
    dbg2["pending_mode"] = pending_mode
    dbg2["frame_index_mode"] = frame_index_mode
    dbg2["sampled_input"] = sampled
    dbg2["first_vid"] = _state_video_frame_idx(states[0], 0)
    dbg2["last_vid"] = last_vid
    dbg2["frame_fens_len"] = len(frame_fens)

    return PipelineResult(frame_fens, moves_uci, moves_san, move_frames), dbg2


def _safe_mean_median_delay_seconds(
        delays_frames: List[int],
        video_fps: float | None,
) -> Tuple[float | None, float | None]:
    if not delays_frames:
        return None, None
    if not video_fps or video_fps <= 0:
        return None, None
    delays_s = [d / float(video_fps) for d in delays_frames]
    return float(sum(delays_s) / len(delays_s)), float(statistics.median(delays_s))


def _iter_games(manifest_path: Path) -> Iterable[Dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    games = manifest.get("games", [])
    if not games:
        raise ValueError("Manifest contains no games.")
    for g in games:
        yield g


def _metrics_row(
        *,
        pipeline_name: str,
        variant: str,
        game: str,
        alpha: float | None,
        min_confirm_frames: int | None,
        start_confirm_frames: int | None,
        use_labels: bool | None,
        use_legal_moves: bool | None,
        pending_mode: str | None,
        gt: Any,
        res: PipelineResult,
        video_fps: float | None,
        debug_counters: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    # If GT uses a larger frame index space than the produced frame_fens, pad with the last state.
    if gt.frame_for_ply and res.frame_fens:
        needed = max(gt.frame_for_ply.values()) + 1
        if len(res.frame_fens) < needed:
            log.warning(
                "Padding frame_fens for %s/%s from %d to %d (GT max frame exceeds predictions).",
                game,
                variant,
                len(res.frame_fens),
                needed,
            )
            res = PipelineResult(
                frame_fens=res.frame_fens + [res.frame_fens[-1]] * (needed - len(res.frame_fens)),
                moves_uci=res.moves_uci,
                moves_san=res.moves_san,
                move_frames=res.move_frames,
            )

    delays_frames = move_detection_delays(res.moves_uci, res.move_frames, gt)
    mean_delay_s, median_delay_s = _safe_mean_median_delay_seconds(delays_frames, video_fps)

    row: Dict[str, Any] = {
        "pipeline": pipeline_name,
        "variant": variant,
        "game": game,
        "alpha": alpha,
        "min_confirm_frames": min_confirm_frames,
        "start_confirm_frames": start_confirm_frames,
        "use_labels": use_labels,
        "use_legal_moves": use_legal_moves,
        "pending_mode": pending_mode,
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

    if debug_counters:
        for k, v in debug_counters.items():
            row[f"dbg_{k}"] = v

    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--skip_baseline", action="store_true")

    ap.add_argument(
        "--game3_homography",
        type=str,
        default=None,
        help="Path to saved homography (json, npy, or txt). Applied only when game name resolves to game3.",
    )

    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    alpha_default = config.MOVE_FILTER_ALPHA
    occ_threshold = config.MOVE_FILTER_THRESHOLD
    min_confirm_default = config.MOVE_MIN_CONFIRM_FRAMES
    debug = config.MOVE_DEBUG

    start_confirm_frames = min_confirm_default

    variants: List[Tuple[str, Dict[str, Any]]] = [
        (
            "full",
            dict(
                alpha=alpha_default,
                min_confirm=min_confirm_default,
                use_labels=True,
                use_legal_moves=True,
                pending_mode="full",
            ),
        ),
        (
            "no_ema",
            dict(
                alpha=1.0,
                min_confirm=min_confirm_default,
                use_labels=True,
                use_legal_moves=True,
                pending_mode="full",
            ),
        ),
        (
            "no_rules",
            dict(
                alpha=alpha_default,
                min_confirm=min_confirm_default,
                use_labels=True,
                use_legal_moves=False,
                pending_mode="full",
            ),
        ),
        (
            "no_labels",
            dict(
                alpha=alpha_default,
                min_confirm=min_confirm_default,
                use_labels=False,
                use_legal_moves=True,
                pending_mode="full",
            ),
        ),
        (
            "no_confirm",
            dict(
                alpha=alpha_default,
                min_confirm=1,
                use_labels=True,
                use_legal_moves=True,
                pending_mode="full",
            ),
        ),
        (
            "no_candidate_set_memory",
            dict(
                alpha=alpha_default,
                min_confirm=min_confirm_default,
                use_labels=True,
                use_legal_moves=True,
                pending_mode="no_candidate_set_memory",
            ),
        ),
    ]

    rows: List[Dict[str, Any]] = []

    for game in _iter_games(manifest_path):
        name = str(game["name"])
        det_path = Path(game["detections"])
        gt_path = Path(game["gt"])

        homography_override: Path | None = None
        if _is_game3(name):
            if args.game3_homography:
                homography_override = Path(args.game3_homography)
            else:
                for k in ("homography", "homography_path", "saved_homography", "saved_homography_path"):
                    if k in game and game[k]:
                        homography_override = Path(str(game[k]))
                        break

        det_log = _load_detections_with_optional_homography(det_path, homography_override)
        gt = load_ground_truth(str(gt_path))

        detections = det_log.detections
        video_fps = getattr(det_log, "video_fps", None)
        detector_fps = getattr(det_log, "detector_fps", None)

        gt_max_frame = max(gt.frame_for_ply.values()) if gt.frame_for_ply else -1
        # Important: default to "state" to match compare_pipelines behavior.
        # Only switch to "video" if GT frame indices exceed len(states) - 1.
        frame_index_mode = "video" if gt_max_frame >= len(detections) else "state"

        log.info(
            "Game %s: detections=%d, video_fps=%.3f, detector_fps=%.3f, gt_moves=%d, gt_max_frame=%d, index_mode=%s",
            name,
            len(detections),
            float(video_fps or 0.0),
            float(detector_fps or 0.0),
            len(gt.moves_uci),
            int(gt_max_frame),
            frame_index_mode,
        )

        if homography_override is not None:
            log.info("Game %s: using homography override: %s", name, str(homography_override))

        if not args.skip_baseline:
            log.info("Running baseline_singleframe on %s", name)
            base_res = run_baseline(
                detections,
                video_fps=video_fps,
                detector_fps=detector_fps,
            )
            rows.append(
                _metrics_row(
                    pipeline_name="baseline_singleframe",
                    variant="default",
                    game=name,
                    alpha=None,
                    min_confirm_frames=None,
                    start_confirm_frames=None,
                    use_labels=None,
                    use_legal_moves=None,
                    pending_mode=None,
                    gt=gt,
                    res=base_res,
                    video_fps=video_fps,
                    debug_counters=None,
                )
            )

        for var_name, cfg in variants:
            log.info("Running multistage variant=%s on %s", var_name, name)
            ms_res, dbg = _run_multistage_variant(
                detections,
                video_fps=video_fps,
                detector_fps=detector_fps,
                alpha=float(cfg["alpha"]),
                occ_threshold=occ_threshold,
                min_confirm_frames=int(cfg["min_confirm"]),
                start_confirm_frames=int(start_confirm_frames),
                debug=debug,
                use_labels=bool(cfg["use_labels"]),
                use_legal_moves=bool(cfg["use_legal_moves"]),
                pending_mode=str(cfg["pending_mode"]),
                frame_index_mode=frame_index_mode,
            )
            rows.append(
                _metrics_row(
                    pipeline_name="multistage",
                    variant=var_name,
                    game=name,
                    alpha=float(cfg["alpha"]),
                    min_confirm_frames=int(cfg["min_confirm"]),
                    start_confirm_frames=int(start_confirm_frames),
                    use_labels=bool(cfg["use_labels"]),
                    use_legal_moves=bool(cfg["use_legal_moves"]),
                    pending_mode=str(cfg["pending_mode"]),
                    gt=gt,
                    res=ms_res,
                    video_fps=video_fps,
                    debug_counters=dbg,
                )
            )

    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    log.info("Wrote: %s", out_path)


if __name__ == "__main__":
    main()
