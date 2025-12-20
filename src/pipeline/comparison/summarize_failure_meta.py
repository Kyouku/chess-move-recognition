from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from src.common.app_logging import get_logger
except Exception:  # pragma: no cover
    from src.common.io_utils import get_logger  # type: ignore

_log = get_logger(__name__)


def _find_meta_files(inp: Path) -> List[Path]:
    if inp.is_file():
        return [inp]
    # directory: search recursively
    return sorted(inp.rglob("failure_frames_meta.json"))


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _safe_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return list(x)


def _event_key(item: Dict[str, Any]) -> Tuple[str, int, int]:
    """
    Defines a "unique failure event" key.
    One event = (pipeline, frame, ply).
    This collapses move_wrong + fen_wrong exports for the same commit.
    """
    pipeline = str(item.get("pipeline", ""))
    frame = int(item.get("frame", -1))
    ply = int(item.get("ply", -1)) if item.get("ply") is not None else -1
    return (pipeline, frame, ply)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _table_counter_to_rows(counter: Counter, name_col: str) -> List[Dict[str, Any]]:
    total = sum(counter.values()) or 1
    rows: List[Dict[str, Any]] = []
    for k, v in counter.most_common():
        rows.append(
            {
                name_col: k,
                "count": int(v),
                "share_percent": round(100.0 * float(v) / float(total), 1),
            }
        )
    return rows


def _crosstab_rows(
        counts: Dict[Tuple[str, str], int],
        row_name: str,
        col_name: str,
) -> List[Dict[str, Any]]:
    # Collect all row/col keys
    rows = sorted({rk for rk, _ in counts.keys()})
    cols = sorted({ck for _, ck in counts.keys()})
    out: List[Dict[str, Any]] = []
    for r in rows:
        row: Dict[str, Any] = {row_name: r}
        for c in cols:
            row[c] = int(counts.get((r, c), 0))
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Either a single failure_frames_meta.json or a directory containing game subfolders.",
    )
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument(
        "--tag_mode",
        choices=["any", "primary"],
        default="any",
        help="any: count every tag occurrence; primary: count only tags[0].",
    )
    args = ap.parse_args()

    meta_files = _find_meta_files(args.input)
    if not meta_files:
        raise SystemExit(f"No meta files found under: {args.input}")

    # Export-level counters (each exported frame record)
    c_pipeline = Counter()
    c_reason = Counter()
    c_tag = Counter()
    c_game = Counter()
    xt_pipeline_reason: Dict[Tuple[str, str], int] = defaultdict(int)
    xt_pipeline_tag: Dict[Tuple[str, str], int] = defaultdict(int)

    # Event-level counters (deduped by (pipeline, frame, ply))
    e_pipeline = Counter()
    e_reason = Counter()
    e_tag = Counter()
    xt_e_pipeline_reason: Dict[Tuple[str, str], int] = defaultdict(int)
    xt_e_pipeline_tag: Dict[Tuple[str, str], int] = defaultdict(int)

    seen_events: set[Tuple[str, int, int]] = set()

    total_exports = 0
    total_events = 0

    per_file_rows: List[Dict[str, Any]] = []

    for mf in meta_files:
        data = _read_json(mf)
        game = str(data.get("game", mf.parent.name))
        exports = _safe_list(data.get("exports", []))

        per_file_rows.append(
            {
                "game": game,
                "meta_path": str(mf),
                "exports_in_file": len(exports),
            }
        )

        for item in exports:
            if not isinstance(item, dict):
                continue

            pipeline = str(item.get("pipeline", "unknown"))
            reason = str(item.get("reason", "unknown"))
            tags = [str(t) for t in _safe_list(item.get("tags", [])) if str(t).strip()]

            total_exports += 1
            c_game[game] += 1
            c_pipeline[pipeline] += 1
            c_reason[reason] += 1
            xt_pipeline_reason[(pipeline, reason)] += 1

            if args.tag_mode == "primary":
                if tags:
                    t0 = tags[0]
                    c_tag[t0] += 1
                    xt_pipeline_tag[(pipeline, t0)] += 1
            else:
                for t in tags:
                    c_tag[t] += 1
                    xt_pipeline_tag[(pipeline, t)] += 1

            # Dedup into events
            key = _event_key(item)
            if key not in seen_events:
                seen_events.add(key)
                total_events += 1
                e_pipeline[pipeline] += 1

                # Event-level reason: keep first seen reason for that event
                e_reason[reason] += 1
                xt_e_pipeline_reason[(pipeline, reason)] += 1

                if args.tag_mode == "primary":
                    if tags:
                        e_tag[tags[0]] += 1
                        xt_e_pipeline_tag[(pipeline, tags[0])] += 1
                else:
                    for t in tags:
                        e_tag[t] += 1
                        xt_e_pipeline_tag[(pipeline, t)] += 1

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write per-file overview
    _write_csv(
        out_dir / "meta_files_overview.csv",
        per_file_rows,
        ["game", "meta_path", "exports_in_file"],
    )

    # Export-level summaries
    _write_csv(out_dir / "counts_pipeline_exports.csv", _table_counter_to_rows(c_pipeline, "pipeline"),
               ["pipeline", "count", "share_percent"])
    _write_csv(out_dir / "counts_reason_exports.csv", _table_counter_to_rows(c_reason, "reason"),
               ["reason", "count", "share_percent"])
    _write_csv(out_dir / "counts_tag_exports.csv", _table_counter_to_rows(c_tag, "tag"),
               ["tag", "count", "share_percent"])

    _write_csv(
        out_dir / "crosstab_pipeline_reason_exports.csv",
        _crosstab_rows(xt_pipeline_reason, "pipeline", "reason"),
        ["pipeline"] + sorted({c for _, c in xt_pipeline_reason.keys()}),
    )
    _write_csv(
        out_dir / "crosstab_pipeline_tag_exports.csv",
        _crosstab_rows(xt_pipeline_tag, "pipeline", "tag"),
        ["pipeline"] + sorted({c for _, c in xt_pipeline_tag.keys()}),
    )

    # Event-level summaries
    _write_csv(out_dir / "counts_pipeline_events.csv", _table_counter_to_rows(e_pipeline, "pipeline"),
               ["pipeline", "count", "share_percent"])
    _write_csv(out_dir / "counts_reason_events.csv", _table_counter_to_rows(e_reason, "reason"),
               ["reason", "count", "share_percent"])
    _write_csv(out_dir / "counts_tag_events.csv", _table_counter_to_rows(e_tag, "tag"),
               ["tag", "count", "share_percent"])

    _write_csv(
        out_dir / "crosstab_pipeline_reason_events.csv",
        _crosstab_rows(xt_e_pipeline_reason, "pipeline", "reason"),
        ["pipeline"] + sorted({c for _, c in xt_e_pipeline_reason.keys()}),
    )
    _write_csv(
        out_dir / "crosstab_pipeline_tag_events.csv",
        _crosstab_rows(xt_e_pipeline_tag, "pipeline", "tag"),
        ["pipeline"] + sorted({c for _, c in xt_e_pipeline_tag.keys()}),
    )

    # Log quick summary
    _log.info("Meta files: %d", len(meta_files))
    _log.info("Total exported items: %d", total_exports)
    _log.info("Total unique events (pipeline,frame,ply): %d", total_events)
    _log.info("Exports by pipeline: %s", dict(c_pipeline))
    _log.info("Events by pipeline: %s", dict(e_pipeline))
    _log.info("Exports by reason: %s", dict(c_reason))
    _log.info("Events by reason: %s", dict(e_reason))
    _log.info("Exports by tag: %s", dict(c_tag))
    _log.info("Events by tag: %s", dict(e_tag))


if __name__ == "__main__":
    main()
