from __future__ import annotations

"""
Factory and entry point for live pipelines.

Picks between the multistage tracker and the single frame baseline
based on configuration or function arguments, and starts the selected
pipeline with a configured capture source.
"""

import logging
from typing import Optional

from src import config
from src.common.app_logging import get_logger, set_log_level
from src.pipeline.live_base import get_capture_source, CaptureSource, BaseLivePipeline
from src.pipeline.multistage.live_multistage_main import MultiStagePipeline
from src.pipeline.singleframe.live_singleframe_main import SingleFramePipeline

_log = get_logger(__name__)


def _default_frame_size() -> tuple[int, int]:
    """
    Resolve default frame size from config.
    """
    width = int(getattr(config, "FRAME_WIDTH", 1280))
    height = int(getattr(config, "FRAME_HEIGHT", 720))
    return width, height


def create_pipeline(
        *,
        mode: Optional[str] = None,
        source: Optional[CaptureSource] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
) -> BaseLivePipeline:
    """
    Create the appropriate live pipeline based on mode and source.

    mode:
      - "multistage" or aliases "multi", "tracking"
      - "singleframe" or aliases "single", "baseline"
      - None: uses config.PIPELINE_MODE

    source:
      - camera index (int)
      - video path (str)
      - None: get_capture_source()

    width/height:
      - requested capture size, used as hints for the backend
    """
    if mode is None:
        mode = getattr(config, "PIPELINE_MODE", "multistage")

    mode_norm = (mode or "").strip().lower()

    if source is None:
        source = get_capture_source()

    if width is None or height is None:
        default_w, default_h = _default_frame_size()
        if width is None:
            width = default_w
        if height is None:
            height = default_h

    if mode_norm in ("multistage", "multi", "tracking"):
        board_size_px = int(getattr(config, "BOARD_SIZE_PX", 640))
        _log.info(
            "Creating MultiStagePipeline (mode=%r, source=%r, %dx%d, board=%d)",
            mode_norm,
            source,
            width,
            height,
            board_size_px,
        )
        return MultiStagePipeline(
            source=source,
            width=width,
            height=height,
            board_size_px=board_size_px,
        )

    if mode_norm in ("singleframe", "single", "baseline"):
        _log.info(
            "Creating SingleFramePipeline (mode=%r, source=%r, %dx%d)",
            mode_norm,
            source,
            width,
            height,
        )
        return SingleFramePipeline(
            source=source,
            width=width,
            height=height,
        )

    raise ValueError(f"Unknown PIPELINE_MODE: {mode!r}")


def run_from_config() -> None:
    """
    Convenience entry point that reads configuration and runs a pipeline.

      - set log level from config.LOG_LEVEL
      - pick source from config
      - pick mode from config.PIPELINE_MODE
      - run the selected pipeline
    """
    log_level = getattr(config, "LOG_LEVEL", logging.DEBUG)
    set_log_level(log_level)

    mode = getattr(config, "PIPELINE_MODE", "multistage")
    source = get_capture_source()
    width, height = _default_frame_size()

    pipeline = create_pipeline(
        mode=mode,
        source=source,
        width=width,
        height=height,
    )

    _log.info("Starting live pipeline with mode=%r", mode)
    pipeline.run()


def main() -> None:
    run_from_config()


if __name__ == "__main__":
    main()
