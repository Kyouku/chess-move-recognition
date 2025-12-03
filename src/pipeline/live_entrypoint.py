from __future__ import annotations

import logging
from typing import Optional

from src import config
from src.app_logging import get_logger, set_log_level
from src.pipeline.live_base import get_capture_source, CaptureSource
from src.pipeline.multistage.live_multistage_main import MultiStagePipeline
from src.pipeline.singleframe.live_singleframe_main import SingleFramePipeline

_log = get_logger(__name__)


def create_pipeline(
        *,
        mode: Optional[str] = None,
        source: Optional[CaptureSource] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
) -> BaseException | MultiStagePipeline | SingleFramePipeline:
    """
    Create the appropriate live pipeline based on mode and source.

    mode:
      - "multistage" or aliases "multi", "tracking"
      - "singleframe" or aliases "single", "baseline"
      - None -> uses config.PIPELINE_MODE

    source:
      - camera index (int)
      - video path (str)
      - None -> get_capture_source()
    """

    if mode is None:
        mode = getattr(config, "PIPELINE_MODE", "multistage")

    mode_norm = (mode or "").strip().lower()

    if source is None:
        source = get_capture_source()

    if width is None:
        width = int(getattr(config, "FRAME_WIDTH", 1280))
    if height is None:
        height = int(getattr(config, "FRAME_HEIGHT", 720))

    if mode_norm in ("multistage", "multi", "tracking"):
        _log.info(
            "Creating MultiStagePipeline (mode=%r, source=%r, %dx%d)",
            mode_norm,
            source,
            width,
            height,
        )
        return MultiStagePipeline(
            source=source,
            width=width,
            height=height,
            board_size_px=int(getattr(config, "BOARD_SIZE_PX", 640)),
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
    Convenience function:
      - set log level
      - get source from config
      - choose mode via config.PIPELINE_MODE
      - start pipeline.run()
    """
    # Logging
    log_level = getattr(config, "LOG_LEVEL", logging.DEBUG)
    set_log_level(log_level)

    mode = getattr(config, "PIPELINE_MODE", "multistage")
    source = get_capture_source()

    pipeline = create_pipeline(
        mode=mode,
        source=source,
        width=int(getattr(config, "FRAME_WIDTH", 1280)),
        height=int(getattr(config, "FRAME_HEIGHT", 720)),
    )

    _log.info("Starting live pipeline with mode=%r", mode)
    pipeline.run()


def main() -> None:
    run_from_config()


if __name__ == "__main__":
    main()
