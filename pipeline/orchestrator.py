"""
==============================================================================
PIPELINE ORCHESTRATOR — entrypoint mới, thay cho `main.py::main()`
==============================================================================

So sánh:
    main.py::main()   2.007 dòng, complexity 355   -> không test được
    orchestrator.py     ~90 dòng, complexity  < 10 -> test được

Chạy:
    python -m forecast_system.pipeline.orchestrator --mode daily
    forecast-run --mode full                    # sau khi pip install -e .
    forecast-run --mode daily --only ensemble_forecast persist_results
"""

from __future__ import annotations

import argparse
import sys
import time

from forecast_system.config.settings import validate_startup_config
from forecast_system.core.exceptions import ConfigurationError, StageFailedError
from forecast_system.pipeline.stages import (
    DEFAULT_PIPELINE,
    ForecastMode,
    PipelineContext,
    Stage,
)
from forecast_system.utils.logger import get_logger, setup_logger

logger = get_logger(__name__)


def run_pipeline(
    mode: ForecastMode = "daily",
    stages: list[type[Stage]] | None = None,
    only: list[str] | None = None,
) -> PipelineContext:
    """
    Chạy pipeline forecast.

    Args:
        mode: 'daily' (30 ngày, nhanh) hoặc 'full' (3 tháng, đầy đủ).
        stages: Danh sách stage. None -> DEFAULT_PIPELINE.
        only: Chỉ chạy các stage có tên trong danh sách này (dùng để resume).

    Returns:
        PipelineContext chứa kết quả và metric của lần chạy.

    Raises:
        ConfigurationError: thiếu biến môi trường bắt buộc.
        StageFailedError: một stage `critical=True` thất bại.
    """
    validate_startup_config()  # fail-fast TRƯỚC khi tốn 20 phút training

    ctx = PipelineContext(mode=mode)
    pipeline = stages or DEFAULT_PIPELINE
    if only:
        pipeline = [s for s in pipeline if s.name in set(only)]

    logger.info("=" * 70)
    logger.info(
        "FORECAST PIPELINE | run_id=%s | mode=%s | %d stages", ctx.run_id, mode, len(pipeline)
    )
    logger.info("=" * 70)

    started = time.perf_counter()

    for stage_cls in pipeline:
        stage = stage_cls()
        try:
            stage.run(ctx)
        except StageFailedError as exc:
            if exc.critical:
                logger.exception("💥 Stage bắt buộc thất bại, dừng pipeline")
                raise
            logger.warning("⚠️  Bỏ qua stage không bắt buộc: %s", exc)
            ctx.warnings.append(str(exc))

    _log_summary(ctx, time.perf_counter() - started)
    return ctx


def _log_summary(ctx: PipelineContext, total_seconds: float) -> None:
    """In bảng tổng kết thời gian và chất lượng của lần chạy."""
    logger.info("-" * 70)
    logger.info("TỔNG KẾT | run_id=%s | tổng %.1fs", ctx.run_id, total_seconds)
    for name, seconds in sorted(ctx.stage_durations.items(), key=lambda kv: kv[1], reverse=True):
        share = seconds / total_seconds * 100 if total_seconds else 0
        logger.info("  %-28s %7.1fs  (%4.1f%%)", name, seconds, share)

    if ctx.degraded_count:
        logger.warning("⚠️  %d nhà hàng có kết quả DEGRADED — cần review", ctx.degraded_count)
    for warning in ctx.warnings:
        logger.warning("  • %s", warning)
    logger.info("-" * 70)


def cli() -> int:
    """Console entrypoint (khai báo ở [project.scripts] trong pyproject.toml)."""
    parser = argparse.ArgumentParser(
        prog="forecast-run",
        description="AI Forecast System — pipeline dự báo lượng khách",
    )
    parser.add_argument(
        "--mode",
        choices=["daily", "full"],
        default="daily",
        help="daily = 30 ngày (nhanh); full = 3 tháng (đầy đủ). Mặc định: daily",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="STAGE",
        help="Chỉ chạy các stage chỉ định (dùng để resume sau khi lỗi)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logger(level=args.log_level)

    try:
        run_pipeline(mode=args.mode, only=args.only)
    except ConfigurationError as exc:
        logger.critical("Lỗi cấu hình: %s", exc)
        logger.critical("Kiểm tra config/.env — tham khảo .env.example")
        return 2
    except StageFailedError as exc:
        logger.critical("Pipeline thất bại: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Người dùng huỷ pipeline.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(cli())
