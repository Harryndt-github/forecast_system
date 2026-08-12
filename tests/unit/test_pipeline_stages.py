"""Test cho lớp trừu tượng Stage — chứng minh pipeline đã test được."""

from __future__ import annotations

import pytest

from forecast_system.core.exceptions import (
    InsufficientDataError,
    StageFailedError,
)
from forecast_system.pipeline.stages import PipelineContext, Stage


class _OkStage(Stage):
    name = "ok_stage"

    def execute(self, ctx: PipelineContext) -> None:
        ctx.active_restaurants = ["R001", "R002"]


class _RecoverableStage(Stage):
    name = "recoverable_stage"
    critical = False

    def execute(self, ctx: PipelineContext) -> None:
        raise InsufficientDataError("R999", available=3, required=30)


class _FatalStage(Stage):
    name = "fatal_stage"
    critical = True

    def execute(self, ctx: PipelineContext) -> None:
        raise ValueError("unexpected")


class _SkippedStage(Stage):
    name = "skipped_stage"

    def should_skip(self, ctx: PipelineContext) -> bool:
        return ctx.mode == "daily"

    def execute(self, ctx: PipelineContext) -> None:
        raise AssertionError("Không được chạy khi mode=daily")


@pytest.mark.unit
def test_stage_thanh_cong_ghi_thoi_gian():
    ctx = PipelineContext()
    _OkStage().run(ctx)

    assert ctx.active_restaurants == ["R001", "R002"]
    assert "ok_stage" in ctx.stage_durations
    assert ctx.stage_durations["ok_stage"] >= 0


@pytest.mark.unit
def test_loi_recoverable_khong_lam_dung_pipeline():
    """Đây là điểm khác biệt cốt lõi so với `except Exception` trần:
    lỗi được GHI NHẬN vào warnings thay vì biến mất."""
    ctx = PipelineContext()
    _RecoverableStage().run(ctx)

    assert len(ctx.warnings) == 1
    assert "R999" in ctx.warnings[0]


@pytest.mark.unit
def test_loi_khong_luong_truoc_duoc_boc_thanh_stage_failed():
    ctx = PipelineContext()
    with pytest.raises(StageFailedError) as exc_info:
        _FatalStage().run(ctx)

    assert exc_info.value.stage_name == "fatal_stage"
    assert isinstance(exc_info.value.cause, ValueError)


@pytest.mark.unit
def test_should_skip_duoc_ton_trong():
    ctx = PipelineContext(mode="daily")
    _SkippedStage().run(ctx)          # không raise = đã skip
    assert "skipped_stage" not in ctx.stage_durations


@pytest.mark.unit
def test_dem_ket_qua_degraded():
    from forecast_system.core.exceptions import QualityFlag

    ctx = PipelineContext()
    ctx.mark_degraded("R001", QualityFlag.DEGRADED_FALLBACK)
    ctx.mark_degraded("R002", QualityFlag.UNRELIABLE)
    ctx.mark_degraded("R003", QualityFlag.OK)

    assert ctx.degraded_count == 2
