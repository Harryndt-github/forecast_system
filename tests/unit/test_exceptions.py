"""Test cho exception hierarchy — nền tảng của việc thay thế `except Exception`."""

from __future__ import annotations

import pytest

from forecast_system.core.exceptions import (
    AllModelsFailedError,
    DatabaseConnectionError,
    FatalError,
    ForecastSystemError,
    InsufficientDataError,
    LLMTimeoutError,
    ModelTrainingError,
    QualityFlag,
    RecoverableError,
    StageFailedError,
)


@pytest.mark.unit
def test_moi_loi_deu_ke_thua_tu_base():
    """Bắt ForecastSystemError = bắt được mọi lỗi của hệ thống ta viết ra."""
    for exc_cls in (
        DatabaseConnectionError, InsufficientDataError,
        ModelTrainingError, LLMTimeoutError, AllModelsFailedError,
    ):
        assert issubclass(exc_cls, ForecastSystemError)


@pytest.mark.unit
def test_phan_loai_recoverable_vs_fatal():
    """Phân loại này quyết định pipeline dừng hay chạy tiếp — phải đúng."""
    assert issubclass(InsufficientDataError, RecoverableError)
    assert issubclass(ModelTrainingError, RecoverableError)
    assert issubclass(LLMTimeoutError, RecoverableError)   # LLM luôn có fallback ML

    assert issubclass(DatabaseConnectionError, FatalError)
    assert issubclass(AllModelsFailedError, FatalError)


@pytest.mark.unit
def test_context_duoc_dua_vao_thong_diep():
    """Exception phải mang đủ context để debug mà không cần đọc log xung quanh."""
    exc = InsufficientDataError(restaurant_code="R042", available=5, required=30)

    message = str(exc)
    assert "R042" in message
    assert "5" in message
    assert "30" in message
    assert exc.quality_flag is QualityFlag.DEGRADED_FALLBACK


@pytest.mark.unit
def test_stage_failed_giu_nguyen_nhan_goc():
    original = ValueError("shape mismatch")
    exc = StageFailedError("ensemble_forecast", original, critical=True)

    assert exc.stage_name == "ensemble_forecast"
    assert exc.cause is original
    assert exc.critical is True
    assert "ensemble_forecast" in str(exc)


@pytest.mark.unit
def test_llm_that_bai_khong_lam_chet_pipeline():
    """
    Quy tắc kiến trúc: LLM là lớp tăng cường, không phải phụ thuộc bắt buộc.
    Mọi lỗi LLM phải recoverable để ensemble chạy 100% bằng ML.
    """
    assert issubclass(LLMTimeoutError, RecoverableError)
    assert not issubclass(LLMTimeoutError, FatalError)
