"""
==============================================================================
DOMAIN EXCEPTION HIERARCHY
==============================================================================
Mục đích: thay thế 219 khối `except Exception` trần trong codebase.

VẤN ĐỀ VỚI `except Exception`
-----------------------------
    try:
        result = train_model(df)
    except Exception as e:
        logger.warning(f"failed: {e}")
        result = fallback()          # <-- pipeline chạy tiếp, trả số SAI

Với hệ thống forecast phục vụ vận hành, đây là rủi ro NGHIỆP VỤ chứ không chỉ
là vấn đề code: nhà hàng nhận số liệu sai mà không ai biết model đã chết.

NGUYÊN TẮC THAY THẾ
-------------------
1. Bắt exception CỤ THỂ nhất có thể.
2. Phân biệt rõ hai loại lỗi:
   - RecoverableError -> có fallback hợp lệ, ghi log WARNING, đánh dấu degraded.
   - FatalError       -> không thể tiếp tục, log ERROR và dừng pipeline.
3. Mọi fallback PHẢI được đánh dấu vào kết quả (`quality_flag`), để downstream
   và dashboard biết con số này kém tin cậy.

VÍ DỤ ĐÚNG
----------
    try:
        result = train_model(df)
    except InsufficientDataError as exc:
        logger.warning("Không đủ dữ liệu cho %s: %s", exc.restaurant_code, exc)
        result = baseline_forecast(df)
        result.quality_flag = QualityFlag.DEGRADED_FALLBACK
    except ModelTrainingError:
        logger.exception("Training thất bại cho %s", code)
        raise                        # không nuốt lỗi không lường trước
"""

from __future__ import annotations

from enum import Enum
from typing import Any


# ==============================================================================
# QUALITY FLAG — gắn vào mọi output để theo dõi độ tin cậy
# ==============================================================================
class QualityFlag(str, Enum):
    """Mức độ tin cậy của một kết quả forecast."""

    OK = "ok"  # chạy đủ pipeline như thiết kế
    DEGRADED_FALLBACK = "degraded_fallback"  # một model lỗi, đã dùng fallback
    DEGRADED_PARTIAL = "degraded_partial"  # thiếu một phần dữ liệu đầu vào
    UNRELIABLE = "unreliable"  # chỉ dùng baseline, cần review tay


# ==============================================================================
# BASE
# ==============================================================================
class ForecastSystemError(Exception):
    """Lớp gốc cho mọi lỗi của hệ thống. Bắt lớp này = bắt lỗi của CHÍNH ta."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        if not self.context:
            return self.message
        detail = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} [{detail}]"


class RecoverableError(ForecastSystemError):
    """Lỗi CÓ fallback hợp lệ. Pipeline tiếp tục nhưng phải đánh dấu degraded."""

    quality_flag: QualityFlag = QualityFlag.DEGRADED_FALLBACK


class FatalError(ForecastSystemError):
    """Lỗi KHÔNG thể tiếp tục. Pipeline phải dừng."""


# ==============================================================================
# CONFIGURATION
# ==============================================================================
class ConfigurationError(FatalError):
    """Thiếu hoặc sai biến môi trường / tham số cấu hình."""


class MissingCredentialError(ConfigurationError):
    """Thiếu credential bắt buộc (DB, API key)."""


# ==============================================================================
# DATA LAYER
# ==============================================================================
class DataError(ForecastSystemError):
    """Lỗi thuộc tầng dữ liệu."""


class DatabaseConnectionError(FatalError, DataError):
    """Không kết nối được datamart. Không có dữ liệu -> không forecast được."""


class DataValidationError(DataError, RecoverableError):
    """Dữ liệu về được nhưng sai schema / sai kiểu / vi phạm ràng buộc."""


class InsufficientDataError(DataError, RecoverableError):
    """Không đủ số điểm dữ liệu để train (nhà hàng mới, mới mở lại)."""

    quality_flag = QualityFlag.DEGRADED_FALLBACK

    def __init__(
        self,
        restaurant_code: str,
        available: int,
        required: int,
    ) -> None:
        super().__init__(
            "Không đủ dữ liệu lịch sử để huấn luyện",
            restaurant_code=restaurant_code,
            available=available,
            required=required,
        )
        self.restaurant_code = restaurant_code
        self.available = available
        self.required = required


class MasterFileError(DataError):
    """Lỗi đọc/ghi file Excel master tracking."""


# ==============================================================================
# MODEL LAYER
# ==============================================================================
class ModelError(ForecastSystemError):
    """Lỗi thuộc tầng mô hình."""


class ModelTrainingError(ModelError, RecoverableError):
    """Một base model train thất bại. Ensemble có thể chạy với các model còn lại."""

    def __init__(self, model_name: str, restaurant_code: str, cause: str) -> None:
        super().__init__(
            "Huấn luyện mô hình thất bại",
            model_name=model_name,
            restaurant_code=restaurant_code,
            cause=cause,
        )
        self.model_name = model_name
        self.restaurant_code = restaurant_code


class ModelPredictionError(ModelError, RecoverableError):
    """Model đã train nhưng predict lỗi."""


class AllModelsFailedError(ModelError, FatalError):
    """Toàn bộ base model thất bại — không còn gì để ensemble."""


class ModelArtifactNotFoundError(ModelError, RecoverableError):
    """Không tìm thấy model đã lưu -> phải train lại."""


# ==============================================================================
# LLM / RAG LAYER
# ==============================================================================
class LLMError(RecoverableError):
    """Lỗi tầng LLM. LUÔN recoverable — ensemble phải chạy được 100% bằng ML."""


class LLMTimeoutError(LLMError):
    """Inference vượt AI_TIMEOUT_PER_RESTAURANT."""


class LLMResponseParseError(LLMError):
    """LLM trả về nội dung không parse được thành JSON theo schema mong đợi."""


class KnowledgeStoreError(RecoverableError):
    """Lỗi vector store / RAG retrieval."""


# ==============================================================================
# PIPELINE
# ==============================================================================
class PipelineError(ForecastSystemError):
    """Lỗi điều phối pipeline."""


class StageFailedError(PipelineError):
    """Một stage thất bại. `critical` quyết định pipeline dừng hay tiếp tục."""

    def __init__(self, stage_name: str, cause: Exception, critical: bool = True) -> None:
        super().__init__(
            f"Stage '{stage_name}' thất bại",
            stage=stage_name,
            cause=type(cause).__name__,
            critical=critical,
        )
        self.stage_name = stage_name
        self.cause = cause
        self.critical = critical


__all__ = [
    "AllModelsFailedError",
    "ConfigurationError",
    "DataError",
    "DataValidationError",
    "DatabaseConnectionError",
    "FatalError",
    "ForecastSystemError",
    "InsufficientDataError",
    "KnowledgeStoreError",
    "LLMError",
    "LLMResponseParseError",
    "LLMTimeoutError",
    "MasterFileError",
    "MissingCredentialError",
    "ModelArtifactNotFoundError",
    "ModelError",
    "ModelPredictionError",
    "ModelTrainingError",
    "PipelineError",
    "QualityFlag",
    "RecoverableError",
    "StageFailedError",
]
