"""
==============================================================================
PIPELINE STAGES — tái cấu trúc `main()` (2.007 dòng, complexity 355)
==============================================================================

VẤN ĐỀ HIỆN TẠI
---------------
`main.py::main()` là một hàm 2.007 dòng với cyclomatic complexity 355
(ngưỡng khuyến nghị: <= 10). Hệ quả:
  - Không thể unit-test: muốn test bước "holiday calibration" phải chạy cả
    kết nối DB, load data, train model.
  - Mỗi thay đổi đều có rủi ro hồi quy trên toàn pipeline.
  - Không thể chạy lại một bước riêng lẻ khi lỗi.

GIẢI PHÁP
---------
Bản thân `main()` đã có sẵn các mốc `# STEP N:` rất rõ ràng. Ta chỉ cần
"vật chất hoá" từng STEP thành một class `Stage` độc lập:

    STEP 1: DATABASE CONNECTION      -> ConnectDatabaseStage
    STEP 2: LOAD DATA                -> LoadDataStage
    STEP 3: ANALYSIS                 -> AnalysisStage
    STEP 5: PREPARE FORECAST         -> PrepareForecastStage
    STEP 5.5-5.7: HOLIDAY            -> HolidayCalibrationStage
    STEP 5.8: SPECIAL EVENT          -> EventCalibrationStage
    STEP 5.9: NEURALPROPHET GLOBAL   -> TrainGlobalModelStage
    STEP 5.95: BOOKING DATA          -> LoadBookingStage
    STEP 6: ENSEMBLE FORECAST LOOP   -> EnsembleForecastStage
    STEP 7: SAVE RESULTS             -> PersistResultsStage
    STEP 8: SUMMARY REPORT           -> SummaryReportStage
    STEP 9: MONITORING               -> MonitoringStage
    STEP 10: BRAIN INSIGHTS          -> BrainInsightsStage

LỢI ÍCH ĐO ĐƯỢC
---------------
  - Mỗi stage < 150 dòng, complexity < 12 -> test được từng cái.
  - `critical=False` cho phép stage phụ (report, insights) fail mà không
    làm hỏng cả lần chạy.
  - Resume: chạy lại từ stage bị lỗi thay vì chạy lại 20 phút từ đầu.
  - Quan sát được: mỗi stage tự log thời gian + trạng thái.

HƯỚNG DẪN MIGRATE (làm dần, không big-bang)
-------------------------------------------
  1. Copy nguyên khối code của một STEP từ `main()` vào `execute()` của stage
     tương ứng. CHƯA sửa logic — chỉ di chuyển.
  2. Thay biến cục bộ bằng `ctx.<field>`.
  3. Viết test cho stage đó.
  4. Xoá khối code cũ khỏi `main()`, gọi stage thay thế.
  5. Lặp lại. Mỗi stage là một PR riêng, review được.
"""

from __future__ import annotations

import abc
import datetime
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from forecast_system.core.exceptions import (
    ForecastSystemError,
    QualityFlag,
    RecoverableError,
    StageFailedError,
)
from forecast_system.utils.logger import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

ForecastMode = Literal["daily", "full"]


# ==============================================================================
# CONTEXT — thay thế ~80 biến cục bộ đang trôi nổi trong main()
# ==============================================================================
@dataclass
class PipelineContext:
    """
    State dùng chung giữa các stage.

    Thay vì hàng chục biến cục bộ trong một hàm 2.000 dòng, mọi state đi qua
    một object có kiểu rõ ràng. Đây cũng là thứ ta mock trong unit test.
    """

    # --- Input ---
    mode: ForecastMode = "daily"
    run_id: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    current_date: datetime.date = field(default_factory=datetime.date.today)

    # --- Data layer ---
    engine: Any | None = None
    df_history: pd.DataFrame | None = None
    df_booking: pd.DataFrame | None = None
    active_restaurants: list[str] = field(default_factory=list)

    # --- Analysis / calibration ---
    analysis_result: dict[str, Any] = field(default_factory=dict)
    holiday_calibration: dict[str, Any] = field(default_factory=dict)
    event_calibration: dict[str, Any] = field(default_factory=dict)
    closure_schedule: dict[str, Any] = field(default_factory=dict)

    # --- Model layer ---
    global_model: Any | None = None
    forecast_days: list[datetime.date] = field(default_factory=list)

    # --- Output ---
    forecast_results: pd.DataFrame | None = None
    quality_flags: dict[str, QualityFlag] = field(default_factory=dict)

    # --- Observability ---
    stage_durations: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def mark_degraded(self, restaurant_code: str, flag: QualityFlag) -> None:
        """Đánh dấu một nhà hàng có kết quả kém tin cậy."""
        self.quality_flags[restaurant_code] = flag

    @property
    def degraded_count(self) -> int:
        return sum(1 for f in self.quality_flags.values() if f is not QualityFlag.OK)


# ==============================================================================
# STAGE BASE
# ==============================================================================
class Stage(abc.ABC):
    """
    Một bước độc lập trong pipeline.

    Contract:
      - `execute()` nhận context, biến đổi nó tại chỗ, không trả về gì.
      - Stage chỉ bắt exception mà nó BIẾT cách xử lý.
      - Lỗi ngoài dự kiến để bay lên orchestrator — không nuốt.
    """

    name: str = "unnamed"
    critical: bool = True  # False = pipeline vẫn tiếp tục nếu stage này lỗi

    @abc.abstractmethod
    def execute(self, ctx: PipelineContext) -> None: ...

    def should_skip(self, ctx: PipelineContext) -> bool:  # noqa: ARG002
        """Override để bỏ qua stage tuỳ điều kiện (ví dụ chỉ chạy ở mode=full)."""
        return False

    def run(self, ctx: PipelineContext) -> None:
        """Wrapper lo timing, logging và phân loại lỗi. KHÔNG override hàm này."""
        if self.should_skip(ctx):
            logger.info("⏭️  Bỏ qua stage: %s", self.name)
            return

        logger.info("▶️  Bắt đầu stage: %s", self.name)
        started = time.perf_counter()
        try:
            self.execute(ctx)
        except RecoverableError as exc:
            elapsed = time.perf_counter() - started
            ctx.stage_durations[self.name] = elapsed
            ctx.warnings.append(f"{self.name}: {exc}")
            logger.warning("⚠️  Stage '%s' degraded (%.1fs): %s", self.name, elapsed, exc)
        except ForecastSystemError as exc:
            raise StageFailedError(self.name, exc, critical=self.critical) from exc
        except Exception as exc:
            logger.exception("❌ Lỗi không lường trước ở stage '%s'", self.name)
            raise StageFailedError(self.name, exc, critical=self.critical) from exc
        else:
            elapsed = time.perf_counter() - started
            ctx.stage_durations[self.name] = elapsed
            logger.info("✅ Hoàn tất stage: %s (%.1fs)", self.name, elapsed)


# ==============================================================================
# STAGE SKELETONS — điền code từ main.py theo mốc STEP tương ứng
# ==============================================================================
class ConnectDatabaseStage(Stage):
    """STEP 1 — mở SQLAlchemy engine, kiểm tra kết nối sống."""

    name = "connect_database"
    critical = True

    def execute(self, ctx: PipelineContext) -> None:
        # Import cục bộ: sqlalchemy nặng, chỉ nạp khi stage này thực sự chạy.
        from sqlalchemy import create_engine, text  # noqa: PLC0415

        from forecast_system.config.settings import (  # noqa: PLC0415
            get_connection_string,
            get_safe_connection_string,
        )

        logger.info("Kết nối tới %s", get_safe_connection_string())  # đã che credential
        ctx.engine = create_engine(
            get_connection_string(),
            pool_recycle=3600,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 60},
        )
        with ctx.engine.connect() as conn:
            conn.execute(text("SELECT 1"))


class LoadDataStage(Stage):
    """STEP 2 — nạp lịch sử giao dịch, xác định nhà hàng đang hoạt động."""

    name = "load_data"
    critical = True

    def execute(self, ctx: PipelineContext) -> None:
        # TODO(migrate): chuyển khối main.py:461-514 vào đây
        raise NotImplementedError("Chuyển STEP 2 từ main.py vào stage này")


class AnalysisStage(Stage):
    """STEP 3 — thống kê mô tả, phân khúc volume, phát hiện drift."""

    name = "analysis"
    critical = True

    def execute(self, ctx: PipelineContext) -> None:
        raise NotImplementedError("Chuyển STEP 3 từ main.py vào stage này")


class HolidayCalibrationStage(Stage):
    """STEP 5.5-5.7 — phát hiện đóng cửa lễ + hiệu chỉnh hệ số tác động."""

    name = "holiday_calibration"
    critical = False  # lỗi ở đây -> dùng hệ số mặc định trong settings

    def execute(self, ctx: PipelineContext) -> None:
        raise NotImplementedError("Chuyển STEP 5.5-5.7 từ main.py vào stage này")


class EventCalibrationStage(Stage):
    """STEP 5.8 — hiệu chỉnh sự kiện đặc biệt từ dữ liệu."""

    name = "event_calibration"
    critical = False

    def execute(self, ctx: PipelineContext) -> None:
        raise NotImplementedError("Chuyển STEP 5.8 từ main.py vào stage này")


class TrainGlobalModelStage(Stage):
    """STEP 5.9 — train NeuralProphet global một lần, dùng chung mọi nhà hàng."""

    name = "train_global_model"
    critical = False  # ensemble vẫn chạy được thiếu NeuralProphet

    def execute(self, ctx: PipelineContext) -> None:
        raise NotImplementedError("Chuyển STEP 5.9 từ main.py vào stage này")


class EnsembleForecastStage(Stage):
    """
    STEP 6 — vòng lặp forecast chính.

    LƯU Ý REFACTOR: đây là khối lớn nhất (~830 dòng). Nên tách tiếp thành:
      - `_forecast_one_restaurant(ctx, code)`  <- unit-test được
      - `_apply_closure_rules(...)`            <- 4 mức priority hiện đang lồng nhau
      - `_split_by_shift(...)`
    Mỗi hàm con giữ complexity < 12.
    """

    name = "ensemble_forecast"
    critical = True

    def execute(self, ctx: PipelineContext) -> None:
        raise NotImplementedError("Chuyển STEP 6 từ main.py vào stage này")


class PersistResultsStage(Stage):
    """STEP 7 — ghi kết quả ra master file + database."""

    name = "persist_results"
    critical = True

    def execute(self, ctx: PipelineContext) -> None:
        raise NotImplementedError("Chuyển STEP 7 từ main.py vào stage này")


class MonitoringStage(Stage):
    """STEP 9 — theo dõi accuracy, drift, sinh báo cáo."""

    name = "monitoring"
    critical = False

    def execute(self, ctx: PipelineContext) -> None:
        raise NotImplementedError("Chuyển STEP 9 từ main.py vào stage này")


class BrainInsightsStage(Stage):
    """STEP 10 — ForecastBrain hấp thụ kết quả, sinh correction cho lần sau."""

    name = "brain_insights"
    critical = False

    def execute(self, ctx: PipelineContext) -> None:
        raise NotImplementedError("Chuyển STEP 10 từ main.py vào stage này")


# ==============================================================================
# PIPELINE DEFINITION
# ==============================================================================
DEFAULT_PIPELINE: list[type[Stage]] = [
    ConnectDatabaseStage,
    LoadDataStage,
    AnalysisStage,
    HolidayCalibrationStage,
    EventCalibrationStage,
    TrainGlobalModelStage,
    EnsembleForecastStage,
    PersistResultsStage,
    MonitoringStage,
    BrainInsightsStage,
]
