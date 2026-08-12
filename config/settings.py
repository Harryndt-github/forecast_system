"""
==============================================
FORECAST SYSTEM - CENTRALIZED CONFIGURATION
==============================================
Tất cả cấu hình được load từ .env file.
Không hardcode credentials trong code.
"""

import calendar
import datetime
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env từ thư mục config (không commit file này - xem .env.example)
_config_dir = Path(__file__).parent
_env_path = _config_dir / '.env'
load_dotenv(_env_path)


class ConfigurationError(RuntimeError):
    """Raise khi thiếu cấu hình bắt buộc. Fail-fast thay vì chạy với default sai."""


def _require_env(key: str, hint: str = "") -> str:
    """
    Đọc biến môi trường BẮT BUỘC.

    Không bao giờ trả về default cho secrets hoặc thông tin hạ tầng nội bộ.
    Nếu thiếu -> dừng ngay với thông báo rõ ràng, thay vì âm thầm kết nối sai host.
    """
    value = os.getenv(key)
    if not value:
        raise ConfigurationError(
            f"Thiếu biến môi trường bắt buộc: {key}. "
            f"Khai báo trong config/.env (tham khảo .env.example). {hint}".strip()
        )
    return value


def _env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("true", "1", "yes", "on")

# ==========================================
# DATABASE CONFIG
# ==========================================
# SECURITY: KHÔNG hardcode host/user/password/database.
# Toàn bộ giá trị đến từ biến môi trường; thiếu -> ConfigurationError.
DB_CONFIG = {
    'driver': os.getenv('DB_DRIVER', 'mysql+pymysql'),
    'user': None,       # lazy-load, xem _resolve_db_config()
    'password': None,
    'host': None,
    'port': None,
    'database': None,
}

_DB_RESOLVED = False


def _resolve_db_config() -> dict:
    """Lazy-resolve DB config. Cho phép import module mà không cần .env (ví dụ khi chạy test)."""
    global _DB_RESOLVED
    if not _DB_RESOLVED:
        DB_CONFIG.update({
            'driver': os.getenv('DB_DRIVER', 'mysql+pymysql'),
            'user': _require_env('DB_USER'),
            'password': _require_env('DB_PASSWORD'),
            'host': _require_env('DB_HOST', hint='Ví dụ: datamart.internal.example.com'),
            'port': os.getenv('DB_PORT', '3306'),
            'database': _require_env('DB_NAME'),
        })
        _DB_RESOLVED = True
    return DB_CONFIG


def get_connection_string() -> str:
    """
    Build SQLAlchemy connection string.

    SECURITY: password được URL-encode để tránh vỡ URI khi chứa ký tự đặc biệt.
    KHÔNG log/print giá trị trả về của hàm này - dùng get_safe_connection_string().
    """
    from urllib.parse import quote_plus

    c = _resolve_db_config()
    user = quote_plus(str(c['user']))
    password = quote_plus(str(c['password']))
    return f"{c['driver']}://{user}:{password}@{c['host']}:{c['port']}/{c['database']}"


def get_safe_connection_string() -> str:
    """Phiên bản đã che credential - DÙNG CHO LOG."""
    c = _resolve_db_config()
    return f"{c['driver']}://***:***@{c['host']}:{c['port']}/{c['database']}"

# ==========================================
# LM STUDIO (AI LOCAL) CONFIG
# ==========================================
LM_STUDIO_CONFIG = {
    'base_url': os.getenv('LM_STUDIO_URL', 'http://127.0.0.1:1234/v1'),
    'api_key': os.getenv('LM_STUDIO_API_KEY', ''),  # SECURITY: không hardcode key
    'model': os.getenv('LM_STUDIO_MODEL', 'openai/gpt-oss-20b'),
    'timeout': int(os.getenv('LM_STUDIO_TIMEOUT', '120')),
    'max_retries': int(os.getenv('LM_STUDIO_MAX_RETRIES', '3')),
}

# ==========================================
# FORECAST PARAMETERS
# ==========================================
_current_date_override = os.getenv('FORECAST_CURRENT_DATE')
if _current_date_override:
    CURRENT_DATE = datetime.date.fromisoformat(_current_date_override)
else:
    CURRENT_DATE = datetime.date.today()
FORECAST_START_DATE = datetime.date(2026, 1, 15)
FORECAST_HORIZON = int(os.getenv('FORECAST_HORIZON', '31'))  # Legacy fallback
FORECAST_MONTHS_AHEAD = int(os.getenv('FORECAST_MONTHS_AHEAD', '3'))  # Forecast 3 full months ahead (chỉ khi mode=full)
DAILY_FORECAST_DAYS = int(os.getenv('DAILY_FORECAST_DAYS', '30'))  # Số ngày forecast cho daily job (mode=daily)
INACTIVE_THRESHOLD = int(os.getenv('INACTIVE_THRESHOLD', '30'))

# ==========================================
# AI INFERENCE OPTIMIZATION
# ==========================================
# Skip LLM inference for restaurants with AI weight below this threshold.
# HIGH_VOLUME_ML restaurants (AI weight=0.15) will use 100% ML instead,
# saving ~175s per restaurant. Only restaurants with meaningful AI contribution
# (ENSEMBLE_WEIGHTED=0.50, AI_PRIMARY=0.70, AI_ONLY=1.00) will run LLM.
AI_INFERENCE_THRESHOLD = float(os.getenv('AI_INFERENCE_THRESHOLD', '0.50'))

# Max seconds for a single LLM inference call. If exceeded, skip AI for that restaurant.
AI_TIMEOUT_PER_RESTAURANT = int(os.getenv('AI_TIMEOUT_PER_RESTAURANT', '300'))

# ==========================================
# FORECAST MODE: SHORT-TERM vs LONG-TERM
# ==========================================
# SHORT_HORIZON_DAYS: Số ngày đầu tiên sẽ forecast chi tiết (theo ca)
#   - Trong SHORT_HORIZON_DAYS ngày đầu: forecast theo ngày, theo ca (Sáng 8h-15h30 + Tối 15h30-23h)
#   - Sau SHORT_HORIZON_DAYS ngày: forecast CHỈ theo nhà hàng + ngày (daily total)
# Mục đích: Tiết kiệm thời gian/tài nguyên cho forecast xa, vì độ chính xác
# theo ca giảm dần khi horizon xa hơn
SHORT_HORIZON_DAYS = int(os.getenv('SHORT_HORIZON_DAYS', '30'))

# ==========================================
# SHIFT DEFINITIONS (Phase 8: Shift-Based Forecast)
# ==========================================
# Forecast theo 2 ca làm việc: Sáng (8h-15h30) + Tối (15h30-23h)
# Mục đích: Giảm noise từ hourly variance, tăng accuracy target 85-90%
SHIFT_DEFINITIONS: dict[str, dict[str, object]] = {
    'MORNING': {
        'name': 'Ca Sáng',
        'name_en': 'Morning',
        'start_hour': 8,
        'end_hour': 15,      # Inclusive (8, 9, 10, 11, 12, 13, 14, 15)
        'hours': list(range(8, 16)),  # 8h → 15h (ca sáng kết thúc 15:30)
    },
    'EVENING': {
        'name': 'Ca Tối',
        'name_en': 'Evening',
        'start_hour': 16,
        'end_hour': 23,      # Inclusive (16, 17, 18, 19, 20, 21, 22, 23)
        'hours': list(range(16, 24)),  # 16h → 23h (ca tối từ 15:30, kết thúc 23h)
    },
}

# All operating hours (union of both shifts)
_MORNING_HOURS: list[int] = list(SHIFT_DEFINITIONS['MORNING']['hours'])  # type: ignore[call-overload]
_EVENING_HOURS: list[int] = list(SHIFT_DEFINITIONS['EVENING']['hours'])  # type: ignore[call-overload]
ALL_OPERATING_HOURS: list[int] = _MORNING_HOURS + _EVENING_HOURS

# Feature importance threshold for auto-pruning
FEATURE_IMPORTANCE_THRESHOLD = float(os.getenv('FEATURE_IMPORTANCE_THRESHOLD', '0.005'))

# ==========================================
# V4 UPGRADE CONFIG
# ==========================================
# Booking threshold: booking_ratio > this → booking_flag = 1
BOOKING_THRESHOLD_RATIO = float(os.getenv('BOOKING_THRESHOLD_RATIO', '0.3'))

# Meta-learner: True = use LightGBM meta-learner, False = use Ridge/weighted avg
META_LEARNER_ENABLED = os.getenv('META_LEARNER_ENABLED', 'true').lower() in ('true', '1', 'yes')

# Low-volume threshold: avg daily guests < this → use SMAPE instead of MAPE
LOW_VOLUME_THRESHOLD = int(os.getenv('LOW_VOLUME_THRESHOLD', '30'))

# ==========================================
# V5 TREND UPGRADE CONFIG
# ==========================================
# Trend detection thresholds
TREND_SPIKE_THRESHOLD = float(os.getenv('TREND_SPIKE_THRESHOLD', '1.2'))   # 3d > 7d × 1.2 → spike
TREND_DROP_THRESHOLD = float(os.getenv('TREND_DROP_THRESHOLD', '0.8'))     # 3d < 7d × 0.8 → drop

# Trend adjustment layer (post-prediction multiplier bounds)
TREND_ADJUST_SPIKE_MIN = float(os.getenv('TREND_ADJUST_SPIKE_MIN', '1.05'))  # uptrend → ×1.05~1.15
TREND_ADJUST_SPIKE_MAX = float(os.getenv('TREND_ADJUST_SPIKE_MAX', '1.15'))
TREND_ADJUST_DROP_MIN = float(os.getenv('TREND_ADJUST_DROP_MIN', '0.85'))    # downtrend → ×0.85~0.95
TREND_ADJUST_DROP_MAX = float(os.getenv('TREND_ADJUST_DROP_MAX', '0.95'))

# ML-heavy weight distribution (v5: boost ML, reduce Prophet)
# Inside _weighted_combine: ml_weight is split as ML_STACKING_SHARE / NP_SHARE / PROPHET_SHARE
ML_STACKING_SHARE = float(os.getenv('ML_STACKING_SHARE', '0.60'))   # 60% → ML stacking (was 40%)
NP_SHARE = float(os.getenv('NP_SHARE', '0.20'))                     # 20% → NeuralProphet (was 30%)
PROPHET_SHARE = float(os.getenv('PROPHET_SHARE', '0.20'))            # 20% → Prophet (was 30%)

# ==========================================
# V6 VOLUME SEGMENTATION CONFIG
# ==========================================
# Volume segmentation thresholds (avg daily guests)
LOW_VOLUME_DAILY_THRESHOLD = int(os.getenv('LOW_VOLUME_DAILY_THRESHOLD', '20'))    # <20 → baseline model
MEDIUM_VOLUME_DAILY_THRESHOLD = int(os.getenv('MEDIUM_VOLUME_DAILY_THRESHOLD', '80'))  # 20-80 → ensemble
# >80 → high volume ML-primary model
# Low volume rounding: if forecast < this, round to nearest integer
LOW_VOLUME_ROUND_THRESHOLD = int(os.getenv('LOW_VOLUME_ROUND_THRESHOLD', '5'))

# ==========================================
# V7 HOLIDAY FORECAST CONFIG
# ==========================================
# Holiday curve: day-by-day impact multiplier relative to nearest holiday
# Key = days_to_holiday (-3 = 3 days before, 0 = holiday, +2 = 2 days after)
HOLIDAY_CURVE_HIGH_VOLUME = {
    -3: 1.10,   # 3 days before: slight increase (early gatherings)
    -2: 1.20,   # 2 days before: moderate increase
    -1: 1.40,   # 1 day before: strong increase (tất niên, pre-celebration)
     0: 0.20,   # Holiday itself: most restaurants closed
    +1: 0.60,   # 1 day after: slow recovery
    +2: 0.80,   # 2 days after: moderate recovery
    +3: 0.90,   # 3 days after: near normal
}

HOLIDAY_CURVE_LOW_VOLUME = {
    -3: 1.05,   # Low volume restaurants less sensitive to pre-holiday
    -2: 1.10,
    -1: 1.20,
     0: 0.10,   # Even more likely to be closed
    +1: 0.50,
    +2: 0.70,
    +3: 0.85,
}

HOLIDAY_CURVE_MEDIUM_VOLUME = {
    -3: 1.08,
    -2: 1.15,
    -1: 1.30,
     0: 0.15,
    +1: 0.55,
    +2: 0.75,
    +3: 0.88,
}

# Closure detection: if historical closure rate > this, force forecast = 0
HOLIDAY_CLOSURE_RATE_THRESHOLD = float(os.getenv('HOLIDAY_CLOSURE_RATE_THRESHOLD', '0.80'))

# Booking override: if booking_ratio > this during holiday window, override factor
HOLIDAY_BOOKING_OVERRIDE_THRESHOLD = float(os.getenv('HOLIDAY_BOOKING_OVERRIDE_THRESHOLD', '1.5'))

# ==========================================
# V8 WEEKEND×EVENING OPTIMIZATION CONFIG
# ==========================================
# Adaptive alpha-blend for shift distribution
# alpha = 1.0 → trust ML fully; alpha = 0.0 → trust historical fully
SHIFT_ALPHA_DEFAULT = float(os.getenv('SHIFT_ALPHA_DEFAULT', '0.7'))       # ML stable → lean ML
SHIFT_ALPHA_VOLATILE = float(os.getenv('SHIFT_ALPHA_VOLATILE', '0.3'))     # ML noisy → lean historical
SHIFT_ALPHA_CV_STABLE = float(os.getenv('SHIFT_ALPHA_CV_STABLE', '0.15')) # CV < this → stable
SHIFT_ALPHA_CV_NOISY = float(os.getenv('SHIFT_ALPHA_CV_NOISY', '0.30'))   # CV > this → noisy

# Sample weighting: boost Weekend×EVENING in training
WEEKEND_EVENING_WEIGHT_BASE = float(os.getenv('WEEKEND_EVENING_WEIGHT_BASE', '3.0'))  # All weekend evening
WEEKEND_EVENING_WEIGHT_HIGH = float(os.getenv('WEEKEND_EVENING_WEIGHT_HIGH', '5.0'))  # High-volume weekend evening

# Shift Residual Corrector
SHIFT_RESIDUAL_MIN_SAMPLES = int(os.getenv('SHIFT_RESIDUAL_MIN_SAMPLES', '100'))
SHIFT_RESIDUAL_MIN_R2 = float(os.getenv('SHIFT_RESIDUAL_MIN_R2', '0.10'))  # Min R² to save model
SHIFT_RESIDUAL_MAX_CORRECTION_PCT = float(os.getenv('SHIFT_RESIDUAL_MAX_CORRECTION_PCT', '0.40'))  # ±40%
SHIFT_RESIDUAL_LOOKBACK_DAYS = int(os.getenv('SHIFT_RESIDUAL_LOOKBACK_DAYS', '90'))


def get_forecast_end_date(
    reference_date: datetime.date | None = None,
    months_ahead: int | None = None,
) -> datetime.date:
    """
    Tính ngày kết thúc forecast = cuối tháng thứ N kể từ tháng hiện tại.

    Ví dụ (months_ahead=3):
        - 23/02/2026 → 31/05/2026 (hết tháng 2 + 3 tháng: Mar, Apr, May)
        - 01/03/2026 → 30/06/2026 (hết tháng 3 + 3 tháng: Apr, May, Jun)
        - 15/12/2026 → 31/03/2027 (hết tháng 12 + 3 tháng: Jan, Feb, Mar 2027)

    Args:
        reference_date: Ngày tham chiếu (default: CURRENT_DATE)
        months_ahead: Số tháng dự phóng tiếp theo (default: FORECAST_MONTHS_AHEAD)

    Returns:
        datetime.date - ngày cuối cùng của tháng thứ N
    """
    if reference_date is None:
        reference_date = CURRENT_DATE
    if months_ahead is None:
        months_ahead = FORECAST_MONTHS_AHEAD

    # Tính tháng đích = tháng hiện tại + months_ahead
    target_month = reference_date.month + months_ahead
    target_year = reference_date.year

    # Xử lý khi vượt qua năm
    while target_month > 12:
        target_month -= 12
        target_year += 1

    # Lấy ngày cuối cùng của tháng đích
    last_day = calendar.monthrange(target_year, target_month)[1]
    return datetime.date(target_year, target_month, last_day)
ROLLING_WINDOW_WEEKS = int(os.getenv('ROLLING_WINDOW_WEEKS', '4'))
START_DATE_DATA = os.getenv('START_DATE_DATA', '2024-01-01')
DATA_LOOKBACK_DAYS = int(os.getenv('DATA_LOOKBACK_DAYS', '400'))

# ==========================================
# FILE PATHS
# ==========================================
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Coding/
MASTER_FILE_NAME = str(PROJECT_ROOT / "Master_Forecast_Tracking.xlsx")
SHIFT_FILE_NAME = str(PROJECT_ROOT / "Shift_Forecast_Summary.xlsx")
ACCURACY_REPORT_FILE = str(PROJECT_ROOT / "Accuracy_Report.xlsx")
ACCURACY_HISTORY_FILE = str(PROJECT_ROOT / "accuracy_history.json")
MODEL_PERFORMANCE_FILE = str(PROJECT_ROOT / "Model_Performance_Report.xlsx")
MODEL_CACHE_DIR = str(PROJECT_ROOT / "model_cache")
LOG_DIR = str(PROJECT_ROOT / "logs")
LUNAR_NY_CLOSURE_FILE = str(PROJECT_ROOT / "Close_lunar_NY_2026.xlsx")
OPEN_CLOSE_FILE = str(PROJECT_ROOT / "Open_Close.xlsx")

# ==========================================
# ANALYSIS THRESHOLDS
# ==========================================
ANALYSIS_CONFIG = {
    # Gap Detection
    'min_gap_days': 7,              # Khoảng cách tối thiểu (ngày) để tính là "gap"
    'max_gap_ratio': 0.5,           # Nếu tỷ lệ gap > 50% thì loại bỏ nhà hàng
    'min_active_days': 7,           # Số ngày active tối thiểu để forecast (giảm từ 30 cho NH mới)
    'new_restaurant_forecast_days': 7,   # [FIX] < 7 ngày → dùng chain-average hỗ trợ (giảm từ 14 để NH có 7-13d data chạy ensemble Phase 8)

    # Outlier Detection
    'outlier_iqr_threshold': 2.5,   # Hệ số IQR để phát hiện outlier

    # Growth Classification
    'strong_growth_threshold': 10,   # > 10% → STRONG_GROWTH
    'mild_growth_threshold': 3,      # > 3% → MILD_GROWTH
    'mild_decline_threshold': -3,    # > -3% → STABLE
    'strong_decline_threshold': -10, # > -10% → MILD_DECLINE, else STRONG_DECLINE

    # Restaurant Classification
    'new_restaurant_days': 7,         # [FIX] < 7 ngày → NEW (giảm từ 14 để NH có 7-13d data được classify YOUNG thay vì NEW)
    'young_restaurant_days': 45,     # < 45 ngày → YOUNG
    'volatile_cv_threshold': 0.5,    # CV > 0.5 → VOLATILE
    'high_volume_threshold': 200,    # Avg daily > 200 → HIGH_VOLUME
}

# ==========================================
# MONITORING CONFIG (Phase 4)
# ==========================================
MONITORING_CONFIG = {
    # Drift Detection
    'drift_threshold_pct': 15,       # MAPE increase > 15% → alert

    # Hit Rate (per restaurant per day)
    # Logic: Prediction is "hit" if |predicted - actual| ≤ 15 guests
    # Đơn giản, trực quan, không phụ thuộc vào volume
    'hit_rate_threshold_abs': 15,    # Error ≤ 15 guests → hit

    # Restaurant Alerts
    'retune_mape_threshold': 40,     # MAPE > 40% → needs retuning
    'min_samples_for_alert': 30,     # Min samples to trigger alert

    # Per-Restaurant Metrics
    'per_restaurant_metrics': True,  # Generate metrics per restaurant, not global

    # History
    'history_retention_days': 90,    # Keep 90 days of accuracy history
}

# ==========================================
# ENSEMBLE WEIGHTS (Strategy → ML/AI weights)
# ==========================================
STRATEGY_WEIGHTS = {
    'AI_ONLY':                  {'ml': 0.0, 'ai': 1.0},
    'AI_PRIMARY_ML_SECONDARY':  {'ml': 0.3, 'ai': 0.7},
    'ENSEMBLE_WEIGHTED':        {'ml': 0.5, 'ai': 0.5},
    'ML_PRIMARY_AI_VALIDATE':   {'ml': 0.7, 'ai': 0.3},
    'ENSEMBLE_EQUAL':           {'ml': 0.5, 'ai': 0.5},
    # ⭐ v6: Volume segmentation strategies
    'BASELINE_ONLY':            {'ml': 0.0, 'ai': 0.0},  # Low volume: median baseline
    'HIGH_VOLUME_ML':           {'ml': 0.85, 'ai': 0.15}, # High volume: ML dominant
}

# ==========================================
# DB ENGINE OPTIONS
# ==========================================
ENGINE_OPTIONS = {
    'pool_recycle': 3600,
    'pool_pre_ping': True,
    'connect_args': {'connect_timeout': 60},
}

# ==========================================
# PARALLEL PROCESSING CONFIG (Phase 5)
# ==========================================
PARALLEL_CONFIG = {
    'max_workers': 0,             # 0 = auto-detect (CPU cores + 2)
    'use_threads': True,          # True=ThreadPool, False=ProcessPool
    'timeout_per_restaurant': 300,  # 5 min timeout per restaurant
}

# ==========================================
# AUTO-TUNER CONFIG (Phase 5)
# ==========================================
AUTOTUNER_CONFIG = {
    'quick_trials': 20,           # QUICK mode
    'standard_trials': 50,        # STANDARD mode
    'thorough_trials': 100,       # THOROUGH mode
    'params_max_age_days': 7,     # Re-tune if params older than 7 days
    'cv_splits': 3,               # Walk-forward CV splits
}

# ==========================================
# RAG + LOCAL LLM CONFIG (Phase 6 - Self-Learning)
# ==========================================
RAG_LLM_CONFIG = {
    # Local LLM settings (Phase 6: Upgraded to 7B model)
    'model_path': os.getenv(
        'LOCAL_LLM_PATH',
        str(PROJECT_ROOT / 'models' / 'Qwen2.5-7B-Instruct-Q4_K_M.gguf')
    ),
    'model_fallback_path': str(PROJECT_ROOT / 'models' / 'qwen2.5-1.5b-instruct-q4_k_m.gguf'),
    'model_type': os.getenv('LOCAL_LLM_TYPE', 'gguf'),  # 'gguf' or 'transformers'
    'max_tokens': 2000,
    'temperature': 0.1,
    'n_ctx': 8192,            # Larger context for 7B model
    'n_gpu_layers': -1,       # -1 = all on GPU/Metal (Apple Silicon)

    # Knowledge Store
    'knowledge_db_path': str(PROJECT_ROOT / 'knowledge_db'),
    'embedding_model': 'all-MiniLM-L6-v2',

    # RAG retrieval
    'rag_top_k': 8,           # Items to retrieve per query
    'max_retries': 2,

    # Self-learning
    'learn_lookback_days': 14, # Learn from last 14 days of actuals
    'max_feedback_records': 2000,
}

# Use RAG agent instead of LM Studio?
USE_RAG_AGENT = os.getenv('USE_RAG_AGENT', 'true').lower() in ('true', '1', 'yes')

# Use fine-tuned LLM (LoRA adapter) if available?
USE_FINETUNED_LLM = os.getenv('USE_FINETUNED_LLM', 'true').lower() in ('true', '1', 'yes')

# ==========================================
# LLM FINE-TUNING CONFIG (Phase 7 - Domain Adaptation)
# ==========================================
FINETUNE_CONFIG = {
    # Base model (HuggingFace name, for training - NOT GGUF)
    'base_model': os.getenv('FINETUNE_MODEL_NAME', 'Qwen/Qwen2.5-1.5B-Instruct'),

    # LoRA hyperparameters
    'lora_r': int(os.getenv('FINETUNE_LORA_R', '16')),
    'lora_alpha': int(os.getenv('FINETUNE_LORA_ALPHA', '32')),

    # Training settings
    'num_epochs': int(os.getenv('FINETUNE_EPOCHS', '3')),
    'batch_size': int(os.getenv('FINETUNE_BATCH_SIZE', '4')),
    'learning_rate': float(os.getenv('FINETUNE_LR', '2e-4')),
    'max_seq_length': int(os.getenv('FINETUNE_MAX_SEQ', '2048')),

    # Data generation
    'lookback_days': int(os.getenv('FINETUNE_LOOKBACK', '60')),
    'min_training_pairs': int(os.getenv('FINETUNE_MIN_PAIRS', '100')),
    'max_training_pairs': int(os.getenv('FINETUNE_MAX_PAIRS', '5000')),

    # Retraining schedule
    'retrain_interval_days': int(os.getenv('FINETUNE_RETRAIN_DAYS', '7')),
    'auto_retrain': os.getenv('FINETUNE_AUTO_RETRAIN', 'false').lower() in ('true', '1', 'yes'),

    # Directories
    'finetune_dir': str(PROJECT_ROOT / 'finetune'),
    'adapter_dir': str(PROJECT_ROOT / 'finetune' / 'adapters'),
    'training_data_dir': str(PROJECT_ROOT / 'finetune' / 'training_data'),
}

# ==========================================
# DASHBOARD CONFIG (Phase 5) — ĐÃ THAY THẾ
# ==========================================
# Dict cũ bind '0.0.0.0' và không có auth. Đã thay bằng các biến
# DASHBOARD_* ở cuối file (đọc từ env, mặc định loopback, bắt buộc API key).
# Giữ lại dạng alias để code cũ không vỡ; sẽ xoá ở Giai đoạn 3.
DASHBOARD_CONFIG = {
    'host': os.getenv('DASHBOARD_HOST', '127.0.0.1'),
    'port': int(os.getenv('DASHBOARD_PORT', '5050')),
    'debug': False,
}


# ==========================================
# DASHBOARD / API SECURITY  (bổ sung khi hardening)
# ==========================================
# Bind mặc định về loopback. Muốn expose ra ngoài PHẢI đặt biến môi trường
# và đứng sau reverse proxy (nginx/traefik) có TLS.
DASHBOARD_HOST = os.getenv('DASHBOARD_HOST', '127.0.0.1')
DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', '5050'))

# API key cho toàn bộ endpoint /api/*. Bắt buộc trừ khi tắt tường minh.
DASHBOARD_API_KEY = os.getenv('DASHBOARD_API_KEY')
DASHBOARD_AUTH_ENABLED = _env_bool('DASHBOARD_AUTH_ENABLED', True)

# Không bao giờ bật debug ngoài máy dev: Werkzeug debugger = RCE.
DASHBOARD_DEBUG = _env_bool('DASHBOARD_DEBUG', False)

# Whitelist origin cho CORS (rỗng = không cho cross-origin)
DASHBOARD_CORS_ORIGINS = [
    o.strip() for o in os.getenv('DASHBOARD_CORS_ORIGINS', '').split(',') if o.strip()
]

# ==========================================
# RUNTIME ENVIRONMENT
# ==========================================
# dev | staging | production - dùng để siết chặt hành vi ở production
APP_ENV = os.getenv('APP_ENV', 'dev').strip().lower()
IS_PRODUCTION = APP_ENV == 'production'


def validate_startup_config() -> None:
    """
    Kiểm tra cấu hình bắt buộc TRƯỚC khi pipeline chạy.

    Gọi ở đầu main() để fail-fast thay vì chết giữa chừng sau 20 phút training.
    """
    _resolve_db_config()

    if DASHBOARD_AUTH_ENABLED and not DASHBOARD_API_KEY:
        raise ConfigurationError(
            "DASHBOARD_AUTH_ENABLED=true nhưng thiếu DASHBOARD_API_KEY. "
            "Sinh key: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )

    if IS_PRODUCTION:
        if DASHBOARD_DEBUG:
            raise ConfigurationError("DASHBOARD_DEBUG không được bật ở production.")
        if not DASHBOARD_AUTH_ENABLED:
            raise ConfigurationError("Không được tắt auth ở production.")
        if DASHBOARD_HOST == '0.0.0.0' and not DASHBOARD_CORS_ORIGINS:  # noqa: S104 - so sánh, không bind
            raise ConfigurationError(
                "Dashboard bind 0.0.0.0 ở production mà không khai báo CORS origin. "
                "Đặt sau reverse proxy và khai báo DASHBOARD_CORS_ORIGINS."
            )
