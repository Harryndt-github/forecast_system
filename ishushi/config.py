"""
==============================================
ISHUSHI FORECAST - CONFIGURATION
==============================================
Danh mục SAP codes và cấu hình cho model Ishushi.
"""

import datetime
from pathlib import Path

# ==========================================
# ISHUSHI SAP CODE CATALOG
# ==========================================
# Danh mục các món ăn cần dự đoán
ISHUSHI_SAP_CATALOG = {
    # California tôm tempura variants
    60011044: "California tôm tempura",
    60017993: "California tôm tempura 1 miếng",
    60026523: "California tôm tempura 1/2",
    60017438: "California tôm tempura 2 miếng",
    60018402: "California tôm tempura phu bơ",
    60018407: "California tôm tempura phu bơ",
    60024300: "California tôm tempura phu bơ (4 miếng)",
    60026525: "California tôm tempura phu bơ 1/2",
    60025657: "California tôm tempura phu bơ_nửa giá",
    60029194: "California tôm tempura thân thiết",
    60025661: "California tôm tempura_nửa giá",
    
    # Nabeyaki udon
    60019550: "Nabeyaki udon",
    60024278: "Nabeyaki udon",
    60028763: "Nabeyaki udon",
    
    # Temaki tôm tempura
    60007632: "Temaki tôm tempura",
    60025689: "Temaki tôm Tempura_nửa giá",
    60000720: "Tempura temaki",
    
    # Tempura tổng hợp
    60000693: "Tempura tổng hợp",
    60000776: "Tempura tổng hợp",
    60009320: "Tempura tổng hợp FF",
}

# Danh sách tất cả SAP codes
ISHUSHI_SAP_CODES = list(ISHUSHI_SAP_CATALOG.keys())

# Nhóm SAP codes theo danh mục
ISHUSHI_SAP_GROUPS = {
    "California tôm tempura": [
        60011044, 60017993, 60026523, 60017438,
        60018402, 60018407, 60024300, 60026525,
        60025657, 60029194, 60025661,
    ],
    "Nabeyaki udon": [60019550, 60024278, 60028763],
    "Temaki tôm tempura": [60007632, 60025689, 60000720],
    "Tempura tổng hợp": [60000693, 60000776, 60009320],
}

# ==========================================
# SHIFT DEFINITIONS (CA LÀM VIỆC)
# ==========================================
# MORNING (Ca Sáng): 8h – 15h30 → hours 8-15
# EVENING (Ca Tối): 15h30 – 23h → hours 16-23
ISHUSHI_SHIFT_HOURS = {
    'MORNING': list(range(8, 16)),    # 8, 9, 10, 11, 12, 13, 14, 15
    'EVENING': list(range(16, 24)),   # 16, 17, 18, 19, 20, 21, 22, 23
}

# Reverse mapping: hour → shift
ISHUSHI_HOUR_TO_SHIFT = {}
for _shift, _hours in ISHUSHI_SHIFT_HOURS.items():
    for _h in _hours:
        ISHUSHI_HOUR_TO_SHIFT[_h] = _shift

# ==========================================
# FORECAST PARAMETERS
# ==========================================
ISHUSHI_CONFIG = {
    # Data loading
    'data_start_date': datetime.date(2024, 1, 1),    # Lấy dữ liệu từ 2024
    'lookback_days': 730,                              # ~2 năm dữ liệu quá khứ
    
    # Forecast horizon
    'forecast_days': 30,                               # Dự đoán 30 ngày tới
    
    # Model parameters
    'test_days': 30,                                   # 30 ngày cuối để test
    'min_data_days': 30,                               # Tối thiểu 30 ngày data
    
    # Feature engineering
    'rolling_windows': [7, 14, 28],                    # Rolling average windows
    'lag_days': [1, 7, 14, 28],                        # Lag features
    
    # Shift distribution (legacy - kept for fallback)
    'shift_historical_weight': 0.6,                    # 60% historical ratio
    'shift_ml_weight': 0.4,                            # 40% ML predicted ratio
    
    # ⭐ v4: Booking features
    'booking_threshold_ratio': 0.3,                    # booking_flag = 1 if booking/avg_daily > 0.3
    
    # ⭐ v4: Meta-learner ensemble
    'meta_learner_enabled': True,                      # Use LightGBM meta-learner instead of weighted avg
    
    # ⭐ v4: Low-volume handling
    'low_volume_threshold': 30,                        # Restaurants with avg < 30 guests/day = low-volume
    
    # Output
    'output_dir': str(Path(__file__).parent.parent.parent / 'ishushi_output'),
}
