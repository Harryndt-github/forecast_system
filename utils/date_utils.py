"""
==============================================
DATE & CALENDAR UTILITIES (ENHANCED)
==============================================
Các hàm xử lý ngày tháng, lịch âm, ngày lễ Việt Nam.
Tập trung logic date để tránh duplicate code.

Holiday Types:
    TET_DUONG_LICH     - Tết Dương lịch (1/1)
    TET_NGUYEN_DAN     - Tết Nguyên Đán (Giao Thừa + Mùng 1-5)
    HUNG_KINGS         - Giỗ Tổ Hùng Vương (10/3 Âm Lịch)
    LIBERATION_DAY     - Ngày Giải Phóng (30/4)
    LABOR_DAY          - Quốc Tế Lao Động (1/5)
    NATIONAL_DAY       - Quốc Khánh (2/9)

Pre/Post Holiday:
    PRE_HOLIDAY  - 2 ngày trước kỳ nghỉ (khách thường đông hơn)
    POST_HOLIDAY - 2 ngày sau kỳ nghỉ (khách thường vắng hơn)

Impact Profiles (default, tunable per restaurant):
    TET_NGUYEN_DAN:  -80% to -100% (nhiều nhà hàng đóng cửa)
    NATIONAL_DAY:    +10% to +30%
    HUNG_KINGS:      +5% to +20%
    LIBERATION_DAY:  +50% to +90% (kỳ nghỉ dài, khách tăng mạnh - thực tế 2025: +~80%)
    LABOR_DAY:       +50% to +90% (kỳ nghỉ dài, khách tăng mạnh - thực tế 2025: +~70%)
    TET_DUONG_LICH:  +5% to +15%
    PRE_HOLIDAY:     +10% to +20% (ăn tất niên, du lịch)
    POST_HOLIDAY:    -10% to -20%
"""

import datetime
import json
import holidays
from pathlib import Path
from lunardate import LunarDate  # type: ignore[reportMissingModuleSource]
from typing import Dict, List, Optional, Tuple, Set
from forecast_system.utils.logger import get_logger

logger = get_logger('date_utils')


# ==========================================
# CALIBRATION CACHE (data-driven impacts)
# ==========================================
_calibration_cache = None
_calibration_loaded = False


def _load_calibration():
    """Load calibrated holiday impacts from JSON (cached)."""
    global _calibration_cache, _calibration_loaded
    if _calibration_loaded:
        return _calibration_cache
    _calibration_loaded = True

    candidates = [
        Path(__file__).parent.parent / 'holiday_calibration.json',
        Path(__file__).parent.parent.parent / 'holiday_calibration.json',
    ]
    cal_file = next((p for p in candidates if p.exists()), None)
    if cal_file is not None:
        try:
            with open(cal_file, 'r', encoding='utf-8') as f:
                _calibration_cache = json.load(f)
            logger.debug(f"Loaded calibration from {cal_file}")
        except Exception as e:
            logger.warning(f"Could not load calibration: {e}")
    return _calibration_cache


def invalidate_calibration_cache():
    """Force reload calibration on next call (after re-calibration)."""
    global _calibration_cache, _calibration_loaded
    _calibration_cache = None
    _calibration_loaded = False


def _get_calibrated_impact(holiday_type, offset, direction):
    """
    Lookup calibrated impact for a specific holiday type + offset.

    Args:
        holiday_type: e.g. 'TET_NGUYEN_DAN'
        offset: int (days from holiday start/end)
        direction: 'pre', 'post', or 'holiday'

    Returns:
        float or None (None = no calibration, use default)
    """
    cal = _load_calibration()
    if not cal:
        return None

    ht = cal.get('holiday_types', {}).get(holiday_type, {})
    agg = ht.get('aggregate', {})

    if direction == 'holiday':
        hol = agg.get('holiday')
        if isinstance(hol, (int, float)):
            return hol
        return hol.get('impact') if isinstance(hol, dict) else None

    offset_data = agg.get(direction, {})
    val = offset_data.get(str(offset))
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, dict):
        return val.get('impact')
    return None


# ==========================================
# HOLIDAY TYPE CONSTANTS
# ==========================================

HOLIDAY_TYPES = {
    'TET_DUONG_LICH': {
        'name': 'Tết Dương Lịch',
        'impact': 'slight_increase',
        'default_factor': 1.10,  # +10%
        'closed_likely': False,
    },
    'TET_NGUYEN_DAN': {
        'name': 'Tết Nguyên Đán',
        'impact': 'major_decrease',
        'default_factor': 0.10,  # -90% (nhiều NHà hàng đóng cửa)
        'closed_likely': True,
    },
    'HUNG_KINGS': {
        'name': 'Giỗ Tổ Hùng Vương',
        'impact': 'moderate_increase',
        'default_factor': 1.15,  # +15%
        'closed_likely': False,
    },
    'LIBERATION_DAY': {
        'name': 'Ngày Giải Phóng 30/4',
        'impact': 'major_increase',
        'default_factor': 1.70,  # +70% (kỳ nghỉ dài 30/4-1/5: thực tế 2025 ~107k khách vs ~60k avg)
        'closed_likely': False,
        'needs_historical_calibration': True,  # [FIX #3] Load prior-year data for calibration
    },
    'LABOR_DAY': {
        'name': 'Quốc Tế Lao Động 1/5',
        'impact': 'major_increase',
        'default_factor': 1.70,  # +70% (kỳ nghỉ dài 30/4-1/5: thực tế 2025 ~98k khách vs ~60k avg)
        'closed_likely': False,
        'needs_historical_calibration': True,  # [FIX #3] Load prior-year data for calibration
    },
    'NATIONAL_DAY': {
        'name': 'Quốc Khánh 2/9',
        'impact': 'major_increase',
        'default_factor': 1.50,  # +50% (kỳ nghỉ dài, khách tăng mạnh)
        'closed_likely': False,
        'needs_historical_calibration': True,  # Load prior-year data for calibration
    },
    'PRE_HOLIDAY': {
        'name': 'Trước kỳ nghỉ',
        'impact': 'slight_increase',
        'default_factor': 1.15,  # +15%
        'closed_likely': False,
    },
    'POST_HOLIDAY': {
        'name': 'Sau kỳ nghỉ',
        'impact': 'slight_decrease',
        'default_factor': 0.90,  # -10%
        'closed_likely': False,
    },
    'PRE_TET': {
        'name': 'Trước Tết Nguyên Đán',
        'impact': 'increase',
        'default_factor': 1.25,  # +25% (ăn tất niên)
        'closed_likely': False,
    },
    'POST_TET': {
        'name': 'Sau Tết Nguyên Đán',
        'impact': 'decrease',
        'default_factor': 0.70,  # -30% (khách chưa về)
        'closed_likely': False,
    },
}

# ==========================================
# SPECIAL EVENTS CALENDAR (Unofficial but commercially important)
# ==========================================
# These are NOT official holidays but have SIGNIFICANT impact on restaurant traffic.
# Impact factors are estimated and can be auto-calibrated from historical data.

SPECIAL_EVENTS = {
    # Format: (month, day): {name, impact_factor, event_type, description}
    (2, 14): {
        'name': 'Valentine\'s Day',
        'event_type': 'VALENTINE',
        'default_factor': 1.60,   # +60% (couples dining out)
        'peak_hours': [17, 18, 19, 20, 21],  # Dinner peak
        'category': 'romantic',
        'pre_days': 1,   # Day before also gets slight boost
        'pre_factor': 1.10,
    },
    (3, 8): {
        'name': 'Quốc tế Phụ nữ',
        'event_type': 'WOMENS_DAY',
        'default_factor': 1.45,   # +45% (family/group dining)
        'peak_hours': [11, 12, 13, 17, 18, 19, 20],
        'category': 'celebration',
        'pre_days': 1,
        'pre_factor': 1.10,
    },
    (10, 20): {
        'name': 'Ngày Phụ nữ Việt Nam',
        'event_type': 'VN_WOMENS_DAY',
        'default_factor': 1.50,   # +50% (similar to 8/3 but stronger locally)
        'peak_hours': [11, 12, 13, 17, 18, 19, 20],
        'category': 'celebration',
        'pre_days': 1,
        'pre_factor': 1.10,
    },
    (11, 20): {
        'name': 'Ngày Nhà giáo Việt Nam',
        'event_type': 'TEACHERS_DAY',
        'default_factor': 1.30,   # +30% (group dining)
        'peak_hours': [11, 12, 13, 17, 18, 19],
        'category': 'celebration',
        'pre_days': 1,
        'pre_factor': 1.05,
    },
    (12, 24): {
        'name': 'Christmas Eve',
        'event_type': 'CHRISTMAS',
        'default_factor': 1.50,   # +50%
        'peak_hours': [17, 18, 19, 20, 21],
        'category': 'festive',
        'pre_days': 0,
        'pre_factor': 1.0,
    },
    (12, 25): {
        'name': 'Christmas Day',
        'event_type': 'CHRISTMAS',
        'default_factor': 1.35,   # +35%
        'peak_hours': [11, 12, 13, 17, 18, 19, 20],
        'category': 'festive',
        'pre_days': 0,
        'pre_factor': 1.0,
    },
    (10, 31): {
        'name': 'Halloween',
        'event_type': 'HALLOWEEN',
        'default_factor': 1.20,   # +20% (moderate)
        'peak_hours': [17, 18, 19, 20, 21],
        'category': 'festive',
        'pre_days': 0,
        'pre_factor': 1.0,
    },
    (6, 1): {
        'name': 'Ngày Quốc tế Thiếu nhi',
        'event_type': 'CHILDRENS_DAY',
        'default_factor': 1.25,   # +25% (family dining)
        'peak_hours': [10, 11, 12, 13, 17, 18, 19],
        'category': 'family',
        'pre_days': 0,
        'pre_factor': 1.0,
    },
    (12, 31): {
        'name': 'New Year\'s Eve',
        'event_type': 'NEW_YEARS_EVE',
        'default_factor': 1.55,   # +55% (group celebrations)
        'peak_hours': [18, 19, 20, 21, 22],
        'category': 'festive',
        'pre_days': 0,
        'pre_factor': 1.0,
    },
    (3, 14): {
        'name': 'White Day',
        'event_type': 'WHITE_DAY',
        'default_factor': 1.15,   # +15% (lighter than Valentine)
        'peak_hours': [17, 18, 19, 20],
        'category': 'romantic',
        'pre_days': 0,
        'pre_factor': 1.0,
    },
    (6, 21): {
        'name': 'Father\'s Day (VN)',
        'event_type': 'FATHERS_DAY',
        'default_factor': 1.15,   # +15%
        'peak_hours': [11, 12, 13, 17, 18, 19],
        'category': 'family',
        'pre_days': 0,
        'pre_factor': 1.0,
    },
    (5, 11): {
        'name': 'Mother\'s Day (VN)',
        'event_type': 'MOTHERS_DAY',
        'default_factor': 1.20,   # +20%
        'peak_hours': [11, 12, 13, 17, 18, 19],
        'category': 'family',
        'pre_days': 0,
        'pre_factor': 1.0,
    },
}

# Mid-Autumn Festival (Trung thu) - date varies by lunar calendar
# Handled dynamically using lunar_month=8, lunar_day=15


def get_special_event_info(date_obj) -> Optional[Dict]:
    """
    Check if a date is a special event (unofficial holiday).
    Also checks mid-autumn festival via lunar calendar.
    
    Args:
        date_obj: datetime.date
    
    Returns:
        Dict with event info or None
    """
    key = (date_obj.month, date_obj.day)
    event = SPECIAL_EVENTS.get(key)
    
    if event:
        return {
            'is_special_event': True,
            'event_type': event['event_type'],
            'event_name': event['name'],
            'event_impact': event['default_factor'],
            'peak_hours': event.get('peak_hours', []),
            'category': event.get('category', 'other'),
        }
    
    # Check pre-event days
    for offset in range(1, 3):  # Check up to 2 days ahead
        future = date_obj + datetime.timedelta(days=offset)
        future_key = (future.month, future.day)
        future_event = SPECIAL_EVENTS.get(future_key)
        if future_event:
            _pre_days = future_event.get('pre_days', 0)
            if isinstance(_pre_days, int) and offset <= _pre_days:
                return {
                    'is_special_event': True,
                    'event_type': f"PRE_{future_event['event_type']}",
                    'event_name': f"Before {future_event['name']}",
                    'event_impact': future_event.get('pre_factor', 1.05),
                    'peak_hours': future_event.get('peak_hours', []),
                    'category': future_event.get('category', 'other'),
                }
    
    # Check Mid-Autumn Festival (Trung thu) via lunar calendar
    try:
        lunar = get_lunar_info(date_obj)
        if lunar['lunar_month'] == 8 and lunar['lunar_day'] == 15:
            return {
                'is_special_event': True,
                'event_type': 'MID_AUTUMN',
                'event_name': 'Tết Trung Thu',
                'event_impact': 1.30,  # +30%
                'peak_hours': [17, 18, 19, 20, 21],
                'category': 'festive',
            }
    except Exception:
        pass
    
    return None


# Holiday impact factors for AI prompt
HOLIDAY_IMPACT_GUIDE = """
HOLIDAY IMPACT GUIDE (for Vietnam restaurants):
  TET_NGUYEN_DAN:  -80% to -100% (most restaurants CLOSED during Lunar New Year)
  PRE_TET:         +20% to +30% (year-end parties, tất niên dinners)
  POST_TET:        -20% to -40% (many people still on vacation)
  NATIONAL_DAY:    +10% to +30% (people go out, celebration meals)
  HUNG_KINGS:      +5% to +20% (short holiday, moderate increase)
  LIBERATION_DAY:  +10% to +25% (30/4, people travel and eat out)
  LABOR_DAY:       +10% to +25% (1/5, combined with 30/4 for long weekend)
  TET_DUONG_LICH:  +5% to +15% (New Year celebration, moderate)
  PRE_HOLIDAY:     +5% to +15% (anticipation, early gatherings)
  POST_HOLIDAY:    -5% to -15% (recovery period)

SPECIAL EVENTS (unofficial but high impact):
  VALENTINE (14/2):    +50% to +80% (couples dining, dinner peak)
  WOMENS_DAY (8/3):    +40% to +60% (group celebrations)
  VN_WOMENS_DAY (20/10): +40% to +60% (local celebration, very strong)
  TEACHERS_DAY (20/11): +25% to +40% (group dining)
  CHRISTMAS (24-25/12): +30% to +50% (festive season)
  NEW_YEARS_EVE (31/12): +40% to +60% (celebrations)
  CHILDRENS_DAY (1/6): +20% to +30% (family dining)
  MID_AUTUMN:          +25% to +40% (Trung Thu - family)
"""


# ==========================================
# HOLIDAY CLASSIFICATION
# ==========================================

def get_vn_holidays(years=None):
    """
    Lấy danh sách ngày lễ Việt Nam cho các năm chỉ định.
    
    Args:
        years: List các năm hoặc None (sẽ dùng năm hiện tại + 1)
    
    Returns:
        holidays.VN object
    """
    if years is None:
        current_year = datetime.date.today().year
        years = [current_year, current_year + 1]
    
    try:
        return holidays.Vietnam(years=years)  # type: ignore[attr-defined]
    except AttributeError:
        return holidays.VN(years=years)  # type: ignore[attr-defined]


def classify_holiday(date_obj, vn_holidays) -> Optional[str]:
    """
    Phân loại ngày lễ thành holiday type cụ thể.
    
    Args:
        date_obj: datetime.date
        vn_holidays: holidays.VN object
        
    Returns:
        str: Holiday type hoặc None nếu không phải ngày lễ
    """
    if date_obj not in vn_holidays:
        return None
    
    name = vn_holidays.get(date_obj, '').lower()
    
    # Tết Nguyên Đán
    if any(kw in name for kw in ['nguyên đán', 'giao thừa', 'mùng']):
        return 'TET_NGUYEN_DAN'
    
    # Hùng Vương
    if 'hùng vương' in name:
        return 'HUNG_KINGS'
    
    # 30/4
    if 'chiến thắng' in name or 'giải phóng' in name:
        return 'LIBERATION_DAY'
    
    # 1/5
    if 'lao động' in name:
        return 'LABOR_DAY'
    
    # 2/9
    if 'quốc khánh' in name:
        return 'NATIONAL_DAY'
    
    # 1/1
    if 'dương lịch' in name or date_obj.month == 1 and date_obj.day == 1:
        return 'TET_DUONG_LICH'
    
    return 'TET_DUONG_LICH'  # Default fallback


def get_holiday_periods(year: int, vn_holidays) -> List[Dict]:
    """
    Xác định các kỳ nghỉ lễ (liên tục) trong năm.
    Group các ngày lễ liền kề thành 1 kỳ nghỉ.
    
    Returns:
        List[Dict]: [{
            'type': 'TET_NGUYEN_DAN',
            'start': date, 'end': date,
            'dates': [date, ...],
            'pre_dates': [date, ...],  # 2 ngày trước
            'post_dates': [date, ...], # 2 ngày sau
        }]
    """
    # Get all holidays in this year
    year_holidays = []
    for d in sorted(vn_holidays.keys()):
        if hasattr(d, 'year') and d.year == year:
            h_type = classify_holiday(d, vn_holidays)
            year_holidays.append((d, h_type))
    
    if not year_holidays:
        return []
    
    # Group consecutive holidays of same type
    periods = []
    current_group = [year_holidays[0]]
    
    for i in range(1, len(year_holidays)):
        prev_date = current_group[-1][0]
        curr_date, curr_type = year_holidays[i]
        prev_type = current_group[-1][1]
        
        # Same type and within 2 days = same period
        if curr_type == prev_type and (curr_date - prev_date).days <= 2:
            current_group.append(year_holidays[i])
        # LIBERATION_DAY + LABOR_DAY are always combined (30/4 + 1/5)
        elif {prev_type, curr_type} == {'LIBERATION_DAY', 'LABOR_DAY'} and (curr_date - prev_date).days <= 2:
            current_group.append(year_holidays[i])
        else:
            periods.append(current_group)
            current_group = [year_holidays[i]]
    
    periods.append(current_group)
    
    # Build period info with pre/post buffers
    result = []
    for group in periods:
        dates = [d for d, _ in group]
        types = [t for _, t in group]
        
        # Primary type (most frequent or most significant)
        primary_type = types[0]
        if 'TET_NGUYEN_DAN' in types:
            primary_type = 'TET_NGUYEN_DAN'
        elif 'LIBERATION_DAY' in types or 'LABOR_DAY' in types:
            primary_type = 'LIBERATION_DAY'
        
        start = min(dates)
        end = max(dates)
        
        # Pre/Post buffer
        pre_days = 3 if primary_type == 'TET_NGUYEN_DAN' else 2
        post_days = 3 if primary_type == 'TET_NGUYEN_DAN' else 2
        
        pre_dates = [
            start - datetime.timedelta(days=i)
            for i in range(1, pre_days + 1)
        ]
        post_dates = [
            end + datetime.timedelta(days=i)
            for i in range(1, post_days + 1)
        ]
        
        result.append({
            'type': primary_type,
            'name': HOLIDAY_TYPES.get(primary_type, {}).get('name', primary_type),
            'start': start,
            'end': end,
            'duration': (end - start).days + 1,
            'dates': dates,
            'pre_dates': pre_dates,
            'post_dates': post_dates,
            'closed_likely': HOLIDAY_TYPES.get(primary_type, {}).get('closed_likely', False),
        })
    
    return result


def get_holiday_info(date_obj, vn_holidays) -> Dict:
    """
    Lấy thông tin holiday đầy đủ cho 1 ngày.
    
    Returns:
        Dict: {
            'is_holiday': bool,
            'holiday_type': str or None,
            'holiday_name': str or None,
            'is_pre_holiday': bool,
            'is_post_holiday': bool,
            'pre_post_type': str or None,  # e.g. 'PRE_TET', 'POST_HOLIDAY'
            'holiday_impact': float,  # default factor
            'closed_likely': bool,
        }
    """
    result = {
        'is_holiday': False,
        'holiday_type': None,
        'holiday_name': None,
        'is_pre_holiday': False,
        'is_post_holiday': False,
        'pre_post_type': None,
        'holiday_impact': 1.0,
        'closed_likely': False,
        # ⭐ v7: Distance-based holiday features
        'days_to_holiday': 0,        # 0=holiday, -1=1 day before, +1=1 day after
        'is_holiday_window': False,  # True if within ±3 days of any holiday
        # Special events (unofficial holidays)
        'is_special_event': False,
        'event_type': None,
        'event_name': None,
        'event_impact': 1.0,
    }
    
    # Check if it's a holiday
    h_type = classify_holiday(date_obj, vn_holidays)
    if h_type:
        result['is_holiday'] = True
        result['holiday_type'] = h_type
        result['holiday_name'] = HOLIDAY_TYPES.get(h_type, {}).get('name', h_type)
        result['closed_likely'] = bool(HOLIDAY_TYPES.get(h_type, {}).get('closed_likely', False))
        # ⭐ v7: distance = 0 (on the holiday itself)
        result['days_to_holiday'] = 0
        result['is_holiday_window'] = True
        # Use calibrated impact if available, otherwise default
        calibrated = _get_calibrated_impact(h_type, 0, 'holiday')
        if calibrated is not None:
            result['holiday_impact'] = calibrated
            result['holiday_impact_source'] = 'calibrated'
        else:
            result['holiday_impact'] = float(HOLIDAY_TYPES.get(h_type, {}).get('default_factor', 1.0))
            result['holiday_impact_source'] = 'default'
        
        # Also check if official holiday date coincides with a special event
        se_info = get_special_event_info(date_obj)
        if se_info:
            result['is_special_event'] = True
            result['event_type'] = se_info['event_type']
            result['event_name'] = se_info['event_name']
            result['event_impact'] = se_info['event_impact']
        
        return result
    
    # ==========================================
    # CHECK SPECIAL EVENTS (unofficial holidays)
    # ==========================================
    se_info = get_special_event_info(date_obj)
    if se_info:
        result['is_special_event'] = True
        result['event_type'] = se_info['event_type']
        result['event_name'] = se_info['event_name']
        result['event_impact'] = se_info['event_impact']
        # Special events use event_impact as the main holiday_impact
        # so all downstream code (ensemble, brain) picks it up
        result['holiday_impact'] = se_info['event_impact']
        result['holiday_impact_source'] = 'special_event'
    
    # ==========================================
    # EXTENDED PRE/POST HOLIDAY DETECTION
    # ==========================================
    # Pre-Tết: up to 14 days before with gradual impact
    # Pre-other: up to 5 days before
    # Post-Tết: up to 7 days after with gradual recovery
    # Post-other: up to 5 days after
    
    # --- PRE-HOLIDAY CHECK (check future days) ---
    # Check Tết first (up to 14 days ahead)
    for offset in range(1, 15):
        future = date_obj + datetime.timedelta(days=offset)
        future_type = classify_holiday(future, vn_holidays)
        
        if future_type == 'TET_NGUYEN_DAN':
            result['is_pre_holiday'] = True
            result['pre_post_type'] = 'PRE_TET'
            # ⭐ v7: distance-based feature (negative = before holiday)
            result['days_to_holiday'] = -offset
            result['is_holiday_window'] = offset <= 3
            
            # Try calibrated impact first (data-driven)
            calibrated = _get_calibrated_impact('TET_NGUYEN_DAN', offset, 'pre')
            if calibrated is not None:
                result['holiday_impact'] = calibrated
                result['holiday_impact_source'] = 'calibrated'
            else:
                # Fallback: hardcoded gradual impact curve
                if offset <= 3:
                    result['holiday_impact'] = 1.25 - (offset - 1) * 0.05
                elif offset <= 7:
                    result['holiday_impact'] = 0.85 + (offset - 4) * 0.033
                else:
                    result['holiday_impact'] = 0.90 + (offset - 8) * 0.01
                result['holiday_impact_source'] = 'default'
            
            # Don't override special event if already detected
            return result
        
        if future_type and offset <= 5:
            # Non-Tết holidays: 5-day pre-window
            result['is_pre_holiday'] = True
            result['pre_post_type'] = 'PRE_HOLIDAY'
            # ⭐ v7: distance-based feature
            result['days_to_holiday'] = -offset
            result['is_holiday_window'] = offset <= 3
            if not result['is_special_event']:  # Don't override special event impact
                if offset <= 2:
                    result['holiday_impact'] = HOLIDAY_TYPES['PRE_HOLIDAY']['default_factor']
                else:
                    result['holiday_impact'] = 1.0 + (5 - offset) * 0.025
            return result
    
    # --- POST-HOLIDAY CHECK (check past days) ---
    for offset in range(1, 8):
        past = date_obj - datetime.timedelta(days=offset)
        past_type = classify_holiday(past, vn_holidays)
        
        if past_type == 'TET_NGUYEN_DAN':
            result['is_post_holiday'] = True
            result['pre_post_type'] = 'POST_TET'
            # ⭐ v7: distance-based feature (positive = after holiday)
            result['days_to_holiday'] = offset
            result['is_holiday_window'] = offset <= 3
            
            # Try calibrated impact first (data-driven)
            calibrated = _get_calibrated_impact('TET_NGUYEN_DAN', offset, 'post')
            if calibrated is not None:
                result['holiday_impact'] = calibrated
                result['holiday_impact_source'] = 'calibrated'
            else:
                if offset <= 2:
                    result['holiday_impact'] = 0.50 + (offset - 1) * 0.05
                elif offset <= 5:
                    result['holiday_impact'] = 0.65 + (offset - 3) * 0.075
                else:
                    result['holiday_impact'] = 0.85 + (offset - 6) * 0.05
                result['holiday_impact_source'] = 'default'
            
            return result
        
        if past_type and offset <= 5:
            result['is_post_holiday'] = True
            result['pre_post_type'] = 'POST_HOLIDAY'
            # ⭐ v7: distance-based feature
            result['days_to_holiday'] = offset
            result['is_holiday_window'] = offset <= 3
            if not result['is_special_event']:
                if offset <= 2:
                    result['holiday_impact'] = HOLIDAY_TYPES['POST_HOLIDAY']['default_factor']
                else:
                    result['holiday_impact'] = 0.90 + (offset - 1) * 0.025
            return result
    
    return result


# ==========================================
# CLOSED RESTAURANT DETECTION
# ==========================================

def detect_holiday_closures(
    df_train,
    vn_holidays,
    years: Optional[List[int]] = None,
    threshold_ratio: float = 0.1,
) -> Dict[str, Set[str]]:
    """
    Phân tích data lịch sử để phát hiện nhà hàng đóng cửa vào dịp lễ.
    
    Logic:
    - Lấy average daily guests bình thường (non-holiday)
    - Nếu holiday guests < threshold_ratio * average → coi như đóng cửa
    - Nếu == 0 guests → chắc chắn đóng cửa
    
    Args:
        df_train: Transaction data (restaurant_code, date, guest_count)
        vn_holidays: holidays.VN object
        years: Years to analyze
        threshold_ratio: Ratio below which = closed (default 10%)
    
    Returns:
        Dict: {
            'TET_NGUYEN_DAN': {'101', '102', ...},  # restaurants closed during Tet
            'NATIONAL_DAY': {'201', '202', ...},
            ...
        }
    """
    import pandas as pd
    import numpy as np
    
    if df_train.empty:
        return {}
    
    if years is None:
        years = list(set(
            d.year for d in pd.to_datetime(df_train['date'], errors='coerce').dt.date
            if hasattr(d, 'year')
        ))
    
    df = df_train.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    
    # Calculate normal average per restaurant (non-holiday weekdays)
    df['_is_holiday'] = df['date'].apply(lambda d: d in vn_holidays if d else False)
    
    normal_avg = df[~df['_is_holiday']].groupby('restaurant_code')['guest_count'].agg(
        normal_daily_avg=lambda x: x.groupby(df.loc[x.index, 'date']).sum().mean()
    )
    
    # Simplified: average guests per day per restaurant (non-holiday)
    normal_daily = df[~df['_is_holiday']].groupby(
        ['restaurant_code', 'date']
    )['guest_count'].sum().reset_index()
    
    normal_avg = normal_daily.groupby('restaurant_code')['guest_count'].mean()
    normal_avg = normal_avg.to_dict()
    
    closures = {}
    
    for year in years:
        periods = get_holiday_periods(year, vn_holidays)
        
        for period in periods:
            h_type = period['type']
            if h_type not in closures:
                closures[h_type] = set()
            
            # Get data during this holiday period
            period_dates = set(period['dates'])
            df_period = df[df['date'].isin(period_dates)]
            
            if df_period.empty:
                continue
            
            # Check each restaurant
            period_daily = df_period.groupby(
                ['restaurant_code', 'date']
            )['guest_count'].sum().reset_index()
            
            period_avg = period_daily.groupby('restaurant_code')['guest_count'].mean()
            
            for res_code, hol_avg in period_avg.items():
                norm = normal_avg.get(res_code, 0)
                if norm <= 0:
                    continue
                
                ratio = hol_avg / norm
                
                # Closed: very few guests compared to normal
                if ratio <= threshold_ratio or hol_avg <= 1:
                    closures[h_type].add(str(res_code))
            
            # Restaurants with NO data during holiday = likely closed
            all_restaurants = set(df['restaurant_code'].unique())
            restaurants_with_data = set(df_period['restaurant_code'].unique())
            no_data_restaurants = all_restaurants - restaurants_with_data
            
            # Only flag as closed if restaurant was active before/after
            for res_code in no_data_restaurants:
                before = period['start'] - datetime.timedelta(days=7)
                after = period['end'] + datetime.timedelta(days=7)
                
                has_before = not df[
                    (df['restaurant_code'] == res_code) & 
                    (df['date'] >= before) & (df['date'] < period['start'])
                ].empty
                
                has_after = not df[
                    (df['restaurant_code'] == res_code) & 
                    (df['date'] > period['end']) & (df['date'] <= after)
                ].empty
                
                if has_before or has_after:
                    closures[h_type].add(str(res_code))
    
    # Log summary
    for h_type, closed_set in closures.items():
        if closed_set:
            logger.info(
                f"🏪 {h_type}: {len(closed_set)} restaurants likely CLOSED "
                f"(based on historical data)"
            )
    
    return closures


# ==========================================
# LUNAR CALENDAR
# ==========================================

def get_lunar_info(date_obj):
    """
    Lấy thông tin âm lịch cho một ngày.
    
    Args:
        date_obj: datetime.date object
    
    Returns:
        dict with keys: lunar_day, lunar_month, is_veg
    """
    try:
        if hasattr(date_obj, 'year'):
            l = LunarDate.fromSolarDate(date_obj.year, date_obj.month, date_obj.day)
        else:
            l = LunarDate.fromSolarDate(date_obj.year, date_obj.month, date_obj.day)
        
        return {
            'lunar_day': l.day,
            'lunar_month': l.month,
            'is_veg': l.day in [1, 15]
        }
    except Exception:
        return {
            'lunar_day': 15,
            'lunar_month': 6,
            'is_veg': False
        }


def is_veg_day(date_obj):
    """Check xem ngày có phải ngày ăn chay (mùng 1, 15 Âm Lịch) không"""
    info = get_lunar_info(date_obj)
    return info['is_veg']


# ==========================================
# FORECAST DAYS BUILDER (ENHANCED)
# ==========================================

def build_forecast_days(start_date, num_days, vn_holidays):
    """
    Tạo danh sách các ngày cần forecast với đầy đủ metadata.
    Enhanced: thêm holiday_type, pre/post holiday, impact factor.
    
    Args:
        start_date: Ngày bắt đầu (datetime.date)
        num_days: Số ngày forecast
        vn_holidays: holidays.VN object
    
    Returns:
        List[dict] với keys: date, date_str, weekday, is_holiday, 
            holiday_type, holiday_name, is_pre_holiday, is_post_holiday,
            pre_post_type, holiday_impact, closed_likely,
            is_veg, lunar_day, lunar_month
    """
    days = []
    for i in range(num_days):
        d = start_date + datetime.timedelta(days=i)
        lunar = get_lunar_info(d)
        h_info = get_holiday_info(d, vn_holidays)
        
        days.append({
            'date': d,
            'date_str': str(d),
            'weekday': d.strftime('%A'),
            # Holiday info
            'is_holiday': h_info['is_holiday'],
            'holiday_type': h_info['holiday_type'],
            'holiday_name': h_info['holiday_name'],
            'is_pre_holiday': h_info['is_pre_holiday'],
            'is_post_holiday': h_info['is_post_holiday'],
            'pre_post_type': h_info['pre_post_type'],
            'holiday_impact': h_info['holiday_impact'],
            'closed_likely': h_info['closed_likely'],
            # ⭐ v7: Distance-based holiday features
            'days_to_holiday': h_info.get('days_to_holiday', 0),
            'is_holiday_window': h_info.get('is_holiday_window', False),
            # Special events (unofficial holidays)
            'is_special_event': h_info.get('is_special_event', False),
            'event_type': h_info.get('event_type'),
            'event_name': h_info.get('event_name'),
            'event_impact': h_info.get('event_impact', 1.0),
            # Lunar
            'is_veg': lunar['is_veg'],
            'lunar_day': lunar['lunar_day'],
            'lunar_month': lunar['lunar_month'],
        })
    
    return days


# ==========================================
# OTHER UTILITIES
# ==========================================

def get_weekday_name(date_obj):
    """Get weekday name from date"""
    return date_obj.strftime('%A')


def date_range_str(start_date, end_date):
    """Format date range for display"""
    return f"{start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}"
