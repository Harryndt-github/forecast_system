"""
==============================================
EVENT IMPACT CALIBRATOR (AUTO-CALIBRATION)
==============================================
Tự động tính impact factors cho SPECIAL EVENTS từ data thực tế.
Thay vì dùng estimated impact (VD: Valentine = +60%), agent này
tính impact thực từ historical data (VD: Valentine thực tế = +137%).

Hỗ trợ 2 chế độ:
    1. AGGREGATE: Tính impact trung bình trên toàn hệ thống (fallback)
    2. PER-RESTAURANT: Tính impact riêng cho từng nhà hàng (ưu tiên)

Flow:
    1. Cho mỗi special event trong SPECIAL_EVENTS dictionary
    2. Tìm data của event đó trong năm trước
    3. So sánh guest count với "ngày bình thường" cùng weekday
    4. Tính actual_impact_factor = event_day_avg / normal_day_avg
    5. Override SPECIAL_EVENTS defaults nếu data available

Được dùng bởi main.py trước khi forecast loop.
"""

import datetime
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Any

from forecast_system.config.settings import PROJECT_ROOT
from forecast_system.utils.logger import get_logger
from forecast_system.utils.date_utils import SPECIAL_EVENTS, get_special_event_info

logger = get_logger('event_calibrator')


class EventCalibrator:
    """
    Auto-calibrate special event impact factors from historical data.
    
    Supports:
    - Aggregate calibration (system-wide, used as fallback)
    - Per-restaurant calibration (preferred, uses each restaurant's own data)
    """
    
    # Cache calibrated results (aggregate)
    _calibration_cache = None
    _cache_timestamp = None
    CACHE_TTL_HOURS = 12
    
    # Per-restaurant cache: { res_code: { event_type: calibrated_factor } }
    _per_restaurant_cache = None
    
    @staticmethod
    def calibrate(
        df_data: pd.DataFrame,
        current_date: datetime.date = None,  # type: ignore[reportArgumentType]
        min_restaurants_for_calibration: int = 10,
    ) -> Dict[str, Any]:
        """
        Main calibration method — computes BOTH aggregate and per-restaurant factors.
        
        Args:
            df_data: Transaction data with columns [restaurant_code, date, hour, guest_count]
            current_date: Reference date (default: today)
            min_restaurants_for_calibration: Minimum # of restaurants needed for aggregate
            
        Returns:
            Dict with calibrated factors:
            {
                'events': {
                    'VALENTINE': {
                        'default_factor': 1.60,
                        'calibrated_factor': 2.37,       # aggregate
                        'event_date': '2025-02-14',
                        'normal_avg': 150.3,
                        'event_avg': 356.2,
                        'n_restaurants': 423,
                        'confidence': 'HIGH',
                    },
                    ...
                },
                'per_restaurant': {
                    'RES001': {
                        'VALENTINE': {'calibrated_factor': 1.85, 'confidence': 'MEDIUM'},
                        'WOMENS_DAY': {'calibrated_factor': 1.32, 'confidence': 'HIGH'},
                    },
                    ...
                },
                'status': 'calibrated',
                'n_events_calibrated': 5,
            }
        """
        if current_date is None:
            current_date = datetime.date.today()
        
        # Check cache
        if (EventCalibrator._calibration_cache is not None and
            EventCalibrator._cache_timestamp is not None):
            age_hours = (
                datetime.datetime.now() - EventCalibrator._cache_timestamp
            ).total_seconds() / 3600
            if age_hours < EventCalibrator.CACHE_TTL_HOURS:
                return EventCalibrator._calibration_cache
        
        logger.info("📐 Starting Special Event Impact Calibration...")
        
        result = {
            'events': {},
            'per_restaurant': {},
            'status': 'no_data',
            'n_events_calibrated': 0,
        }
        
        if df_data.empty:
            return result
        
        # Prepare data
        df = df_data.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df['hour'] = df['hour'].astype(int)
        
        # Get date range in data
        min_date = df['date'].min()
        max_date = df['date'].max()
        data_span_days = (max_date - min_date).days
        
        logger.info(f"   Data range: {min_date} → {max_date} ({data_span_days} days)")
        
        if data_span_days < 60:
            logger.warning("   Not enough data span for calibration (need > 60 days)")
            return result
        
        # Daily guest totals per restaurant
        daily = df.groupby(['restaurant_code', 'date'])['guest_count'].sum().reset_index()
        daily.columns = ['res_code', 'date', 'total_guests']
        daily['weekday'] = pd.to_datetime(daily['date']).dt.dayofweek
        
        # Calculate "normal" average per weekday per restaurant
        # Exclude holiday/event dates from normal calculation
        event_dates = set()
        for year in range(min_date.year, max_date.year + 1):
            for (month, day), event_info in SPECIAL_EVENTS.items():
                try:
                    event_date = datetime.date(year, month, day)
                    event_dates.add(event_date)
                    # Also exclude pre-event days
                    pre_days = event_info.get('pre_days', 0)
                    for offset in range(1, pre_days + 1):
                        event_dates.add(event_date - datetime.timedelta(days=offset))
                except ValueError:
                    continue
        
        daily['is_event_date'] = daily['date'].isin(event_dates)  # type: ignore[reportArgumentType]
        normal_days = daily[~daily['is_event_date']]
        
        # Normal average per weekday per restaurant
        normal_avg = normal_days.groupby(['res_code', 'weekday'])['total_guests'].mean()
        normal_avg.name = 'normal_avg'
        
        # Per-restaurant results
        per_restaurant = {}
        
        # Process each special event
        calibrated_count = 0
        
        for (month, day), event_info in SPECIAL_EVENTS.items():
            event_type = event_info['event_type']
            event_name = event_info['name']
            
            # Find this event in historical data
            event_dates_for_type = []
            for year in range(min_date.year, max_date.year + 1):
                try:
                    event_date = datetime.date(year, month, day)
                    if min_date <= event_date <= max_date:
                        event_dates_for_type.append(event_date)
                except ValueError:
                    continue
            
            if not event_dates_for_type:
                logger.debug(f"   {event_type}: No data found in range")
                continue
            
            # Get data for event dates
            event_daily = daily[daily['date'].isin(event_dates_for_type)]
            
            if event_daily.empty:
                continue
            
            # Calculate actual impact per restaurant
            impacts = []
            for _, row in event_daily.iterrows():
                # Get normal average for same weekday + same restaurant
                try:
                    norm = normal_avg.loc[(row['res_code'], row['weekday'])]
                    if norm > 0:
                        impact = row['total_guests'] / norm
                        impacts.append({
                            'res_code': row['res_code'],
                            'event_guests': row['total_guests'],
                            'normal_avg': norm,
                            'impact': impact,
                        })
                        
                        # Store per-restaurant result
                        res_code = row['res_code']
                        if res_code not in per_restaurant:
                            per_restaurant[res_code] = {}
                        
                        # If multiple years, average them
                        if event_type in per_restaurant[res_code]:
                            prev = per_restaurant[res_code][event_type]
                            # Rolling average over multiple years
                            n = prev.get('n_years', 1)
                            prev_factor = prev['calibrated_factor']
                            new_factor = (prev_factor * n + impact) / (n + 1)
                            per_restaurant[res_code][event_type] = {
                                'calibrated_factor': round(float(new_factor), 3),
                                'n_years': n + 1,
                                'latest_event_guests': int(row['total_guests']),
                                'normal_avg': round(float(norm), 1),
                            }
                        else:
                            per_restaurant[res_code][event_type] = {
                                'calibrated_factor': round(float(impact), 3),
                                'n_years': 1,
                                'latest_event_guests': int(row['total_guests']),
                                'normal_avg': round(float(norm), 1),
                            }
                        
                except (KeyError, ZeroDivisionError):
                    continue
            
            # Aggregate calibration (needs minimum sample size)
            if len(impacts) >= min_restaurants_for_calibration:
                # Calculate calibrated factor (use median for robustness)
                impact_values = [i['impact'] for i in impacts]
                calibrated_factor = float(np.median(impact_values))
                
                # Confidence based on sample size and consistency
                std = float(np.std(impact_values))
                cv = std / calibrated_factor if calibrated_factor > 0 else float('inf')
                
                if len(impacts) >= 100 and cv < 0.5:
                    confidence = 'HIGH'
                elif len(impacts) >= 50 and cv < 0.8:
                    confidence = 'MEDIUM'
                else:
                    confidence = 'LOW'
                
                # Store aggregate result
                result['events'][event_type] = {
                    'default_factor': event_info['default_factor'],
                    'calibrated_factor': round(calibrated_factor, 3),
                    'event_dates': [str(d) for d in event_dates_for_type],
                    'normal_avg': round(float(np.mean([i['normal_avg'] for i in impacts])), 1),
                    'event_avg': round(float(np.mean([i['event_guests'] for i in impacts])), 1),
                    'n_restaurants': len(impacts),
                    'confidence': confidence,
                    'cv': round(cv, 3),
                    'percentile_25': round(float(np.percentile(impact_values, 25)), 3),
                    'percentile_75': round(float(np.percentile(impact_values, 75)), 3),
                }
                
                calibrated_count += 1
                
                diff = calibrated_factor - event_info['default_factor']
                direction = '↑' if diff > 0 else '↓'
                logger.info(
                    f"   ✅ {event_type} ({event_name}): "
                    f"{event_info['default_factor']:.2f} → {calibrated_factor:.2f} "
                    f"({direction}{abs(diff):.2f}) "
                    f"[{len(impacts)} restaurants, {confidence}]"
                )
            else:
                logger.debug(
                    f"   {event_type}: Only {len(impacts)} records "
                    f"(need {min_restaurants_for_calibration}) — per-restaurant only"
                )
        
        # Add per-restaurant confidence levels
        for res_code, events in per_restaurant.items():
            for event_type, cal_info in events.items():
                n_years = cal_info.get('n_years', 1)
                if n_years >= 3:
                    cal_info['confidence'] = 'HIGH'
                elif n_years >= 2:
                    cal_info['confidence'] = 'MEDIUM'
                else:
                    cal_info['confidence'] = 'LOW'
        
        result['per_restaurant'] = per_restaurant
        result['n_events_calibrated'] = calibrated_count
        result['n_restaurants_calibrated'] = len(per_restaurant)
        result['status'] = 'calibrated' if (calibrated_count > 0 or per_restaurant) else 'no_data'
        
        # Cache
        EventCalibrator._calibration_cache = result
        EventCalibrator._per_restaurant_cache = per_restaurant
        EventCalibrator._cache_timestamp = datetime.datetime.now()
        
        n_per_res = sum(len(v) for v in per_restaurant.values())
        logger.info(
            f"   📊 Per-restaurant calibration: "
            f"{len(per_restaurant)} restaurants, {n_per_res} event-restaurant pairs"
        )
        
        return result
    
    @staticmethod
    def apply_calibration_to_forecast_days(
        next_days: List[Dict],
        calibration: Dict,
        res_code: str = None,  # type: ignore[reportArgumentType]
    ) -> List[Dict]:
        """
        Override default event impact factors with calibrated ones.
        
        PRIORITY ORDER:
            1. Per-restaurant calibrated factor (highest priority)
            2. Aggregate calibrated factor (fallback)
            3. Default factor from SPECIAL_EVENTS (lowest priority)
        
        Args:
            next_days: List of day info dicts from build_forecast_days()
            calibration: Result from calibrate()
            res_code: Restaurant code for per-restaurant lookup (optional)
            
        Returns:
            Updated next_days with calibrated event impacts
        """
        if not calibration or not calibration.get('events'):
            # Even without aggregate, try per-restaurant
            if not calibration or not calibration.get('per_restaurant'):
                return next_days
        
        cal_events = calibration.get('events', {})
        per_res = calibration.get('per_restaurant', {})
        res_calibration = per_res.get(res_code, {}) if res_code else {}
        
        n_updated = 0
        
        for d_info in next_days:
            event_type = d_info.get('event_type')
            if not event_type:
                continue
            
            # Handle PRE_ prefix
            base_type = event_type.replace('PRE_', '')
            is_pre = event_type.startswith('PRE_')
            
            # --- Priority 1: Per-restaurant calibrated factor ---
            if base_type in res_calibration:
                cal = res_calibration[base_type]
                old_impact = d_info.get('holiday_impact', 1.0)
                
                if is_pre:
                    # Pre-event: scale pre_factor proportionally
                    pre_factor = SPECIAL_EVENTS.get(  # type: ignore[reportCallIssue]
                        _find_event_key(base_type), {}  # type: ignore[reportArgumentType]
                    ).get('pre_factor', 1.05)
                    default_main = SPECIAL_EVENTS.get(  # type: ignore[reportCallIssue]
                        _find_event_key(base_type), {}  # type: ignore[reportArgumentType]
                    ).get('default_factor', 1.0)
                    if default_main > 1.0:
                        scale = cal['calibrated_factor'] / default_main
                        new_impact = 1.0 + (pre_factor - 1.0) * scale
                    else:
                        new_impact = pre_factor
                else:
                    new_impact = cal['calibrated_factor']
                
                d_info['holiday_impact'] = new_impact
                d_info['event_impact'] = new_impact
                d_info['holiday_impact_source'] = 'per_restaurant_calibrated'
                d_info['event_calibration_confidence'] = cal.get('confidence', 'LOW')
                
                if abs(new_impact - old_impact) > 0.01:
                    n_updated += 1
                continue  # Per-restaurant takes priority, skip aggregate
            
            # --- Priority 2: Aggregate calibrated factor ---
            if base_type in cal_events:
                cal = cal_events[base_type]
                old_impact = d_info.get('holiday_impact', 1.0)
                
                if is_pre:
                    pre_factor = SPECIAL_EVENTS.get(  # type: ignore[reportCallIssue]
                        _find_event_key(base_type), {}  # type: ignore[reportArgumentType]
                    ).get('pre_factor', 1.05)
                    default_main = cal['default_factor']
                    if default_main > 1.0:
                        scale = cal['calibrated_factor'] / default_main
                        new_impact = 1.0 + (pre_factor - 1.0) * scale
                    else:
                        new_impact = pre_factor
                else:
                    new_impact = cal['calibrated_factor']
                
                d_info['holiday_impact'] = new_impact
                d_info['event_impact'] = new_impact
                d_info['holiday_impact_source'] = 'event_calibrated'
                d_info['event_calibration_confidence'] = cal['confidence']
                
                if abs(new_impact - old_impact) > 0.01:
                    n_updated += 1
        
        source_str = f"(per-res: {res_code})" if res_code else "(aggregate)"
        if n_updated > 0:
            logger.info(
                f"   📊 Updated {n_updated} forecast days with calibrated event impacts {source_str}"
            )
        
        return next_days
    
    @staticmethod
    def get_calibrated_factor(
        event_type: str,
        res_code: str = None,  # type: ignore[reportArgumentType]
    ) -> Optional[float]:
        """
        Get the calibrated factor for a specific event type.
        
        Priority: per-restaurant > aggregate > None
        """
        if EventCalibrator._calibration_cache is None:
            return None
        
        # Try per-restaurant first
        if res_code and EventCalibrator._per_restaurant_cache:
            res_cal = EventCalibrator._per_restaurant_cache.get(res_code, {})
            if event_type in res_cal:
                return res_cal[event_type]['calibrated_factor']
        
        # Fallback to aggregate
        events = EventCalibrator._calibration_cache.get('events', {})
        cal = events.get(event_type)
        
        if cal:
            return cal['calibrated_factor']
        return None
    
    @staticmethod
    def invalidate_cache():
        """Clear calibration cache."""
        EventCalibrator._calibration_cache = None
        EventCalibrator._per_restaurant_cache = None
        EventCalibrator._cache_timestamp = None


def _find_event_key(event_type: str) -> Optional[tuple]:
    """Find the (month, day) key for a given event type."""
    for key, info in SPECIAL_EVENTS.items():
        if info['event_type'] == event_type:
            return key
    return None
