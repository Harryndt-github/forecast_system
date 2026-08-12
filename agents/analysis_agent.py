"""
==============================================
ANALYSIS AGENT (MỚI HOÀN TOÀN)
==============================================
Trách nhiệm:
- Phân tích xu hướng tăng trưởng/suy giảm từng nhà hàng
- Phát hiện khoảng trống hoạt động (gap detection)
- Phát hiện dữ liệu bất thường (outlier detection)
- Phân loại nhà hàng để chọn strategy forecast phù hợp
- Tạo analysis report để enrich AI prompt

Agent này hoàn toàn mới, không có trong code gốc.
Nó giải quyết yêu cầu:
  "suy luận tỷ lệ trượt, tăng trưởng của mỗi nhà hàng"
  "loại bỏ dữ liệu lọc nhiễu đối với nhà hàng có thời gian hoạt động ngắt quãng"
"""

import pandas as pd
import numpy as np
import datetime
from typing import Dict, List, Tuple, Optional, Any

from forecast_system.config.settings import (
    ANALYSIS_CONFIG, CURRENT_DATE,
    LOW_VOLUME_DAILY_THRESHOLD, MEDIUM_VOLUME_DAILY_THRESHOLD,
)
from forecast_system.utils.logger import get_logger

logger = get_logger('analysis_agent')


class AnalysisAgent:
    """
    Agent phân tích dữ liệu lịch sử nhà hàng:
    - Growth/Decline trends
    - Activity gap detection  
    - Outlier detection
    - Restaurant classification
    """
    
    # ==========================================
    # 1. GROWTH RATE ANALYSIS
    # ==========================================
    
    @staticmethod
    def calculate_growth_rate(
        df_res: pd.DataFrame,
        windows: Optional[List[int]] = None
    ) -> Dict:
        """
        Tính tỷ lệ tăng trưởng/suy giảm cho MỘT nhà hàng.
        
        Logic:
        - Chia mỗi window thành 2 nửa (first_half vs second_half)
        - So sánh trung bình guest count giữa 2 nửa
        - Tính % thay đổi = (avg_second - avg_first) / avg_first * 100
        
        Args:
            df_res: DataFrame transactions của 1 nhà hàng
            windows: List các window days để phân tích. Default [30, 60, 90]
        
        Returns:
            dict with keys:
                growth_30d, growth_60d, growth_90d: % thay đổi
                avg_first_30d, avg_second_30d, ...: Trung bình mỗi nửa
                trend: STRONG_GROWTH | MILD_GROWTH | STABLE | MILD_DECLINE | STRONG_DECLINE
                trend_score: float (-100 to 100) cho weighted calculation
        """
        if windows is None:
            windows = [30, 60, 90]
        
        if df_res.empty:
            return {
                'trend': 'NO_DATA',
                'trend_score': 0,
                **{f'growth_{w}d': 0 for w in windows}
            }
        
        # Aggregate to daily
        daily = df_res.groupby('date')['guest_count'].sum().reset_index()
        daily = daily.sort_values('date')
        daily['date'] = pd.to_datetime(daily['date']).dt.date
        
        ref_date = max(daily['date'].max(), CURRENT_DATE)
        results = {}
        
        for w in windows:
            start = ref_date - datetime.timedelta(days=w)
            mid = ref_date - datetime.timedelta(days=w // 2)
            
            first_half = daily[(daily['date'] >= start) & (daily['date'] < mid)]
            second_half = daily[(daily['date'] >= mid) & (daily['date'] <= ref_date)]
            
            avg_first = first_half['guest_count'].mean() if not first_half.empty else 0
            avg_second = second_half['guest_count'].mean() if not second_half.empty else 0
            
            if avg_first > 0:
                growth = ((avg_second - avg_first) / avg_first) * 100
            else:
                growth = 0 if avg_second == 0 else 100  # Nếu trước đó không có data
            
            results[f'growth_{w}d'] = round(growth, 2)
            results[f'avg_first_{w}d'] = round(avg_first, 1)
            results[f'avg_second_{w}d'] = round(avg_second, 1)
            results[f'sample_first_{w}d'] = len(first_half)
            results[f'sample_second_{w}d'] = len(second_half)
        
        # Trend classification (dựa trên 30-day growth)
        g30 = results.get('growth_30d', 0)
        cfg = ANALYSIS_CONFIG
        
        if g30 > cfg['strong_growth_threshold']:
            results['trend'] = 'STRONG_GROWTH'
        elif g30 > cfg['mild_growth_threshold']:
            results['trend'] = 'MILD_GROWTH'
        elif g30 > cfg['mild_decline_threshold']:
            results['trend'] = 'STABLE'
        elif g30 > cfg['strong_decline_threshold']:
            results['trend'] = 'MILD_DECLINE'
        else:
            results['trend'] = 'STRONG_DECLINE'
        
        # Trend score cho weighted ensemble (-100 to 100)
        results['trend_score'] = round(max(min(g30, 100), -100), 1)
        
        return results
    
    # ==========================================
    # 2. GAP DETECTION (Phát hiện ngắt quãng)
    # ==========================================
    
    @staticmethod
    def detect_activity_gaps(
        df_res: pd.DataFrame,
        min_gap_days: Optional[int] = None
    ) -> List[Dict]:
        """
        Phát hiện nhà hàng có hoạt động ngắt quãng.
        
        Logic:
        - Sắp xếp theo ngày, tìm khoảng cách giữa các ngày có transaction
        - Nếu khoảng cách >= min_gap_days → đánh dấu là gap
        - Phân loại gap: TEMPORARY_CLOSURE (7-30 ngày), RENOVATION (>30 ngày)
        
        Args:
            df_res: DataFrame transactions của 1 nhà hàng
            min_gap_days: Khoảng cách tối thiểu để tính gap
        
        Returns:
            List[dict] với keys: gap_start, gap_end, gap_days, reason
        """
        if min_gap_days is None:
            min_gap_days = ANALYSIS_CONFIG['min_gap_days']
        
        if df_res.empty:
            return []
        
        # Lấy danh sách ngày có transaction (unique, sorted)
        daily = df_res.groupby('date')['guest_count'].sum().reset_index()
        daily['date'] = pd.to_datetime(daily['date']).dt.date
        dates = sorted(daily['date'].unique())
        
        if len(dates) < 2:
            return []
        
        gaps = []
        for i in range(1, len(dates)):
            gap_days = (dates[i] - dates[i - 1]).days
            
            if gap_days >= min_gap_days:
                # Phân loại lý do gap
                if gap_days > 60:
                    reason = 'LONG_CLOSURE'
                elif gap_days > 30:
                    reason = 'RENOVATION'
                elif gap_days > 14:
                    reason = 'EXTENDED_CLOSURE'
                else:
                    reason = 'TEMPORARY_CLOSURE'
                
                gaps.append({
                    'gap_start': dates[i - 1],
                    'gap_end': dates[i],
                    'gap_days': gap_days,
                    'reason': reason,
                })
        
        if gaps:
            logger.debug(f"Found {len(gaps)} gaps "
                        f"(max: {max(g['gap_days'] for g in gaps)} days)")
        
        return gaps
    
    @staticmethod
    def should_exclude_restaurant(
        df_res: pd.DataFrame,
        gaps: Optional[List[Dict]] = None,
        min_active_days: Optional[int] = None,
        max_gap_ratio: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Quyết định có nên loại bỏ nhà hàng khỏi forecast không.
        
        Tiêu chí loại bỏ:
        1. Tổng gap > max_gap_ratio (50%) thời gian hoạt động
        2. Số ngày active < min_active_days (30 ngày)
        3. Không có data gì cả
        
        Args:
            df_res: DataFrame transactions
            gaps: List gaps từ detect_activity_gaps (hoặc None để tự detect)
            min_active_days: Số ngày active tối thiểu
            max_gap_ratio: Tỷ lệ gap tối đa cho phép
        
        Returns:
            (should_exclude: bool, reason: str)
        """
        cfg = ANALYSIS_CONFIG
        min_active_days_val = int(min_active_days) if min_active_days is not None else int(cfg['min_active_days'])
        max_gap_ratio_val = float(max_gap_ratio) if max_gap_ratio is not None else float(cfg['max_gap_ratio'])
        
        if df_res.empty:
            return True, "NO_DATA"
        
        # Detect gaps nếu chưa có
        if gaps is None:
            gaps = AnalysisAgent.detect_activity_gaps(df_res)
        
        # Tính tổng thời gian
        dates = pd.to_datetime(df_res['date']).dt.date
        date_min, date_max = dates.min(), dates.max()
        total_span = (date_max - date_min).days
        
        if total_span <= 0:
            return True, "SINGLE_DAY_DATA"
        
        # Số ngày thực sự active
        actual_active_days = df_res['date'].nunique()
        
        # Tổng ngày gap
        total_gap_days = sum(g['gap_days'] for g in gaps)
        
        # Tỷ lệ gap
        gap_ratio = total_gap_days / total_span if total_span > 0 else 0
        
        # Check tiêu chí loại bỏ
        if gap_ratio > max_gap_ratio_val:
            return True, f"HIGH_GAP_RATIO ({gap_ratio:.0%}, threshold: {max_gap_ratio_val:.0%})"
        
        if actual_active_days < min_active_days_val:
            return True, f"LOW_ACTIVE_DAYS ({actual_active_days}d, min: {min_active_days_val}d)"
        
        # Check nếu gap gần nhất vẫn đang diễn ra (nhà hàng có thể đã đóng cửa)
        if gaps:
            latest_gap = max(gaps, key=lambda g: g['gap_end'])
            if latest_gap['gap_end'] >= CURRENT_DATE - datetime.timedelta(days=3):
                # Gap kết thúc trong 3 ngày gần → có thể vẫn đang đóng
                return True, f"CURRENTLY_IN_GAP (since {latest_gap['gap_start']})"
        
        return False, "OK"
    
    # ==========================================
    # 3. OUTLIER DETECTION
    # ==========================================
    
    @staticmethod
    def detect_outliers(
        df_res: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> Tuple[List[Dict], pd.DataFrame]:
        """
        Phát hiện ngày bất thường cho từng nhà hàng.
        
        Logic:
        - Group theo weekday riêng biệt (vì pattern khác nhau)
        - Dùng IQR method để detect outlier per weekday
        - Return danh sách outliers + cleaned DataFrame
        
        Args:
            df_res: DataFrame transactions
            threshold: IQR multiplier (default từ config)
        
        Returns:
            (outliers: List[dict], df_tagged: pd.DataFrame)
            outliers: each dict has date, weekday, guest_count, expected_range, type
            df_tagged: DataFrame giữ lại ngày outlier và thêm outlier_weight
        """
        threshold_val = float(threshold) if threshold is not None else float(ANALYSIS_CONFIG['outlier_iqr_threshold'])
        
        if df_res.empty:
            return [], df_res
        
        # Daily aggregation
        daily: pd.DataFrame = df_res.groupby(['date', 'weekday'])['guest_count'].sum().reset_index() # type: ignore
        
        outliers = []
        outlier_dates = set()
        
        for wd in daily['weekday'].unique():
            subset = daily[daily['weekday'] == wd]
            
            if len(subset) < 4:
                # Quá ít data để detect outlier
                continue
            
            subset_gc = pd.Series(subset['guest_count'])
            q1 = float(subset_gc.quantile(0.25))
            q3 = float(subset_gc.quantile(0.75))
            iqr = q3 - q1
            
            # Nếu IQR = 0 (tất cả giống nhau), bỏ qua
            if iqr == 0:
                continue
            
            lower = max(0, q1 - threshold_val * iqr)  # Không cho lower < 0
            upper = q3 + threshold_val * iqr
            
            anomalies = subset[
                (subset['guest_count'] < lower) | (subset['guest_count'] > upper)
            ]
            
            anomalies_df = pd.DataFrame(anomalies)
            for _, row in anomalies_df.iterrows():
                outlier_dates.add(row['date'])
                outliers.append({
                    'date': row['date'],
                    'weekday': row['weekday'],
                    'guest_count': int(row['guest_count']),
                    'expected_range': f"{int(lower)}-{int(upper)}",
                    'q1': round(q1, 1),
                    'q3': round(q3, 1),
                    'type': 'LOW' if row['guest_count'] < lower else 'HIGH'
                })
        
        # Keep outlier days in training but tag/downweight them. Removing
        # high-demand days made the model learn artificially low demand.
        df_tagged = df_res.copy()
        df_tagged['is_outlier_day'] = 0
        df_tagged['outlier_weight'] = 1.0
        if outlier_dates:
            outlier_dates_list = list(outlier_dates)
            df_tagged.loc[df_tagged['date'].isin(outlier_dates_list), 'is_outlier_day'] = 1

            high_dates = {o['date'] for o in outliers if o.get('type') == 'HIGH'}
            low_dates = {o['date'] for o in outliers if o.get('type') == 'LOW'}
            # High outliers are often real demand peaks: light downweight.
            df_tagged.loc[df_tagged['date'].isin(list(high_dates)), 'outlier_weight'] = 0.70
            # Low outliers may be closures/data gaps: stronger downweight.
            df_tagged.loc[df_tagged['date'].isin(list(low_dates)), 'outlier_weight'] = 0.40

            logger.info(
                f"Tagged {len(outliers)} outlier days for downweighting "
                f"(kept {len(df_tagged)} transactions)"
            )
        
        return outliers, df_tagged
    
    # ==========================================
    # 4. RESTAURANT CLASSIFICATION
    # ==========================================
    
    @staticmethod
    def compute_smart_max_daily(df_res: pd.DataFrame) -> Dict:
        """
        Tính max_daily THÔNG MINH, phân biệt ngày thường vs ngày lễ.
        
        Logic:
        - Ngày thường: max khách trong 3 THÁNG GẦN NHẤT (loại trừ ngày lễ)
        - Ngày lễ/holiday: max khách của ĐÚNG ngày lễ đó trong năm trước
        - Special event: max khách của ĐÚNG event đó trong năm trước
        
        Returns:
            {
                'max_daily': int,              # Overall max (backward compat)
                'max_daily_normal': int,        # Max ngày thường (3 tháng gần)
                'min_daily_normal': int,        # Min ngày thường (3 tháng gần) — floor
                'min_daily_normal_weekday': int, # Min Mon-Fri (3 tháng) — floor
                'min_daily_normal_weekend': int, # Min Sat-Sun (3 tháng) — floor
                'max_daily_by_holiday': {       # Max theo từng holiday type
                    'TET_NGUYEN_DAN': int,
                    'VALENTINE': int,
                    ...
                }
            }
        """
        from forecast_system.utils.date_utils import (
            get_vn_holidays, classify_holiday, SPECIAL_EVENTS
        )
        
        result = {
            'max_daily': 0,
            'max_daily_normal': 0,
            'min_daily_normal': 0,
            'min_daily_normal_weekday': 0,
            'min_daily_normal_weekend': 0,
            'max_daily_by_holiday': {},
        }
        
        if df_res.empty:
            return result
        
        df = df_res.copy()
        df['_date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # ---- Group by date to get daily totals ----
        daily = df.groupby('date')['guest_count'].sum()
        result['max_daily'] = int(daily.max())
        
        # ---- Identify holiday dates in historical data ----
        all_dates = pd.to_datetime(daily.index)
        years_in_data = sorted(set(d.year for d in all_dates if hasattr(d, 'year')))
        vn_holidays = get_vn_holidays(years=years_in_data if years_in_data else None)
        
        holiday_dates = {}      # date_str → holiday_type
        special_dates = {}      # date_str → event_type
        
        for d in all_dates:
            d_date = d.date() if hasattr(d, 'date') else d
            
            # Check official holidays
            h_type = classify_holiday(d_date, vn_holidays)
            if h_type:
                holiday_dates[str(d_date)] = h_type
            
            # Check special events (Valentine, Women's Day, etc.)
            key = (d_date.month, d_date.day)
            if key in SPECIAL_EVENTS:
                special_dates[str(d_date)] = SPECIAL_EVENTS[key]['event_type']
        
        # Combine all non-normal dates
        all_special_dates = set(holiday_dates.keys()) | set(special_dates.keys())
        
        # ---- MAX NGÀY THƯỜNG: chỉ 3 tháng gần nhất, loại trừ lễ ----
        # TÁCH weekday vs weekend để tránh ngày thường kéo cap cuối tuần xuống
        cutoff_3m = CURRENT_DATE - datetime.timedelta(days=90)
        normal_weekday_values = []  # Mon-Fri
        normal_weekend_values = []  # Sat-Sun
        normal_daily_values = []    # All (backward compat)
        
        for d_str, total in daily.items():
            d_parsed = pd.to_datetime(str(d_str))
            # Chỉ lấy 3 tháng gần nhất VÀ không phải ngày lễ/special
            if d_parsed.date() >= cutoff_3m and str(d_str) not in all_special_dates:
                val = int(total)
                normal_daily_values.append(val)
                if d_parsed.dayofweek >= 5:  # Saturday=5, Sunday=6
                    normal_weekend_values.append(val)
                else:
                    normal_weekday_values.append(val)
        
        if normal_daily_values:
            result['max_daily_normal'] = max(normal_daily_values)
            result['min_daily_normal'] = min(normal_daily_values)
        else:
            # Fallback: nếu không có data 3 tháng, dùng all-time (trừ lễ)
            all_normal = [
                int(v) for d_str, v in daily.items()
                if str(d_str) not in all_special_dates
            ]
            result['max_daily_normal'] = max(all_normal) if all_normal else result['max_daily']
            result['min_daily_normal'] = min(all_normal) if all_normal else 0
        
        # Weekend/Weekday specific caps
        result['max_daily_normal_weekday'] = (
            max(normal_weekday_values) if normal_weekday_values
            else result['max_daily_normal']
        )
        result['max_daily_normal_weekend'] = (
            max(normal_weekend_values) if normal_weekend_values
            else result['max_daily_normal']
        )
        
        # Weekend/Weekday specific floors (min)
        result['min_daily_normal_weekday'] = (
            min(normal_weekday_values) if normal_weekday_values
            else result['min_daily_normal']
        )
        result['min_daily_normal_weekend'] = (
            min(normal_weekend_values) if normal_weekend_values
            else result['min_daily_normal']
        )
        
        # ---- MAX NGÀY LỄ: theo từng holiday_type, lấy từ năm trước ----
        max_by_holiday = {}
        
        # Official holidays
        for d_str, h_type in holiday_dates.items():
            total_val = daily.get(d_str, 0)
            total = int(total_val) if total_val is not None else 0
            if total > 0:
                if h_type not in max_by_holiday or total > max_by_holiday[h_type]:
                    max_by_holiday[h_type] = total
        
        # Special events  
        for d_str, e_type in special_dates.items():
            total_val = daily.get(d_str, 0)
            total = int(total_val) if total_val is not None else 0
            if total > 0:
                if e_type not in max_by_holiday or total > max_by_holiday[e_type]:
                    max_by_holiday[e_type] = total
        
        result['max_daily_by_holiday'] = max_by_holiday
        
        logger.debug(
            f"Smart max_daily: normal={result['max_daily_normal']} "
            f"min_daily: normal={result['min_daily_normal']} "
            f"(3mo), holidays={max_by_holiday}"
        )
        
        return result
    
    @staticmethod
    def classify_restaurant(df_res: pd.DataFrame) -> Tuple[str, Dict]:
        """
        Phân loại nhà hàng để chọn model/strategy forecast phù hợp.
        
        Categories:
        - NEW: < 14 ngày data → Dùng AI primarily
        - YOUNG: 14-45 ngày → AI chính, ML phụ
        - VOLATILE: CV > 0.5 → Ensemble weighted
        - HIGH_VOLUME: Avg daily > 200 khách → ML chính, AI validate
        - STANDARD: Trường hợp còn lại → Ensemble cân bằng
        
        Returns:
            (category: str, profile: dict)
        """
        cfg = ANALYSIS_CONFIG
        
        if df_res.empty:
            return 'NEW', {
                'strategy': 'AI_ONLY',
                'confidence': 'VERY_LOW',
                'active_days': 0,
                'avg_daily': 0,
                'cv': None
            }
        
        daily = df_res.groupby('date')['guest_count'].sum()
        avg_daily = daily.mean()
        active_days = len(daily)
        std_daily = daily.std() if len(daily) > 1 else 0
        cv = std_daily / avg_daily if avg_daily > 0 else 999
        
        # Smart max_daily: phân biệt ngày thường vs ngày lễ
        smart_max = AnalysisAgent.compute_smart_max_daily(df_res)
        
        profile = {
            'active_days': active_days,
            'avg_daily': round(avg_daily, 1),
            'std_daily': round(std_daily, 1),
            'cv': round(cv, 3),
            'min_daily': int(daily.min()),
            'max_daily': smart_max['max_daily'],             # Overall (backward compat)
            'max_daily_normal': smart_max['max_daily_normal'],  # Ngày thường 3 tháng
            'max_daily_normal_weekday': smart_max['max_daily_normal_weekday'],  # Mon-Fri
            'max_daily_normal_weekend': smart_max['max_daily_normal_weekend'],  # Sat-Sun
            'min_daily_normal': smart_max['min_daily_normal'],  # Floor ngày thường 3 tháng
            'min_daily_normal_weekday': smart_max['min_daily_normal_weekday'],  # Floor Mon-Fri
            'min_daily_normal_weekend': smart_max['min_daily_normal_weekend'],  # Floor Sat-Sun
            'max_daily_by_holiday': smart_max['max_daily_by_holiday'],  # Theo holiday type
            'median_daily': round(daily.median(), 1),
        }
        
        # ⭐ v5: Calculate trend_short for trend adjustment layer
        daily_sorted = daily.sort_index()
        if len(daily_sorted) >= 7:
            mean_3d = daily_sorted.tail(3).mean()
            mean_7d = daily_sorted.tail(7).mean()
            mean_14d = daily_sorted.tail(14).mean() if len(daily_sorted) >= 14 else mean_7d
            profile['trend_short'] = round(mean_3d / max(mean_7d, 1), 4)
            profile['trend_medium'] = round(mean_7d / max(mean_14d, 1), 4)
        else:
            profile['trend_short'] = 1.0
            profile['trend_medium'] = 1.0
        
        if active_days < cfg['new_restaurant_days']:
            category = 'NEW'
            profile['strategy'] = 'AI_ONLY'
            profile['confidence'] = 'LOW'
            
        elif active_days < cfg['young_restaurant_days']:
            category = 'YOUNG'
            profile['strategy'] = 'AI_PRIMARY_ML_SECONDARY'
            profile['confidence'] = 'MEDIUM'
            
        elif cv > cfg['volatile_cv_threshold']:
            category = 'VOLATILE'
            profile['strategy'] = 'ENSEMBLE_WEIGHTED'
            profile['confidence'] = 'MEDIUM'
            
        elif avg_daily > cfg['high_volume_threshold']:
            category = 'HIGH_VOLUME'
            profile['strategy'] = 'ML_PRIMARY_AI_VALIDATE'
            profile['confidence'] = 'HIGH'
            
        else:
            category = 'STANDARD'
            profile['strategy'] = 'ENSEMBLE_EQUAL'
            profile['confidence'] = 'HIGH'
        
        # ⭐ v6: Volume Segmentation Layer
        # Override strategy based on volume level for optimized model selection
        if avg_daily < LOW_VOLUME_DAILY_THRESHOLD:
            profile['volume_segment'] = 'LOW_VOLUME'
            # Low volume: ML models are noisy, use baseline median
            if category not in ('NEW', 'YOUNG'):  # Don't override NEW/YOUNG routing
                profile['strategy'] = 'BASELINE_ONLY'
                profile['confidence'] = 'MEDIUM'
                logger.debug(
                    f"📊 v6: LOW_VOLUME (avg={avg_daily:.1f} < {LOW_VOLUME_DAILY_THRESHOLD}) "
                    f"→ BASELINE_ONLY"
                )
        elif avg_daily < MEDIUM_VOLUME_DAILY_THRESHOLD:
            profile['volume_segment'] = 'MEDIUM_VOLUME'
            # Medium volume: full ensemble with trend + booking features
            # Keep existing strategy but ensure ensemble is used
            if category not in ('NEW', 'YOUNG'):
                profile['strategy'] = 'ENSEMBLE_WEIGHTED'
        else:
            profile['volume_segment'] = 'HIGH_VOLUME'
            # High volume: ML-primary, peak behavior handled by weekend features
            if category not in ('NEW', 'YOUNG'):
                profile['strategy'] = 'HIGH_VOLUME_ML'
                profile['confidence'] = 'HIGH'
        
        return category, profile
    
    # ==========================================
    # 5. COMPREHENSIVE ANALYSIS REPORT
    # ==========================================
    
    @staticmethod
    def generate_restaurant_report(
        res_code: str,
        df_res: pd.DataFrame
    ) -> Dict:
        """
        Tạo BÁO CÁO PHÂN TÍCH TOÀN DIỆN cho 1 nhà hàng.
        Kết hợp tất cả analysis methods.
        
        Report này sẽ được:
        1. Truyền vào AI prompt để enrich context
        2. Dùng để chọn forecast strategy
        3. Lưu để monitoring
        
        Returns:
            dict (complete analysis report)
        """
        report: Dict[str, Any] = {
            'restaurant_code': res_code,
            'analysis_date': str(CURRENT_DATE),
        }
        
        # 1. Growth Analysis
        growth = AnalysisAgent.calculate_growth_rate(df_res)
        report['growth'] = growth
        report['trend'] = growth.get('trend', 'NO_DATA')
        report['trend_score'] = growth.get('trend_score', 0)
        
        # 2. Gap Detection
        gaps = AnalysisAgent.detect_activity_gaps(df_res)
        report['gaps'] = gaps
        report['total_gaps'] = len(gaps)
        report['total_gap_days'] = sum(g['gap_days'] for g in gaps)
        
        # 3. Exclusion Check
        should_exclude, exclude_reason = AnalysisAgent.should_exclude_restaurant(
            df_res, gaps=gaps
        )
        report['should_exclude'] = should_exclude
        report['exclude_reason'] = exclude_reason
        
        # 4. Outlier Detection
        outliers, df_cleaned = AnalysisAgent.detect_outliers(df_res)
        report['outliers'] = outliers
        report['outlier_count'] = len(outliers)
        report['cleaned_data_rows'] = len(df_cleaned)
        report['original_data_rows'] = len(df_res)
        
        # 5. Restaurant Classification
        category, profile = AnalysisAgent.classify_restaurant(df_cleaned)
        report['category'] = category
        report['profile'] = profile
        report['strategy'] = profile.get('strategy', 'ENSEMBLE_EQUAL')
        report['confidence'] = profile.get('confidence', 'MEDIUM')
        
        # 6. Seasonality Analysis (weekday patterns)
        if not df_res.empty:
            daily = pd.DataFrame(df_res.groupby(['date', 'weekday'])['guest_count'].sum().reset_index())
            weekday_stats = {}
            for wd in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                wd_data = pd.Series(daily[daily['weekday'] == wd]['guest_count'])
                if not wd_data.empty:
                    weekday_stats[wd] = {
                        'avg': round(wd_data.mean(), 1),
                        'std': round(wd_data.std(), 1) if len(wd_data) > 1 else 0,
                        'min': int(wd_data.min()),
                        'max': int(wd_data.max()),
                        'count': len(wd_data),
                    }
            report['weekday_patterns'] = weekday_stats
            
            # Weekend vs Weekday ratio
            wd_avg = np.mean([
                weekday_stats.get(d, {}).get('avg', 0) 
                for d in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            ])
            we_avg = np.mean([
                weekday_stats.get(d, {}).get('avg', 0) 
                for d in ['Saturday', 'Sunday']
            ])
            report['weekend_weekday_ratio'] = round(we_avg / wd_avg, 2) if wd_avg > 0 else 1.0
        
        return report
    
    # ==========================================
    # 6. CHAIN-LEVEL OUTLIER DETECTION
    # ==========================================
    
    @staticmethod
    def detect_outlier_restaurants(
        df_train: pd.DataFrame,
        active_restaurants: list,
        iqr_multiplier: float = 5.0
    ) -> set:
        """
        Detect restaurants whose avg daily guest count is an extreme outlier
        compared to the entire chain.
        
        Example: Restaurant 587 averaging 577 guests/hr when the chain
        averages 17.5 guests/hr → clear data quality issue.
        
        Uses IQR method across all restaurants' daily averages.
        
        Args:
            df_train: Full training DataFrame
            active_restaurants: List of restaurant codes
            iqr_multiplier: IQR multiplier for outlier detection (default 5.0 = very conservative)
        
        Returns:
            set of restaurant codes that are chain-level outliers
        """
        if len(active_restaurants) < 5:
            return set()  # Need enough restaurants for IQR to be meaningful
        
        # Calculate average daily guests per restaurant
        res_avgs = {}
        for res_code in active_restaurants:
            df_res = df_train[df_train['restaurant_code'] == res_code]
            if df_res.empty:
                continue
            daily = df_res.groupby('date')['guest_count'].sum()
            if len(daily) > 0:
                res_avgs[res_code] = daily.mean()
        
        if len(res_avgs) < 5:
            return set()
        
        avgs = np.array(list(res_avgs.values()))
        q1 = np.percentile(avgs, 25)
        q3 = np.percentile(avgs, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            # All restaurants similar, use median-based detection
            median = np.median(avgs) # type: ignore[reportCallIssue]
            upper_bound = median * 10  # 10x median is suspicious
        else:
            upper_bound = q3 + iqr_multiplier * iqr
        
        outlier_codes = set()
        for res_code, avg in res_avgs.items():
            if avg > upper_bound:
                outlier_codes.add(res_code)
                logger.warning(
                    f"⚠️ CHAIN-LEVEL OUTLIER: {res_code} "
                    f"avg_daily={avg:.0f} vs chain Q3={q3:.0f}, "
                    f"upper_bound={upper_bound:.0f}"
                )
        
        return outlier_codes
    
    # ==========================================
    # 7. BATCH ANALYSIS (All Restaurants)
    # ==========================================
    
    @staticmethod
    def analyze_all_restaurants(
        df_train: pd.DataFrame,
        active_restaurants: list
    ) -> Dict[str, Dict]:
        """
        Phân tích TẤT CẢ nhà hàng active, trả về dict of reports.
        
        Includes chain-level outlier detection to auto-exclude restaurants
        with abnormal volume (e.g. data quality issues).
        
        Args:
            df_train: Full training DataFrame
            active_restaurants: List of restaurant codes
        
        Returns:
            dict: {restaurant_code: analysis_report}
        """
        logger.info(f"📊 Analyzing {len(active_restaurants)} restaurants...")
        
        # Step 0: Detect chain-level outlier restaurants
        chain_outliers = AnalysisAgent.detect_outlier_restaurants(
            df_train, active_restaurants
        )
        if chain_outliers:
            logger.warning(
                f"⚠️ Found {len(chain_outliers)} chain-level outlier restaurant(s): "
                f"{chain_outliers}"
            )
        
        reports = {}
        excluded = []
        categories_count = {}
        
        for i, res_code in enumerate(active_restaurants):
            if (i + 1) % 50 == 0 or i == 0:
                logger.info(f"   Analyzing... ({i+1}/{len(active_restaurants)})")
            
            df_res = pd.DataFrame(df_train[df_train['restaurant_code'] == res_code])
            report = AnalysisAgent.generate_restaurant_report(res_code, df_res)
            
            # Mark chain-level outliers as excluded
            if res_code in chain_outliers and not report['should_exclude']:
                report['should_exclude'] = True
                avg_daily = report.get('profile', {}).get('avg_daily', 0)
                report['exclude_reason'] = (
                    f"CHAIN_OUTLIER (avg_daily={avg_daily:.0f}, "
                    f"abnormally high vs chain)"
                )
            
            reports[res_code] = report
            
            # Track stats
            cat = report['category']
            categories_count[cat] = categories_count.get(cat, 0) + 1
            
            if report['should_exclude']:
                excluded.append({
                    'code': res_code,
                    'reason': report['exclude_reason']
                })
        
        # Summary
        logger.info("=" * 50)
        logger.info("📊 ANALYSIS SUMMARY")
        logger.info(f"   Total analyzed: {len(active_restaurants)}")
        logger.info(f"   Excluded: {len(excluded)}")
        logger.info(f"   To forecast: {len(active_restaurants) - len(excluded)}")
        logger.info(f"   Categories: {categories_count}")
        if chain_outliers:
            logger.info(f"   Chain outliers: {chain_outliers}")
        
        if excluded:
            logger.info(f"   Excluded restaurants:")
            for ex in excluded[:10]:  # Show max 10
                logger.info(f"     - {ex['code']}: {ex['reason']}")
            if len(excluded) > 10:
                logger.info(f"     ... and {len(excluded) - 10} more")
        
        logger.info("=" * 50)
        
        return reports
    
    # ==========================================
    # 7. FORMAT FOR AI PROMPT
    # ==========================================
    
    @staticmethod
    def format_report_for_prompt(report: Dict) -> str:
        """
        Format analysis report thành text để inject vào AI prompt.
        Cho LM Studio biết context về nhà hàng trước khi dự đoán.
        
        Returns:
            str: Formatted analysis text
        """
        lines = [
            "📊 RESTAURANT ANALYSIS REPORT:",
            f"  Category: {report.get('category', 'UNKNOWN')}",
            f"  Trend: {report.get('trend', 'N/A')} (score: {report.get('trend_score', 0)})",
            f"  Confidence: {report.get('confidence', 'N/A')}",
        ]
        
        # Growth details
        growth = report.get('growth', {})
        if growth:
            lines.append("")
            lines.append("📈 GROWTH ANALYSIS:")
            for w in [30, 60, 90]:
                g = growth.get(f'growth_{w}d')
                if g is not None:
                    direction = "↑" if g > 0 else "↓" if g < 0 else "→"
                    lines.append(f"  {w}-day: {direction} {abs(g):.1f}% "
                               f"(avg: {growth.get(f'avg_first_{w}d', 0):.0f} → "
                               f"{growth.get(f'avg_second_{w}d', 0):.0f})")
        
        # Weekday patterns
        weekday_patterns = report.get('weekday_patterns', {})
        if weekday_patterns:
            lines.append("")
            lines.append("📅 WEEKDAY PATTERNS:")
            for wd in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                stats = weekday_patterns.get(wd)
                if stats:
                    lines.append(f"  {wd}: avg={stats['avg']:.0f} "
                               f"(range: {stats['min']}-{stats['max']}, "
                               f"samples: {stats['count']})")
        
        # Weekend ratio
        ratio = report.get('weekend_weekday_ratio', 1.0)
        lines.append(f"\n  Weekend/Weekday ratio: {ratio:.2f}x")
        
        # Profile
        profile = report.get('profile', {})
        if profile:
            lines.append("")
            lines.append("📋 PROFILE:")
            lines.append(f"  Active days: {profile.get('active_days', 0)}")
            lines.append(f"  Avg daily guests: {profile.get('avg_daily', 0):.0f}")
            lines.append(f"  Volatility (CV): {profile.get('cv', 0):.2f}")
        
        # Gaps
        total_gaps = report.get('total_gaps', 0)
        if total_gaps > 0:
            lines.append("")
            lines.append(f"⚠️ ACTIVITY GAPS: {total_gaps} gaps "
                        f"({report.get('total_gap_days', 0)} total days)")
        
        # Outliers
        outlier_count = report.get('outlier_count', 0)
        if outlier_count > 0:
            lines.append(f"⚠️ OUTLIERS DETECTED: {outlier_count} anomalous days removed")
        
        return "\n".join(lines)
