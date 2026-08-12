"""
==============================================
HOLIDAY IMPACT CALIBRATOR (AUTO-CALIBRATION)
==============================================
Tự động tính holiday impact factors từ data thực tế,
thay thế các con số hardcoded "educated guesses".

Methodology:
  1. Load data lịch sử quanh các dịp lễ (VD: Tết 2025)
  2. Với mỗi nhà hàng, tính:
     - Baseline: median khách ngày thường (cùng thứ, non-holiday)
     - Actual: khách thực tế mỗi ngày trong window lễ
     - Ratio = actual / baseline
  3. Aggregate: median ratio qua tất cả nhà hàng (robust)
  4. Lưu calibration vào holiday_calibration.json
  5. Apply vào forecast days

Key Insight:
  - Ngày Tết chính: đa phần NHà hàng ĐÓNG CỬA → xử lý bởi closure detection
  - PRE-Tết & POST-Tết: calibration RẤT QUAN TRỌNG (multiplier trực tiếp)
"""

import datetime
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict

from forecast_system.config.settings import PROJECT_ROOT
from forecast_system.utils.date_utils import (
    get_vn_holidays, get_holiday_periods, classify_holiday, HOLIDAY_TYPES
)
from forecast_system.utils.logger import get_logger

logger = get_logger('holiday_calibrator')


class HolidayCalibrator:
    """Auto-calibrate holiday impact factors from actual transaction data."""

    CALIBRATION_FILE = PROJECT_ROOT / 'holiday_calibration.json'

    # Analysis windows per holiday type
    WINDOWS = {
        'TET_NGUYEN_DAN': {'pre': 14, 'post': 10},
        'DEFAULT': {'pre': 5, 'post': 5},
    }

    # Baseline computation parameters
    BASELINE_BUFFER_DAYS = 45
    MIN_BASELINE_GUESTS = 5
    MIN_BASELINE_DAYS = 3
    MIN_RESTAURANTS = 10

    # ==========================================
    # MAIN ENTRY POINT
    # ==========================================

    @staticmethod
    def calibrate(df_data, vn_holidays, engine=None):
        """
        Main entry: calibrate holiday impact factors from historical data.

        Args:
            df_data: Transaction data (restaurant_code, date, guest_count, hour)
            vn_holidays: holidays.VN object
            engine: Optional SQLAlchemy engine for loading extended baseline data

        Returns:
            Dict: calibration results (also saved to JSON)
        """
        logger.info("=" * 60)
        logger.info("📐 HOLIDAY IMPACT CALIBRATOR - Data-Driven")
        logger.info("=" * 60)

        if df_data.empty:
            logger.warning("No data provided for calibration")
            return {}

        # 1. Aggregate to daily per restaurant
        df_daily = HolidayCalibrator._aggregate_daily(df_data)
        logger.info(
            f"   📊 Daily data: {len(df_daily):,} rows, "
            f"{df_daily['restaurant_code'].nunique()} restaurants"
        )

        # 2. Find years in data
        years = sorted(df_daily['date'].apply(lambda d: d.year).unique())
        logger.info(f"   📅 Years: {years}")
        logger.info(
            f"   📅 Range: {df_daily['date'].min()} → {df_daily['date'].max()}"
        )

        # 3. Initialize calibration structure
        calibration = {
            'version': 1,
            'calibration_date': str(datetime.date.today()),
            'years_analyzed': [int(y) for y in years],
            'holiday_types': {},
        }

        # 4. Analyze each holiday period found in data
        for year in years:
            periods = get_holiday_periods(year, vn_holidays)

            for period in periods:
                h_type = period['type']
                logger.info(
                    f"\n🎌 Analyzing: {h_type} {year} "
                    f"({period['start']} → {period['end']})"
                )

                # Ensure sufficient baseline data
                df_analysis = HolidayCalibrator._ensure_sufficient_data(
                    df_daily, period, engine
                )
                if df_analysis.empty:
                    logger.warning(f"   ⚠️ Insufficient data for {h_type} {year}")
                    continue

                # Analyze this period
                result = HolidayCalibrator._analyze_period(
                    df_analysis, period, vn_holidays
                )
                if result:
                    if h_type not in calibration['holiday_types']:
                        calibration['holiday_types'][h_type] = {
                            'periods': [],
                            'aggregate': {},
                        }
                    calibration['holiday_types'][h_type]['periods'].append(result)

        # 5. Compute aggregates across years
        for h_type, data in calibration['holiday_types'].items():
            if data['periods']:
                data['aggregate'] = HolidayCalibrator._compute_aggregate(
                    data['periods']
                )

        # 6. Save and report
        HolidayCalibrator.save_calibration(calibration)
        HolidayCalibrator.print_report(calibration)

        return calibration

    # ==========================================
    # DATA PREPARATION
    # ==========================================

    @staticmethod
    def _aggregate_daily(df_data):
        """Aggregate transactions to daily guests per restaurant."""
        df = df_data.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df = df.dropna(subset=['date'])
        df_daily = df.groupby(
            ['restaurant_code', 'date']
        )['guest_count'].sum().reset_index()
        return df_daily

    @staticmethod
    def _ensure_sufficient_data(df_daily, period, engine=None):
        """
        Ensure enough baseline data exists around the holiday period.
        If insufficient, attempt to load more via DB engine.
        """
        buffer = HolidayCalibrator.BASELINE_BUFFER_DAYS
        needed_start = period['start'] - datetime.timedelta(days=buffer)
        needed_end = period['end'] + datetime.timedelta(days=buffer)

        existing_dates = set(df_daily['date'].unique())
        before = sum(1 for d in existing_dates if needed_start <= d < period['start'])
        after = sum(1 for d in existing_dates if period['end'] < d <= needed_end)

        logger.info(f"   Data coverage: {before} days before, {after} days after")

        if before >= 14 and after >= 14:
            return df_daily

        if engine is None:
            if before < 3 and after < 3:
                return pd.DataFrame()
            return df_daily

        # Load additional data from DB
        logger.info("   📥 Loading extended baseline data from DB...")
        try:
            from forecast_system.agents.data_agent import DataAgent
            df_extra = DataAgent.load_date_range(engine, needed_start, needed_end)

            if not df_extra.empty:
                df_extra_daily = HolidayCalibrator._aggregate_daily(df_extra)
                df_combined = pd.concat(
                    [df_daily, df_extra_daily], ignore_index=True
                ).drop_duplicates(
                    subset=['restaurant_code', 'date'], keep='first'
                )
                logger.info(
                    f"   ✅ Extended: {len(df_daily):,} → {len(df_combined):,} rows"
                )
                return df_combined
        except Exception as e:
            logger.warning(f"   Could not load extended data: {e}")

        return df_daily

    # ==========================================
    # CORE ANALYSIS
    # ==========================================

    @staticmethod
    def _analyze_period(df_daily, period, vn_holidays):
        """
        Analyze one holiday period: compute actual/baseline ratios
        for each offset day (pre, holiday, post).
        """
        h_type = period['type']
        holiday_start = period['start']
        holiday_end = period['end']
        holiday_dates = set(period['dates'])

        windows = HolidayCalibrator.WINDOWS.get(
            h_type, HolidayCalibrator.WINDOWS['DEFAULT']
        )
        pre_window = windows['pre']
        post_window = windows['post']

        # Exclude zone: holiday + analysis window
        exclude_start = holiday_start - datetime.timedelta(days=pre_window)
        exclude_end = holiday_end + datetime.timedelta(days=post_window)

        # Compute baselines (normal behavior per restaurant per weekday)
        baselines = HolidayCalibrator._compute_baselines(
            df_daily, exclude_start, exclude_end, vn_holidays
        )
        if not baselines:
            logger.warning("   Could not compute baselines")
            return None

        n_restaurants = len(set(k[0] for k in baselines.keys()))
        logger.info(f"   Baselines: {n_restaurants} restaurants, "
                     f"{len(baselines)} (restaurant, weekday) pairs")

        result = {
            'year': holiday_start.year,
            'period_start': str(holiday_start),
            'period_end': str(holiday_end),
            'n_baseline_restaurants': n_restaurants,
            'pre': {},
            'holiday': {},
            'post': {},
            'per_restaurant': {},
            'is_future': False,  # Will be set True if holiday dates are all in future
        }

        # [FIX] If all holiday dates are in the future, we cannot calibrate this year.
        # Skip to avoid polluting the aggregate with zero-data periods.
        today = datetime.date.today()
        if holiday_start > today:
            logger.info(
                f"   ⏭️ Skipping {h_type} {holiday_start.year}: holiday dates are in the future"
            )
            result['is_future'] = True
            return result  # Return early with empty pre/holiday/post

        # ── PRE-HOLIDAY ratios ──
        for offset in range(1, pre_window + 1):
            date = holiday_start - datetime.timedelta(days=offset)
            ratios, per_res = HolidayCalibrator._compute_day_ratios(
                df_daily, date, baselines
            )
            if ratios:
                result['pre'][str(offset)] = HolidayCalibrator._summarize_ratios(
                    ratios, date
                )
                # Store per-restaurant
                for res_code, ratio in per_res.items():
                    result['per_restaurant'].setdefault(res_code, {})
                    result['per_restaurant'][res_code][f'pre_{offset}'] = round(ratio, 3)

        # ── HOLIDAY ratios ──
        all_holiday_ratios = []
        for date in sorted(holiday_dates):
            ratios, per_res = HolidayCalibrator._compute_day_ratios(
                df_daily, date, baselines
            )
            if ratios:
                all_holiday_ratios.extend(ratios)

        if all_holiday_ratios:
            result['holiday'] = HolidayCalibrator._summarize_ratios(
                all_holiday_ratios, holiday_start, label='holiday_days'
            )

        # ── POST-HOLIDAY ratios ──
        for offset in range(1, post_window + 1):
            date = holiday_end + datetime.timedelta(days=offset)
            ratios, per_res = HolidayCalibrator._compute_day_ratios(
                df_daily, date, baselines
            )
            if ratios:
                result['post'][str(offset)] = HolidayCalibrator._summarize_ratios(
                    ratios, date
                )
                for res_code, ratio in per_res.items():
                    result['per_restaurant'].setdefault(res_code, {})
                    result['per_restaurant'][res_code][f'post_{offset}'] = round(ratio, 3)

        return result

    @staticmethod
    def _compute_baselines(df_daily, exclude_start, exclude_end, vn_holidays):
        """
        Compute normal daily guest baseline per (restaurant, weekday).
        Uses data OUTSIDE the exclude zone and non-holiday days.

        Returns:
            Dict[(res_code, weekday_int), float]: median daily guests
        """
        # Filter: outside exclude zone, not a holiday
        mask = ~(
            (df_daily['date'] >= exclude_start) &
            (df_daily['date'] <= exclude_end)
        )
        df_normal = df_daily[mask].copy()

        # Remove holiday dates
        df_normal = df_normal[
            ~df_normal['date'].apply(lambda d: d in vn_holidays)
        ]

        if df_normal.empty:
            return {}

        df_normal['weekday'] = df_normal['date'].apply(lambda d: d.weekday())

        baselines = {}
        for (res_code, weekday), group in df_normal.groupby(
            ['restaurant_code', 'weekday']
        ):
            if len(group) >= HolidayCalibrator.MIN_BASELINE_DAYS:
                median_val = group['guest_count'].median()
                if median_val >= HolidayCalibrator.MIN_BASELINE_GUESTS:
                    baselines[(res_code, weekday)] = float(median_val)

        return baselines

    @staticmethod
    def _compute_day_ratios(df_daily, date, baselines):
        """
        Compute actual/baseline ratio for each restaurant on a given date.

        Returns:
            (list_of_ratios, dict_per_restaurant)
            Returns ([], {}) if date is in the future or has no data at all.
        """
        # [FIX] Skip future dates entirely - they have NO data, not because they are closed
        # This prevents POST-holiday offsets for future periods from getting impact=0
        today = datetime.date.today()
        if date > today:
            return [], {}

        weekday = date.weekday()
        df_day = df_daily[df_daily['date'] == date]

        # [FIX] If no restaurants have ANY data for this date, it means it's outside
        # the loaded data range (not that everyone was closed) → skip it
        if df_day.empty:
            return [], {}

        ratios = []
        per_restaurant = {}

        for _, row in df_day.iterrows():
            res_code = str(row['restaurant_code'])
            actual = float(row['guest_count'])
            baseline = baselines.get((row['restaurant_code'], weekday))

            if baseline is None:
                continue

            ratio = actual / baseline
            ratios.append(ratio)
            per_restaurant[res_code] = ratio

        # Restaurants with baseline but NO data on this date → ratio = 0 (CLOSED)
        # Only do this when there IS some data for the date (ie, not a future/missing date)
        restaurants_with_baseline = set(
            rc for (rc, wd) in baselines.keys() if wd == weekday
        )
        restaurants_with_data = set(df_day['restaurant_code'].unique())
        missing = restaurants_with_baseline - restaurants_with_data

        for res_code in missing:
            ratios.append(0.0)
            per_restaurant[str(res_code)] = 0.0

        return ratios, per_restaurant

    @staticmethod
    def _summarize_ratios(ratios, date, label=None):
        """Summarize a list of ratios into statistics."""
        arr = np.array(ratios)
        open_ratios = arr[arr >= 0.1]  # Restaurants that were open

        summary = {
            'impact_all': round(float(np.median(arr)), 3),
            'impact_open': round(float(np.median(open_ratios)), 3) if len(open_ratios) > 0 else None,
            'mean_all': round(float(np.mean(arr)), 3),
            'std': round(float(np.std(arr)), 3),
            'n_total': len(arr),
            'n_open': int(np.sum(arr >= 0.1)),
            'n_closed': int(np.sum(arr < 0.1)),
            'pct_closed': round(float(np.sum(arr < 0.1)) / max(len(arr), 1) * 100, 1),
            'p25': round(float(np.percentile(arr, 25)), 3),
            'p75': round(float(np.percentile(arr, 75)), 3),
            'date': str(date),
            'weekday': date.strftime('%A') if hasattr(date, 'strftime') else '',
        }
        # Use impact_open for actual impact (closed handled separately)
        summary['impact'] = summary['impact_open'] if summary['impact_open'] else summary['impact_all']
        return summary

    # ==========================================
    # AGGREGATION (MULTI-YEAR)
    # ==========================================

    @staticmethod
    def _compute_aggregate(periods_data):
        """
        Merge calibration results from multiple years into final aggregate.
        Uses weighted average by n_total (more data = more weight).
        
        [FIX] Filters out future periods (is_future=True) and periods where
        pct_closed=100 for holiday days due to missing future data.
        """
        # [FIX] Filter out future periods before aggregating
        today = datetime.date.today()
        valid_periods = []
        for p in periods_data:
            # Skip if explicitly marked as future
            if p.get('is_future', False):
                continue
            # Skip if period start is in the future (belt-and-suspenders)
            period_start = p.get('period_start', '')
            try:
                ps_date = datetime.date.fromisoformat(period_start)
                if ps_date > today:
                    logger.info(f"   ⏭️ Skipping future period {period_start} from aggregate")
                    continue
            except (ValueError, TypeError):
                pass
            # Skip if all holiday days show 100% closed (all future, no real data)
            hol = p.get('holiday', {})
            if hol and hol.get('pct_closed', 0) >= 99.0 and hol.get('impact_open') is None:
                logger.info(
                    f"   ⏭️ Skipping period {period_start}: "
                    f"holiday shows 100% closed (no data, future period)"
                )
                continue
            valid_periods.append(p)
        
        if not valid_periods:
            # All periods were future/invalid — return sensible defaults
            logger.warning("   No valid historical periods for aggregate, using defaults")
            return {'pre': {}, 'holiday': 1.0, 'post': {}}
        
        periods_data = valid_periods

        if len(periods_data) == 1:
            p = periods_data[0]
            return {
                'pre': {k: v.get('impact', 1.0) for k, v in p['pre'].items()},
                'holiday': p['holiday'].get('impact', 0.1) if p['holiday'] else 0.1,
                'post': {k: v.get('impact', 1.0) for k, v in p['post'].items()},
                'detail': {
                    'pre': p['pre'],
                    'holiday': p['holiday'],
                    'post': p['post'],
                },
            }

        # Weighted average across years
        agg_pre = {}
        agg_post = {}
        holiday_impacts = []
        holiday_weights = []

        for p in periods_data:
            for offset, stats in p['pre'].items():
                if offset not in agg_pre:
                    agg_pre[offset] = {'impacts': [], 'weights': []}
                agg_pre[offset]['impacts'].append(stats.get('impact', 1.0))
                agg_pre[offset]['weights'].append(stats.get('n_total', 1))

            if p['holiday']:
                holiday_impacts.append(p['holiday'].get('impact', 0.1))
                holiday_weights.append(p['holiday'].get('n_total', 1))

            for offset, stats in p['post'].items():
                if offset not in agg_post:
                    agg_post[offset] = {'impacts': [], 'weights': []}
                agg_post[offset]['impacts'].append(stats.get('impact', 1.0))
                agg_post[offset]['weights'].append(stats.get('n_total', 1))

        def weighted_avg(impacts, weights):
            total_w = sum(weights)
            if total_w == 0:
                return np.mean(impacts)
            return sum(i * w for i, w in zip(impacts, weights)) / total_w

        return {
            'pre': {
                k: round(weighted_avg(v['impacts'], v['weights']), 3)
                for k, v in agg_pre.items()
            },
            'holiday': round(
                weighted_avg(holiday_impacts, holiday_weights), 3
            ) if holiday_impacts else 0.1,
            'post': {
                k: round(weighted_avg(v['impacts'], v['weights']), 3)
                for k, v in agg_post.items()
            },
            'detail': {
                'pre': periods_data[-1]['pre'],
                'holiday': periods_data[-1]['holiday'],
                'post': periods_data[-1]['post'],
            },
        }

    # ==========================================
    # PERSISTENCE
    # ==========================================

    @staticmethod
    def save_calibration(calibration):
        """Save calibration to JSON file (including per-restaurant data)."""
        cal_save = json.loads(json.dumps(calibration, default=str))
        
        # Build per-restaurant aggregate from all periods
        per_res_aggregate = {}
        for h_type, data in cal_save.get('holiday_types', {}).items():
            for period in data.get('periods', []):
                per_res = period.get('per_restaurant', {})
                for res_code, impacts in per_res.items():
                    per_res_aggregate.setdefault(res_code, {})
                    per_res_aggregate[res_code].update(impacts)
                # Remove from period (keep only aggregate)
                period.pop('per_restaurant', None)
        
        # Store per-restaurant aggregate
        if per_res_aggregate:
            for h_type in cal_save.get('holiday_types', {}):
                cal_save['holiday_types'][h_type]['per_restaurant'] = per_res_aggregate
                n_res = len(per_res_aggregate)
                logger.info(f"   🏠 Per-restaurant data: {n_res} restaurants")
        
        with open(HolidayCalibrator.CALIBRATION_FILE, 'w', encoding='utf-8') as f:
            json.dump(cal_save, f, ensure_ascii=False, indent=2)

        logger.info(f"\n💾 Calibration saved: {HolidayCalibrator.CALIBRATION_FILE}")

    @staticmethod
    def load_calibration():
        """Load calibration from JSON file."""
        if not HolidayCalibrator.CALIBRATION_FILE.exists():
            return None
        try:
            with open(HolidayCalibrator.CALIBRATION_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load calibration: {e}")
            return None

    # ==========================================
    # APPLY TO FORECAST
    # ==========================================

    @staticmethod
    def apply_to_forecast_days(next_days, calibration=None):
        """
        Update next_days list with calibrated holiday impacts.
        Replaces hardcoded values with data-driven ones.

        Args:
            next_days: List[Dict] from build_forecast_days()
            calibration: Dict or None (will load from file)

        Returns:
            List[Dict]: Updated next_days with calibrated impacts
        """
        if calibration is None:
            calibration = HolidayCalibrator.load_calibration()

        if not calibration:
            logger.debug("No calibration data available, using defaults")
            return next_days

        updated_count = 0

        for d in next_days:
            h_type = d.get('holiday_type')
            pre_post = d.get('pre_post_type')
            old_impact = d.get('holiday_impact', 1.0)
            new_impact = None

            # Case 1: Actual holiday day
            if d.get('is_holiday') and h_type:
                # [FIX] LABOR_DAY (1/5) is always part of LIBERATION_DAY period (30/4-1/5)
                # Calibration tracks them as one period under LIBERATION_DAY
                lookup_type = h_type
                if h_type == 'LABOR_DAY' and 'LABOR_DAY' not in calibration.get('holiday_types', {}):
                    lookup_type = 'LIBERATION_DAY'  # Use same calibration
                    
                ht_data = calibration.get('holiday_types', {}).get(lookup_type, {})
                agg = ht_data.get('aggregate', {})
                if 'holiday' in agg:
                    hol_val = agg['holiday']
                    new_impact = hol_val if isinstance(hol_val, (int, float)) else hol_val.get('impact')

            # Case 2: Pre-holiday
            elif d.get('is_pre_holiday') and pre_post:
                # Determine holiday type from pre_post_type
                if pre_post == 'PRE_TET':
                    target_type = 'TET_NGUYEN_DAN'
                else:
                    target_type = None  # Generic pre-holiday

                if target_type:
                    ht_data = calibration.get('holiday_types', {}).get(target_type, {})
                    agg = ht_data.get('aggregate', {})
                    # Find offset
                    offset = HolidayCalibrator._find_pre_offset(d, next_days)
                    if offset and str(offset) in agg.get('pre', {}):
                        new_impact = agg['pre'][str(offset)]

            # Case 3: Post-holiday
            elif d.get('is_post_holiday') and pre_post:
                if pre_post == 'POST_TET':
                    target_type = 'TET_NGUYEN_DAN'
                else:
                    target_type = None

                if target_type:
                    ht_data = calibration.get('holiday_types', {}).get(target_type, {})
                    agg = ht_data.get('aggregate', {})
                    offset = HolidayCalibrator._find_post_offset(d, next_days)
                    if offset and str(offset) in agg.get('post', {}):
                        new_impact = agg['post'][str(offset)]

            # Apply calibrated impact
            if new_impact is not None and new_impact != old_impact:
                d['holiday_impact'] = round(new_impact, 3)
                d['holiday_impact_source'] = 'calibrated'
                d['holiday_impact_default'] = old_impact
                updated_count += 1
            else:
                d['holiday_impact_source'] = 'default'

        if updated_count > 0:
            logger.info(
                f"   📐 Calibration applied: {updated_count}/{len(next_days)} "
                f"days updated with data-driven impacts"
            )

        return next_days

    @staticmethod
    def _find_pre_offset(day_info, next_days):
        """Find how many days before the next holiday this day is."""
        target_date = day_info['date']
        for d in next_days:
            if d.get('is_holiday') and d.get('holiday_type'):
                diff = (d['date'] - target_date).days
                if 0 < diff <= 14:
                    return diff
        return None

    @staticmethod
    def _find_post_offset(day_info, next_days):
        """Find how many days after the last holiday this day is."""
        target_date = day_info['date']
        last_holiday = None
        for d in next_days:
            if d.get('is_holiday') and d.get('holiday_type'):
                if d['date'] < target_date:
                    last_holiday = d['date']
        if last_holiday:
            return (target_date - last_holiday).days
        return None

    @staticmethod
    def get_calibrated_impact(holiday_type, offset, direction='pre', res_code=None):
        """
        Get calibrated impact value.
        Supports per-restaurant lookup with fallback to aggregate.

        Args:
            holiday_type: e.g. 'TET_NGUYEN_DAN'
            offset: int (days from holiday)
            direction: 'pre', 'post', or 'holiday'
            res_code: Optional restaurant code for per-restaurant calibration

        Returns:
            float or None
        """
        cal = HolidayCalibrator.load_calibration()
        if not cal:
            return None

        ht = cal.get('holiday_types', {}).get(holiday_type, {})
        agg = ht.get('aggregate', {})
        
        # [PER-RESTAURANT] Try restaurant-specific first
        if res_code:
            per_res = ht.get('per_restaurant', {})
            res_data = per_res.get(str(res_code), {})
            if res_data:
                key = f"{direction}_{offset}" if direction != 'holiday' else 'holiday_0'
                if key in res_data:
                    return res_data[key]

        # Fallback to aggregate
        if direction == 'holiday':
            hol = agg.get('holiday')
            return hol if isinstance(hol, (int, float)) else (hol.get('impact') if hol else None)

        return agg.get(direction, {}).get(str(offset))
    
    @staticmethod
    def get_per_restaurant_impact(res_code, holiday_type='TET_NGUYEN_DAN'):
        """
        Get all calibrated impacts for a specific restaurant.
        
        Returns:
            Dict with pre/post offsets and their impacts, or None
        """
        cal = HolidayCalibrator.load_calibration()
        if not cal:
            return None
        
        ht = cal.get('holiday_types', {}).get(holiday_type, {})
        per_res = ht.get('per_restaurant', {})
        return per_res.get(str(res_code))

    # ==========================================
    # REPORTING
    # ==========================================

    @staticmethod
    def print_report(calibration):
        """Print a formatted calibration report."""
        logger.info("\n" + "=" * 65)
        logger.info("📐 CALIBRATION REPORT")
        logger.info("=" * 65)

        for h_type, data in calibration.get('holiday_types', {}).items():
            logger.info(f"\n🎌 {h_type}")
            logger.info(f"   Years analyzed: {calibration.get('years_analyzed', [])}")

            agg = data.get('aggregate', {})
            detail = agg.get('detail', {})

            # Pre-holiday
            pre_data = detail.get('pre', agg.get('pre', {}))
            if pre_data:
                logger.info(f"\n   📅 PRE-{h_type}:")
                logger.info(f"   {'Offset':>8} {'Date':>12} {'Day':>10} "
                           f"{'Impact':>8} {'Open%':>6} {'N':>5} {'vs Default':>12}")
                logger.info(f"   {'─' * 65}")

                for offset in sorted(pre_data.keys(), key=lambda x: int(x)):
                    stats = pre_data[offset]
                    if isinstance(stats, dict):
                        impact = stats.get('impact', '?')
                        date_str = stats.get('date', '')[:10]
                        weekday = stats.get('weekday', '')[:3]
                        n = stats.get('n_total', 0)
                        pct_open = 100 - stats.get('pct_closed', 0)

                        # Compare to default
                        off_int = int(offset)
                        if h_type == 'TET_NGUYEN_DAN':
                            if off_int <= 3:
                                default = 1.25 - (off_int - 1) * 0.05
                            elif off_int <= 7:
                                default = 0.85 + (off_int - 4) * 0.033
                            else:
                                default = 0.90 + (off_int - 8) * 0.01
                        else:
                            default = 1.0

                        diff = (impact - default) * 100 if isinstance(impact, (int, float)) else 0

                        logger.info(
                            f"   {'-' + offset + 'd':>8} {date_str:>12} {weekday:>10} "
                            f"{impact:>8.3f} {pct_open:>5.0f}% {n:>5} "
                            f"{'↑' if diff > 0 else '↓'}{abs(diff):>+5.1f}pp"
                        )

            # Holiday days
            hol_data = detail.get('holiday', agg.get('holiday'))
            if hol_data:
                if isinstance(hol_data, dict):
                    logger.info(f"\n   🧧 HOLIDAY DAYS:")
                    logger.info(
                        f"   Impact(all): {hol_data.get('impact_all', '?'):.3f}  "
                        f"Impact(open): {hol_data.get('impact_open', '?')}  "
                        f"Closed: {hol_data.get('pct_closed', '?'):.0f}%  "
                        f"N: {hol_data.get('n_total', '?')}"
                    )
                    default_hol = HOLIDAY_TYPES.get(h_type, {}).get('default_factor', 1.0)
                    logger.info(f"   Default was: {default_hol:.2f}")

            # Post-holiday
            post_data = detail.get('post', agg.get('post', {}))
            if post_data:
                logger.info(f"\n   📅 POST-{h_type}:")
                logger.info(f"   {'Offset':>8} {'Date':>12} {'Day':>10} "
                           f"{'Impact':>8} {'Open%':>6} {'N':>5} {'vs Default':>12}")
                logger.info(f"   {'─' * 65}")

                for offset in sorted(post_data.keys(), key=lambda x: int(x)):
                    stats = post_data[offset]
                    if isinstance(stats, dict):
                        impact = stats.get('impact', '?')
                        date_str = stats.get('date', '')[:10]
                        weekday = stats.get('weekday', '')[:3]
                        n = stats.get('n_total', 0)
                        pct_open = 100 - stats.get('pct_closed', 0)

                        off_int = int(offset)
                        if h_type == 'TET_NGUYEN_DAN':
                            if off_int <= 2:
                                default = 0.50 + (off_int - 1) * 0.05
                            elif off_int <= 5:
                                default = 0.65 + (off_int - 3) * 0.075
                            else:
                                default = 0.85 + (off_int - 6) * 0.05
                        else:
                            default = 0.90

                        diff = (impact - default) * 100 if isinstance(impact, (int, float)) else 0

                        logger.info(
                            f"   {'+' + offset + 'd':>8} {date_str:>12} {weekday:>10} "
                            f"{impact:>8.3f} {pct_open:>5.0f}% {n:>5} "
                            f"{'↑' if diff > 0 else '↓'}{abs(diff):>+5.1f}pp"
                        )

        logger.info("\n" + "=" * 65)
