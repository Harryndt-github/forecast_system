"""
==============================================
FORECAST BRAIN - CLOSED-LOOP SELF-CORRECTION
==============================================
Bộ não trung tâm của hệ thống forecast.

Core Philosophy:
    Model dự đoán sai → Brain ghi nhớ sai ở đâu → Lần sau tự sửa.
    
Architecture:
    1. ERROR MEMORY: Lưu patterns lỗi per restaurant (bias, weekday, hourly, seasonal)
    2. ISSUE DETECTOR: Phân tích nguyên nhân chênh lệch >25% MAPE
    3. AUTO CORRECTOR: Áp dụng correction trước khi lưu kết quả
    4. STRATEGY OPTIMIZER: Tự chọn strategy tối ưu per restaurant
    5. INSIGHT GENERATOR: Sinh báo cáo "tại sao model sai"

Memory Structure (brain_memory.json):
    {
        "version": 2,
        "last_updated": "2026-02-08",
        "global_patterns": {
            "holiday_overpredict_pct": 12.5,
            "weekend_bias": -3.2,
            "seasonal_factors": {...}
        },
        "restaurants": {
            "R001": {
                "overall_bias": +3.2,
                "correction_factor": 0.92,
                "weekday_bias": {"Monday": -2.1, ...},
                "hourly_bias": {"10": +1.5, ...},
                "mape_history": [45, 38, 32, 28],
                "best_strategy": "ML_PRIMARY_AI_VALIDATE",
                "ml_mape": 30, "ai_mape": 45,
                "issues": [
                    {"date": "2026-01-20", "type": "HOLIDAY_SPIKE", "error_pct": 35},
                    ...
                ],
                "learned_at": "2026-02-08T10:00:00"
            }
        },
        "learning_log": [
            {"date": "2026-02-08", "action": "bias_correction", "details": "..."},
        ]
    }
"""

import os
import json
import datetime
import traceback
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from forecast_system.config.settings import (
    CURRENT_DATE, PROJECT_ROOT, STRATEGY_WEIGHTS, MONITORING_CONFIG,
)
from forecast_system.utils.logger import get_logger

logger = get_logger('forecast_brain')


class ForecastBrain:
    """
    Bộ não tự học của hệ thống forecast.
    
    Workflow mỗi lần chạy pipeline:
    
    1. LEARN  (sau mỗi lần có actual data mới)
       brain.learn_from_errors(df_master)
       → Phân tích errors, ghi nhớ bias patterns, detect issues
    
    2. CORRECT (trước khi lưu forecast mới)
       corrected = brain.apply_corrections(predictions, res_code)
       → Trừ bias, điều chỉnh theo weekday, hourly patterns
    
    3. RECOMMEND (trước khi chọn strategy)
       strategy = brain.get_optimal_strategy(res_code)
       → Dựa trên history, recommend ML_PRIMARY vs AI_PRIMARY
    
    4. DIAGNOSE (khi cần debug)
       report = brain.diagnose_restaurant(res_code, df_master)
       → Báo cáo chi tiết tại sao model sai
    """
    
    # File paths
    BRAIN_FILE = str(PROJECT_ROOT / "brain_memory.json")
    
    # Thresholds
    MAPE_TARGET = 25.0           # Target: MAPE < 25%
    SIGNIFICANT_BIAS = 2.0       # [FIX A3] Raised from 1.0 → 2.0 (less aggressive bias correction)
    MIN_SAMPLES_LEARN = 10       # Tối thiểu 10 samples để học
    MAX_CORRECTION = 0.35        # [FIX A3] Reduced from 0.60 → 0.35 (prevent feedback loop)
    MAX_CORRECTION_SPECIAL_EVENT = 0.50  # [FIX A3] Reduced from 0.80 → 0.50
    BIAS_SMOOTHING = 0.3         # [FIX A3] Reduced from 0.5 → 0.3 (smoother corrections)
    ISSUE_RETENTION_DAYS = 60    # Giữ issues 60 ngày

    # ==========================================
    # MEMORY MANAGEMENT
    # ==========================================
    
    @staticmethod
    def load_memory() -> Dict:
        """Load brain memory từ JSON file."""
        if os.path.exists(ForecastBrain.BRAIN_FILE):
            try:
                with open(ForecastBrain.BRAIN_FILE, 'r', encoding='utf-8') as f:
                    memory = json.load(f)
                return memory
            except Exception as e:
                logger.warning(f"Failed to load brain memory: {e}")
        
        return ForecastBrain._create_empty_memory()
    
    @staticmethod
    def save_memory(memory: Dict):
        """Save brain memory to JSON file."""
        memory['last_updated'] = str(CURRENT_DATE)
        memory['version'] = 2
        
        try:
            # Atomic write: tmp file + os.replace so a crash mid-write
            # cannot corrupt/erase the learned memory.
            tmp_file = ForecastBrain.BRAIN_FILE + '.tmp'
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(memory, f, indent=2, default=str)
            os.replace(tmp_file, ForecastBrain.BRAIN_FILE)
            logger.info(f"🧠 Brain memory saved ({len(memory.get('restaurants', {}))} restaurants)")
        except Exception as e:
            logger.error(f"Failed to save brain memory: {e}")
    
    @staticmethod
    def _create_empty_memory() -> Dict:
        return {
            'version': 2,
            'last_updated': str(CURRENT_DATE),
            'global_patterns': {
                'holiday_bias_pct': 0,
                'weekend_bias': 0,
                'weekday_biases': {},
                'hourly_biases': {},
                'overall_mape_trend': [],
                'seasonal_trends': {},  # Global monthly averages: {'1': avg_guests, '2': ..., '12': avg_guests}
            },
            'restaurants': {},
            'learning_log': [],
        }
    
    @staticmethod
    def _get_restaurant_memory(memory: Dict, res_code: str) -> Dict:
        """Get or create restaurant-specific memory."""
        if res_code not in memory['restaurants']:
            memory['restaurants'][res_code] = {
                'overall_bias': 0.0,
                'correction_factor': 1.0,
                'weekday_bias': {},
                'shift_bias': {},
                'weekday_shift_bias': {},
                'hourly_bias': {},
                'holiday_bias': 0.0,
                'event_biases': {},    # Per-event-type bias: {'VALENTINE': 5.2, 'TET_NGUYEN_DAN': -12.3, ...}
                'actual_ratio_anchor': None,  # Learned actual/predicted median ratio
                'mape_history': [],
                'hit_rate_history': [],  # Track hit rate over time for trend detection
                'consecutive_hr_drops': 0,  # Counter for consecutive Hit Rate drops
                'best_strategy': None,
                'ml_mape': None,
                'ai_mape': None,
                'issues': [],
                'correction_count': 0,
                'last_mape': None,
                'last_hit_rate': None,
                'learned_at': None,
                'seasonal_memory': {},  # Monthly patterns: {'1': {'avg_actual': 50, 'avg_predicted': 55, 'bias': 5}, ...}
                'preferred_granularity': 'mixed',
            }
        # Migrate old memory format
        res = memory['restaurants'][res_code]
        if 'shift_bias' not in res:
            res['shift_bias'] = {}
        if 'weekday_shift_bias' not in res:
            res['weekday_shift_bias'] = {}
        if 'event_biases' not in res:
            res['event_biases'] = {}
        if 'actual_ratio_anchor' not in res:
            res['actual_ratio_anchor'] = None
        # Migrate: add hit_rate tracking fields
        if 'hit_rate_history' not in res:
            res['hit_rate_history'] = []
        if 'consecutive_hr_drops' not in res:
            res['consecutive_hr_drops'] = 0
        if 'last_hit_rate' not in res:
            res['last_hit_rate'] = None
        if 'seasonal_memory' not in res:
            res['seasonal_memory'] = {}
        if 'preferred_granularity' not in res:
            res['preferred_granularity'] = 'mixed'
        # ⭐ Memento: Skill Utility Scoring
        if 'skill_scores' not in res:
            res['skill_scores'] = {}   # {model_name: {'mape': float, 'utility': float, 'samples': int}}
        if 'dynamic_weights' not in res:
            res['dynamic_weights'] = {}  # {'ml': float, 'ai': float} – override STRATEGY_WEIGHTS
        # ⭐ Memento: Direction-Aware Contextual Bias
        if 'contextual_bias' not in res:
            res['contextual_bias'] = {}  # key: '{weekday}__{shift}__holiday|normal' → bias
        if 'bias_direction' not in res:
            res['bias_direction'] = 'UNKNOWN'   # CONSISTENT_OVER / CONSISTENT_UNDER / MIXED
        if 'segment_corrections' not in res:
            res['segment_corrections'] = {}  # key: SHIFT|DAY_TYPE|SIZE_GROUP
        return res

    @staticmethod
    def _select_learning_rows(df_res: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """
        Chọn đúng granularity để học.

        Khi nhà hàng đã có dữ liệu shift-based thì ưu tiên CHỈ học từ shift rows,
        tránh trộn legacy hourly rows với MORNING/EVENING rows trong cùng memory.
        """
        if df_res.empty:
            return df_res.copy(), 'empty'

        shift_series = (
            df_res['Shift']
            if 'Shift' in df_res.columns
            else pd.Series(index=df_res.index, dtype='object')
        )
        shift_mask = shift_series.isin(['MORNING', 'EVENING'])

        if 'Forecast_Mode' in df_res.columns:
            mode_series = df_res['Forecast_Mode'].astype(str)
            shift_mode_mask = shift_mask & mode_series.eq('shift')
            if int(shift_mode_mask.sum()) >= ForecastBrain.MIN_SAMPLES_LEARN:
                return df_res[shift_mode_mask].copy(), 'shift'

        if int(shift_mask.sum()) >= ForecastBrain.MIN_SAMPLES_LEARN:
            return df_res[shift_mask].copy(), 'shift'

        if 'Hour' in df_res.columns:
            hourly_mask = df_res['Hour'].notna()
            if int(hourly_mask.sum()) >= ForecastBrain.MIN_SAMPLES_LEARN:
                return df_res[hourly_mask].copy(), 'hourly'

        return df_res.copy(), 'mixed'

    # ==========================================
    # 1. LEARN FROM ERRORS
    # ==========================================
    
    @staticmethod
    def learn_from_errors(df_master: pd.DataFrame) -> Dict:
        """
        Phân tích toàn bộ errors từ master file và ghi nhớ patterns.
        
        Học:
        - Per-restaurant bias (over/under predict trung bình)
        - Weekday-specific bias (VD: Thứ 7 luôn over 15%)
        - Hourly bias (VD: giờ 12h luôn under 20%)
        - Holiday effect
        - Trend direction
        - Issues causing MAPE > 25%
        
        Returns:
            Dict: learning summary
        """
        memory = ForecastBrain.load_memory()
        
        if df_master.empty:
            return {'status': 'no_data'}
        
        # Filter valid rows (có cả predicted và actual)
        df = df_master.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        
        mask = (
            pd.notna(df['Final_Predicted_Guests']) &
            pd.notna(df['Actual_Guest']) &
            (df['Actual_Guest'] >= 0) &
            (df['Final_Predicted_Guests'] >= 0)
        )
        df_valid = df[mask].copy()
        
        if len(df_valid) < ForecastBrain.MIN_SAMPLES_LEARN:
            return {'status': 'insufficient_data', 'samples': len(df_valid)}
        
        # Calculate errors
        df_valid['error'] = df_valid['Final_Predicted_Guests'] - df_valid['Actual_Guest']
        df_valid['abs_error'] = df_valid['error'].abs()  # type: ignore[reportAttributeAccessIssue]
        df_valid['pct_error'] = np.where(
            df_valid['Actual_Guest'] > 0,
            (df_valid['abs_error'] / df_valid['Actual_Guest']) * 100,
            np.nan
        )
        df_valid['signed_pct'] = np.where(
            df_valid['Actual_Guest'] > 0,
            (df_valid['error'] / df_valid['Actual_Guest']) * 100,
            np.nan
        )
        
        summary = {
            'status': 'learned',
            'total_samples': len(df_valid),
            'restaurants_learned': 0,
            'issues_found': 0,
            'corrections_updated': 0,
        }
        
        # --- Learn Global Patterns ---
        ForecastBrain._learn_global_patterns(memory, df_valid)  # type: ignore[reportArgumentType]
        
        # --- Learn Per-Restaurant ---
        for res_code, df_res_all in df_valid.groupby('Restaurant_Code'):
            df_res, granularity = ForecastBrain._select_learning_rows(df_res_all)
            if len(df_res) < ForecastBrain.MIN_SAMPLES_LEARN:
                continue
            
            res_mem = ForecastBrain._get_restaurant_memory(memory, res_code)  # type: ignore[reportArgumentType]
            res_mem['preferred_granularity'] = granularity
            before_segment_corrections = copy.deepcopy(res_mem.get('segment_corrections', {}))
            
            # Learn bias
            ForecastBrain._learn_restaurant_bias(res_mem, df_res)  # type: ignore[reportArgumentType]
            
            # Learn weekday patterns
            ForecastBrain._learn_weekday_bias(res_mem, df_res)  # type: ignore[reportArgumentType]

            # Learn shift-aware patterns for MORNING/EVENING forecasts
            ForecastBrain._learn_shift_bias(res_mem, df_res)  # type: ignore[reportArgumentType]
            ForecastBrain._learn_weekday_shift_bias(res_mem, df_res)  # type: ignore[reportArgumentType]
            
            # Learn hourly patterns only for legacy hourly forecasts
            if granularity != 'shift':
                ForecastBrain._learn_hourly_bias(res_mem, df_res)  # type: ignore[reportArgumentType]
            
            # Learn holiday effect
            ForecastBrain._learn_holiday_bias(res_mem, df_res)  # type: ignore[reportArgumentType]
            
            # Learn event-specific biases (Valentine, 8/3, Tet, etc.)
            ForecastBrain._learn_event_biases(res_mem, df_res)  # type: ignore[reportArgumentType]
            
            # Learn seasonal (monthly) trends — 1-year memory
            ForecastBrain._learn_seasonal_trends(res_mem, df_res)  # type: ignore[reportArgumentType]

            # ⭐ Memento: Direction-Aware Contextual Bias
            ForecastBrain._learn_contextual_bias(res_mem, df_res)  # type: ignore[reportArgumentType]

            # Validated Learning Loop: segment-level correction rules
            ForecastBrain._learn_segment_corrections(res_mem, df_res)  # type: ignore[reportArgumentType]

            # ⭐ Memento: Skill Utility Scores (per-model MAPE)
            ForecastBrain._learn_skill_scores(res_mem, df_res)  # type: ignore[reportArgumentType]
            
            # Detect issues (high error events)
            issues = ForecastBrain._detect_issues(res_code, df_res)  # type: ignore[reportArgumentType]
            res_mem['issues'] = ForecastBrain._merge_issues(
                res_mem.get('issues', []), issues
            )
            summary['issues_found'] += len(issues)
            
            # Track MAPE history
            # [FIX] Use per-DAY MAPE (not per-row), avoids inflated MAPE from shift/hourly mixing
            daily_agg = df_res.groupby('Date').agg(
                Predicted=('Final_Predicted_Guests', 'sum'),
                Actual=('Actual_Guest', 'sum'),
            ).reset_index()
            daily_agg = daily_agg[daily_agg['Actual'] > 0]
            if not daily_agg.empty:
                daily_mape = ((daily_agg['Predicted'] - daily_agg['Actual']).abs() / daily_agg['Actual'] * 100).mean()  # type: ignore[reportAttributeAccessIssue]
                current_mape = round(float(daily_mape), 1) if pd.notna(daily_mape) else None
            else:
                current_mape = None
            
            if current_mape is not None:
                res_mem['last_mape'] = current_mape
                mape_hist = res_mem.get('mape_history', [])
                mape_hist.append(current_mape)
                res_mem['mape_history'] = mape_hist[-30:]  # Keep last 30
            
            # Calculate correction factor
            ForecastBrain._update_correction_factor(res_mem, df_res)  # type: ignore[reportArgumentType]
            
            # Determine best strategy
            ForecastBrain._determine_best_strategy(res_mem, df_res)  # type: ignore[reportArgumentType]
            
            # ⭐ Directional escalation (replaces generic auto_escalate for consistent bias)
            ForecastBrain._directional_escalate(res_mem)
            # Generic auto-escalate for stagnant MAPE (keeps legacy logic)
            ForecastBrain._auto_escalate(res_mem)
            
            # [FIX #1] Clamp stale escalation levels from old code (now max is 2)
            stale_esc = res_mem.get('escalation_level', 0)
            if stale_esc > 2:
                res_mem['escalation_level'] = 2
            
            # [FIX #3] Reset needs_retune after a full learning cycle
            # The brain has now re-learned correction factors, so retune is done
            if res_mem.get('needs_retune', False):
                res_mem['needs_retune'] = False
                res_mem['retune_completed_at'] = str(CURRENT_DATE)

            # Register changed segment corrections for later validation.
            try:
                from forecast_system.agents.correction_validator import CorrectionValidator
                after_segments = res_mem.get('segment_corrections', {})
                for segment_key, after_state in after_segments.items():
                    before_state = before_segment_corrections.get(segment_key)
                    if before_state == after_state:
                        continue
                    factor_changed = (
                        before_state is None or
                        abs(float(after_state.get('factor', 1.0)) - float(before_state.get('factor', 1.0))) >= 0.015
                    )
                    bias_changed = (
                        before_state is None or
                        abs(float(after_state.get('bias', 0.0)) - float(before_state.get('bias', 0.0))) >= 2.0
                    )
                    if factor_changed or bias_changed:
                        if CorrectionValidator.record_segment_checkpoint(
                            str(res_code),
                            str(segment_key),
                            df_valid,
                            before_state or {},
                            copy.deepcopy(after_state),
                            reason=after_state.get('direction', 'SEGMENT_CORRECTION'),
                        ):
                            summary['corrections_updated'] += 1
            except Exception as _ve:
                logger.warning(f"Segment validation checkpoint skipped for {res_code}: {_ve}")
            
            res_mem['learned_at'] = datetime.datetime.now().isoformat()
            summary['restaurants_learned'] += 1
        
        # Log learning event
        memory['learning_log'].append({
            'date': str(CURRENT_DATE),
            'timestamp': datetime.datetime.now().isoformat(),
            'restaurants_learned': summary['restaurants_learned'],
            'issues_found': summary['issues_found'],
            'total_samples': summary['total_samples'],
        })
        memory['learning_log'] = memory['learning_log'][-90:]
        
        ForecastBrain.save_memory(memory)
        
        logger.info(f"🧠 Brain learned: {summary['restaurants_learned']} restaurants, "
                    f"{summary['issues_found']} issues found")
        
        return summary
    
    @staticmethod
    def _learn_global_patterns(memory: Dict, df: pd.DataFrame):
        """Học patterns chung toàn hệ thống."""
        gp = memory['global_patterns']
        
        # Weekend bias
        if 'Weekday' in df.columns:
            weekend_mask = df['Weekday'].isin(['Saturday', 'Sunday'])
            if weekend_mask.any():  # type: ignore[reportGeneralTypeIssues]
                weekend_bias = df.loc[weekend_mask, 'error'].mean()
                gp['weekend_bias'] = round(float(weekend_bias), 2)
            
            # Per-weekday bias
            wd_bias = df.groupby('Weekday')['error'].mean()
            gp['weekday_biases'] = {
                str(k): round(float(v), 2) for k, v in wd_bias.items()
            }
        
        # Hourly bias
        if 'Hour' in df.columns:
            h_bias = df.groupby('Hour')['error'].mean()
            gp['hourly_biases'] = {
                str(int(k)): round(float(v), 2) for k, v in h_bias.items()  # type: ignore[reportArgumentType]
            }
        
        # Holiday bias
        if 'Is_Holiday' in df.columns:
            hol_mask = df['Is_Holiday'] == True
            if hol_mask.any() and (~hol_mask).any():
                hol_err = df.loc[hol_mask, 'signed_pct'].dropna().mean()
                gp['holiday_bias_pct'] = round(float(hol_err), 1) if pd.notna(hol_err) else 0
        
        # Overall MAPE trend (weekly)
        overall_mape = df['pct_error'].dropna().mean()
        if pd.notna(overall_mape):  # type: ignore[reportGeneralTypeIssues]
            trend = gp.get('overall_mape_trend', [])
            trend.append({
                'date': str(CURRENT_DATE),
                'mape': round(float(overall_mape), 1),
            })
            gp['overall_mape_trend'] = trend[-30:]
        
        # Global seasonal trends (monthly avg across all restaurants)
        if 'Date' in df.columns:
            df_work = df.copy()
            df_work['_month'] = pd.to_datetime(df_work['Date'], errors='coerce').dt.month
            month_stats = df_work.groupby('_month').agg(
                avg_actual=('Actual_Guest', 'mean'),
                avg_predicted=('Final_Predicted_Guests', 'mean'),
                avg_bias=('error', 'mean'),
                count=('error', 'size'),
            )
            seasonal = gp.get('seasonal_trends', {})
            for month_num, row in month_stats.iterrows():
                if pd.isna(month_num):  # type: ignore[reportGeneralTypeIssues]
                    continue
                seasonal[str(int(month_num))] = {  # type: ignore[reportArgumentType]
                    'avg_actual': round(float(row['avg_actual']), 1),
                    'avg_predicted': round(float(row['avg_predicted']), 1),
                    'bias': round(float(row['avg_bias']), 2),
                    'count': int(row['count']),
                }
            gp['seasonal_trends'] = seasonal
    
    @staticmethod
    def _learn_restaurant_bias(res_mem: Dict, df_res: pd.DataFrame):
        """
        Học overall bias cho restaurant.
        Dùng adaptive exponential smoothing:
        - MAPE cao → α cao (học nhanh hơn)
        - MAPE improving → α thấp (giữ ổn định)
        """
        new_bias = df_res['error'].mean()
        if pd.isna(new_bias):  # type: ignore[reportGeneralTypeIssues]
            return
        
        old_bias = res_mem.get('overall_bias', 0)
        alpha = ForecastBrain._adaptive_alpha(res_mem)
        
        # Exponential smoothing: new = α * observed + (1-α) * old
        smoothed_bias = alpha * float(new_bias) + (1 - alpha) * old_bias
        res_mem['overall_bias'] = round(smoothed_bias, 2)
        
        # [FIX LOOP] Track bias history for consistency check in _auto_escalate
        bias_hist = res_mem.get('bias_history', [])
        bias_hist.append(round(smoothed_bias, 2))
        res_mem['bias_history'] = bias_hist[-10:]  # Keep last 10
    
    @staticmethod
    def _learn_weekday_bias(res_mem: Dict, df_res: pd.DataFrame):
        """Học bias theo ngày trong tuần."""
        if 'Weekday' not in df_res.columns:
            return
        
        alpha = ForecastBrain._adaptive_alpha(res_mem)
        old_biases = res_mem.get('weekday_bias', {})
        new_biases = {}
        
        for wd, grp in df_res.groupby('Weekday'):
            if len(grp) < 3:
                continue
            new_val = float(grp['error'].mean())
            old_val = old_biases.get(str(wd), 0)
            smoothed = alpha * new_val + (1 - alpha) * old_val
            new_biases[str(wd)] = round(smoothed, 2)
        
        # Merge with existing (keep old if no new data)
        for wd, val in old_biases.items():
            if wd not in new_biases:
                new_biases[wd] = val
        
        res_mem['weekday_bias'] = new_biases

    @staticmethod
    def _learn_shift_bias(res_mem: Dict, df_res: pd.DataFrame):
        """Học bias theo ca MORNING/EVENING."""
        if 'Shift' not in df_res.columns:
            return

        valid = df_res[df_res['Shift'].isin(['MORNING', 'EVENING'])].copy()
        if valid.empty:
            return

        alpha = ForecastBrain._adaptive_alpha(res_mem)
        old_biases = res_mem.get('shift_bias', {})
        new_biases = {}

        for shift, grp in valid.groupby('Shift'):
            if len(grp) < 3:
                continue
            new_val = float(grp['error'].mean())
            old_val = old_biases.get(str(shift), 0)
            smoothed = alpha * new_val + (1 - alpha) * old_val
            new_biases[str(shift)] = round(smoothed, 2)

        for shift, val in old_biases.items():
            if shift not in new_biases:
                new_biases[shift] = val

        res_mem['shift_bias'] = new_biases

    @staticmethod
    def _learn_weekday_shift_bias(res_mem: Dict, df_res: pd.DataFrame):
        """Học bias theo tổ hợp Weekday × Shift để bắt lỗi cuối tuần ca tối."""
        if 'Weekday' not in df_res.columns or 'Shift' not in df_res.columns:
            return

        valid = df_res[df_res['Shift'].isin(['MORNING', 'EVENING'])].copy()
        if valid.empty:
            return

        alpha = ForecastBrain._adaptive_alpha(res_mem)
        old_biases = res_mem.get('weekday_shift_bias', {})
        new_biases = {}

        for (weekday, shift), grp in valid.groupby(['Weekday', 'Shift']):
            if len(grp) < 3:
                continue
            key = f"{weekday}__{shift}"
            new_val = float(grp['error'].mean())
            old_val = old_biases.get(key, 0)
            smoothed = alpha * new_val + (1 - alpha) * old_val
            new_biases[key] = round(smoothed, 2)

        for key, val in old_biases.items():
            if key not in new_biases:
                new_biases[key] = val

        res_mem['weekday_shift_bias'] = new_biases
    
    @staticmethod
    def _learn_hourly_bias(res_mem: Dict, df_res: pd.DataFrame):
        """Học bias theo giờ."""
        if 'Hour' not in df_res.columns:
            return
        
        alpha = ForecastBrain._adaptive_alpha(res_mem)
        old_biases = res_mem.get('hourly_bias', {})
        new_biases = {}
        
        for hour, grp in df_res.groupby('Hour'):
            if len(grp) < 3:
                continue
            new_val = float(grp['error'].mean())
            old_val = old_biases.get(str(int(hour)), 0)  # type: ignore[reportArgumentType]
            smoothed = alpha * new_val + (1 - alpha) * old_val
            new_biases[str(int(hour))] = round(smoothed, 2)  # type: ignore[reportArgumentType]
        
        for h, val in old_biases.items():
            if h not in new_biases:
                new_biases[h] = val
        
        res_mem['hourly_bias'] = new_biases
    
    @staticmethod
    def _learn_holiday_bias(res_mem: Dict, df_res: pd.DataFrame):
        """Học bias riêng cho ngày lễ."""
        if 'Is_Holiday' not in df_res.columns:
            return
        
        hol_rows = df_res[df_res['Is_Holiday'] == True]
        if len(hol_rows) < 3:
            return
        
        new_bias = float(hol_rows['error'].mean())
        old_bias = res_mem.get('holiday_bias', 0)
        alpha = ForecastBrain._adaptive_alpha(res_mem)
        
        res_mem['holiday_bias'] = round(
            alpha * new_bias + (1 - alpha) * old_bias, 2
        )
    
    @staticmethod
    def _learn_event_biases(res_mem: Dict, df_res: pd.DataFrame):
        """
        Học bias RIÊNG CHO TỪNG LOẠI SỰ KIỆN.
        
        Vd: Valentine model luôn over-predict 20% → event_biases['VALENTINE'] = +8.5
            Tết AI under-predict 30% → event_biases['TET_NGUYEN_DAN'] = -15.2
            8/3 luôn đúng → event_biases['WOMENS_DAY'] ≈ 0
        
        Dùng cột Weekday chứa tên ngày, và Date để match event dates.
        """
        from forecast_system.utils.date_utils import SPECIAL_EVENTS, HOLIDAY_TYPES
        
        alpha = ForecastBrain._adaptive_alpha(res_mem)
        old_event_biases = res_mem.get('event_biases', {})
        new_event_biases = dict(old_event_biases)  # Start from old
        
        # Build event date → event_type mapping from historical data
        dates_in_data = set(df_res['Date'].unique())
        
        # Check SPECIAL_EVENTS
        event_date_map = {}
        for (month, day), event_info in SPECIAL_EVENTS.items():
            event_type = event_info['event_type']
            for year in range(2024, 2027):
                try:
                    ed = datetime.date(year, month, day)
                    if ed in dates_in_data:
                        event_date_map[ed] = event_type
                except ValueError:
                    continue
        
        # Check official HOLIDAYS (Tet, 30/4, 1/5, 2/9, etc.)
        from forecast_system.utils.date_utils import get_vn_holidays, classify_holiday
        vn_hols = get_vn_holidays(years=list(range(2024, 2027)))
        for d in dates_in_data:
            if not isinstance(d, datetime.date):
                continue
            hol_type = classify_holiday(d, vn_hols)
            if hol_type and d not in event_date_map:
                event_date_map[d] = hol_type
        
        # Group errors by event type
        event_errors = {}
        for date_val, event_type in event_date_map.items():
            rows = df_res[df_res['Date'] == date_val]
            if rows.empty:
                continue
            err = rows['error'].mean()
            if pd.notna(err):  # type: ignore[reportGeneralTypeIssues]
                if event_type not in event_errors:
                    event_errors[event_type] = []
                event_errors[event_type].append(float(err))
        
        # Smooth and store
        for event_type, errors in event_errors.items():
            if len(errors) < 1:
                continue
            new_bias = np.mean(errors)
            old_bias = old_event_biases.get(event_type, 0)
            smoothed = alpha * new_bias + (1 - alpha) * old_bias
            new_event_biases[event_type] = round(smoothed, 2)
        
        res_mem['event_biases'] = new_event_biases
    
    @staticmethod
    def _learn_seasonal_trends(res_mem: Dict, df_res: pd.DataFrame):
        """
        Học xu hướng theo tháng (seasonal memory 1 năm).
        
        Ghi nhớ per-month:
        - avg_actual: Trung bình khách thực tế
        - avg_predicted: Trung bình dự đoán
        - bias: Sai lệch trung bình (predicted - actual)
        - count: Số lượng samples
        
        Dùng exponential smoothing để cập nhật mượt mà.
        Khi forecast tháng X, brain sẽ biết tháng X lịch sử có pattern gì.
        """
        if 'Date' not in df_res.columns:
            return
        
        alpha = ForecastBrain._adaptive_alpha(res_mem)
        old_seasonal = res_mem.get('seasonal_memory', {})
        new_seasonal = dict(old_seasonal)
        
        # Extract month from date
        df_work = df_res.copy()
        df_work['_month'] = pd.to_datetime(df_work['Date'], errors='coerce').dt.month
        
        for month, grp in df_work.groupby('_month'):
            if pd.isna(month) or len(grp) < 3:  # type: ignore[reportGeneralTypeIssues]
                continue
            
            month_key = str(int(month))  # type: ignore[reportArgumentType]
            
            # Current month stats
            curr_avg_actual = float(grp['Actual_Guest'].mean())
            curr_avg_predicted = float(grp['Final_Predicted_Guests'].mean())
            curr_bias = float(grp['error'].mean())
            curr_count = len(grp)
            
            # Old values
            old_entry = old_seasonal.get(month_key, {})
            old_avg_actual = old_entry.get('avg_actual', curr_avg_actual)
            old_avg_predicted = old_entry.get('avg_predicted', curr_avg_predicted)
            old_bias = old_entry.get('bias', curr_bias)
            old_count = old_entry.get('count', 0)
            
            # Exponential smoothing
            new_seasonal[month_key] = {
                'avg_actual': round(alpha * curr_avg_actual + (1 - alpha) * old_avg_actual, 1),
                'avg_predicted': round(alpha * curr_avg_predicted + (1 - alpha) * old_avg_predicted, 1),
                'bias': round(alpha * curr_bias + (1 - alpha) * old_bias, 2),
                'count': old_count + curr_count,
                'last_updated': str(CURRENT_DATE),
            }
        
        res_mem['seasonal_memory'] = new_seasonal
    
    @staticmethod
    def _adaptive_alpha(res_mem: Dict) -> float:
        """
        Tính α động dựa trên tình trạng accuracy.
        
        Logic (conservative to prevent feedback loops):
        - MAPE > 80%  → α = 0.5 (learn moderately fast, but stay stable)
        - MAPE > 50%  → α = 0.4
        - MAPE 25-50% → α = 0.3 (standard)
        - MAPE < 25%  → α = 0.2 (good accuracy, prioritize stability)
        
        [FIX LOOP] Max α capped at 0.5 (was 0.8-0.9).
        Removed MAPE-worsening boost (it accelerated the feedback loop).
        """
        last_mape = res_mem.get('last_mape', 0) or 0
        
        # Base alpha from current MAPE (capped at 0.5)
        if last_mape > 80:
            alpha = 0.5
        elif last_mape > 50:
            alpha = 0.4
        elif last_mape > 25:
            alpha = 0.3
        else:
            alpha = 0.2
        
        # [FIX LOOP] Removed: boosting α when MAPE worsens.
        # That was counter-productive — it made the system over-react
        # to temporary spikes, causing oscillation.
        
        return alpha

    # ==========================================
    # ⭐ MEMENTO: SKILL UTILITY SCORING
    # ==========================================

    @staticmethod
    def _learn_skill_scores(res_mem: Dict, df_res: pd.DataFrame):
        """
        Học utility score cho từng model dựa trên cột ML/AI prediction trong master file.

        Cột được track (nếu có trong df_res):
          - ML_Predicted / System_Predicted_Before_Brain → xgboost/ensemble ML MAPE
          - AI_Raw_Daily_Forecast → AI LLM MAPE
          - Prophet_Predicted     → Prophet MAPE (nếu có)

        Utility score = max(0, 1 - MAPE/100), clip [0, 1].
        Dynamic weights được tính từ utility scores.
        """
        alpha = ForecastBrain._adaptive_alpha(res_mem)
        old_scores = res_mem.get('skill_scores', {})
        new_scores = dict(old_scores)

        model_cols = {
            'ml':     ['ML_Predicted', 'System_Predicted_Before_Brain'],
            'ai':     ['AI_Raw_Daily_Forecast'],
            'prophet':['Prophet_Predicted'],
        }

        actual_col = 'Actual_Guest'
        if actual_col not in df_res.columns:
            return

        valid = df_res[df_res[actual_col] > 0].copy()
        if valid.empty:
            return

        for skill_name, col_candidates in model_cols.items():
            col = next((c for c in col_candidates if c in valid.columns), None)
            if col is None:
                continue
            rows = valid.dropna(subset=[col])
            if len(rows) < ForecastBrain.MIN_SAMPLES_LEARN:
                continue

            # Daily MAPE
            daily = rows.groupby('Date').agg(
                Predicted=(col, 'sum'),
                Actual=(actual_col, 'sum'),
            ).reset_index()
            daily = daily[daily['Actual'] > 0]
            if daily.empty:
                continue
            mape = float(((daily['Predicted'] - daily['Actual']).abs() / daily['Actual'] * 100).mean())
            utility = float(max(0.0, min(1.0, 1.0 - mape / 100.0)))

            old_entry = old_scores.get(skill_name, {})
            old_mape    = old_entry.get('mape', mape)
            old_utility = old_entry.get('utility', utility)

            new_scores[skill_name] = {
                'mape':    round(alpha * mape    + (1 - alpha) * old_mape,    2),
                'utility': round(alpha * utility + (1 - alpha) * old_utility, 4),
                'samples': len(daily),
                'updated': str(CURRENT_DATE),
            }

        res_mem['skill_scores'] = new_scores

        # ── Recompute dynamic weights from utility scores ──
        ForecastBrain._recompute_dynamic_weights(res_mem)

    @staticmethod
    def _recompute_dynamic_weights(res_mem: Dict):
        """
        Tính dynamic ML/AI weights từ skill_scores.
        Dùng softmax trên utility để ra weights tổng = 1.
        """
        scores = res_mem.get('skill_scores', {})
        if not scores:
            return

        # We only set ml/ai ratio (prophet is part of ml)
        ml_utility  = scores.get('ml',  {}).get('utility', 0.5)
        ai_utility  = scores.get('ai',  {}).get('utility', 0.3)
        prophet_u   = scores.get('prophet', {}).get('utility', 0.4)

        # Blend prophet into ml utility
        ml_combined = 0.7 * ml_utility + 0.3 * prophet_u

        # Softmax-style normalization
        total = ml_combined + ai_utility
        if total <= 0:
            return

        ml_w = round(ml_combined / total, 3)
        ai_w = round(ai_utility  / total, 3)

        # Clamp to reasonable range [0.15, 0.85]
        ml_w = max(0.15, min(0.85, ml_w))
        ai_w = round(1.0 - ml_w, 3)

        res_mem['dynamic_weights'] = {'ml': ml_w, 'ai': ai_w}

    @staticmethod
    def get_dynamic_weights(res_code: str) -> Optional[Dict]:
        """
        Trả về dynamic weights cho restaurant nếu đã học đủ data.
        Returns None nếu chưa có (fallback về STRATEGY_WEIGHTS).
        """
        memory = ForecastBrain.load_memory()
        res_mem = memory.get('restaurants', {}).get(str(res_code), {})
        dw = res_mem.get('dynamic_weights', {})
        if not dw or abs(dw.get('ml', 0.5) - 0.5) < 0.02:
            return None   # Not enough differentiation yet
        return dw

    # ==========================================
    # ⭐ MEMENTO: DIRECTION-AWARE CONTEXTUAL BIAS
    # ==========================================

    @staticmethod
    def _strict_hit_rate(df: pd.DataFrame) -> float:
        """Strict KPI: small actual <100 uses +/-10 guests, large uses +/-10%."""
        valid = df[df['Actual_Guest'] > 0].copy()
        if valid.empty:
            return 0.0
        abs_error = (valid['Final_Predicted_Guests'] - valid['Actual_Guest']).abs()
        pct_error = abs_error / valid['Actual_Guest']
        small = valid['Actual_Guest'] < 100
        hits = ((small & (abs_error <= 10)) | (~small & (pct_error <= 0.10))).sum()
        return round(float(hits) / len(valid), 4)

    @staticmethod
    def _add_segment_columns(df_res: pd.DataFrame) -> pd.DataFrame:
        df = df_res.copy()
        if 'Shift' in df.columns:
            df['_shift'] = df['Shift'].astype(str)
        else:
            df['_shift'] = 'DAILY'

        if 'Is_Holiday' in df.columns:
            is_holiday = df['Is_Holiday'].fillna(False).astype(bool)
        else:
            is_holiday = pd.Series(False, index=df.index)

        if 'Weekday' in df.columns:
            weekday = df['Weekday'].astype(str)
        elif 'Date' in df.columns:
            weekday = pd.to_datetime(df['Date'], errors='coerce').dt.day_name()
        else:
            weekday = pd.Series('', index=df.index)

        df['_day_type'] = np.where(
            is_holiday,
            'HOLIDAY',
            np.where(weekday.isin(['Saturday', 'Sunday']), 'WEEKEND', 'WEEKDAY'),
        )
        df['_size_group'] = np.where(df['Actual_Guest'] >= 100, 'LARGE', 'SMALL')
        df['_segment_key'] = df['_shift'] + '|' + df['_day_type'] + '|' + df['_size_group']
        return df

    @staticmethod
    def _learn_segment_corrections(res_mem: Dict, df_res: pd.DataFrame):
        """
        Learn correction rules by shift/day-type/size. These rules are stored
        separately from global correction_factor so validation can confirm or
        rollback each segment independently.
        """
        required = {'Actual_Guest', 'Final_Predicted_Guests'}
        if not required.issubset(df_res.columns):
            return

        valid = df_res.dropna(subset=['Actual_Guest', 'Final_Predicted_Guests']).copy()
        valid = valid[(valid['Actual_Guest'] > 0) & (valid['Final_Predicted_Guests'] > 0)]
        if len(valid) < ForecastBrain.MIN_SAMPLES_LEARN:
            return

        valid = ForecastBrain._add_segment_columns(valid)
        alpha = ForecastBrain._adaptive_alpha(res_mem)
        segment_mem = res_mem.get('segment_corrections', {})
        learned = dict(segment_mem)

        for segment_key, grp in valid.groupby('_segment_key'):
            if len(grp) < 5:
                continue

            bias = float((grp['Final_Predicted_Guests'] - grp['Actual_Guest']).mean())
            ratios = (grp['Actual_Guest'] / grp['Final_Predicted_Guests']).replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if ratios.empty:
                continue

            observed_factor = float(ratios.median())
            size_group = str(segment_key).split('|')[-1]
            if size_group == 'LARGE':
                observed_factor = max(0.75, min(1.45, observed_factor))
            else:
                observed_factor = max(0.85, min(1.25, observed_factor))

            old = segment_mem.get(str(segment_key), {})
            old_factor = old.get('factor', 1.0)
            old_bias = old.get('bias', 0.0)
            new_factor = alpha * observed_factor + (1 - alpha) * old_factor
            new_bias = alpha * bias + (1 - alpha) * old_bias

            mape = float((grp['Final_Predicted_Guests'] - grp['Actual_Guest']).abs().div(grp['Actual_Guest']).mean() * 100)
            hit_rate = ForecastBrain._strict_hit_rate(grp)
            direction = 'MIXED'
            if new_bias > 2:
                direction = 'CONSISTENT_OVER'
            elif new_bias < -2:
                direction = 'CONSISTENT_UNDER'

            learned[str(segment_key)] = {
                'factor': round(float(new_factor), 4),
                'bias': round(float(new_bias), 2),
                'direction': direction,
                'mape': round(mape, 2),
                'hit_rate': hit_rate,
                'samples': int(len(grp)),
                'confidence': min(0.95, max(0.30, old.get('confidence', 0.45) + 0.03)),
                'validated': bool(old.get('validated', False)),
                'updated': str(CURRENT_DATE),
            }

        res_mem['segment_corrections'] = learned

    @staticmethod
    def _learn_contextual_bias(res_mem: Dict, df_res: pd.DataFrame):
        """
        Học bias theo tổ hợp (weekday × shift × holiday/normal).

        Key format: '{Weekday}__{Shift}__holiday' hoặc '{Weekday}__{Shift}__normal'
        VD: 'Saturday__EVENING__normal' → bias = -8.2 (luôn under 8 khách)

        Nếu Saturday EVENING luôn under 20% → correction tăng đúng context đó,
        không ảnh hưởng các ca/ngày khác.
        """
        if 'Weekday' not in df_res.columns or 'Shift' not in df_res.columns:
            return

        valid = df_res[df_res['Shift'].isin(['MORNING', 'EVENING'])].copy()
        if valid.empty:
            return

        # Add holiday flag
        holiday_flag = (
            valid['Is_Holiday'].astype(bool)
            if 'Is_Holiday' in valid.columns
            else pd.Series(False, index=valid.index)
        )
        valid['_ctx_key'] = (
            valid['Weekday'].astype(str) + '__' +
            valid['Shift'].astype(str) + '__' +
            holiday_flag.map({True: 'holiday', False: 'normal'})
        )

        alpha = ForecastBrain._adaptive_alpha(res_mem)
        old_ctx = res_mem.get('contextual_bias', {})
        new_ctx = dict(old_ctx)

        for ctx_key, grp in valid.groupby('_ctx_key'):
            if len(grp) < 3:
                continue
            new_bias = float(grp['error'].mean())
            old_bias = old_ctx.get(str(ctx_key), 0.0)
            smoothed = alpha * new_bias + (1 - alpha) * old_bias
            new_ctx[str(ctx_key)] = round(smoothed, 2)

        res_mem['contextual_bias'] = new_ctx

        # ── Update bias_direction ──
        bias_hist = res_mem.get('bias_history', [])[-10:]
        if len(bias_hist) >= 5:
            all_over  = all(b >  2 for b in bias_hist)
            all_under = all(b < -2 for b in bias_hist)
            if all_over:
                res_mem['bias_direction'] = 'CONSISTENT_OVER'
            elif all_under:
                res_mem['bias_direction'] = 'CONSISTENT_UNDER'
            else:
                res_mem['bias_direction'] = 'MIXED'

    # ==========================================
    # ⭐ DIRECTIONAL ESCALATION (replaces generic _auto_escalate logic)
    # ==========================================

    @staticmethod
    def _directional_escalate(res_mem: Dict):
        """
        Directional escalation: chỉ tăng correction theo đúng hướng bias.
        Nếu bias oscillate → không escalate (tránh feedback loop).
        """
        bias_hist = res_mem.get('bias_history', [])[-10:]
        if len(bias_hist) < 5:
            return

        all_over  = all(b >  2 for b in bias_hist[-5:])
        all_under = all(b < -2 for b in bias_hist[-5:])

        last_mape = res_mem.get('last_mape', 0) or 0
        if last_mape <= ForecastBrain.MAPE_TARGET:
            # Already good → decay CF toward 1.0
            old_cf = res_mem.get('correction_factor', 1.0)
            if abs(old_cf - 1.0) > 0.01:
                res_mem['correction_factor'] = round(
                    old_cf + (1.0 - old_cf) * 0.10, 4
                )
            return

        if all_over:
            # Consistent over-predict → reduce CF
            old_cf = res_mem.get('correction_factor', 1.0)
            push   = 0.02 if last_mape > 50 else 0.01
            new_cf = max(1 - ForecastBrain.MAX_CORRECTION, old_cf - push)
            res_mem['correction_factor'] = round(new_cf, 4)
            res_mem['bias_direction'] = 'CONSISTENT_OVER'
            logger.debug(
                f"🧠 Directional escalate OVER: CF {old_cf:.4f}→{new_cf:.4f}"
            )
        elif all_under:
            # Consistent under-predict → increase CF
            old_cf = res_mem.get('correction_factor', 1.0)
            push   = 0.02 if last_mape > 50 else 0.01
            new_cf = min(1 + ForecastBrain.MAX_CORRECTION, old_cf + push)
            res_mem['correction_factor'] = round(new_cf, 4)
            res_mem['bias_direction'] = 'CONSISTENT_UNDER'
            logger.debug(
                f"🧠 Directional escalate UNDER: CF {old_cf:.4f}→{new_cf:.4f}"
            )
        else:
            res_mem['bias_direction'] = 'MIXED'
            # Oscillating → decay slightly toward 1.0 (stabilize)
            old_cf = res_mem.get('correction_factor', 1.0)
            if abs(old_cf - 1.0) > 0.01:
                res_mem['correction_factor'] = round(
                    old_cf + (1.0 - old_cf) * 0.05, 4
                )

    @staticmethod
    def _auto_escalate(res_mem: Dict):
        """
        AUTO-ESCALATION: Nếu MAPE không cải thiện sau nhiều lần học,
        tự động tăng aggressiveness của correction.
        
        [FIX LOOP] Added guards:
        - Minimum 10 MAPE history entries (was 5)
        - Bias consistency check: only escalate if bias direction is
          consistent for last 3 cycles
        - Correction factor decay: CF moves 5% toward 1.0 each cycle
        """
        mape_hist = res_mem.get('mape_history', [])
        if len(mape_hist) < 10:  # [FIX LOOP] Need more history (was 5)
            return
        
        last_mape = res_mem.get('last_mape', 0) or 0
        if last_mape <= ForecastBrain.MAPE_TARGET:
            # Already good — decay correction factor toward 1.0
            old_factor = res_mem.get('correction_factor', 1.0)
            if abs(old_factor - 1.0) > 0.01:
                decayed = old_factor + (1.0 - old_factor) * 0.10  # 10% decay when good
                res_mem['correction_factor'] = round(decayed, 4)
            res_mem.pop('escalation_level', None)
            return
        
        # [FIX LOOP] Decay correction factor 5% toward 1.0 every cycle
        # This prevents old corrections from persisting indefinitely
        old_factor = res_mem.get('correction_factor', 1.0)
        if abs(old_factor - 1.0) > 0.005:
            decayed = old_factor + (1.0 - old_factor) * 0.05
            res_mem['correction_factor'] = round(decayed, 4)
            old_factor = res_mem['correction_factor']  # Use decayed value below
        
        # Check if MAPE is improving
        recent_3 = np.mean(mape_hist[-3:])
        older_3 = np.mean(mape_hist[-6:-3]) if len(mape_hist) >= 6 else np.mean(mape_hist[:3])
        
        is_improving = recent_3 < older_3 * 0.97  # Improving by >3%
        is_stagnant = abs(recent_3 - older_3) / max(older_3, 1) < 0.03  # Within 3%
        
        if is_improving:
            # Good progress, reduce escalation level
            level = max(0, res_mem.get('escalation_level', 0) - 1)
            res_mem['escalation_level'] = level
            return
        
        # [FIX LOOP] Guard: only escalate if bias direction is consistent
        overall_bias = res_mem.get('overall_bias', 0)
        bias_hist = res_mem.get('bias_history', [])
        if len(bias_hist) >= 3:
            recent_biases = bias_hist[-3:]
            all_positive = all(b > 0 for b in recent_biases)
            all_negative = all(b < 0 for b in recent_biases)
            if not (all_positive or all_negative):
                # Bias is oscillating — correction would make it worse
                return
        
        if is_stagnant:
            # [FIX LOOP] Cap at level 2 (was 3/5) — gentler escalation
            level = min(2, res_mem.get('escalation_level', 0) + 1)
            res_mem['escalation_level'] = level
            
            # [FIX LOOP] Even gentler push amounts
            push = level * 0.005  # Level 1: +0.005, Level 2: +0.01
            
            if overall_bias > 0:
                new_factor = old_factor - push
            elif overall_bias < 0:
                new_factor = old_factor + push
            else:
                return
            
            # Clamp
            max_f = 1 + ForecastBrain.MAX_CORRECTION
            min_f = 1 - ForecastBrain.MAX_CORRECTION
            new_factor = max(min_f, min(max_f, new_factor))
            res_mem['correction_factor'] = round(new_factor, 4)
            
            logger.debug(
                f"🧠 Auto-escalate level {level}: "
                f"CF {old_factor:.4f}→{new_factor:.4f} "
                f"(MAPE {older_3:.1f}%→{recent_3:.1f}%, bias={overall_bias:.1f})"
            )
    
    @staticmethod
    def _hard_reset_restaurant(res_mem: Dict, reason: str = ''):
        """
        Hard reset learned corrections cho 1 restaurant.
        
        Khi Brain detect rằng corrections hiện tại đang HARMFUL
        (VD: Hit Rate drop liên tục 3+ lần), clear mọi stale state
        và bắt đầu học lại từ đầu.
        
        KHÔNG xóa: mape_history, hit_rate_history (cần cho trend tracking)
        CÓ xóa: correction_factor, bias, escalation, ratio anchor
        """
        old_cf = res_mem.get('correction_factor', 1.0)
        old_bias = res_mem.get('overall_bias', 0)
        
        # Reset corrections về neutral
        res_mem['correction_factor'] = 1.0
        res_mem['overall_bias'] = 0.0
        res_mem['weekday_bias'] = {}
        res_mem['shift_bias'] = {}
        res_mem['weekday_shift_bias'] = {}
        res_mem['hourly_bias'] = {}
        res_mem['holiday_bias'] = 0.0
        res_mem['event_biases'] = {}
        res_mem['actual_ratio_anchor'] = None
        res_mem['preferred_granularity'] = 'mixed'
        
        # Clear escalation
        res_mem['escalation_level'] = 0
        res_mem['consecutive_hr_drops'] = 0
        
        # Mark as needing retune
        res_mem['needs_retune'] = True
        res_mem['hard_reset_at'] = str(CURRENT_DATE)
        res_mem['hard_reset_reason'] = reason
        
        # Ghi nhận reset event trong issues
        res_mem.setdefault('issues', []).append({
            'date': str(CURRENT_DATE),
            'type': 'HARD_RESET',
            'reason': reason,
            'old_correction_factor': old_cf,
            'old_bias': old_bias,
        })
        # Keep issues trimmed
        res_mem['issues'] = res_mem['issues'][-30:]
        
        logger.warning(
            f"🧠🔄 HARD RESET: CF {old_cf:.4f}→1.0, bias {old_bias:.1f}→0. "
            f"Reason: {reason}"
        )
    
    @staticmethod
    def _detect_issues(res_code: str, df_res: pd.DataFrame) -> List[Dict]:
        """
        Phát hiện các sự kiện gây MAPE > 25%.
        
        Issue types:
        - HOLIDAY_SPIKE: Ngày lễ gây sai lệch lớn
        - WEEKEND_ANOMALY: Weekend có pattern khác thường
        - TREND_SHIFT: Xu hướng thay đổi đột ngột
        - EXTREME_ERROR: Sai lệch quá lớn (>50%)
        - CONSISTENT_OVERPREDICT: Liên tục dự đoán cao hơn thực tế
        - CONSISTENT_UNDERPREDICT: Liên tục dự đoán thấp hơn thực tế
        """
        issues = []
        
        # Group by date for daily analysis
        daily = df_res.groupby('Date').agg({
            'error': 'sum',
            'Final_Predicted_Guests': 'sum',
            'Actual_Guest': 'sum',
            'pct_error': 'mean',
        }).reset_index()
        
        daily['daily_pct'] = np.where(
            daily['Actual_Guest'] > 0,
            (daily['error'].abs() / daily['Actual_Guest']) * 100,
            np.nan
        )
        daily['signed_daily_pct'] = np.where(
            daily['Actual_Guest'] > 0,
            (daily['error'] / daily['Actual_Guest']) * 100,
            np.nan
        )
        
        for _, day in daily.iterrows():
            if pd.isna(day['daily_pct']) or day['daily_pct'] <= ForecastBrain.MAPE_TARGET:  # type: ignore[reportGeneralTypeIssues]
                continue
            
            issue = {
                'date': str(day['Date']),
                'error_pct': round(float(day['daily_pct']), 1),
                'error_guests': int(round(day['error'])),  # type: ignore[reportArgumentType]
                'predicted': int(round(day['Final_Predicted_Guests'])),  # type: ignore[reportArgumentType]
                'actual': int(round(day['Actual_Guest'])),  # type: ignore[reportArgumentType]
                'detected_at': str(CURRENT_DATE),
            }
            
            # Classify issue type
            is_holiday = False
            if 'Is_Holiday' in df_res.columns:
                day_rows = df_res[df_res['Date'] == day['Date']]
                is_holiday = day_rows['Is_Holiday'].any()
            
            is_weekend = False
            if 'Weekday' in df_res.columns:
                day_rows = df_res[df_res['Date'] == day['Date']]
                is_weekend = day_rows['Weekday'].isin(['Saturday', 'Sunday']).any()  # type: ignore[reportAttributeAccessIssue]
            
            if is_holiday:  # type: ignore[reportGeneralTypeIssues]
                issue['type'] = 'HOLIDAY_SPIKE'
                issue['cause'] = 'Ngày lễ gây sai lệch lớn'
            elif day['daily_pct'] > 50:
                issue['type'] = 'EXTREME_ERROR'
                issue['cause'] = f"Sai lệch cực lớn ({day['daily_pct']:.0f}%)"
            elif day['signed_daily_pct'] > 25:
                issue['type'] = 'OVERPREDICT'
                issue['cause'] = f"Dự đoán cao hơn thực tế {day['signed_daily_pct']:.0f}%"
            elif day['signed_daily_pct'] < -25:
                issue['type'] = 'UNDERPREDICT'
                issue['cause'] = f"Dự đoán thấp hơn thực tế {abs(day['signed_daily_pct']):.0f}%"
            elif is_weekend:  # type: ignore[reportGeneralTypeIssues]
                issue['type'] = 'WEEKEND_ANOMALY'
                issue['cause'] = 'Weekend có pattern khác thường'
            else:
                issue['type'] = 'HIGH_ERROR'
                issue['cause'] = f"MAPE {day['daily_pct']:.0f}% vượt ngưỡng"
            
            issues.append(issue)
        
        # Detect consistent bias pattern
        if len(daily) >= 7:
            recent = daily.tail(7)
            overpredict_count = (recent['signed_daily_pct'] > 10).sum()
            underpredict_count = (recent['signed_daily_pct'] < -10).sum()
            
            if overpredict_count >= 5:
                issues.append({
                    'date': str(CURRENT_DATE),
                    'type': 'CONSISTENT_OVERPREDICT',
                    'cause': f'5/7 ngày gần nhất over-predict >10%',
                    'error_pct': round(float(recent['signed_daily_pct'].mean()), 1),
                    'detected_at': str(CURRENT_DATE),
                })
            
            if underpredict_count >= 5:
                issues.append({
                    'date': str(CURRENT_DATE),
                    'type': 'CONSISTENT_UNDERPREDICT',
                    'cause': f'5/7 ngày gần nhất under-predict >10%',
                    'error_pct': round(float(recent['signed_daily_pct'].mean()), 1),
                    'detected_at': str(CURRENT_DATE),
                })
        
        return issues
    
    @staticmethod
    def _merge_issues(old_issues: List, new_issues: List) -> List:
        """Merge old + new issues, keep unique, remove expired."""
        # Remove expired
        cutoff = str(CURRENT_DATE - datetime.timedelta(days=ForecastBrain.ISSUE_RETENTION_DAYS))
        
        all_issues = []
        seen = set()
        
        # Add new issues first (priority)
        for issue in new_issues:
            key = f"{issue.get('date', '')}_{issue.get('type', '')}"
            if key not in seen:
                all_issues.append(issue)
                seen.add(key)
        
        # Add old issues that aren't expired or duplicated
        for issue in old_issues:
            key = f"{issue.get('date', '')}_{issue.get('type', '')}"
            issue_date = issue.get('date', '')
            if key not in seen and issue_date >= cutoff:
                all_issues.append(issue)
                seen.add(key)
        
        # Keep last 100 issues max
        return all_issues[-100:]
    
    @staticmethod
    def _update_correction_factor(res_mem: Dict, df_res: pd.DataFrame):
        """
        Tính correction factor tổng hợp.
        
        [FIX A3] Reduced dynamic clamp range and unified with MAX_CORRECTION.
        """
        valid = df_res.dropna(subset=['Actual_Guest', 'Final_Predicted_Guests'])
        valid = valid[valid['Final_Predicted_Guests'] > 0]
        
        if len(valid) < ForecastBrain.MIN_SAMPLES_LEARN:
            return
        
        # Ratio: actual / predicted
        ratios = valid['Actual_Guest'] / valid['Final_Predicted_Guests']
        ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()  # type: ignore[reportAttributeAccessIssue]
        
        if ratios.empty:
            return
        
        # Use median ratio (robust to outliers)
        median_ratio = float(ratios.median())
        
        # Store actual_ratio_anchor for fallback scaling in apply_corrections.
        # Keep it separate from correction_factor, but do not rely on both
        # simultaneously during inference.
        old_anchor = res_mem.get('actual_ratio_anchor')
        alpha = ForecastBrain._adaptive_alpha(res_mem)
        if old_anchor is not None:
            new_anchor = alpha * median_ratio + (1 - alpha) * old_anchor
        else:
            new_anchor = median_ratio
        # [FIX A3] Tighter clamp for anchor (was 0.2~3.0, now 0.5~2.0)
        new_anchor = max(0.5, min(2.0, new_anchor))
        res_mem['actual_ratio_anchor'] = round(new_anchor, 4)
        
        # [FIX #6] Dynamic correction range for extreme outliers
        # Restaurants with MAPE > 100% AND consistent bias direction get extended range
        last_mape = res_mem.get('last_mape', 0) or 0
        overall_bias = res_mem.get('overall_bias', 0)
        bias_hist = res_mem.get('bias_history', [])
        
        if last_mape > 100 and len(bias_hist) >= 5:
            # Check if bias direction is consistent (all same sign for last 5)
            recent_bias = bias_hist[-5:]
            all_negative = all(b < -2 for b in recent_bias)
            all_positive = all(b > 2 for b in recent_bias)
            if all_negative or all_positive:
                # Extended range for persistent outliers: allow up to ×1.75
                extended_max = 1.75
                extended_min = 0.40
                logger.debug(
                    f"🧠 Extended correction range for extreme outlier "
                    f"(MAPE={last_mape:.0f}%, consistent bias={overall_bias:.1f})"
                )
                median_ratio = max(extended_min, min(extended_max, median_ratio))
            else:
                # Normal cap
                max_corr = 1 + ForecastBrain.MAX_CORRECTION  # 1.35
                min_corr = 1 - ForecastBrain.MAX_CORRECTION  # 0.65
                median_ratio = max(min_corr, min(max_corr, median_ratio))
        else:
            # [FIX A3] Standard clamp using MAX_CORRECTION
            max_corr = 1 + ForecastBrain.MAX_CORRECTION  # 1.35
            min_corr = 1 - ForecastBrain.MAX_CORRECTION  # 0.65
            median_ratio = max(min_corr, min(max_corr, median_ratio))
        
        # Use adaptive alpha
        old_factor = res_mem.get('correction_factor', 1.0)
        new_factor = alpha * median_ratio + (1 - alpha) * old_factor
        
        res_mem['correction_factor'] = round(new_factor, 4)
    
    @staticmethod
    def _determine_best_strategy(res_mem: Dict, df_res: pd.DataFrame):
        """
        So sánh ML vs AI accuracy cho restaurant này.
        Recommend strategy tối ưu.
        """
        system_pred_col = (
            'System_Predicted_Before_Brain'
            if 'System_Predicted_Before_Brain' in df_res.columns
            else 'ML_Predicted'
        )

        if system_pred_col not in df_res.columns or 'AI_Raw_Daily_Forecast' not in df_res.columns:
            return
        
        valid = df_res.dropna(subset=['Actual_Guest'])
        valid = valid[valid['Actual_Guest'] > 0]
        
        # System pre-brain accuracy at daily level
        system_valid = valid.dropna(subset=[system_pred_col])  # type: ignore[reportCallIssue]
        if len(system_valid) >= 10:
            daily_system = system_valid.groupby('Date').agg({
                system_pred_col: 'sum',
                'Actual_Guest': 'sum',
            }).reset_index()
            daily_system = daily_system[daily_system['Actual_Guest'] > 0]
            if len(daily_system) >= 5:
                system_err = ((daily_system[system_pred_col] - daily_system['Actual_Guest']).abs() /
                              daily_system['Actual_Guest'] * 100)
                res_mem['ml_mape'] = round(float(system_err.mean()), 1)
        
        # AI accuracy (daily level)
        ai_valid = valid.dropna(subset=['AI_Raw_Daily_Forecast'])  # type: ignore[reportCallIssue]
        if len(ai_valid) >= 10:
            # AI gives daily total, compare at daily level
            daily = ai_valid.groupby('Date').agg({
                'AI_Raw_Daily_Forecast': 'first',
                'Actual_Guest': 'sum',
            }).reset_index()
            daily = daily[daily['Actual_Guest'] > 0]
            
            if len(daily) >= 5:
                ai_err = ((daily['AI_Raw_Daily_Forecast'] - daily['Actual_Guest']).abs() /  # type: ignore[reportAttributeAccessIssue]
                          daily['Actual_Guest'] * 100)
                res_mem['ai_mape'] = round(float(ai_err.mean()), 1)
        
        # Determine best strategy
        ml_mape = res_mem.get('ml_mape')
        ai_mape = res_mem.get('ai_mape')
        
        if ml_mape is not None and ai_mape is not None:
            if ml_mape < ai_mape * 0.7:
                res_mem['best_strategy'] = 'ML_PRIMARY_AI_VALIDATE'
            elif ai_mape < ml_mape * 0.7:
                res_mem['best_strategy'] = 'AI_PRIMARY_ML_SECONDARY'
            elif ml_mape < ai_mape:
                res_mem['best_strategy'] = 'ENSEMBLE_WEIGHTED'
            else:
                res_mem['best_strategy'] = 'ENSEMBLE_EQUAL'

    # ==========================================
    # 2. APPLY CORRECTIONS
    # ==========================================
    
    @staticmethod
    def apply_corrections(
        predictions: List[Dict],
        res_code: str,
        analysis_report: Dict = None,  # type: ignore[reportArgumentType]
    ) -> List[Dict]:
        """
        Áp dụng corrections từ brain memory vào predictions.
        
        Correction pipeline:
        1. Overall bias correction (trừ systematic bias)
        2. Weekday-specific correction
        3. Hourly-specific correction
        4. Holiday correction
        5. Correction factor (scale ratio)
        
        Args:
            predictions: List of prediction dicts từ EnsembleForecastAgent
            res_code: Restaurant code
            analysis_report: Optional analysis report
            
        Returns:
            List[Dict]: Corrected predictions
        """
        memory = ForecastBrain.load_memory()
        res_mem = memory.get('restaurants', {}).get(res_code)
        
        if not res_mem:
            return predictions  # No memory → no correction
        
        correction_factor = res_mem.get('correction_factor', 1.0)
        overall_bias = res_mem.get('overall_bias', 0)
        weekday_bias = res_mem.get('weekday_bias', {})
        shift_bias = res_mem.get('shift_bias', {})
        weekday_shift_bias = res_mem.get('weekday_shift_bias', {})
        hourly_bias = res_mem.get('hourly_bias', {})
        holiday_bias = res_mem.get('holiday_bias', 0)
        segment_corrections = res_mem.get('segment_corrections', {})
        
        # Skip if corrections are truly negligible
        if (abs(correction_factor - 1.0) < 0.01 and 
            abs(overall_bias) < 0.5 and
            not weekday_bias and not shift_bias and not weekday_shift_bias and not hourly_bias and
            not segment_corrections and
            abs(holiday_bias) < 0.5):
            return predictions
        
        corrected = []
        n_corrected = 0
        last_mape = res_mem.get('last_mape', 0) or 0
        escalation_level = res_mem.get('escalation_level', 0)
        event_biases = res_mem.get('event_biases', {})
        actual_ratio_anchor = res_mem.get('actual_ratio_anchor')
        profile = (analysis_report or {}).get('profile', {}) if isinstance(analysis_report, dict) else {}
        avg_daily = (
            profile.get('avg_daily') or
            profile.get('avg_daily_guests') or
            profile.get('avg_guests') or
            0
        )
        
        # Correction strength: higher when MAPE is worse or escalated
        if last_mape > 80 or escalation_level >= 2:
            bias_strength = 0.9
            weekday_strength = 0.8
            shift_strength = 0.8
            hourly_strength = 0.7
            holiday_strength = 0.8
            event_strength = 0.9
        elif last_mape > 50 or escalation_level >= 1:
            bias_strength = 0.8
            weekday_strength = 0.7
            shift_strength = 0.7
            hourly_strength = 0.6
            holiday_strength = 0.7
            event_strength = 0.8
        elif last_mape > 25:
            bias_strength = 0.7
            weekday_strength = 0.6
            shift_strength = 0.6
            hourly_strength = 0.5
            holiday_strength = 0.6
            event_strength = 0.7
        else:
            # Already good accuracy — be conservative
            bias_strength = 0.5
            weekday_strength = 0.4
            shift_strength = 0.4
            hourly_strength = 0.3
            holiday_strength = 0.4
            event_strength = 0.5
        
        for pred in predictions:
            p = pred.copy()
            original = p.get('forecast', 0)
            
            if original <= 0:
                corrected.append(p)
                continue
            
            adjusted = float(original)
            
            # Step 1: Apply correction factor (scale) — most impactful correction
            if abs(correction_factor - 1.0) >= 0.01:
                adjusted *= correction_factor

            # Step 1.5: Segment correction (shift x day_type x size_group).
            # This is the validated-learning layer; it targets the exact
            # segments that have been under/over forecast historically.
            weekday = p.get('weekday', '')
            shift = p.get('shift') or p.get('Shift')
            shift_key = str(shift) if shift is not None else ''
            is_holiday = p.get('is_holiday', False)
            if is_holiday:
                day_type = 'HOLIDAY'
            elif weekday in ('Saturday', 'Sunday'):
                day_type = 'WEEKEND'
            else:
                day_type = 'WEEKDAY'
            size_group = 'LARGE' if float(avg_daily or original) >= 100 else 'SMALL'
            segment_key = f"{shift_key or 'DAILY'}|{day_type}|{size_group}"
            seg = segment_corrections.get(segment_key)
            if seg:
                seg_factor = float(seg.get('factor', 1.0))
                seg_bias = float(seg.get('bias', 0.0))
                confidence = float(seg.get('confidence', 0.4))
                seg_strength = min(0.85, max(0.25, confidence))

                if segment_key == 'EVENING|WEEKDAY|LARGE':
                    # Keep this uplift: 2026-06-01 audit showed it reduced
                    # under-forecast and improved MAE/hit-rate.
                    pass
                elif segment_key == 'MORNING|WEEKDAY|LARGE':
                    # This segment flipped from under to over after uplift.
                    seg_strength = min(seg_strength, 0.35)
                    if seg_factor > 1.0:
                        seg_factor = 1.0 + (seg_factor - 1.0) * 0.45
                    if seg_bias < 0:
                        seg_bias *= 0.45
                elif segment_key == 'MORNING|WEEKDAY|SMALL':
                    # Do not add positive uplift here; it was the main source
                    # of over-forecast and hit-rate loss on 2026-06-01.
                    if seg_factor > 1.0:
                        seg_factor = 1.0
                    if seg_bias < 0:
                        seg_bias = 0.0
                    seg_strength = min(seg_strength, 0.20)

                if abs(seg_factor - 1.0) >= 0.01:
                    adjusted = adjusted * (1 + (seg_factor - 1.0) * seg_strength)
                if abs(seg_bias) >= 1.0:
                    adjusted -= seg_bias * min(0.75, seg_strength)
            
            # Step 2: Subtract overall bias (adaptive strength)
            if abs(overall_bias) >= ForecastBrain.SIGNIFICANT_BIAS:
                adjusted -= overall_bias * bias_strength
            
            # Step 3: Weekday correction (adaptive strength)
            weekday_shift_key = (
                f"{weekday}__{shift_key}"
                if weekday and shift_key in ('MORNING', 'EVENING')
                else ''
            )
            has_specific_shift_bias = weekday_shift_key in weekday_shift_bias

            # ⭐ Step 3.0: Contextual bias (highest specificity — weekday×shift×holiday)
            contextual_bias = res_mem.get('contextual_bias', {})
            if contextual_bias and weekday and shift_key in ('MORNING', 'EVENING'):
                is_hol = p.get('is_holiday', False)
                ctx_suffix = 'holiday' if is_hol else 'normal'
                ctx_key = f"{weekday}__{shift_key}__{ctx_suffix}"
                cb = contextual_bias.get(ctx_key, 0.0)
                if abs(cb) >= 1.0:   # Only apply if signal is meaningful
                    adjusted -= cb * shift_strength
                    # If contextual bias covers this case, skip generic shift bias below
                    has_specific_shift_bias = True  # suppress double-correction

            if weekday and str(weekday) in weekday_bias and not has_specific_shift_bias:
                wb = weekday_bias[str(weekday)]
                if abs(wb) >= 0.5:
                    adjusted -= wb * weekday_strength

            # Step 4: Shift-aware correction for MORNING/EVENING forecasts
            if has_specific_shift_bias:
                wsb = weekday_shift_bias[weekday_shift_key]
                if abs(wsb) >= 0.5:
                    adjusted -= wsb * shift_strength
            elif shift_key in shift_bias:
                sb = shift_bias[shift_key]
                if abs(sb) >= 0.5:
                    adjusted -= sb * shift_strength
            
            # Step 5: Hourly correction only for legacy hourly predictions
            hour = p.get('hour', '')
            if shift_key not in ('MORNING', 'EVENING') and str(hour) in hourly_bias:
                hb = hourly_bias[str(hour)]
                if abs(hb) >= 0.3:
                    adjusted -= hb * hourly_strength
            
            # Step 6: Holiday correction (adaptive strength)
            if is_holiday and abs(holiday_bias) >= 1.0:
                adjusted -= holiday_bias * holiday_strength
            
            # Step 7: EVENT-SPECIFIC correction (Valentine, 8/3, Tet, etc.)
            event_type = p.get('event_type')
            is_special_event = p.get('is_special_event', False)
            if event_type and event_type in event_biases:
                eb = event_biases[event_type]
                if abs(eb) >= 0.5:
                    adjusted -= eb * event_strength
            elif is_special_event and is_holiday:
                # Check holiday_type from prediction metadata
                holiday_type = p.get('holiday_type', '')
                if holiday_type and holiday_type in event_biases:
                    eb = event_biases[holiday_type]
                    if abs(eb) >= 0.5:
                        adjusted -= eb * event_strength
            
            # Step 7.5: SEASONAL monthly correction
            seasonal_mem = res_mem.get('seasonal_memory', {})
            forecast_date = p.get('date')
            if forecast_date and seasonal_mem:
                try:
                    if isinstance(forecast_date, str):
                        forecast_month = str(pd.to_datetime(forecast_date).month)
                    else:
                        forecast_month = str(forecast_date.month)
                    
                    month_data = seasonal_mem.get(forecast_month)
                    if month_data and month_data.get('count', 0) >= 30:
                        seasonal_bias = month_data.get('bias', 0)
                        if abs(seasonal_bias) >= 1.5:
                            # Conservative seasonal strength (lower than other corrections)
                            seasonal_strength = min(0.4, bias_strength * 0.5)
                            adjusted -= seasonal_bias * seasonal_strength
                except (ValueError, AttributeError):
                    pass
            
            # Step 8: actual_ratio_anchor fallback scaling.
            # Avoid stacking both ratio-based corrections aggressively.
            use_anchor = (
                actual_ratio_anchor is not None and
                abs(actual_ratio_anchor - 1.0) > 0.05 and
                abs(correction_factor - 1.0) < 0.03
            )
            if use_anchor:
                anchor_strength = min(0.3, last_mape / 300)  # Max 30% anchor influence (was 50%)
                anchor_adj = original * actual_ratio_anchor
                adjusted = adjusted * (1 - anchor_strength) + anchor_adj * anchor_strength
            
            # Determine correction clamp — now proportional to MAPE
            needs_retune = res_mem.get('needs_retune', False)
            
            if last_mape > 200 or needs_retune:
                effective_max_correction = 0.85
            elif p.get('is_special_event', False):
                effective_max_correction = ForecastBrain.MAX_CORRECTION_SPECIAL_EVENT
            elif last_mape > 100:
                effective_max_correction = 0.70
            elif last_mape > 50:
                effective_max_correction = ForecastBrain.MAX_CORRECTION
            else:
                effective_max_correction = 0.40
            
            # Clamp
            max_adj = original * (1 + effective_max_correction)
            min_adj = original * (1 - effective_max_correction)
            adjusted = max(min_adj, min(max_adj, adjusted))
            
            # Floor at 0
            adjusted = max(0, int(round(adjusted)))
            
            if adjusted != original:
                n_corrected += 1
            
            p['forecast'] = adjusted
            p['forecast_before_correction'] = original
            p['correction_applied'] = round(adjusted - original, 1)
            
            corrected.append(p)
        
        if n_corrected > 0:
            logger.debug(
                f"🧠 {res_code}: Corrected {n_corrected}/{len(predictions)} predictions "
                f"(factor={correction_factor:.3f}, bias={overall_bias:.1f}, "
                f"shift_bias={len(shift_bias)}, weekday_shift_bias={len(weekday_shift_bias)}, "
                f"segment_corrections={len(segment_corrections)}, "
                f"anchor={actual_ratio_anchor}, events={len(event_biases)}, "
                f"strength={bias_strength:.1f}, esc_level={escalation_level})"
            )
        
        return corrected
    
    # ==========================================
    # 3. STRATEGY RECOMMENDATION
    # ==========================================
    
    @staticmethod
    def get_optimal_strategy(res_code: str) -> Optional[str]:
        """
        Recommend strategy tối ưu cho restaurant.
        
        Dựa trên brain memory:
        - So sánh ML vs AI performance
        - Chọn strategy cho accuracy tốt nhất
        
        Returns:
            str: Strategy name hoặc None (dùng default)
        """
        memory = ForecastBrain.load_memory()
        res_mem = memory.get('restaurants', {}).get(res_code)
        
        if not res_mem:
            return None
        
        best = res_mem.get('best_strategy')
        
        # Validate strategy exists
        if best and best in STRATEGY_WEIGHTS:
            return best
        
        return None
    
    @staticmethod
    def get_all_strategy_overrides() -> Dict[str, str]:
        """
        Lấy tất cả strategy overrides cho toàn bộ restaurants.
        
        Returns:
            Dict: {res_code: recommended_strategy}
        """
        memory = ForecastBrain.load_memory()
        overrides = {}
        
        for res_code, res_mem in memory.get('restaurants', {}).items():
            best = res_mem.get('best_strategy')
            if best and best in STRATEGY_WEIGHTS:
                overrides[res_code] = best
        
        return overrides

    # ==========================================
    # 4. DIAGNOSE RESTAURANT
    # ==========================================
    
    @staticmethod
    def diagnose_restaurant(res_code: str, df_master: pd.DataFrame = None) -> Dict:  # type: ignore[reportArgumentType]
        """
        Chẩn đoán chi tiết tại sao restaurant có MAPE cao.
        
        Returns:
            Dict: {
                'restaurant': code,
                'current_mape': float,
                'diagnosis': str,
                'root_causes': list,
                'recommendations': list,
                'correction_info': dict,
                'issue_history': list,
            }
        """
        memory = ForecastBrain.load_memory()
        res_mem = memory.get('restaurants', {}).get(res_code, {})
        
        diagnosis = {
            'restaurant': res_code,
            'current_mape': res_mem.get('last_mape'),
            'correction_factor': res_mem.get('correction_factor', 1.0),
            'overall_bias': res_mem.get('overall_bias', 0),
            'best_strategy': res_mem.get('best_strategy'),
            'ml_mape': res_mem.get('ml_mape'),
            'ai_mape': res_mem.get('ai_mape'),
            'mape_trend': res_mem.get('mape_history', []),
            'root_causes': [],
            'recommendations': [],
            'issue_history': res_mem.get('issues', []),
        }
        
        mape = res_mem.get('last_mape', 0) or 0
        bias = res_mem.get('overall_bias', 0)
        factor = res_mem.get('correction_factor', 1.0)
        
        # --- Root Cause Analysis ---
        causes = []
        recommendations = []
        
        # 1. Systematic bias
        if abs(bias) >= 5:
            direction = 'over-predict' if bias > 0 else 'under-predict'
            causes.append(
                f"Systematic {direction}: Trung bình sai {abs(bias):.1f} guests/giờ"
            )
            recommendations.append(
                f"Brain đã học correction_factor={factor:.3f} để bù bias"
            )
        
        # 2. Weekday problems
        wd_bias = res_mem.get('weekday_bias', {})
        bad_days = {k: v for k, v in wd_bias.items() if abs(v) > 3}
        if bad_days:
            worst_day = max(bad_days, key=lambda k: abs(bad_days[k]))
            causes.append(
                f"Ngày {worst_day} có bias lớn nhất: {bad_days[worst_day]:+.1f} guests/giờ"
            )
            recommendations.append(
                f"Brain áp dụng weekday correction cho {len(bad_days)} ngày"
            )
        
        # 3. Hourly problems
        h_bias = res_mem.get('hourly_bias', {})
        bad_hours = {k: v for k, v in h_bias.items() if abs(v) > 2}
        if bad_hours:
            worst_hour = max(bad_hours, key=lambda k: abs(bad_hours[k]))
            causes.append(
                f"Giờ {worst_hour}:00 có bias lớn nhất: {bad_hours[worst_hour]:+.1f} guests"
            )
        
        # 4. ML vs AI mismatch
        ml_mape = res_mem.get('ml_mape')
        ai_mape = res_mem.get('ai_mape')
        if ml_mape and ai_mape:
            if ml_mape < ai_mape * 0.6:
                causes.append(
                    f"AI kém hơn ML nhiều: AI MAPE={ai_mape}%, ML MAPE={ml_mape}%"
                )
                recommendations.append("Brain recommend ML_PRIMARY strategy")
            elif ai_mape < ml_mape * 0.6:
                causes.append(
                    f"ML kém hơn AI: ML MAPE={ml_mape}%, AI MAPE={ai_mape}%"
                )
                recommendations.append("Brain recommend AI_PRIMARY strategy")
        
        # 5. Holiday effect
        hol_bias = res_mem.get('holiday_bias', 0)
        if abs(hol_bias) > 5:
            direction = 'over' if hol_bias > 0 else 'under'
            causes.append(
                f"Ngày lễ thường {direction}-predict {abs(hol_bias):.1f} guests/giờ"
            )
        
        # 6. Data issues (from issues log)
        issues = res_mem.get('issues', [])
        extreme_count = sum(1 for i in issues if i.get('type') == 'EXTREME_ERROR')
        if extreme_count >= 3:
            causes.append(
                f"Có {extreme_count} sự kiện extreme error (>50% MAPE)"
            )
            recommendations.append(
                "Kiểm tra data quality hoặc đặc thù nhà hàng"
            )
        
        # 7. MAPE trend
        mape_hist = res_mem.get('mape_history', [])
        if len(mape_hist) >= 3:
            recent = mape_hist[-3:]
            if all(recent[i] <= recent[i-1] for i in range(1, len(recent))):
                recommendations.append(
                    f"MAPE đang giảm dần: {' → '.join(str(m) for m in recent)}% ✅"
                )
            elif all(recent[i] >= recent[i-1] for i in range(1, len(recent))):
                causes.append(
                    f"MAPE đang tăng: {' → '.join(str(m) for m in recent)}%"
                )
                recommendations.append("Cần review data hoặc retune model")
        
        if not causes:
            causes.append("Không phát hiện root cause rõ ràng")
            recommendations.append("Model đang hoạt động bình thường")
        
        diagnosis['root_causes'] = causes
        diagnosis['recommendations'] = recommendations
        
        # Summary diagnosis
        if mape <= 25:
            diagnosis['diagnosis'] = f'✅ GOOD - MAPE {mape}% (dưới target 25%)'
        elif mape <= 40:
            diagnosis['diagnosis'] = f'⚠️ FAIR - MAPE {mape}% (cần cải thiện)'
        else:
            diagnosis['diagnosis'] = f'❌ POOR - MAPE {mape}% (cần xử lý gấp)'
        
        return diagnosis

    # ==========================================
    # 5. BATCH CORRECTION (cho main pipeline)
    # ==========================================
    
    @staticmethod
    def correct_all_predictions(
        all_predictions: Dict[str, List[Dict]],
    ) -> Dict[str, List[Dict]]:
        """
        Áp dụng brain correction cho TẤT CẢ restaurants cùng lúc.
        
        Args:
            all_predictions: {res_code: [prediction_dicts]}
            
        Returns:
            Dict: {res_code: [corrected_prediction_dicts]}
        """
        memory = ForecastBrain.load_memory()
        known_restaurants = set(memory.get('restaurants', {}).keys())
        
        corrected_all = {}
        total_before = 0
        total_changed = 0
        
        for res_code, preds in all_predictions.items():
            if res_code in known_restaurants:
                corrected = ForecastBrain.apply_corrections(preds, res_code)
                changed = sum(
                    1 for p in corrected 
                    if p.get('correction_applied', 0) != 0
                )
                total_changed += changed
            else:
                corrected = preds
            
            corrected_all[res_code] = corrected
            total_before += len(preds)
        
        if total_changed > 0:
            logger.info(
                f"🧠 Brain corrected {total_changed}/{total_before} predictions "
                f"across {len(known_restaurants & set(all_predictions.keys()))} restaurants"
            )
        
        return corrected_all

    # ==========================================
    # 6. INSIGHTS & REPORTING
    # ==========================================
    
    @staticmethod
    def generate_insights() -> Dict:
        """
        Generate human-readable insights từ brain memory.
        
        Returns:
            Dict: {
                'summary': str,
                'high_mape_restaurants': list,
                'improving': list,
                'degrading': list,
                'top_issues': list,
                'strategy_overrides': dict,
                'global_insights': list,
            }
        """
        memory = ForecastBrain.load_memory()
        restaurants = memory.get('restaurants', {})
        
        insights = {
            'total_restaurants_learned': len(restaurants),
            'last_updated': memory.get('last_updated'),
            'high_mape_restaurants': [],
            'improving': [],
            'degrading': [],
            'top_issues': [],
            'strategy_overrides': {},
            'global_insights': [],
        }
        
        for res_code, res_mem in restaurants.items():
            mape = res_mem.get('last_mape')
            if mape is None:
                continue
            
            # High MAPE
            if mape > ForecastBrain.MAPE_TARGET:
                insights['high_mape_restaurants'].append({
                    'code': res_code,
                    'mape': mape,
                    'bias': res_mem.get('overall_bias', 0),
                    'correction': res_mem.get('correction_factor', 1.0),
                    'issues_count': len(res_mem.get('issues', [])),
                })
            
            # MAPE trend
            hist = res_mem.get('mape_history', [])
            if len(hist) >= 3:
                if hist[-1] < hist[-3] * 0.85:  # Improved >15%
                    insights['improving'].append({
                        'code': res_code,
                        'from_mape': hist[-3],
                        'to_mape': hist[-1],
                    })
                elif hist[-1] > hist[-3] * 1.15:  # Degraded >15%
                    insights['degrading'].append({
                        'code': res_code,
                        'from_mape': hist[-3],
                        'to_mape': hist[-1],
                    })
            
            # Strategy overrides
            best = res_mem.get('best_strategy')
            if best and best in STRATEGY_WEIGHTS:
                insights['strategy_overrides'][res_code] = best
        
        # Sort high MAPE by severity
        insights['high_mape_restaurants'].sort(
            key=lambda x: x['mape'], reverse=True
        )
        
        # Global insights
        gp = memory.get('global_patterns', {})
        
        weekend_bias = gp.get('weekend_bias', 0)
        if abs(weekend_bias) > 3:
            direction = 'over' if weekend_bias > 0 else 'under'
            insights['global_insights'].append(
                f"Weekend: Hệ thống {direction}-predict trung bình {abs(weekend_bias):.1f} guests"
            )
        
        holiday_bias = gp.get('holiday_bias_pct', 0)
        if abs(holiday_bias) > 10:
            insights['global_insights'].append(
                f"Holiday: Sai lệch trung bình {holiday_bias:+.0f}%"
            )
        
        # Overall MAPE trend
        mape_trend = gp.get('overall_mape_trend', [])
        if len(mape_trend) >= 2:
            first = mape_trend[0]['mape']
            last = mape_trend[-1]['mape']
            if last < first:
                insights['global_insights'].append(
                    f"📉 MAPE đang giảm: {first}% → {last}% (cải thiện {first-last:.1f}%)"
                )
            else:
                insights['global_insights'].append(
                    f"📈 MAPE đang tăng: {first}% → {last}% (xấu đi {last-first:.1f}%)"
                )
        
        # Summary
        n_high = len(insights['high_mape_restaurants'])
        n_total = len(restaurants)
        n_improving = len(insights['improving'])
        n_degrading = len(insights['degrading'])
        
        insights['summary'] = (
            f"Brain theo dõi {n_total} restaurants | "
            f"{n_high} có MAPE>{ForecastBrain.MAPE_TARGET}% | "
            f"{n_improving} đang cải thiện | "
            f"{n_degrading} đang xấu đi | "
            f"{len(insights['strategy_overrides'])} strategy overrides"
        )
        
        return insights
    
    @staticmethod
    def print_insights(logger_func=None):
        """Print brain insights to logger."""
        log = logger_func or logger.info
        insights = ForecastBrain.generate_insights()
        
        log("\n" + "=" * 65)
        log("🧠 FORECAST BRAIN - SELF-LEARNING REPORT")
        log("=" * 65)
        log(f"   {insights['summary']}")
        log(f"   Last updated: {insights.get('last_updated', 'Never')}")
        
        # High MAPE
        high = insights.get('high_mape_restaurants', [])
        if high:
            log(f"\n  ❌ HIGH MAPE RESTAURANTS (>{ForecastBrain.MAPE_TARGET}%):")
            for r in high[:15]:
                corr = r['correction']
                bias_str = f"+{r['bias']:.1f}" if r['bias'] > 0 else f"{r['bias']:.1f}"
                log(f"     {r['code']:>10s}: MAPE={r['mape']}% | "
                    f"Bias={bias_str} | Factor={corr:.3f} | "
                    f"Issues={r['issues_count']}")
            if len(high) > 15:
                log(f"     ... and {len(high) - 15} more")
        
        # Improving
        improving = insights.get('improving', [])
        if improving:
            log(f"\n  📉 IMPROVING ({len(improving)} restaurants):")
            for r in improving[:10]:
                log(f"     {r['code']:>10s}: {r['from_mape']}% → {r['to_mape']}% ✅")
        
        # Degrading
        degrading = insights.get('degrading', [])
        if degrading:
            log(f"\n  📈 DEGRADING ({len(degrading)} restaurants):")
            for r in degrading[:10]:
                log(f"     {r['code']:>10s}: {r['from_mape']}% → {r['to_mape']}% ⚠️")
        
        # Global
        global_ins = insights.get('global_insights', [])
        if global_ins:
            log(f"\n  🌐 GLOBAL INSIGHTS:")
            for g in global_ins:
                log(f"     • {g}")
        
        # Strategy overrides count
        overrides = insights.get('strategy_overrides', {})
        if overrides:
            from collections import Counter
            strat_counts = Counter(overrides.values())
            log(f"\n  🎯 STRATEGY OVERRIDES ({len(overrides)} restaurants):")
            for strat, count in strat_counts.most_common():
                log(f"     {strat}: {count} restaurants")
        
        log("=" * 65)

    # ==========================================
    # 7. ABSORB MONITORING REPORT (CLOSED-LOOP)
    # ==========================================
    
    @staticmethod
    def absorb_monitoring_report(report: Dict) -> Dict:
        """
        Hấp thụ kết quả từ MonitoringAgent report để cập nhật Brain memory.
        
        Đây là bước đóng vòng lặp closed-loop:
            MonitoringAgent tính accuracy chính xác
            → Brain đọc kết quả
            → Brain cập nhật memory & điều chỉnh correction
        
        Absorb 4 nguồn thông tin:
        1. Official metrics (MAE, MAPE, Hit Rate) - thay thế tự tính
        2. Drift alerts - tăng correction nếu accuracy đang giảm
        3. Needs_Retune list - đánh dấu + tăng correction mạnh hơn
        4. Accuracy history - xu hướng dài hạn
        
        Args:
            report: Dict từ MonitoringAgent.generate_full_report()
            
        Returns:
            Dict: absorption summary
        """
        memory = ForecastBrain.load_memory()
        
        summary = {
            'metrics_updated': 0,
            'drift_adjustments': 0,
            'retune_flagged': 0,
            'hit_rate_resets': 0,
            'history_insights': [],
        }
        
        # ────────────────────────────────────────
        # 1. Absorb official per-restaurant metrics
        # ────────────────────────────────────────
        per_restaurant = report.get('best_restaurants', []) + report.get('worst_restaurants', [])
        
        # Also get full per-restaurant data if available
        # (MonitoringAgent calculates this internally)
        for res_info in per_restaurant:
            res_code = str(res_info.get('Restaurant_Code', ''))
            if not res_code:
                continue
            
            res_mem = ForecastBrain._get_restaurant_memory(memory, res_code)
            
            # Update with MonitoringAgent's official metrics
            official_mape = res_info.get('MAPE')
            official_mae = res_info.get('MAE')
            official_bias = res_info.get('Bias')
            official_hit_rate = res_info.get('Hit_Rate')
            
            if official_mape is not None and not pd.isna(official_mape):
                res_mem['monitoring_mape'] = round(float(official_mape), 1)
                summary['metrics_updated'] += 1
            
            if official_mae is not None and not pd.isna(official_mae):
                res_mem['monitoring_mae'] = round(float(official_mae), 2)
            
            if official_bias is not None and not pd.isna(official_bias):
                res_mem['monitoring_bias'] = round(float(official_bias), 2)
            
            if official_hit_rate is not None and not pd.isna(official_hit_rate):
                hr_val = round(float(official_hit_rate), 1)
                
                # Store current as last before updating
                old_hr = res_mem.get('monitoring_hit_rate')
                if old_hr is not None:
                    res_mem['last_hit_rate'] = old_hr
                
                res_mem['monitoring_hit_rate'] = hr_val
                
                # Track hit rate history (keep last 30)
                res_mem.setdefault('hit_rate_history', []).append({
                    'date': str(CURRENT_DATE),
                    'value': hr_val,
                })
                res_mem['hit_rate_history'] = res_mem['hit_rate_history'][-30:]
        
        # ────────────────────────────────────────
        # 2. Absorb drift alerts → increase correction
        # ────────────────────────────────────────
        drift = report.get('drift', {})
        
        if drift.get('has_drift'):
            alerts = drift.get('alerts', [])
            changes = drift.get('changes', {})
            
            for alert in alerts:
                level = alert.get('level', 'INFO')
                metric = alert.get('metric', '')
                
                if metric == 'MAPE' and level in ('WARNING', 'CRITICAL'):
                    # MAPE đang tăng → tăng correction strength toàn hệ thống
                    mape_change_pct = changes.get('MAPE', {}).get('change_pct', 0)
                    
                    if mape_change_pct > 0:
                        # Tăng correction factor cho TẤT CẢ restaurants
                        # Mức tăng tỷ lệ với mức drift
                        boost_factor = min(0.05, mape_change_pct / 1000)
                        # boost nhẹ: drift 20% → boost 0.02, drift 50% → boost 0.05
                        
                        for res_code, res_mem in memory.get('restaurants', {}).items():
                            old_factor = res_mem.get('correction_factor', 1.0)
                            old_bias = res_mem.get('overall_bias', 0)
                            
                            # Nếu đang over-predict (bias > 0) → giảm factor thêm
                            # Nếu đang under-predict (bias < 0) → tăng factor thêm
                            if old_bias > ForecastBrain.SIGNIFICANT_BIAS:
                                new_factor = old_factor - boost_factor
                            elif old_bias < -ForecastBrain.SIGNIFICANT_BIAS:
                                new_factor = old_factor + boost_factor
                            else:
                                continue  # Bias nhỏ → không cần điều chỉnh
                            
                            # Clamp
                            max_f = 1 + ForecastBrain.MAX_CORRECTION
                            min_f = 1 - ForecastBrain.MAX_CORRECTION
                            new_factor = max(min_f, min(max_f, new_factor))
                            
                            res_mem['correction_factor'] = round(new_factor, 4)
                            summary['drift_adjustments'] += 1
                        
                        logger.info(
                            f"🧠 Drift absorbed: MAPE drift +{mape_change_pct:.1f}% "
                            f"→ boosted correction for {summary['drift_adjustments']} restaurants"
                        )
                
                if metric == 'Hit_Rate' and level == 'CRITICAL':
                    # ═══════════════════════════════════════════
                    # [FIX] Hit Rate CRITICAL drop → REAL ACTIONS
                    # ═══════════════════════════════════════════
                    
                    # 1. Ghi nhận vào drift_history
                    memory.setdefault('drift_history', []).append({
                        'date': str(CURRENT_DATE),
                        'type': 'HIT_RATE_DROP',
                        'message': alert.get('message', ''),
                    })
                    
                    # 2. Tăng consecutive drop counter cho TẤT CẢ restaurants
                    #    và trigger retune + correction boost
                    hr_change_data = changes.get('Hit_Rate', {})
                    drop_magnitude = abs(
                        hr_change_data.get('this_week', 0) - 
                        hr_change_data.get('last_week', 0)
                    )
                    
                    hard_reset_count = 0
                    boost_count = 0
                    
                    for res_code, res_mem in memory.get('restaurants', {}).items():
                        res_hr = res_mem.get('monitoring_hit_rate')
                        last_hr = res_mem.get('last_hit_rate')
                        
                        # Track consecutive drops per restaurant
                        if res_hr is not None and last_hr is not None:
                            if res_hr < last_hr - 2:  # Drop > 2 points
                                res_mem['consecutive_hr_drops'] = (
                                    res_mem.get('consecutive_hr_drops', 0) + 1
                                )
                            elif res_hr >= last_hr:
                                # Hit rate stable/improving → reset counter
                                res_mem['consecutive_hr_drops'] = 0
                        
                        consecutive = res_mem.get('consecutive_hr_drops', 0)
                        
                        # ── Action 1: 3+ consecutive drops → HARD RESET ──
                        if consecutive >= 3:
                            ForecastBrain._hard_reset_restaurant(
                                res_mem,
                                reason=(
                                    f"Hit Rate dropped {consecutive} consecutive times. "
                                    f"Current: {res_hr}%, corrections likely harmful."
                                )
                            )
                            hard_reset_count += 1
                            summary['retune_flagged'] += 1
                            summary['hit_rate_resets'] += 1
                        
                        # ── Action 2: Bất kỳ CRITICAL drop → boost correction ──
                        elif res_hr is not None and res_hr < 50:
                            # Hit rate < 50%: hơn nửa predictions sai
                            # → boost correction proportional to how bad it is
                            old_factor = res_mem.get('correction_factor', 1.0)
                            old_bias = res_mem.get('overall_bias', 0)
                            boost = min(0.08, (50 - res_hr) / 500)
                            
                            if old_bias > ForecastBrain.SIGNIFICANT_BIAS:
                                new_factor = old_factor - boost
                            elif old_bias < -ForecastBrain.SIGNIFICANT_BIAS:
                                new_factor = old_factor + boost
                            else:
                                new_factor = old_factor  # No clear bias
                            
                            # Clamp
                            max_f = 1 + ForecastBrain.MAX_CORRECTION
                            min_f = 1 - ForecastBrain.MAX_CORRECTION
                            new_factor = max(min_f, min(max_f, new_factor))
                            res_mem['correction_factor'] = round(new_factor, 4)
                            
                            # Mark for retune if really bad
                            if res_hr < 35:
                                res_mem['needs_retune'] = True
                                res_mem['retune_flagged_at'] = str(CURRENT_DATE)
                            
                            # Bump escalation
                            res_mem['escalation_level'] = min(
                                5, res_mem.get('escalation_level', 0) + 1
                            )
                            boost_count += 1
                    
                    if hard_reset_count > 0 or boost_count > 0:
                        logger.warning(
                            f"🧠 Hit Rate CRITICAL → "
                            f"{hard_reset_count} hard resets, "
                            f"{boost_count} correction boosts"
                        )
                        summary['drift_adjustments'] += hard_reset_count + boost_count
            
            # Lưu drift status vào global patterns
            memory['global_patterns']['last_drift_detected'] = str(CURRENT_DATE)
            memory['global_patterns']['drift_severity'] = (
                'CRITICAL' if any(a.get('level') == 'CRITICAL' for a in alerts)
                else 'WARNING'
            )
        else:
            # Không có drift → ghi nhận ổn định
            memory['global_patterns']['last_drift_detected'] = None
            memory['global_patterns']['drift_severity'] = None
        
        # ────────────────────────────────────────
        # 3. Absorb Needs_Retune list
        # ────────────────────────────────────────
        problems = report.get('problem_restaurants', {})
        needs_retune = problems.get('needs_retune', [])
        
        retune_codes = set()
        for res_info in needs_retune:
            res_code = str(res_info.get('Restaurant_Code', ''))
            if not res_code:
                continue
            
            retune_codes.add(res_code)
            res_mem = ForecastBrain._get_restaurant_memory(memory, res_code)
            
            # Flag needs retune
            res_mem['needs_retune'] = True
            res_mem['retune_flagged_at'] = str(CURRENT_DATE)
            res_mem['retune_mape'] = res_info.get('MAPE')
            
            # Tăng correction factor mạnh hơn cho restaurants cần retune
            # Sửa mạnh tay hơn cho restaurants cần retune
            # WAS: 3% fixed boost → NOW: proportional to MAPE severity
            old_factor = res_mem.get('correction_factor', 1.0)
            old_bias = res_mem.get('overall_bias', 0)
            retune_mape = res_info.get('MAPE', res_mem.get('last_mape', 50))
            
            # Boost proportional to how bad the MAPE is
            # MAPE 50% → 5% boost, MAPE 100% → 10% boost, MAPE 200% → 15%
            retune_boost = min(0.15, max(0.03, (retune_mape or 50) / 1000))
            
            if old_bias > 0:
                new_factor = old_factor - retune_boost
            elif old_bias < 0:
                new_factor = old_factor + retune_boost
            else:
                # No clear bias direction → use ratio from data
                new_factor = old_factor
            
            # Clamp
            max_f = 1 + ForecastBrain.MAX_CORRECTION
            min_f = 1 - ForecastBrain.MAX_CORRECTION
            new_factor = max(min_f, min(max_f, new_factor))
            res_mem['correction_factor'] = round(new_factor, 4)
            
            # Also set escalation level for this restaurant
            res_mem['escalation_level'] = min(5, res_mem.get('escalation_level', 0) + 2)
            
            summary['retune_flagged'] += 1
        
        # Clear retune flag for restaurants NOT in needs_retune anymore
        for res_code, res_mem in memory.get('restaurants', {}).items():
            if res_code not in retune_codes and res_mem.get('needs_retune'):
                res_mem['needs_retune'] = False
                res_mem.pop('retune_flagged_at', None)
                res_mem.pop('retune_mape', None)
        
        if retune_codes:
            logger.info(
                f"🧠 Retune absorbed: {len(retune_codes)} restaurants flagged, "
                f"correction boosted"
            )
        
        # ────────────────────────────────────────
        # 4. Absorb accuracy_history.json (long-term trend)
        # ────────────────────────────────────────
        history = ForecastBrain._load_accuracy_history()
        
        if len(history) >= 3:
            # Analyze trend: last 7 entries
            recent = history[-7:]
            mape_values = [
                h.get('MAPE', h.get('mape', 0))
                for h in recent
                if h.get('MAPE', h.get('mape')) is not None
            ]
            
            if len(mape_values) >= 3:
                first_half = np.mean(mape_values[:len(mape_values)//2])
                second_half = np.mean(mape_values[len(mape_values)//2:])
                
                trend_pct = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
                
                memory['global_patterns']['accuracy_trend'] = {
                    'direction': 'improving' if trend_pct < -3 else 'degrading' if trend_pct > 3 else 'stable',
                    'change_pct': round(trend_pct, 1),
                    'recent_mape_avg': round(second_half, 1),
                    'older_mape_avg': round(first_half, 1),
                    'data_points': len(mape_values),
                    'analyzed_at': str(CURRENT_DATE),
                }
                
                direction = memory['global_patterns']['accuracy_trend']['direction']
                summary['history_insights'].append(
                    f"Long-term: {direction} ({trend_pct:+.1f}% MAPE change)"
                )
                
                # Nếu accuracy đang degrading liên tục → tăng correction toàn bộ
                if trend_pct > 10:  # MAPE tăng hơn 10% so với trước
                    degrade_boost = min(0.03, trend_pct / 500)
                    adjusted_count = 0
                    
                    for res_code, res_mem in memory.get('restaurants', {}).items():
                        old_bias = res_mem.get('overall_bias', 0)
                        if abs(old_bias) > ForecastBrain.SIGNIFICANT_BIAS:
                            old_f = res_mem.get('correction_factor', 1.0)
                            if old_bias > 0:
                                res_mem['correction_factor'] = round(
                                    max(1 - ForecastBrain.MAX_CORRECTION, old_f - degrade_boost), 4
                                )
                            else:
                                res_mem['correction_factor'] = round(
                                    min(1 + ForecastBrain.MAX_CORRECTION, old_f + degrade_boost), 4
                                )
                            adjusted_count += 1
                    
                    if adjusted_count > 0:
                        summary['history_insights'].append(
                            f"Degrading trend detected → boosted {adjusted_count} restaurants"
                        )
                
                logger.info(
                    f"🧠 History absorbed: {direction} "
                    f"(MAPE {first_half:.1f}% → {second_half:.1f}%, "
                    f"{trend_pct:+.1f}%)"
                )
        
        # ────────────────────────────────────────
        # Save updated memory
        # ────────────────────────────────────────
        
        # Log absorption event
        memory['learning_log'].append({
            'date': str(CURRENT_DATE),
            'timestamp': datetime.datetime.now().isoformat(),
            'action': 'absorb_monitoring',
            'metrics_updated': summary['metrics_updated'],
            'drift_adjustments': summary['drift_adjustments'],
            'retune_flagged': summary['retune_flagged'],
            'history_insights': summary['history_insights'],
        })
        memory['learning_log'] = memory['learning_log'][-90:]
        
        ForecastBrain.save_memory(memory)
        
        logger.info(
            f"🧠 Monitoring absorption complete: "
            f"{summary['metrics_updated']} metrics updated, "
            f"{summary['drift_adjustments']} drift corrections, "
            f"{summary['retune_flagged']} retune flagged, "
            f"{summary['hit_rate_resets']} hit-rate resets"
        )
        
        return summary
    
    @staticmethod
    def _load_accuracy_history() -> List[Dict]:
        """Load accuracy_history.json cho long-term trend analysis."""
        from forecast_system.agents.monitoring_agent import MonitoringAgent
        
        history_file = MonitoringAgent.ACCURACY_HISTORY_FILE
        
        if not os.path.exists(history_file):
            return []
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data
            return []
        except Exception as e:
            logger.debug(f"Could not load accuracy history: {e}")
            return []
