"""
==============================================
BACKTEST ENGINE - Walk-Forward Simulation
==============================================
Mô phỏng chạy forecast trên dữ liệu lịch sử để đánh giá accuracy.

Logic:
    1. Chọn khoảng thời gian backtest (VD: 2026-01-15 → 2026-02-10)
    2. Với mỗi "simulation date":
       - Giả lập CURRENT_DATE = simulation_date
       - Dùng data TRƯỚC simulation_date để train
       - Forecast N ngày tới
       - So sánh forecast vs actual
    3. Tổng hợp metrics qua tất cả simulations

Usage:
    # CLI
    python -m forecast_system.backtest --start 2026-01-15 --end 2026-02-10 --horizon 7
    
    # Hoặc trong code
    from forecast_system.backtest import BacktestEngine
    engine = BacktestEngine()
    results = engine.run_backtest(
        start_date=date(2026, 1, 15),
        end_date=date(2026, 2, 10),
        forecast_horizon=7,
    )
"""

import os
import sys
import json
import time
import datetime
import argparse
import traceback
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Ensure import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from forecast_system.config.settings import (
    PROJECT_ROOT, DATA_LOOKBACK_DAYS, STRATEGY_WEIGHTS,
    ANALYSIS_CONFIG, MONITORING_CONFIG, LOW_VOLUME_THRESHOLD,
)
from forecast_system.utils.logger import setup_logger, get_logger
from forecast_system.utils.db_utils import create_db_engine
from forecast_system.utils.date_utils import (
    get_vn_holidays, build_forecast_days, get_lunar_info,
)
from forecast_system.agents.data_agent import DataAgent
from forecast_system.agents.analysis_agent import AnalysisAgent
from forecast_system.agents.ml_forecast_agent import MLForecastAgent
from forecast_system.agents.ensemble_agent import (
    EnsembleMLAgent, ProphetDailyAgent, EnsembleForecastAgent,
)
from forecast_system.agents.ai_forecast_agent import AIForecastAgent

logger = get_logger('backtest')


# ==========================================
# BACKTEST RESULT CLASSES
# ==========================================

class SimulationResult:
    """Kết quả 1 lần simulation."""
    def __init__(self, sim_date: datetime.date):
        self.sim_date = sim_date
        self.predictions: List[Dict] = []
        self.actuals: List[Dict] = []
        self.metrics: Dict = {}
        self.restaurant_metrics: Dict = {}
        self.elapsed_seconds: float = 0
        self.n_restaurants: int = 0
        self.n_success: int = 0
        self.n_failed: int = 0
        self.errors: List[str] = []


class BacktestReport:
    """Báo cáo tổng hợp backtest."""
    def __init__(self):
        self.simulations: List[SimulationResult] = []
        self.overall_metrics: Dict = {}
        self.per_restaurant: pd.DataFrame = pd.DataFrame()
        self.per_weekday: pd.DataFrame = pd.DataFrame()
        self.per_horizon: pd.DataFrame = pd.DataFrame()
        self.per_simulation: pd.DataFrame = pd.DataFrame()
        # ⭐ V8 Task 6+7: Shift-level and volume-segmented metrics
        self.per_shift: pd.DataFrame = pd.DataFrame()              # MORNING vs EVENING
        self.per_shift_weekday: pd.DataFrame = pd.DataFrame()      # (Shift × Weekday) breakdown
        self.per_volume_segment: pd.DataFrame = pd.DataFrame()     # LOW/MED/HIGH volume
        self.config: Dict = {}
        self.total_elapsed: float = 0


# ==========================================
# BACKTEST ENGINE
# ==========================================

class BacktestEngine:
    """
    Walk-forward Backtest Engine.
    
    Mô phỏng chạy pipeline tại các ngày trong quá khứ,
    so sánh forecast vs actual, đo lường accuracy.
    """
    
    def __init__(self, use_ai: bool = False, verbose: bool = True):
        """
        Args:
            use_ai: Có gọi LM Studio (AI) trong backtest không?
                    Default False vì chậm và cần server chạy.
            verbose: In chi tiết log.
        """
        self.use_ai = use_ai
        self.verbose = verbose
        self.engine = None  # DB engine
    
    def run_backtest(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        forecast_horizon: int = 7,
        step_days: int = 7,
        lookback_days: int = None,  # type: ignore[reportArgumentType]
        restaurant_filter: List[str] = None,  # type: ignore[reportArgumentType]
        max_restaurants: int = None,  # type: ignore[reportArgumentType]
    ) -> BacktestReport:
        """
        Run walk-forward backtest.
        
        Args:
            start_date: Ngày bắt đầu backtest (simulation_date đầu tiên)
            end_date: Ngày kết thúc backtest (simulation_date cuối cùng)
            forecast_horizon: Số ngày forecast mỗi simulation (default 7)
            step_days: Bước nhảy giữa các simulations (default 7 = mỗi tuần 1 lần)
            lookback_days: Số ngày data lịch sử để train (default 120)
            restaurant_filter: Chỉ backtest các nhà hàng này (None = tất cả)
            max_restaurants: Giới hạn số nhà hàng (để test nhanh)
        
        Returns:
            BacktestReport: Báo cáo tổng hợp
        """
        if lookback_days is None:
            lookback_days = DATA_LOOKBACK_DAYS
        
        total_start = time.time()
        
        logger.info("=" * 65)
        logger.info("🔬 BACKTEST ENGINE - Walk-Forward Simulation")
        logger.info("=" * 65)
        logger.info(f"  Period:          {start_date} → {end_date}")
        logger.info(f"  Horizon:         {forecast_horizon} days per simulation")
        logger.info(f"  Step:            every {step_days} days")
        logger.info(f"  Lookback:        {lookback_days} days for training")
        logger.info(f"  Use AI (LLM):    {'Yes' if self.use_ai else 'No (ML+Prophet only)'}")
        if restaurant_filter:
            logger.info(f"  Restaurants:     {len(restaurant_filter)} specified")
        if max_restaurants:
            logger.info(f"  Max restaurants: {max_restaurants}")
        logger.info("=" * 65)
        
        # Connect DB
        logger.info("\n📡 Connecting to Database...")
        self.engine = create_db_engine(max_retries=3, retry_delay=5)
        if self.engine is None:
            logger.error("Failed to connect to Database. Aborting backtest.")
            report = BacktestReport()
            report.config = self._build_config(
                start_date, end_date, forecast_horizon, step_days, lookback_days
            )
            return report
        
        # Generate simulation dates
        sim_dates = []
        current = start_date
        while current <= end_date:
            sim_dates.append(current)
            current += datetime.timedelta(days=step_days)
        
        logger.info(f"\n📅 {len(sim_dates)} simulation(s) planned:")
        for sd in sim_dates:
            fc_end = sd + datetime.timedelta(days=forecast_horizon - 1)
            logger.info(f"   {sd} → forecast {sd} to {fc_end}")
        
        # Load ALL data needed (from earliest lookback to latest forecast end)
        earliest_data = start_date - datetime.timedelta(days=lookback_days)
        latest_data = end_date + datetime.timedelta(days=forecast_horizon)
        
        logger.info(f"\n📥 Loading all data: {earliest_data} → {latest_data}")
        df_all = self._load_full_data(earliest_data, latest_data)
        
        if df_all.empty:
            logger.error("No data loaded. Aborting backtest.")
            report = BacktestReport()
            report.config = self._build_config(
                start_date, end_date, forecast_horizon, step_days, lookback_days
            )
            return report
        
        logger.info(f"   Total rows: {len(df_all):,}")
        logger.info(f"   Date range: {df_all['date'].min()} → {df_all['date'].max()}")
        logger.info(f"   Restaurants: {df_all['restaurant_code'].nunique()}")
        
        # Filter restaurants
        if restaurant_filter:
            df_all = df_all[df_all['restaurant_code'].isin(restaurant_filter)]
            logger.info(f"   After filter: {df_all['restaurant_code'].nunique()} restaurants")  # type: ignore[reportAttributeAccessIssue]
        
        # Run simulations
        report = BacktestReport()
        report.config = self._build_config(
            start_date, end_date, forecast_horizon, step_days, lookback_days
        )
        
        for i, sim_date in enumerate(sim_dates):
            logger.info(f"\n{'='*50}")
            logger.info(f"🔄 Simulation {i+1}/{len(sim_dates)}: {sim_date}")
            logger.info(f"{'='*50}")
            
            sim_result = self._run_single_simulation(
                df_all=df_all,  # type: ignore[reportArgumentType]
                sim_date=sim_date,
                forecast_horizon=forecast_horizon,
                lookback_days=lookback_days,
                max_restaurants=max_restaurants,
            )
            report.simulations.append(sim_result)
            
            logger.info(
                f"   ✅ Done: {sim_result.n_success}/{sim_result.n_restaurants} restaurants, "
                f"{sim_result.elapsed_seconds:.1f}s"
            )
            if sim_result.metrics:
                logger.info(
                    f"   📊 MAPE: {sim_result.metrics.get('MAPE', 'N/A')}%, "
                    f"MAE: {sim_result.metrics.get('MAE', 'N/A')}, "
                    f"Hit Rate: {sim_result.metrics.get('Hit_Rate', 'N/A')}%"
                )
        
        # Compile overall report
        report.total_elapsed = time.time() - total_start
        self._compile_report(report)
        
        # Print summary
        self._print_summary(report)
        
        # Save results
        self._save_report(report)
        
        return report
    
    # ==========================================
    # SINGLE SIMULATION
    # ==========================================
    
    def _run_single_simulation(
        self,
        df_all: pd.DataFrame,
        sim_date: datetime.date,
        forecast_horizon: int,
        lookback_days: int,
        max_restaurants: int = None,  # type: ignore[reportArgumentType]
    ) -> SimulationResult:
        """Run 1 simulation tại sim_date."""
        
        sim_start = time.time()
        result = SimulationResult(sim_date)
        
        # Split data: train = before sim_date, actual = sim_date to +horizon
        train_start = sim_date - datetime.timedelta(days=lookback_days)
        forecast_end = sim_date + datetime.timedelta(days=forecast_horizon - 1)
        
        df_train = df_all[
            (df_all['date'] >= train_start) &
            (df_all['date'] < sim_date)
        ].copy()
        
        df_actual = df_all[
            (df_all['date'] >= sim_date) &
            (df_all['date'] <= forecast_end)
        ].copy()
        
        if df_train.empty:
            result.errors.append("No training data available")
            return result
        
        # Get active restaurants
        active_restaurants = DataAgent.get_active_restaurants(
            df_train, inactive_threshold=30
        )
        
        if max_restaurants and len(active_restaurants) > max_restaurants:
            active_restaurants = active_restaurants[:max_restaurants]
        
        result.n_restaurants = len(active_restaurants)
        
        # Build forecast days
        vn_holidays = get_vn_holidays([sim_date.year, sim_date.year + 1])
        next_days = build_forecast_days(sim_date, forecast_horizon, vn_holidays)
        
        # Analysis
        analysis_reports = AnalysisAgent.analyze_all_restaurants(
            df_train, active_restaurants  # type: ignore[reportArgumentType]
        )
        
        # Forecast each restaurant
        all_predictions = []
        
        for res_code in active_restaurants:
            try:
                report = analysis_reports.get(res_code, {})
                if report.get('should_exclude', False):
                    continue
                
                strategy = report.get('strategy', 'ENSEMBLE_EQUAL')
                weights = STRATEGY_WEIGHTS.get(strategy, {'ml': 0.5, 'ai': 0.5})
                
                # If not using AI, force ML-only
                if not self.use_ai:
                    weights = {'ml': 1.0, 'ai': 0.0}
                
                df_res = df_train[df_train['restaurant_code'] == res_code]
                if df_res.empty or df_res['date'].nunique() < 7:  # type: ignore[reportAttributeAccessIssue]
                    continue
                
                # Outlier cleaning
                _, df_res_cleaned = AnalysisAgent.detect_outliers(df_res)  # type: ignore[reportArgumentType]
                
                # Feature engineering
                df_processed = MLForecastAgent.prepare_data(df_res_cleaned, vn_holidays)
                if df_processed.empty:
                    continue
                
                # AI forecast (optional)
                ai_daily_map = {}
                if self.use_ai and weights['ai'] > 0:
                    upcoming_days = [d for d in next_days if d['date'] >= sim_date]
                    if upcoming_days:
                        hist_text = AIForecastAgent.prepare_enhanced_prompt(
                            df_res_cleaned, sim_date, vn_holidays,
                            analysis_report=report
                        )
                        ai_resp = AIForecastAgent.generate_forecast(
                            res_code, hist_text, upcoming_days,
                            analysis_report=report
                        )
                        ai_data = AIForecastAgent.parse_response(ai_resp) if ai_resp else None
                        if ai_data:
                            for item in ai_data:
                                ai_daily_map[str(item.get('date'))] = item.get('forecast', 0)
                
                # Ensemble forecast
                predictions, ensemble_info = EnsembleForecastAgent.run_ensemble_forecast(
                    res_code=res_code,
                    df_res_cleaned=df_res_cleaned,
                    df_processed=df_processed,
                    next_days_info=next_days,
                    vn_holidays=vn_holidays,
                    analysis_report=report,
                    ai_daily_map=ai_daily_map,
                )
                
                if predictions:
                    for p in predictions:
                        all_predictions.append({
                            'sim_date': sim_date,
                            'restaurant_code': str(res_code),
                            'date': p['date'],
                            'weekday': p['weekday'],
                            'hour': p['hour'],
                            'predicted': p['forecast'],
                            'strategy': strategy,
                            'models_used': ','.join(ensemble_info.get('models_used', [])),
                        })
                    result.n_success += 1
                
            except Exception as e:
                result.n_failed += 1
                result.errors.append(f"{res_code}: {str(e)[:100]}")
                if self.verbose:
                    logger.debug(f"Error {res_code}: {e}")
                continue
        
        # Match predictions with actuals
        if all_predictions and not df_actual.empty:
            df_pred = pd.DataFrame(all_predictions)
            df_pred['date'] = pd.to_datetime(df_pred['date']).dt.date
            
            # Aggregate actuals by restaurant, date, hour
            df_act = df_actual.groupby(
                ['restaurant_code', 'date', 'hour']
            )['guest_count'].sum().reset_index()
            
            # Merge
            merged = pd.merge(
                df_pred,
                df_act,
                on=['restaurant_code', 'date', 'hour'],
                how='inner',
            )
            
            if not merged.empty:
                result.predictions = merged.to_dict('records')
                
                # Calculate metrics
                result.metrics = self._calculate_metrics(merged)
                
                # Per-restaurant metrics
                for rc, grp in merged.groupby('restaurant_code'):
                    result.restaurant_metrics[rc] = self._calculate_metrics(grp)
        
        result.elapsed_seconds = time.time() - sim_start
        return result
    
    # ==========================================
    # METRICS
    # ==========================================
    
    @staticmethod
    def _calculate_metrics(df: pd.DataFrame) -> Dict:
        """Tính metrics từ merged predictions+actuals.
        ⭐ v4: SMAPE cho low-volume restaurants.
        ⭐ v6: Hybrid metric (MAE for actual<20, MAPE for actual>=20).
        """
        pred = np.array(df['predicted'].values, dtype=float)
        actual = np.array(df['guest_count'].values, dtype=float)
        
        # Filter valid
        valid = (actual >= 0) & (pred >= 0) & np.isfinite(pred) & np.isfinite(actual)
        pred = pred[valid]
        actual = actual[valid]
        
        if len(pred) == 0:
            return {}
        
        errors = pred - actual
        abs_errors = np.abs(errors)
        
        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        bias = float(np.mean(errors))
        
        avg_actual = float(np.mean(actual))
        
        # ⭐ v6: Hybrid metric — MAE for actual<20, MAPE for actual>=20
        low_vol_mask = actual < LOW_VOLUME_THRESHOLD
        high_vol_mask = actual >= LOW_VOLUME_THRESHOLD
        
        # MAPE for high-volume points only (for per-segment reporting)
        nonzero_high = high_vol_mask & (actual > 0)
        if nonzero_high.sum() > 0:
            mape_high = float(np.mean(abs_errors[nonzero_high] / actual[nonzero_high]) * 100)
        else:
            mape_high = np.nan
        
        # MAE for low-volume points
        mae_low = float(np.mean(abs_errors[low_vol_mask])) if low_vol_mask.sum() > 0 else np.nan
        
        # ⭐ v5/v6: WMAPE = sum(|error|) / sum(actual) — replaces old MAPE
        # Old MAPE (|error|/actual per-point average) was REMOVED because it inflates
        # errors for low-volume points (e.g. a 3→5 pred = 67% MAPE but only 2 guests off)
        nonzero = actual > 0
        if nonzero.sum() > 0:
            wmape = float(abs_errors[nonzero].sum() / actual[nonzero].sum() * 100)
        else:
            wmape = np.nan
        
        # Primary MAPE metric = WMAPE (robust, volume-weighted)
        mape = wmape
        metric_type = 'WMAPE'
        
        # ⭐ v6: Hybrid metric — choose best metric per volume segment
        if avg_actual < LOW_VOLUME_THRESHOLD:
            # Low volume: use MAE as primary (MAPE is misleading at small numbers)
            hybrid_metric = mae
            hybrid_type = 'MAE'
        else:
            # Normal/high volume: use WMAPE (more robust than per-point MAPE)
            hybrid_metric = wmape if not np.isnan(wmape) else mape
            hybrid_type = 'WMAPE'
        
        # Hit rate (abs-only: error ≤ 15 guests)
        threshold_abs = MONITORING_CONFIG.get('hit_rate_threshold_abs', 15)
        
        within_abs = abs_errors <= threshold_abs
        hit_rate = float(np.mean(within_abs) * 100)
        
        # Weighted MAE (relative to volume)
        weighted_mae = mae / max(avg_actual, 1)
        
        return {
            'MAE': round(mae, 2),
            'MAPE': round(mape, 1) if not np.isnan(mape) else None,
            'WMAPE': round(wmape, 1) if not np.isnan(wmape) else None,
            'RMSE': round(rmse, 2),
            'Bias': round(bias, 2),
            'Hit_Rate': round(hit_rate, 1),
            'N_samples': int(len(pred)),
            'metric_type': metric_type,              # ⭐ v4
            'avg_actual': round(avg_actual, 1),      # ⭐ v4
            'weighted_mae': round(weighted_mae, 4),  # ⭐ v4
            # ⭐ v6: Hybrid metrics
            'hybrid_metric': round(hybrid_metric, 2) if not np.isnan(hybrid_metric) else None,
            'hybrid_type': hybrid_type,
            'mape_high_vol': round(mape_high, 1) if not np.isnan(mape_high) else None,
            'mae_low_vol': round(mae_low, 2) if not np.isnan(mae_low) else None,
            'n_low_vol_points': int(low_vol_mask.sum()),
            'n_high_vol_points': int(high_vol_mask.sum()),
        }
    
    # ==========================================
    # DATA LOADING
    # ==========================================
    
    def _load_full_data(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> pd.DataFrame:
        """Load tất cả data trong khoảng [start_date, end_date]."""
        try:
            df = DataAgent.load_date_range(self.engine, start_date, end_date)
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    # ==========================================
    # REPORT COMPILATION
    # ==========================================
    
    def _compile_report(self, report: BacktestReport):
        """Tổng hợp kết quả từ tất cả simulations."""
        
        # Collect all predictions
        all_records = []
        for sim in report.simulations:
            all_records.extend(sim.predictions)
        
        if not all_records:
            logger.warning("No prediction records to compile.")
            return
        
        df = pd.DataFrame(all_records)
        
        # --- Overall Metrics ---
        report.overall_metrics = self._calculate_metrics(df)
        
        # --- Per Restaurant ---
        res_metrics = []
        for rc, grp in df.groupby('restaurant_code'):
            m = self._calculate_metrics(grp)
            m['restaurant_code'] = rc
            res_metrics.append(m)
        report.per_restaurant = pd.DataFrame(res_metrics)
        if not report.per_restaurant.empty:
            report.per_restaurant = report.per_restaurant.sort_values(
                'MAPE', ascending=True
            ).reset_index(drop=True)
        
        # --- Per Weekday ---
        wd_metrics = []
        if 'weekday' in df.columns:
            for wd, grp in df.groupby('weekday'):
                m = self._calculate_metrics(grp)
                m['weekday'] = wd
                wd_metrics.append(m)
        report.per_weekday = pd.DataFrame(wd_metrics)
        
        # --- Per Horizon Day (day 1, day 2, ...) ---
        horizon_metrics = []
        if 'sim_date' in df.columns and 'date' in df.columns:
            df['horizon_day'] = (
                pd.to_datetime(df['date']) - pd.to_datetime(df['sim_date'])
            ).dt.days + 1
            
            for hd, grp in df.groupby('horizon_day'):
                m = self._calculate_metrics(grp)
                m['horizon_day'] = int(hd)  # type: ignore[reportArgumentType]
                horizon_metrics.append(m)
        report.per_horizon = pd.DataFrame(horizon_metrics)
        if not report.per_horizon.empty:
            report.per_horizon = report.per_horizon.sort_values('horizon_day')
        
        # --- Per Simulation ---
        sim_rows = []
        for sim in report.simulations:
            sim_rows.append({
                'sim_date': sim.sim_date,
                'n_restaurants': sim.n_restaurants,
                'n_success': sim.n_success,
                'n_failed': sim.n_failed,
                'elapsed_s': round(sim.elapsed_seconds, 1),
                **sim.metrics,
            })
        report.per_simulation = pd.DataFrame(sim_rows)
        
        # ⭐ V8 Task 6: Per-Shift metrics
        shift_metrics = []
        if 'shift' in df.columns:
            for shift, grp in df.groupby('shift'):
                m = self._calculate_metrics(grp)
                m['shift'] = shift
                m['is_weekend_only'] = False
                shift_metrics.append(m)
                
                # Weekend-only per shift (key diagnostic)
                if 'weekday' in grp.columns:
                    we_mask = grp['weekday'].isin(['Saturday', 'Sunday'])
                    if we_mask.sum() >= 3:
                        m_we = self._calculate_metrics(grp[we_mask])
                        m_we['shift'] = f"{shift}_WEEKEND"
                        m_we['is_weekend_only'] = True
                        shift_metrics.append(m_we)
        report.per_shift = pd.DataFrame(shift_metrics)
        
        # ⭐ V8 Task 7: Shift × Weekday cross-breakdown
        shift_wd_metrics = []
        if 'shift' in df.columns and 'weekday' in df.columns:
            for (shift, wd), grp in df.groupby(['shift', 'weekday']):
                if len(grp) >= 2:
                    m = self._calculate_metrics(grp)
                    m['shift'] = shift
                    m['weekday'] = wd
                    shift_wd_metrics.append(m)
        report.per_shift_weekday = pd.DataFrame(shift_wd_metrics)
        
        # ⭐ V8 Task 6: Volume-segmented metrics
        vol_metrics = []
        if 'actual' in df.columns:
            # Classify restaurants by their average actual volume
            res_avg = df.groupby('restaurant_code')['actual'].mean()
            vol_segment_map = {}
            for rc, avg in res_avg.items():
                if avg < 30:
                    vol_segment_map[rc] = 'LOW_VOLUME'
                elif avg < 80:
                    vol_segment_map[rc] = 'MEDIUM_VOLUME'
                else:
                    vol_segment_map[rc] = 'HIGH_VOLUME'
            
            df_vol = df.copy()
            df_vol['volume_segment'] = df_vol['restaurant_code'].map(vol_segment_map)
            
            for seg, grp in df_vol.groupby('volume_segment'):
                m = self._calculate_metrics(grp)
                m['volume_segment'] = seg
                m['n_restaurants'] = grp['restaurant_code'].nunique()
                vol_metrics.append(m)
                
                # Also compute shift-level within each volume segment
                if 'shift' in grp.columns:
                    for shift, shift_grp in grp.groupby('shift'):
                        if len(shift_grp) >= 2:
                            m_s = self._calculate_metrics(shift_grp)
                            m_s['volume_segment'] = f"{seg}_{shift}"
                            m_s['n_restaurants'] = shift_grp['restaurant_code'].nunique()
                            vol_metrics.append(m_s)
        report.per_volume_segment = pd.DataFrame(vol_metrics)
    
    # ==========================================
    # PRINTING & SAVING
    # ==========================================
    
    def _print_summary(self, report: BacktestReport):
        """Print formatted backtest summary."""
        
        logger.info("\n" + "=" * 65)
        logger.info("📊 BACKTEST RESULTS SUMMARY")
        logger.info("=" * 65)
        
        cfg = report.config
        logger.info(f"  Period:           {cfg['start_date']} → {cfg['end_date']}")
        logger.info(f"  Horizon:          {cfg['forecast_horizon']} days")
        logger.info(f"  Simulations:      {len(report.simulations)}")
        logger.info(f"  Total time:       {report.total_elapsed:.0f}s "
                    f"({report.total_elapsed/60:.1f} min)")
        
        if report.overall_metrics:
            m = report.overall_metrics
            logger.info(f"\n  📈 OVERALL ACCURACY:")
            logger.info(f"     MAPE:       {m.get('MAPE', 'N/A')}%")
            logger.info(f"     MAE:        {m.get('MAE', 'N/A')} guests")
            logger.info(f"     RMSE:       {m.get('RMSE', 'N/A')} guests")
            logger.info(f"     Bias:       {m.get('Bias', 'N/A')} guests")
            logger.info(f"     Hit Rate:   {m.get('Hit_Rate', 'N/A')}%")
            logger.info(f"     Samples:    {m.get('N_samples', 0):,}")
        
        # Per-simulation
        if not report.per_simulation.empty:
            logger.info(f"\n  📅 PER SIMULATION:")
            for _, row in report.per_simulation.iterrows():
                logger.info(
                    f"     {row['sim_date']} | "
                    f"MAPE: {row.get('MAPE', 'N/A'):>5}% | "
                    f"MAE: {row.get('MAE', 'N/A'):>6} | "
                    f"Hit: {row.get('Hit_Rate', 'N/A'):>5}% | "
                    f"OK: {row['n_success']}/{row['n_restaurants']} | "
                    f"{row['elapsed_s']:.0f}s"
                )
        
        # Per horizon day (accuracy decay)
        if not report.per_horizon.empty:
            logger.info(f"\n  📉 ACCURACY BY FORECAST DAY (accuracy decay):")
            for _, row in report.per_horizon.iterrows():
                day = int(row['horizon_day'])
                mape = row.get('MAPE', 'N/A')
                bar = "█" * min(50, int(float(mape) / 2)) if isinstance(mape, (int, float)) else ""
                logger.info(f"     Day {day:>2}: MAPE {mape:>5}% {bar}")
        
        # Per weekday
        if not report.per_weekday.empty:
            logger.info(f"\n  📊 ACCURACY BY WEEKDAY:")
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                        'Friday', 'Saturday', 'Sunday']
            wd_df = report.per_weekday.copy()
            wd_df['sort_key'] = wd_df['weekday'].apply(
                lambda x: day_order.index(x) if x in day_order else 99
            )
            wd_df = wd_df.sort_values('sort_key')
            for _, row in wd_df.iterrows():
                logger.info(
                    f"     {row['weekday']:>10}: MAPE {row.get('MAPE', 'N/A'):>5}% | "
                    f"MAE {row.get('MAE', 'N/A'):>6} | "
                    f"Hit {row.get('Hit_Rate', 'N/A'):>5}%"
                )
        
        # Top/Bottom restaurants
        if not report.per_restaurant.empty and len(report.per_restaurant) >= 5:
            n_show = min(10, len(report.per_restaurant))
            logger.info(f"\n  🏆 TOP {n_show} BEST RESTAURANTS (lowest MAPE):")
            for _, row in report.per_restaurant.head(n_show).iterrows():
                logger.info(
                    f"     {row['restaurant_code']:>8}: MAPE {row.get('MAPE', 'N/A'):>5}% | "
                    f"MAE {row.get('MAE', 'N/A'):>6}"
                )
            
            logger.info(f"\n  ⚠️ TOP {n_show} WORST RESTAURANTS (highest MAPE):")
            for _, row in report.per_restaurant.tail(n_show).iterrows():
                logger.info(
                    f"     {row['restaurant_code']:>8}: MAPE {row.get('MAPE', 'N/A'):>5}% | "
                    f"MAE {row.get('MAE', 'N/A'):>6}"
                )
        
        # ⭐ V8 Task 6: Shift-level accuracy
        if not report.per_shift.empty:
            logger.info(f"\n  🔄 ACCURACY BY SHIFT:")
            for _, row in report.per_shift.iterrows():
                we_flag = " 🎯" if row.get('is_weekend_only', False) else ""
                logger.info(
                    f"     {row['shift']:>20}: MAPE {row.get('MAPE', 'N/A'):>5}% | "
                    f"MAE {row.get('MAE', 'N/A'):>6} | "
                    f"N={row.get('N_samples', 0):>4}{we_flag}"
                )
        
        # ⭐ V8 Task 7: Shift × Weekday cross-breakdown (focus on Weekend×EVENING)
        if not report.per_shift_weekday.empty:
            logger.info(f"\n  📋 SHIFT × WEEKDAY BREAKDOWN (Weekend highlighted):")
            swd = report.per_shift_weekday.copy()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                        'Friday', 'Saturday', 'Sunday']
            swd['wd_idx'] = swd['weekday'].apply(
                lambda x: day_order.index(x) if x in day_order else 99
            )
            swd = swd.sort_values(['shift', 'wd_idx'])
            for _, row in swd.iterrows():
                is_target = (
                    row.get('weekday') in ('Saturday', 'Sunday') and 
                    row.get('shift') == 'EVENING'
                )
                flag = " ⚠️ TARGET" if is_target else ""
                logger.info(
                    f"     {row['shift']:>8} × {row['weekday']:>10}: "
                    f"MAPE {row.get('MAPE', 'N/A'):>5}% | "
                    f"MAE {row.get('MAE', 'N/A'):>6}{flag}"
                )
        
        # ⭐ V8 Task 6: Volume-segmented accuracy
        if not report.per_volume_segment.empty:
            logger.info(f"\n  📊 ACCURACY BY VOLUME SEGMENT:")
            for _, row in report.per_volume_segment.iterrows():
                logger.info(
                    f"     {row['volume_segment']:>25}: MAPE {row.get('MAPE', 'N/A'):>5}% | "
                    f"MAE {row.get('MAE', 'N/A'):>6} | "
                    f"N_res={row.get('n_restaurants', 'N/A')}"
                )
        
        logger.info("\n" + "=" * 65)
        logger.info("✅ Backtest completed!")
        logger.info("=" * 65)
    
    def _save_report(self, report: BacktestReport):
        """Save backtest results to Excel + JSON."""
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # --- JSON summary ---
        json_path = str(PROJECT_ROOT / f"backtest_results_{timestamp}.json")
        summary = {
            'config': {k: str(v) for k, v in report.config.items()},
            'overall_metrics': report.overall_metrics,
            'n_simulations': len(report.simulations),
            'total_elapsed_seconds': round(report.total_elapsed, 1),
            'per_simulation': report.per_simulation.to_dict('records') if not report.per_simulation.empty else [],
        }
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"\n💾 JSON saved: {json_path}")
        except Exception as e:
            logger.warning(f"Failed to save JSON: {e}")
        
        # --- Excel report ---
        excel_path = str(PROJECT_ROOT / f"Backtest_Report_{timestamp}.xlsx")
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Sheet 1: Summary
                summary_df = pd.DataFrame([report.overall_metrics])
                summary_df.insert(0, 'Group', 'OVERALL')
                summary_df.to_excel(writer, sheet_name='Overall', index=False)
                
                # Sheet 2: Per Simulation
                if not report.per_simulation.empty:
                    report.per_simulation.to_excel(
                        writer, sheet_name='Per Simulation', index=False
                    )
                
                # Sheet 3: Per Restaurant
                if not report.per_restaurant.empty:
                    report.per_restaurant.to_excel(
                        writer, sheet_name='Per Restaurant', index=False
                    )
                
                # Sheet 4: Per Weekday
                if not report.per_weekday.empty:
                    report.per_weekday.to_excel(
                        writer, sheet_name='Per Weekday', index=False
                    )
                
                # Sheet 5: Per Horizon Day
                if not report.per_horizon.empty:
                    report.per_horizon.to_excel(
                        writer, sheet_name='Accuracy Decay', index=False
                    )
                
                # ⭐ V8 Sheet 6: Per Shift
                if not report.per_shift.empty:
                    report.per_shift.to_excel(
                        writer, sheet_name='Per Shift', index=False
                    )
                
                # ⭐ V8 Sheet 7: Shift × Weekday
                if not report.per_shift_weekday.empty:
                    report.per_shift_weekday.to_excel(
                        writer, sheet_name='Shift x Weekday', index=False
                    )
                
                # ⭐ V8 Sheet 8: Per Volume Segment
                if not report.per_volume_segment.empty:
                    report.per_volume_segment.to_excel(
                        writer, sheet_name='Per Volume Segment', index=False
                    )
                
                # Sheet 9: Config
                config_df = pd.DataFrame([
                    {'parameter': k, 'value': str(v)}
                    for k, v in report.config.items()
                ])
                config_df.to_excel(writer, sheet_name='Config', index=False)
            
            logger.info(f"💾 Excel saved: {excel_path}")
        except Exception as e:
            logger.warning(f"Failed to save Excel: {e}")
    
    # ==========================================
    # HELPERS
    # ==========================================
    
    @staticmethod
    def _build_config(start_date, end_date, horizon, step, lookback) -> Dict:
        return {
            'start_date': start_date,
            'end_date': end_date,
            'forecast_horizon': horizon,
            'step_days': step,
            'lookback_days': lookback,
            'timestamp': datetime.datetime.now().isoformat(),
        }


# ==========================================
# CLI ENTRY POINT
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="🔬 Backtest Forecast System - Walk-Forward Simulation"
    )
    parser.add_argument(
        '--start', type=str, required=True,
        help='Start date (YYYY-MM-DD). First simulation date.'
    )
    parser.add_argument(
        '--end', type=str, required=True,
        help='End date (YYYY-MM-DD). Last simulation date.'
    )
    parser.add_argument(
        '--horizon', type=int, default=7,
        help='Forecast horizon per simulation (days). Default: 7'
    )
    parser.add_argument(
        '--step', type=int, default=7,
        help='Days between simulations. Default: 7'
    )
    parser.add_argument(
        '--lookback', type=int, default=120,
        help='Training data lookback (days). Default: 120'
    )
    parser.add_argument(
        '--max-restaurants', type=int, default=None,
        help='Max restaurants to backtest (for quick testing). Default: all'
    )
    parser.add_argument(
        '--restaurants', type=str, default=None,
        help='Comma-separated restaurant codes to backtest. Default: all'
    )
    parser.add_argument(
        '--use-ai', action='store_true',
        help='Include AI (LM Studio) in backtest. Default: ML+Prophet only.'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Less verbose output.'
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.date.fromisoformat(args.start)
    end_date = datetime.date.fromisoformat(args.end)
    
    # Parse restaurant filter
    restaurant_filter = None
    if args.restaurants:
        restaurant_filter = [r.strip() for r in args.restaurants.split(',')]
    
    # Setup logger
    log_dir = str(PROJECT_ROOT / "forecast_system" / "logs")
    setup_logger('backtest', log_dir=log_dir)
    
    # Run
    engine = BacktestEngine(
        use_ai=args.use_ai,
        verbose=not args.quiet,
    )
    
    report = engine.run_backtest(
        start_date=start_date,
        end_date=end_date,
        forecast_horizon=args.horizon,
        step_days=args.step,
        lookback_days=args.lookback,
        restaurant_filter=restaurant_filter,  # type: ignore[reportArgumentType]
        max_restaurants=args.max_restaurants,
    )
    
    # Exit code based on results
    if report.overall_metrics:
        mape = report.overall_metrics.get('MAPE', 999)
        if mape and mape < 30:
            print(f"\n✅ Backtest PASSED (MAPE: {mape}% < 30%)")
            sys.exit(0)
        else:
            print(f"\n⚠️ Backtest WARNING (MAPE: {mape}% >= 30%)")
            sys.exit(0)
    else:
        print("\n❌ Backtest FAILED (no results)")
        sys.exit(1)


if __name__ == '__main__':
    main()
