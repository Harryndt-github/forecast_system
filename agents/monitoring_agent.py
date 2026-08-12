"""
==============================================
MONITORING AGENT (MỚI - PHASE 4)
==============================================
Trách nhiệm:
- Đo lường accuracy: MAE, MAPE, RMSE, Bias per restaurant
- Tracking performance theo thời gian (daily, weekly)  
- Drift detection: Cảnh báo khi accuracy giảm
- Restaurant-level alerts: Nhà hàng nào cần retune
- Accuracy report generation (text + Excel)
- Model comparison: So sánh ML vs AI vs Ensemble

Metrics (PER RESTAURANT PER DAY):
    MAE  = Mean Absolute Error (đơn vị: guests)
    MAPE = Mean Absolute Percentage Error (%)
    RMSE = Root Mean Squared Error
    Bias = Mean Error (dương = overpredict, âm = underpredict)
    Hit Rate = % ngày mà error ≤ 15% VÀ error ≤ 10 guests (AND logic)
"""

import numpy as np
import pandas as pd
import datetime
import os
import json
import traceback
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from forecast_system.config.settings import (
    CURRENT_DATE, MASTER_FILE_NAME, LOG_DIR, PROJECT_ROOT,
    MONITORING_CONFIG,
)
from forecast_system.utils.logger import get_logger

logger = get_logger('monitoring_agent')


class MonitoringAgent:
    """
    Agent giám sát accuracy của hệ thống forecast.
    
    Features:
    1. Accuracy Calculation
       - Per restaurant, per day, per category
       - Rolling accuracy (7d, 14d, 30d windows)
       
    2. Drift Detection
       - So sánh accuracy tuần này vs tuần trước
       - Alert khi accuracy giảm > threshold
       
    3. Reporting
       - Daily accuracy report
       - Top/Bottom performers
       - Model comparison (ML vs AI)
       
    4. Alerting
       - Restaurants cần retune
       - System-level accuracy degradation
    """
    
    # File paths
    ACCURACY_HISTORY_FILE = str(PROJECT_ROOT / "accuracy_history.json")
    ACCURACY_REPORT_FILE = str(PROJECT_ROOT / "Accuracy_Report.xlsx")
    
    # ==========================================
    # CORE ACCURACY METRICS
    # ==========================================
    
    @staticmethod
    def calculate_metrics(
        df: pd.DataFrame,
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Tính toán accuracy metrics từ master file.
        
        Args:
            df: Master forecast DataFrame (phải có Actual_Guest + Final_Predicted_Guests)
            group_by: Optional grouping column ('Restaurant_Code', 'Date', 'Weekday', etc.)
        
        Returns:
            DataFrame với metrics: MAE, MAPE, RMSE, Bias, Hit_Rate, N_samples
        """
        # Filter: chỉ lấy rows có cả predicted và actual
        df_work = df.copy()
        # Ensure Date column doesn't have NaT (pandas 3.0 compat)
        if 'Date' in df_work.columns:
            df_work['Date'] = pd.to_datetime(df_work['Date'], errors='coerce')
            df_work = df_work.dropna(subset=['Date'])
            df_work['Date'] = df_work['Date'].dt.date
        
        mask = (
            pd.notna(df_work['Final_Predicted_Guests']) &
            pd.notna(df_work['Actual_Guest']) &
            (df_work['Final_Predicted_Guests'] >= 0) &
            (df_work['Actual_Guest'] >= 0)
        )
        df_valid = df_work[mask].copy()
        
        if df_valid.empty:
            logger.warning("No valid prediction/actual pairs found for metrics")
            return pd.DataFrame()
        
        # Calculate errors
        df_valid['abs_error'] = (
            df_valid['Final_Predicted_Guests'] - df_valid['Actual_Guest']
        ).abs()  # type: ignore[reportAttributeAccessIssue]
        df_valid['squared_error'] = (
            df_valid['Final_Predicted_Guests'] - df_valid['Actual_Guest']
        ) ** 2
        df_valid['error'] = (
            df_valid['Final_Predicted_Guests'] - df_valid['Actual_Guest']
        )
        
        # Percentage error (avoid division by zero)
        nonzero_actual = df_valid['Actual_Guest'] > 0
        df_valid.loc[nonzero_actual, 'pct_error'] = (
            df_valid.loc[nonzero_actual, 'abs_error'] /
            df_valid.loc[nonzero_actual, 'Actual_Guest']
        ) * 100
        df_valid.loc[~nonzero_actual, 'pct_error'] = np.nan
        
        # Hit rate: error ≤ 15 guests (đơn giản, abs-only)
        threshold_abs = MONITORING_CONFIG.get('hit_rate_threshold_abs', 15)
        
        within_abs = df_valid['abs_error'] <= threshold_abs
        df_valid['hit'] = within_abs.astype(int)
        
        # Group and aggregate
        def _agg_metrics(group):
            n = len(group)
            if n == 0:
                return pd.Series({
                    'MAE': np.nan, 'MAPE': np.nan, 'WMAPE': np.nan,
                    'RMSE': np.nan, 'Bias': np.nan, 'Hit_Rate': np.nan,
                    'N_samples': 0
                })
            
            mae = group['abs_error'].mean()
            rmse = np.sqrt(group['squared_error'].mean())
            bias = group['error'].mean()
            hit_rate = group['hit'].mean() * 100
            
            # ⭐ v5/v6: WMAPE = sum(|error|) / sum(actual) — replaces old per-point MAPE
            # Old: mape = mean(|error|/actual) — inflates errors for low-volume
            total_actual = group.loc[
                group['Actual_Guest'] > 0, 'Actual_Guest'
            ].sum()
            total_abs_error = group.loc[
                group['Actual_Guest'] > 0, 'abs_error'
            ].sum()
            wmape = (total_abs_error / total_actual * 100) if total_actual > 0 else np.nan
            
            # Primary MAPE = WMAPE (volume-weighted, robust)
            mape = wmape
            
            return pd.Series({
                'MAE': round(mae, 2),
                'MAPE': round(mape, 1) if pd.notna(mape) else np.nan,
                'WMAPE': round(wmape, 1) if pd.notna(wmape) else np.nan,
                'RMSE': round(rmse, 2),
                'Bias': round(bias, 2),
                'Hit_Rate': round(hit_rate, 1),
                'N_samples': n,
            })
        
        if group_by and group_by in df_valid.columns:
            metrics = df_valid.groupby(group_by).apply(
                _agg_metrics, include_groups=False
            ).reset_index()
        else:
            metrics = _agg_metrics(df_valid).to_frame().T
            metrics.insert(0, 'Group', 'OVERALL')
        
        return metrics
    
    @staticmethod
    def calculate_daily_accuracy(df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính accuracy theo ngày (daily trend).
        
        Returns:
            DataFrame: Date, MAE, MAPE, RMSE, Bias, Hit_Rate, N
        """
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce')
        df_copy = df_copy.dropna(subset=['Date'])
        df_copy['Date'] = df_copy['Date'].dt.date
        return MonitoringAgent.calculate_metrics(df_copy, group_by='Date')
    
    @staticmethod
    def calculate_restaurant_accuracy(df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính accuracy per restaurant (hourly-level metrics).
        
        Returns:
            DataFrame sorted by MAPE ascending (best → worst)
        """
        metrics = MonitoringAgent.calculate_metrics(df, group_by='Restaurant_Code')
        if not metrics.empty:
            metrics = metrics.sort_values('MAPE', ascending=True)
        return metrics
    
    @staticmethod
    def calculate_restaurant_daily_accuracy(df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính accuracy PER RESTAURANT PER DAY.
        
        Logic:
        1. Group theo (Restaurant_Code, Date)
        2. Tổng hợp predicted/actual cho cả ngày của nhà hàng đó
        3. Tính error, MAPE, Hit Rate ở cấp ngày
        4. Hit = error ≤ 15% VÀ error ≤ 10 guests
        5. Trả về metrics CHO TỪNG nhà hàng riêng biệt
        
        Returns:
            DataFrame: Restaurant_Code, N_days, MAE, MAPE, WMAPE, Bias,
                       Hit_Rate, Worst_Day_Error, Best_Day_Error
        """
        df_work = df.copy()
        df_work['Date'] = pd.to_datetime(df_work['Date'], errors='coerce')
        df_work = df_work.dropna(subset=['Date'])
        df_work['Date'] = df_work['Date'].dt.date
        
        mask = (
            pd.notna(df_work['Final_Predicted_Guests']) &
            pd.notna(df_work['Actual_Guest']) &
            (df_work['Final_Predicted_Guests'] >= 0) &
            (df_work['Actual_Guest'] >= 0)
        )
        df_valid = df_work[mask].copy()
        
        if df_valid.empty:
            return pd.DataFrame()
        
        # Step 1: Aggregate to daily level per restaurant
        daily = df_valid.groupby(['Restaurant_Code', 'Date']).agg(
            Predicted_Total=('Final_Predicted_Guests', 'sum'),
            Actual_Total=('Actual_Guest', 'sum'),
            N_hours=('Final_Predicted_Guests', 'count'),
        ).reset_index()
        
        # Step 2: Calculate daily errors
        daily['error'] = daily['Predicted_Total'] - daily['Actual_Total']
        daily['abs_error'] = daily['error'].abs()
        daily['pct_error'] = np.where(
            daily['Actual_Total'] > 0,
            daily['abs_error'] / daily['Actual_Total'] * 100,
            np.nan
        )
        
        # Step 3: Hit Rate per day (abs-only: error ≤ 15 guests)
        threshold_abs = MONITORING_CONFIG.get('hit_rate_threshold_abs', 15)
        
        within_abs = daily['abs_error'] <= threshold_abs
        daily['hit'] = within_abs.astype(int)
        
        # Step 4: Aggregate per restaurant
        results = []
        for res_code, grp in daily.groupby('Restaurant_Code'):
            n_days = len(grp)
            mae = grp['abs_error'].mean()
            bias = grp['error'].mean()
            
            # ⭐ v5/v6: WMAPE as primary MAPE metric
            # Old: mape = mean(pct_error per point) — REMOVED
            # WMAPE = sum(|error|) / sum(actual) — robust, volume-weighted
            total_actual = grp['Actual_Total'].sum()
            total_abs_error = grp['abs_error'].sum()
            wmape = (total_abs_error / total_actual * 100) if total_actual > 0 else np.nan
            mape = wmape  # Primary MAPE = WMAPE
            
            # Hit Rate
            hit_rate = grp['hit'].mean() * 100
            
            # Worst / Best day
            worst_day_error = grp['abs_error'].max()
            best_day_error = grp['abs_error'].min()
            
            # Daily actual average
            avg_daily_actual = grp['Actual_Total'].mean()
            
            results.append({
                'Restaurant_Code': res_code,
                'N_days': n_days,
                'Avg_Daily_Actual': round(avg_daily_actual, 1),
                'MAE': round(mae, 2),
                'MAPE': round(mape, 1) if pd.notna(mape) else np.nan,
                'WMAPE': round(wmape, 1) if pd.notna(wmape) else np.nan,
                'Bias': round(bias, 2),
                'Hit_Rate': round(hit_rate, 1),
                'Worst_Day_Error': round(worst_day_error, 1),
                'Best_Day_Error': round(best_day_error, 1),
            })
        
        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df = result_df.sort_values('MAPE', ascending=True).reset_index(drop=True)
        
        return result_df
    
    @staticmethod
    def calculate_weekday_accuracy(df: pd.DataFrame) -> pd.DataFrame:
        """Tính accuracy per weekday (giúp detect weekend/weekday bias)"""
        return MonitoringAgent.calculate_metrics(df, group_by='Weekday')
    
    @staticmethod
    def calculate_hourly_accuracy(df: pd.DataFrame) -> pd.DataFrame:
        """Tính accuracy per hour (giúp detect peak vs off-peak bias)"""
        return MonitoringAgent.calculate_metrics(df, group_by='Hour')
    
    # ==========================================
    # RUN PERFORMANCE (PER FORECAST RUN DATE)
    # ==========================================
    
    @staticmethod
    def calculate_run_performance(df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính hiệu suất của MỖI LẦN chạy model (theo Forecast_Run_Date).
        
        Kết quả bao gồm:
        - Metrics (MAE, MAPE, RMSE, Bias, Hit_Rate) cho từng lần chạy
        - Số lượng nhà hàng, ngày dự đoán, samples
        - So sánh với lần chạy trước (Δ delta columns)
        - Đánh giá tổng quan: Tốt / Xấu / Ổn định
        
        Returns:
            DataFrame với performance cho từng lần chạy model
        """
        df_copy = df.copy()
        
        # Ensure correct types (pandas 3.0 compat: drop NaT before .dt.date)
        df_copy['Forecast_Run_Date'] = pd.to_datetime(
            df_copy['Forecast_Run_Date'], errors='coerce'
        )
        df_copy['Date'] = pd.to_datetime(
            df_copy['Date'], errors='coerce'
        )
        # Drop rows with invalid dates to avoid NaT comparison errors
        df_copy = df_copy.dropna(subset=['Forecast_Run_Date', 'Date'])
        df_copy['Forecast_Run_Date'] = df_copy['Forecast_Run_Date'].dt.date
        df_copy['Date'] = df_copy['Date'].dt.date
        
        df_copy['Actual_Guest'] = pd.to_numeric(
            df_copy['Actual_Guest'], errors='coerce'
        )
        df_copy['Final_Predicted_Guests'] = pd.to_numeric(
            df_copy['Final_Predicted_Guests'], errors='coerce'
        )
        
        # Only rows with valid actual + prediction (for metrics calculation)
        valid = df_copy.dropna(subset=['Actual_Guest', 'Final_Predicted_Guests'])
        valid = valid[valid['Final_Predicted_Guests'] >= 0]
        
        # Get ALL run dates (including those without actuals yet)
        all_run_dates = sorted(df_copy['Forecast_Run_Date'].unique())
        valid_run_dates = set(valid['Forecast_Run_Date'].unique()) if not valid.empty else set()  # type: ignore[reportAttributeAccessIssue]
        
        rows = []
        
        # 1. Runs WITH actuals → calculate full metrics
        for run_date in all_run_dates:
            if run_date not in valid_run_dates:
                continue
            
            run_data = valid[valid['Forecast_Run_Date'] == run_date]
            
            if run_data.empty:  # type: ignore[reportAttributeAccessIssue]
                continue
            
            actual = run_data['Actual_Guest']
            predicted = run_data['Final_Predicted_Guests']
            diff = actual - predicted
            abs_diff = diff.abs()  # type: ignore[reportAttributeAccessIssue]
            
            # Core metrics
            mae = round(abs_diff.mean(), 2)
            
            # ⭐ v5/v6: WMAPE as primary MAPE metric
            # Old: mape = mean(|error|/actual per point) — REMOVED
            non_zero = run_data[actual > 0]
            if not non_zero.empty:  # type: ignore[reportAttributeAccessIssue]
                non_zero_total = non_zero['Actual_Guest'].sum()
                non_zero_abs_error = (non_zero['Actual_Guest'] - non_zero['Final_Predicted_Guests']).abs().sum()  # type: ignore[reportAttributeAccessIssue]
                if non_zero_total > 0:
                    mape = round(non_zero_abs_error / non_zero_total * 100, 1)
                else:
                    mape = None
            else:
                mape = None
            
            rmse = round(np.sqrt((diff ** 2).mean()), 2)
            bias = round(diff.mean(), 2)
            
            # Hit Rate (abs-only: error ≤ 15 guests)
            threshold_abs = MONITORING_CONFIG.get('hit_rate_threshold_abs', 15)
            
            w_abs = abs_diff <= threshold_abs
            hits = w_abs
            
            hit_rate = round(float(pd.Series(hits).mean() * 100), 1)
            
            # Coverage info
            n_restaurants = run_data['Restaurant_Code'].nunique()  # type: ignore[reportAttributeAccessIssue]
            forecast_dates = run_data['Date'].unique()  # type: ignore[reportAttributeAccessIssue]
            n_forecast_days = len(forecast_dates)
            date_range = f"{min(forecast_dates)} → {max(forecast_dates)}"
            n_samples = len(run_data)
            
            # Calculate actual coverage percentage
            all_run_rows = df_copy[df_copy['Forecast_Run_Date'] == run_date]
            total_predictions = len(all_run_rows[all_run_rows['Final_Predicted_Guests'].notna()])  # type: ignore[reportAttributeAccessIssue]
            actual_pct = round(n_samples / max(total_predictions, 1) * 100, 0)
            
            rows.append({
                'Ngày_Chạy_Model': run_date,
                'Số_NH': n_restaurants,
                'Số_Ngày_Dự_Báo': n_forecast_days,
                'Phạm_Vi_Dự_Báo': date_range,
                'N_Samples': n_samples,
                'Actual_%': actual_pct,
                'MAE': mae,
                'MAPE': mape,
                'RMSE': rmse,
                'Bias': bias,
                'Hit_Rate': hit_rate,
            })
        
        # 2. Runs WITHOUT actuals → show as pending
        for run_date in all_run_dates:
            if run_date in valid_run_dates:
                continue
            
            run_all = df_copy[df_copy['Forecast_Run_Date'] == run_date]
            if run_all.empty:
                continue
            
            n_restaurants = run_all['Restaurant_Code'].nunique()  # type: ignore[reportAttributeAccessIssue]
            forecast_dates = run_all['Date'].unique()  # type: ignore[reportAttributeAccessIssue]
            n_forecast_days = len(forecast_dates)
            date_range = f"{min(forecast_dates)} → {max(forecast_dates)}"
            n_predictions = len(run_all[run_all['Final_Predicted_Guests'].notna()])  # type: ignore[reportAttributeAccessIssue]
            
            rows.append({
                'Ngày_Chạy_Model': run_date,
                'Số_NH': n_restaurants,
                'Số_Ngày_Dự_Báo': n_forecast_days,
                'Phạm_Vi_Dự_Báo': date_range,
                'N_Samples': n_predictions,
                'Actual_%': 0,
                'MAE': None,
                'MAPE': None,
                'RMSE': None,
                'Bias': None,
                'Hit_Rate': None,
            })
        
        if not rows:
            return pd.DataFrame()
        
        df_result = pd.DataFrame(rows)
        
        # Sort by run date for proper delta calculation
        df_result = df_result.sort_values('Ngày_Chạy_Model').reset_index(drop=True)
        
        # Add delta columns (compare to previous run - only between runs WITH metrics)
        for metric in ['MAE', 'MAPE', 'RMSE', 'Hit_Rate']:
            delta_col = f'{metric}_Δ'
            df_result[delta_col] = df_result[metric].diff()
            # Round deltas
            df_result[delta_col] = df_result[delta_col].round(2)
        
        # Add performance assessment
        def assess_performance(row):
            """Đánh giá hiệu suất dựa trên thay đổi so với lần trước"""
            # Runs without actuals → pending
            if pd.isna(row.get('MAE')) or row.get('Actual_%', 0) == 0:
                return '⏳ Chờ dữ liệu thực'
            
            mae_d = row.get('MAE_Δ')
            mape_d = row.get('MAPE_Δ')
            hit_d = row.get('Hit_Rate_Δ')
            
            # First run → no comparison
            if pd.isna(mae_d) or pd.isna(mape_d):
                return 'Lần đầu'
            
            # Score: positive = better, negative = worse
            score = 0
            
            # MAE decrease is good
            if mae_d < -0.5:
                score += 2
            elif mae_d < -0.1:
                score += 1
            elif mae_d > 1.0:
                score -= 2
            elif mae_d > 0.3:
                score -= 1
            
            # MAPE decrease is good
            if mape_d is not None and not pd.isna(mape_d):
                if mape_d < -3:
                    score += 2
                elif mape_d < -1:
                    score += 1
                elif mape_d > 5:
                    score -= 2
                elif mape_d > 2:
                    score -= 1
            
            # Hit Rate increase is good
            if hit_d is not None and not pd.isna(hit_d):
                if hit_d > 3:
                    score += 2
                elif hit_d > 1:
                    score += 1
                elif hit_d < -5:
                    score -= 2
                elif hit_d < -2:
                    score -= 1
            
            if score >= 2:
                return '✅ Tốt lên'
            elif score <= -2:
                return '❌ Xấu đi'
            elif score == 0:
                return '➖ Ổn định'
            elif score > 0:
                return '🔼 Cải thiện nhẹ'
            else:
                return '🔽 Giảm nhẹ'
        
        df_result['Đánh_Giá'] = df_result.apply(assess_performance, axis=1)
        
        return df_result
    
    # ==========================================
    # ROLLING ACCURACY (TIME WINDOWS)
    # ==========================================
    
    @staticmethod
    def calculate_rolling_accuracy(
        df: pd.DataFrame,
        windows: List[int] = None  # type: ignore[reportArgumentType]
    ) -> Dict[str, pd.DataFrame]:
        """
        Tính accuracy cho các time windows: 7d, 14d, 30d.
        
        Args:
            df: Master forecast DataFrame
            windows: List of window sizes (days). Default: [7, 14, 30]
        
        Returns:
            Dict: {'7d': metrics_df, '14d': metrics_df, '30d': metrics_df}
        """
        if windows is None:
            windows = [7, 14, 30]
        
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce')
        df_copy = df_copy.dropna(subset=['Date'])
        df_copy['Date'] = df_copy['Date'].dt.date
        
        results = {}
        for w in windows:
            cutoff = CURRENT_DATE - datetime.timedelta(days=w)
            df_window = df_copy[df_copy['Date'] >= cutoff]
            
            metrics = MonitoringAgent.calculate_metrics(df_window)  # type: ignore[reportArgumentType]
            if not metrics.empty:
                results[f'{w}d'] = metrics
        
        return results
    
    # ==========================================
    # MODEL COMPARISON (ML vs AI)
    # ==========================================
    
    @staticmethod
    def compare_ml_vs_ai(df: pd.DataFrame) -> Dict:
        """
        So sánh accuracy giữa ML và AI predictions.
        
        Dùng AI_Raw_Daily_Forecast column để compare.
        
        Returns:
            Dict: {'ensemble': metrics, 'ai_raw': metrics, 'winner': str}
        """
        ai_available = (
            df['AI_Forecast_Available'].fillna(False).astype(bool)
            if 'AI_Forecast_Available' in df.columns
            else True
        )
        mask = (
            pd.notna(df['Final_Predicted_Guests']) &
            pd.notna(df['Actual_Guest']) &
            pd.notna(df['AI_Raw_Daily_Forecast']) &
            (df['Actual_Guest'] > 0) &
            ai_available
        )
        df_valid = df[mask].copy()
        
        if df_valid.empty:
            return {'ensemble': {}, 'ai_raw': {}, 'winner': 'N/A'}
        
        # Ensemble accuracy (daily level)
        daily_ens = df_valid.groupby('Date').agg({
            'Final_Predicted_Guests': 'sum',
            'Actual_Guest': 'sum'
        }).reset_index()
        
        ens_mae = (daily_ens['Final_Predicted_Guests'] - daily_ens['Actual_Guest']).abs().mean()
        # ⭐ v5/v6: WMAPE = sum(|error|) / sum(actual)
        ens_abs_total = (daily_ens['Final_Predicted_Guests'] - daily_ens['Actual_Guest']).abs().sum()
        ens_actual_total = daily_ens['Actual_Guest'].sum()
        ens_mape = (ens_abs_total / ens_actual_total * 100) if ens_actual_total > 0 else np.nan
        
        # AI raw accuracy (daily level)
        daily_ai = df_valid.groupby('Date').agg({
            'AI_Raw_Daily_Forecast': 'first',  # Same daily value per day
            'Actual_Guest': 'sum'
        }).reset_index()
        
        ai_mae = (daily_ai['AI_Raw_Daily_Forecast'] - daily_ai['Actual_Guest']).abs().mean()
        # ⭐ v5/v6: WMAPE
        ai_abs_total = (daily_ai['AI_Raw_Daily_Forecast'] - daily_ai['Actual_Guest']).abs().sum()
        ai_actual_total = daily_ai['Actual_Guest'].sum()
        ai_mape = (ai_abs_total / ai_actual_total * 100) if ai_actual_total > 0 else np.nan
        
        winner = 'ensemble' if ens_mae <= ai_mae else 'ai_raw'
        
        return {
            'ensemble': {
                'MAE': round(ens_mae, 2),
                'MAPE': round(ens_mape, 1),
            },
            'ai_raw': {
                'MAE': round(ai_mae, 2),
                'MAPE': round(ai_mape, 1),
            },
            'winner': winner,
            'improvement': round(
                (ai_mae - ens_mae) / ai_mae * 100, 1
            ) if ai_mae > 0 else 0,
        }
    
    # ==========================================
    # DRIFT DETECTION
    # ==========================================
    
    @staticmethod
    def detect_drift(
        df: pd.DataFrame,
    ) -> Dict:
        """
        So sánh accuracy tuần này vs tuần trước.
        
        Alert nếu:
        - MAPE tăng > drift_threshold (default 15%)
        - MAE tăng > 30%
        - Hit Rate giảm > 10 points
        
        Returns:
            Dict: {
                'has_drift': bool,
                'alerts': list,
                'this_week': metrics,
                'last_week': metrics,
                'changes': dict
            }
        """
        drift_threshold = MONITORING_CONFIG.get('drift_threshold_pct', 15)
        
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce')
        df_copy = df_copy.dropna(subset=['Date'])
        df_copy['Date'] = df_copy['Date'].dt.date
        
        # This week vs last week
        this_week_start = CURRENT_DATE - datetime.timedelta(days=7)
        last_week_start = CURRENT_DATE - datetime.timedelta(days=14)
        
        df_this = df_copy[df_copy['Date'] >= this_week_start]
        df_last = df_copy[
            (df_copy['Date'] >= last_week_start) &
            (df_copy['Date'] < this_week_start)
        ]
        
        metrics_this = MonitoringAgent.calculate_metrics(df_this)  # type: ignore[reportArgumentType]
        metrics_last = MonitoringAgent.calculate_metrics(df_last)  # type: ignore[reportArgumentType]
        
        result = {
            'has_drift': False,
            'alerts': [],
            'this_week': metrics_this.to_dict('records')[0] if not metrics_this.empty else {},
            'last_week': metrics_last.to_dict('records')[0] if not metrics_last.empty else {},
            'changes': {},
        }
        
        if metrics_this.empty or metrics_last.empty:
            return result
        
        tw = metrics_this.iloc[0]
        lw = metrics_last.iloc[0]
        
        # Calculate changes
        changes = {}
        
        for metric in ['MAE', 'MAPE', 'RMSE', 'Hit_Rate']:
            tw_val = tw.get(metric, np.nan)
            lw_val = lw.get(metric, np.nan)
            
            if pd.notna(tw_val) and pd.notna(lw_val) and lw_val > 0:
                pct_change = ((tw_val - lw_val) / lw_val) * 100
                changes[metric] = {
                    'this_week': round(tw_val, 2),
                    'last_week': round(lw_val, 2),
                    'change_pct': round(pct_change, 1),
                }
        
        result['changes'] = changes
        
        # Check for alerts
        alerts = []
        
        # MAPE drift
        if 'MAPE' in changes:
            mape_change = changes['MAPE']['change_pct']
            if mape_change > drift_threshold:
                alerts.append({
                    'level': 'WARNING',
                    'metric': 'MAPE',
                    'message': (
                        f"MAPE increased by {mape_change:.1f}% "
                        f"({changes['MAPE']['last_week']}% → "
                        f"{changes['MAPE']['this_week']}%)"
                    ),
                })
        
        # MAE drift
        if 'MAE' in changes:
            mae_change = changes['MAE']['change_pct']
            if mae_change > 30:
                alerts.append({
                    'level': 'WARNING',
                    'metric': 'MAE',
                    'message': (
                        f"MAE increased by {mae_change:.1f}% "
                        f"({changes['MAE']['last_week']} → "
                        f"{changes['MAE']['this_week']})"
                    ),
                })
        
        # Hit Rate decline
        if 'Hit_Rate' in changes:
            hr_change = (
                changes['Hit_Rate']['this_week'] -
                changes['Hit_Rate']['last_week']
            )
            if hr_change < -10:
                alerts.append({
                    'level': 'CRITICAL',
                    'metric': 'Hit_Rate',
                    'message': (
                        f"Hit Rate dropped by {abs(hr_change):.1f} points "
                        f"({changes['Hit_Rate']['last_week']}% → "
                        f"{changes['Hit_Rate']['this_week']}%)"
                    ),
                })
        
        result['has_drift'] = len(alerts) > 0
        result['alerts'] = alerts
        
        return result
    
    # ==========================================
    # RESTAURANT-LEVEL ALERTS
    # ==========================================
    
    @staticmethod
    def get_problem_restaurants(
        df: pd.DataFrame,
        top_n: int = 20
    ) -> Dict:
        """
        Tìm nhà hàng có accuracy tệ nhất.
        
        Returns:
            Dict: {
                'worst_mape': top N by MAPE,
                'high_bias': restaurants with significant over/under prediction,
                'needs_retune': restaurants that need model retraining
            }
        """
        retune_mape = MONITORING_CONFIG.get('retune_mape_threshold', 40)
        min_samples = MONITORING_CONFIG.get('min_samples_for_alert', 30)
        
        metrics = MonitoringAgent.calculate_restaurant_accuracy(df)
        
        if metrics.empty:
            return {'worst_mape': [], 'high_bias': [], 'needs_retune': []}
        
        # Filter: need enough samples
        metrics_sufficient = metrics[metrics['N_samples'] >= min_samples]
        
        if metrics_sufficient.empty:
            return {'worst_mape': [], 'high_bias': [], 'needs_retune': []}
        
        # Worst MAPE
        worst_mape = metrics_sufficient.nlargest(
            min(top_n, len(metrics_sufficient)), 'MAPE'  # type: ignore[reportArgumentType]
        )[['Restaurant_Code', 'MAE', 'MAPE', 'Bias', 'Hit_Rate', 'N_samples']]
        
        # High bias (over or under predicting consistently)
        high_bias = metrics_sufficient[  # type: ignore[reportCallIssue]
            metrics_sufficient['Bias'].abs() > metrics_sufficient['MAE'] * 0.5  # type: ignore[reportAttributeAccessIssue]
        ].sort_values('Bias', ascending=False)  # type: ignore[reportAttributeAccessIssue]
        
        # Needs retune: MAPE > threshold
        needs_retune = metrics_sufficient[  # type: ignore[reportCallIssue]
            metrics_sufficient['MAPE'] > retune_mape
        ].sort_values('MAPE', ascending=False)  # type: ignore[reportAttributeAccessIssue]
        
        return {
            'worst_mape': worst_mape.to_dict('records'),  # type: ignore[reportAttributeAccessIssue, reportCallIssue]
            'high_bias': high_bias[  # type: ignore[reportCallIssue]
                ['Restaurant_Code', 'Bias', 'MAE', 'MAPE']
            ].to_dict('records'),
            'needs_retune': needs_retune[  # type: ignore[reportCallIssue]
                ['Restaurant_Code', 'MAPE', 'MAE', 'Hit_Rate']
            ].to_dict('records'),
        }
    
    # ==========================================
    # ACCURACY HISTORY TRACKING
    # ==========================================
    
    @staticmethod
    def save_accuracy_snapshot(metrics: Dict):
        """
        Lưu accuracy snapshot hàng ngày vào JSON file.
        Dùng để track accuracy over time.
        """
        history_file = MonitoringAgent.ACCURACY_HISTORY_FILE
        
        # Load existing history
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except Exception:
                history = []
        
        # Add new snapshot
        snapshot = {
            'date': str(CURRENT_DATE),
            'timestamp': datetime.datetime.now().isoformat(),
            'metrics': {},
        }
        
        # Extract overall metrics
        if metrics:
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    snapshot['metrics'][key] = round(val, 2) if isinstance(val, float) else val
                elif isinstance(val, dict):
                    snapshot['metrics'][key] = {
                        k: round(v, 2) if isinstance(v, float) else v
                        for k, v in val.items()
                    }
        
        # Prevent duplicate dates
        history = [h for h in history if h.get('date') != str(CURRENT_DATE)]
        history.append(snapshot)
        
        # Keep last 90 days
        history = history[-90:]
        
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, default=str)
            logger.info(f"Accuracy snapshot saved ({len(history)} entries)")
        except Exception as e:
            logger.warning(f"Failed to save accuracy history: {e}")
    
    @staticmethod
    def load_accuracy_history() -> List[Dict]:
        """Load accuracy history từ JSON file."""
        history_file = MonitoringAgent.ACCURACY_HISTORY_FILE
        
        if not os.path.exists(history_file):
            return []
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    
    # ==========================================
    # REPORT GENERATION
    # ==========================================
    
    @staticmethod
    def generate_full_report(df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive accuracy report.
        
        Returns:
            Dict with all reports and alerts
        """
        report = {
            'report_date': str(CURRENT_DATE),
            'total_rows': len(df),
        }
        
        # Overall metrics
        overall = MonitoringAgent.calculate_metrics(df)
        report['overall'] = overall.to_dict('records')[0] if not overall.empty else {}
        
        # Rolling windows
        rolling = MonitoringAgent.calculate_rolling_accuracy(df)
        report['rolling'] = {
            window: metrics.to_dict('records')[0] if not metrics.empty else {}
            for window, metrics in rolling.items()
        }
        
        # Per restaurant (hourly-level - backward compat)
        per_restaurant = MonitoringAgent.calculate_restaurant_accuracy(df)
        report['per_restaurant_count'] = len(per_restaurant)
        report['best_restaurants'] = (
            per_restaurant.head(10).to_dict('records')
            if not per_restaurant.empty else []
        )
        report['worst_restaurants'] = (
            per_restaurant.tail(10).to_dict('records')
            if not per_restaurant.empty else []
        )
        
        # Per restaurant DAILY (chỉ số riêng từng nhà hàng, tính ở cấp ngày)
        per_restaurant_daily = MonitoringAgent.calculate_restaurant_daily_accuracy(df)
        report['per_restaurant_daily_count'] = len(per_restaurant_daily)
        report['per_restaurant_daily'] = (
            per_restaurant_daily.to_dict('records')
            if not per_restaurant_daily.empty else []
        )
        # Best/worst by daily hit rate
        if not per_restaurant_daily.empty:
            report['best_restaurants_daily'] = (
                per_restaurant_daily.nlargest(10, 'Hit_Rate').to_dict('records')
            )
            report['worst_restaurants_daily'] = (
                per_restaurant_daily.nsmallest(10, 'Hit_Rate').to_dict('records')
            )
        
        # Weekday analysis
        weekday = MonitoringAgent.calculate_weekday_accuracy(df)
        report['weekday'] = weekday.to_dict('records') if not weekday.empty else []
        
        # Hourly analysis
        hourly = MonitoringAgent.calculate_hourly_accuracy(df)
        report['hourly'] = hourly.to_dict('records') if not hourly.empty else []
        
        # Drift detection
        drift = MonitoringAgent.detect_drift(df)
        report['drift'] = drift
        
        # Problem restaurants
        problems = MonitoringAgent.get_problem_restaurants(df)
        report['problem_restaurants'] = problems
        
        # ML vs AI comparison
        comparison = MonitoringAgent.compare_ml_vs_ai(df)
        report['model_comparison'] = comparison
        
        # Run Performance (per Forecast_Run_Date)
        run_perf = MonitoringAgent.calculate_run_performance(df)
        report['run_performance'] = run_perf
        
        # Save snapshot
        MonitoringAgent.save_accuracy_snapshot(report.get('overall', {}))
        
        return report
    
    @staticmethod
    def print_report(report: Dict, logger_func=None):
        """
        Print formatted accuracy report to console/logger.
        """
        log = logger_func or logger.info
        
        log("\n" + "=" * 65)
        log("📊 ACCURACY MONITORING REPORT")
        log(f"   Date: {report.get('report_date', 'N/A')}")
        log("=" * 65)
        
        # Overall
        overall = report.get('overall', {})
        if overall:
            log(f"\n  📋 OVERALL ACCURACY:")
            log(f"     MAE:       {overall.get('MAE', 'N/A')} guests")
            log(f"     MAPE:      {overall.get('MAPE', 'N/A')}%")
            log(f"     RMSE:      {overall.get('RMSE', 'N/A')}")
            log(f"     Bias:      {overall.get('Bias', 'N/A')} "
                f"({'↑ over' if (overall.get('Bias', 0) or 0) > 0 else '↓ under'})")
            log(f"     Hit Rate:  {overall.get('Hit_Rate', 'N/A')}%")
            log(f"     Samples:   {overall.get('N_samples', 0):,}")
        
        # Rolling
        rolling = report.get('rolling', {})
        if rolling:
            log(f"\n  📈 ROLLING ACCURACY (Window → MAPE):")
            for window, metrics in sorted(rolling.items()):
                mape = metrics.get('MAPE', 'N/A')
                mae = metrics.get('MAE', 'N/A')
                hr = metrics.get('Hit_Rate', 'N/A')
                log(f"     {window:>4s}: MAPE={mape}%, MAE={mae}, Hit={hr}%")
        
        # Drift alerts
        drift = report.get('drift', {})
        if drift.get('has_drift'):
            log(f"\n  🚨 DRIFT ALERTS:")
            for alert in drift.get('alerts', []):
                level = alert.get('level', 'INFO')
                msg = alert.get('message', '')
                icon = '🔴' if level == 'CRITICAL' else '🟡'
                log(f"     {icon} [{level}] {msg}")
        else:
            log(f"\n  ✅ No accuracy drift detected")
        
        # Week-over-week changes
        changes = drift.get('changes', {})
        if changes:
            log(f"\n  📊 WEEK-OVER-WEEK:")
            for metric, info in changes.items():
                change = info.get('change_pct', 0)
                arrow = '↑' if change > 0 else '↓'
                log(f"     {metric:>10s}: {info.get('last_week')} → "
                    f"{info.get('this_week')} ({arrow}{abs(change):.1f}%)")
        
        # Model comparison
        comparison = report.get('model_comparison', {})
        if comparison.get('ensemble') and comparison.get('ai_raw'):
            log(f"\n  🤖 MODEL COMPARISON (Daily Level):")
            ens = comparison['ensemble']
            ai = comparison['ai_raw']
            winner = comparison.get('winner', 'N/A')
            improvement = comparison.get('improvement', 0)
            log(f"     Ensemble: MAE={ens.get('MAE')}, MAPE={ens.get('MAPE')}%")
            log(f"     AI Raw:   MAE={ai.get('MAE')}, MAPE={ai.get('MAPE')}%")
            log(f"     Winner:   {winner.upper()} "
                f"(+{improvement}% improvement)" if improvement > 0 else
                f"     Winner:   {winner.upper()}")
        
        # Weekday analysis
        weekday = report.get('weekday', [])
        if weekday:
            log(f"\n  📅 ACCURACY BY WEEKDAY:")
            for wd in weekday:
                day_name = wd.get('Weekday', '?')
                mape = wd.get('MAPE', 'N/A')
                bias = wd.get('Bias', 0) or 0
                direction = '↑' if bias > 0 else '↓'
                log(f"     {day_name:>10s}: MAPE={mape}% | "
                    f"Bias={direction}{abs(bias):.1f}")
        
        # Problem restaurants
        problems = report.get('problem_restaurants', {})
        needs_retune = problems.get('needs_retune', [])
        if needs_retune:
            log(f"\n  ⚠️ RESTAURANTS NEED RETUNING ({len(needs_retune)}):")
            for r in needs_retune[:10]:
                log(f"     {r.get('Restaurant_Code', '?'):>10s}: "
                    f"MAPE={r.get('MAPE', '?')}%, "
                    f"MAE={r.get('MAE', '?')}, "
                    f"Hit={r.get('Hit_Rate', '?')}%")
            if len(needs_retune) > 10:
                log(f"     ... and {len(needs_retune) - 10} more")
        
        # Best performers
        best = report.get('best_restaurants', [])
        if best:
            log(f"\n  🏆 TOP PERFORMERS:")
            for r in best[:5]:
                log(f"     {r.get('Restaurant_Code', '?'):>10s}: "
                    f"MAPE={r.get('MAPE', '?')}%, "
                    f"Hit={r.get('Hit_Rate', '?')}%")
        
        # Per-Restaurant Daily metrics (new)
        per_res_daily = report.get('per_restaurant_daily', [])
        if per_res_daily:
            log(f"\n  📊 PER-RESTAURANT DAILY ACCURACY ({len(per_res_daily)} restaurants):")
            log(f"     {'Code':>10s} | {'Days':>4s} | {'MAPE%':>6s} | {'WMAPE%':>7s} | "
                f"{'Hit%':>5s} | {'MAE':>6s} | {'Bias':>7s}")
            log("     " + "-" * 60)
            
            # Show top 5 best
            best_daily = report.get('best_restaurants_daily', per_res_daily[:5])
            for r in best_daily[:5]:
                log(f"  🟢 {r.get('Restaurant_Code', '?'):>10s} | "
                    f"{r.get('N_days', '?'):>4} | "
                    f"{r.get('MAPE', '?'):>6} | "
                    f"{r.get('WMAPE', '?'):>7} | "
                    f"{r.get('Hit_Rate', '?'):>5} | "
                    f"{r.get('MAE', '?'):>6} | "
                    f"{r.get('Bias', '?'):>7}")
            
            log("     ...")
            
            # Show bottom 5 worst
            worst_daily = report.get('worst_restaurants_daily', per_res_daily[-5:])
            for r in worst_daily[:5]:
                log(f"  🔴 {r.get('Restaurant_Code', '?'):>10s} | "
                    f"{r.get('N_days', '?'):>4} | "
                    f"{r.get('MAPE', '?'):>6} | "
                    f"{r.get('WMAPE', '?'):>7} | "
                    f"{r.get('Hit_Rate', '?'):>5} | "
                    f"{r.get('MAE', '?'):>6} | "
                    f"{r.get('Bias', '?'):>7}")
        
        log("\n" + "=" * 65)
    
    @staticmethod
    def save_report_excel(df: pd.DataFrame, report: Dict):
        """
        Lưu accuracy report ra Excel file với nhiều sheets.
        
        Sheets:
        1. Overall Summary
        2. Per Restaurant
        3. Daily Accuracy
        4. Weekday Analysis
        5. Hourly Analysis
        6. Problem Restaurants
        """
        report_file = MonitoringAgent.ACCURACY_REPORT_FILE
        
        try:
            with pd.ExcelWriter(report_file, engine='xlsxwriter') as writer:
                
                # Sheet 1: Overall Summary
                overall = report.get('overall', {})
                rolling = report.get('rolling', {})
                summary_data = [
                    {'Metric': 'Report Date', 'Value': report.get('report_date')},
                    {'Metric': 'Total Rows', 'Value': report.get('total_rows')},
                    {'Metric': 'MAE', 'Value': overall.get('MAE')},
                    {'Metric': 'MAPE (%)', 'Value': overall.get('MAPE')},
                    {'Metric': 'RMSE', 'Value': overall.get('RMSE')},
                    {'Metric': 'Bias', 'Value': overall.get('Bias')},
                    {'Metric': 'Hit Rate (%)', 'Value': overall.get('Hit_Rate')},
                    {'Metric': 'N Samples', 'Value': overall.get('N_samples')},
                    {'Metric': '---', 'Value': '---'},
                ]
                for window, m in rolling.items():
                    summary_data.append({
                        'Metric': f'MAPE ({window})',
                        'Value': m.get('MAPE')
                    })
                    summary_data.append({
                        'Metric': f'Hit Rate ({window})',
                        'Value': m.get('Hit_Rate')
                    })
                
                pd.DataFrame(summary_data).to_excel(
                    writer, sheet_name='Summary', index=False
                )
                
                # Sheet 2: Per Restaurant (hourly-level - backward compat)
                per_res = MonitoringAgent.calculate_restaurant_accuracy(df)
                if not per_res.empty:
                    per_res.to_excel(
                        writer, sheet_name='Per_Restaurant', index=False
                    )
                
                # Sheet 2B: Per Restaurant DAILY (mới - metrics ở cấp ngày)
                per_res_daily = MonitoringAgent.calculate_restaurant_daily_accuracy(df)
                if not per_res_daily.empty:
                    per_res_daily.to_excel(
                        writer, sheet_name='Per_Restaurant_Daily', index=False
                    )
                
                # Sheet 3: Daily
                daily = MonitoringAgent.calculate_daily_accuracy(df)
                if not daily.empty:
                    daily.to_excel(
                        writer, sheet_name='Daily', index=False
                    )
                
                # Sheet 4: Weekday
                wd_data = report.get('weekday', [])
                if wd_data:
                    pd.DataFrame(wd_data).to_excel(
                        writer, sheet_name='Weekday', index=False
                    )
                
                # Sheet 5: Hourly
                hourly_data = report.get('hourly', [])
                if hourly_data:
                    pd.DataFrame(hourly_data).to_excel(
                        writer, sheet_name='Hourly', index=False
                    )
                
                # Sheet 6: Problem Restaurants
                problems = report.get('problem_restaurants', {})
                retune = problems.get('needs_retune', [])
                if retune:
                    pd.DataFrame(retune).to_excel(
                        writer, sheet_name='Needs_Retune', index=False
                    )
                
                # Sheet 7: Run Performance (So sánh hiệu suất từng lần chạy)
                run_perf = report.get('run_performance')
                if run_perf is not None and isinstance(run_perf, pd.DataFrame) and not run_perf.empty:
                    run_perf.to_excel(
                        writer, sheet_name='Run_Performance', index=False
                    )
                    
                    # Format the sheet with xlsxwriter
                    workbook = writer.book
                    worksheet = writer.sheets['Run_Performance']
                    
                    # Header format
                    header_fmt = workbook.add_format({
                        'bold': True,
                        'bg_color': '#2F5496',
                        'font_color': 'white',
                        'border': 1,
                        'text_wrap': True,
                        'valign': 'vcenter',
                    })
                    
                    # Number formats
                    num_fmt = workbook.add_format({'num_format': '0.00', 'border': 1})
                    pct_fmt = workbook.add_format({'num_format': '0.0', 'border': 1})
                    int_fmt = workbook.add_format({'num_format': '#,##0', 'border': 1})
                    text_fmt = workbook.add_format({'border': 1})
                    
                    # Conditional formats for delta columns
                    good_fmt = workbook.add_format({
                        'bg_color': '#C6EFCE', 'font_color': '#006100',
                        'num_format': '0.00', 'border': 1
                    })
                    bad_fmt = workbook.add_format({
                        'bg_color': '#FFC7CE', 'font_color': '#9C0006',
                        'num_format': '0.00', 'border': 1
                    })
                    
                    # Apply header format
                    for col_num, col_name in enumerate(run_perf.columns):
                        worksheet.write(0, col_num, col_name, header_fmt)
                    
                    # Set column widths
                    col_widths = {
                        'Ngày_Chạy_Model': 16,
                        'Số_NH': 8,
                        'Số_Ngày_Dự_Báo': 14,
                        'Phạm_Vi_Dự_Báo': 28,
                        'N_Samples': 12,
                        'Actual_%': 10,
                        'MAE': 8, 'MAPE': 8, 'RMSE': 10, 'Bias': 8, 'Hit_Rate': 10,
                        'MAE_Δ': 8, 'MAPE_Δ': 8, 'RMSE_Δ': 10, 'Hit_Rate_Δ': 10,
                        'Đánh_Giá': 20,
                    }
                    for col_num, col_name in enumerate(run_perf.columns):
                        width = col_widths.get(col_name, 12)
                        worksheet.set_column(col_num, col_num, width)
                    
                    # Conditional formatting for MAE_Δ (negative = better)
                    mae_delta_col = list(run_perf.columns).index('MAE_Δ') if 'MAE_Δ' in run_perf.columns else None
                    if mae_delta_col is not None:
                        data_rows = len(run_perf)
                        worksheet.conditional_format(
                            1, mae_delta_col, data_rows, mae_delta_col,
                            {'type': 'cell', 'criteria': '<', 'value': 0, 'format': good_fmt}
                        )
                        worksheet.conditional_format(
                            1, mae_delta_col, data_rows, mae_delta_col,
                            {'type': 'cell', 'criteria': '>', 'value': 0, 'format': bad_fmt}
                        )
                    
                    # Conditional formatting for Hit_Rate_Δ (positive = better)
                    hit_delta_col = list(run_perf.columns).index('Hit_Rate_Δ') if 'Hit_Rate_Δ' in run_perf.columns else None
                    if hit_delta_col is not None:
                        worksheet.conditional_format(
                            1, hit_delta_col, data_rows, hit_delta_col,  # type: ignore[reportPossiblyUnboundVariable]
                            {'type': 'cell', 'criteria': '>', 'value': 0, 'format': good_fmt}
                        )
                        worksheet.conditional_format(
                            1, hit_delta_col, data_rows, hit_delta_col,  # type: ignore[reportPossiblyUnboundVariable]
                            {'type': 'cell', 'criteria': '<', 'value': 0, 'format': bad_fmt}
                        )
                    
                    logger.info(f"   ✅ Run_Performance sheet: {len(run_perf)} runs")
            
            logger.info(f"📊 Accuracy report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save accuracy report: {e}")
            traceback.print_exc()
