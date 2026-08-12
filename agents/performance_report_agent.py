"""
==============================================
PERFORMANCE REPORT AGENT
==============================================
Trách nhiệm:
- Tạo Model_Performance_Report.xlsx với 3 chỉ tiêu đánh giá chính:

  KPI 1: Tỷ lệ dự đoán đúng theo ca/ngày cho từng nhà hàng (%)
  KPI 2: Tỷ lệ dự đoán chính xác theo ca trong tuần vs cuối tuần (%)  
  KPI 3: Tỷ lệ sai số trong khoảng ±15 khách/ca + danh sách nhà hàng forecast sai

- Cập nhật Accuracy_Report.xlsx: Chi tiết accuracy mỗi lần chạy

Output - Model_Performance_Report.xlsx:
  Sheet 1: "KPI_1_Restaurant_Shift"   → Accuracy mỗi ca/ngày cho từng nhà hàng
  Sheet 2: "KPI_2_Weekday_Weekend"    → Accuracy ca SÁNG/CHIỀU: ngày thường vs cuối tuần
  Sheet 3: "KPI_3_Tolerance_15"       → Hit Rate ±15 khách + DS nhà hàng sai
  Sheet 4: "Run_History"              → Lịch sử tổng hợp mỗi lần chạy
  Sheet 5: "Daily_Error_Tracking"     → Số nhà hàng sai từng ngày chạy

Output - Accuracy_Report.xlsx:
  Sheet 1: "Summary"                  → Tổng quan accuracy lần chạy mới nhất 
  Sheet 2: "Per_Restaurant_Daily"     → Accuracy từng restaurant
  Sheet 3: "Weekday"                  → Accuracy theo thứ
  Sheet 4: "Needs_Retune"             → Restaurants cần retune
"""

import pandas as pd
import numpy as np
import os
import datetime
import traceback

from forecast_system.config.settings import (
    CURRENT_DATE, MODEL_PERFORMANCE_FILE, ACCURACY_REPORT_FILE,
    ACCURACY_HISTORY_FILE, LOW_VOLUME_THRESHOLD,
)
from forecast_system.utils.logger import get_logger

logger = get_logger('performance_report')

# Tolerance threshold for ±15 guests
TOLERANCE_GUESTS = 15

# Tolerance buckets for shift accuracy chart
TOLERANCE_LEVELS = [5, 10, 15]

# Weekend days (Vietnamese restaurant context)
WEEKEND_DAYS = {'Friday', 'Saturday', 'Sunday'}
WEEKDAY_DAYS = {'Monday', 'Tuesday', 'Wednesday', 'Thursday'}


class PerformanceReportAgent:
    """
    Agent tạo báo cáo hiệu suất model sau mỗi lần chạy.
    Đánh giá theo 3 KPI do quản lý yêu cầu.
    """
    
    # ==========================================
    # CORE DATA PREPARATION
    # ==========================================
    
    @staticmethod
    def _prepare_shift_data(df_master: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn bị dữ liệu shift-level với actual + predicted.
        Chỉ lấy rows có Shift (MORNING/EVENING) VÀ có cả predicted + actual.
        """
        if df_master is None or df_master.empty:
            return pd.DataFrame()
        
        df = df_master.copy()
        
        # Ensure correct dtypes
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Forecast_Run_Date'] = pd.to_datetime(
            df['Forecast_Run_Date'], errors='coerce'
        )
        
        # Filter: chỉ rows có Shift và có cả predicted + actual
        mask = (
            df['Shift'].isin(['MORNING', 'EVENING']) &
            pd.notna(df['Final_Predicted_Guests']) &
            pd.notna(df['Actual_Guest']) &
            (df['Final_Predicted_Guests'] >= 0) &
            (df['Actual_Guest'] >= 0)
        )
        df_shift = df[mask].copy()
        
        if df_shift.empty:
            # Fallback: thử với hourly rows (aggregate to shift)
            mask_hourly = (
                pd.notna(df['Final_Predicted_Guests']) &
                pd.notna(df['Actual_Guest']) &
                pd.notna(df['Hour']) &
                (df['Final_Predicted_Guests'] >= 0) &
                (df['Actual_Guest'] >= 0)
            )
            df_hourly = df[mask_hourly].copy()
            
            if df_hourly.empty:
                return pd.DataFrame()
            
            # Map hour → shift
            df_hourly['Shift'] = df_hourly['Hour'].apply(  # type: ignore[reportAttributeAccessIssue]
                lambda h: 'MORNING' if 9 <= h <= 14 else (
                    'EVENING' if 15 <= h <= 22 else None
                )
            )
            df_hourly = df_hourly[df_hourly['Shift'].notna()].copy()  # type: ignore[reportAttributeAccessIssue]
            
            if df_hourly.empty:  # type: ignore[reportAttributeAccessIssue]
                return pd.DataFrame()
            
            # Aggregate hourly → shift
            df_shift = df_hourly.groupby(  # type: ignore[reportAttributeAccessIssue]
                ['Forecast_Run_Date', 'Restaurant_Code', 'Date',
                 'Weekday', 'Shift']
            ).agg(
                Final_Predicted_Guests=('Final_Predicted_Guests', 'sum'),
                Actual_Guest=('Actual_Guest', 'sum'),
                sap_code=('sap_code', 'first'),
                restaurant_name=('restaurant_name', 'first'),
            ).reset_index()
        
        # Calculate errors
        df_shift['Diff'] = (
            df_shift['Final_Predicted_Guests'] - df_shift['Actual_Guest']
        )
        df_shift['Abs_Diff'] = df_shift['Diff'].abs()  # type: ignore[reportAttributeAccessIssue]
        df_shift['Error_Pct'] = np.where(
            df_shift['Actual_Guest'] > 0,
            df_shift['Abs_Diff'] / df_shift['Actual_Guest'] * 100,
            np.where(df_shift['Final_Predicted_Guests'] == 0, 0, 100)
        )
        
        # Hit: within ±15 guests
        df_shift['Hit_15'] = (df_shift['Abs_Diff'] <= TOLERANCE_GUESTS).astype(int)
        
        # Day type
        df_shift['Day_Type'] = df_shift['Weekday'].apply(  # type: ignore[reportAttributeAccessIssue]
            lambda w: 'Cuối tuần' if w in WEEKEND_DAYS else 'Ngày thường'
        )
        
        return df_shift  # type: ignore[reportReturnType]
    
    # ==========================================
    # KPI 1: Accuracy per SHIFT/DAY per RESTAURANT
    # ==========================================
    
    @staticmethod
    def calculate_kpi1_restaurant_shift(df_shift: pd.DataFrame) -> pd.DataFrame:
        """
        KPI 1: Tỷ lệ dự đoán đúng cho mỗi ca/ngày của một nhà hàng.
        
        Tính:
        - Accuracy = (1 - MAPE) × 100 cho mỗi restaurant + shift
        - Hit Rate ±15 khách
        - MAE per shift
        - ⭐ v6: Hybrid metric (MAE for low vol, WMAPE for normal)
        - ⭐ v6: Volume_Segment column + exclude low vol from global MAPE
        
        Returns: DataFrame với columns:
            Restaurant_Code, sap_code, restaurant_name, Shift,
            N_Shifts (số ca đã so), Accuracy_%, Hit_Rate_15_%, 
            MAE, Avg_Predicted, Avg_Actual, Avg_Error,
            Volume_Segment, Loại_Metric
        """
        if df_shift.empty:
            return pd.DataFrame()
        
        results = []
        
        for (res_code, shift), grp in df_shift.groupby(  # type: ignore[reportGeneralTypeIssues]
            ['Restaurant_Code', 'Shift']
        ):
            n = len(grp)
            
            # MAPE-based accuracy
            total_actual = grp['Actual_Guest'].sum()
            total_abs_error = grp['Abs_Diff'].sum()
            avg_actual = grp['Actual_Guest'].mean()
            
            # ⭐ v6: Volume segmentation for metric selection
            if avg_actual < LOW_VOLUME_THRESHOLD:
                # LOW VOLUME: Use MAE as primary metric (MAPE is misleading)
                mae_val = grp['Abs_Diff'].mean()
                # Accuracy based on MAE relative to avg
                if avg_actual > 0:
                    accuracy = max(0, 100 - (mae_val / avg_actual * 100))
                else:
                    accuracy = 0
                metric_type = 'MAE'
                volume_segment = 'LOW_VOLUME'
            elif total_actual > 0:
                # MEDIUM/HIGH VOLUME: Use WMAPE
                wmape = total_abs_error / total_actual * 100
                accuracy = max(0, 100 - wmape)
                metric_type = 'WMAPE'
                volume_segment = 'HIGH_VOLUME' if avg_actual >= 80 else 'MEDIUM_VOLUME'
            else:
                accuracy = 0
                metric_type = 'MAPE'
                volume_segment = 'MEDIUM_VOLUME'
            
            # Hit Rate ±15 guests
            hit_rate_15 = grp['Hit_15'].mean() * 100
            
            # MAE
            mae = grp['Abs_Diff'].mean()
            
            results.append({
                'Restaurant_Code': res_code,
                'sap_code': grp['sap_code'].iloc[0] if 'sap_code' in grp.columns else '',
                'restaurant_name': grp['restaurant_name'].iloc[0] if 'restaurant_name' in grp.columns else '',
                'Shift': shift,
                'Số_Ca_Đã_So': n,
                'Accuracy_%': round(accuracy, 1),
                'Hit_Rate_±15_Khách_%': round(hit_rate_15, 1),
                'MAE_Khách': round(mae, 1),
                'TB_Dự_Đoán': round(grp['Final_Predicted_Guests'].mean(), 0),
                'TB_Thực_Tế': round(avg_actual, 0),
                'TB_Sai_Số': round(grp['Diff'].mean(), 1),
                'Loại_Metric': metric_type,        # ⭐ v4/v6
                'Volume_Segment': volume_segment,  # ⭐ v6
            })
        
        df_result = pd.DataFrame(results)
        
        if not df_result.empty:
            df_result = df_result.sort_values(
                ['Accuracy_%'], ascending=True
            ).reset_index(drop=True)
        
        return df_result
    
    # ==========================================
    # KPI 2: Accuracy by SHIFT × WEEKDAY/WEEKEND
    # ==========================================
    
    @staticmethod
    def calculate_kpi2_weekday_weekend(df_shift: pd.DataFrame) -> pd.DataFrame:
        """
        KPI 2: Tỷ lệ dự đoán chính xác theo ca của ngày trong tuần vs cuối tuần.
        
        Breakdown:
        - Ca SÁNG × Ngày thường
        - Ca SÁNG × Cuối tuần
        - Ca CHIỀU × Ngày thường
        - Ca CHIỀU × Cuối tuần
        
        Returns: DataFrame với accuracy, hit rate, MAE cho mỗi combination
        """
        if df_shift.empty:
            return pd.DataFrame()
        
        results = []
        
        # ── Part A: Tổng hợp theo Shift × Day_Type ──
        for (shift, day_type), grp in df_shift.groupby(['Shift', 'Day_Type']):  # type: ignore[reportGeneralTypeIssues]
            n = len(grp)
            
            total_actual = grp['Actual_Guest'].sum()
            total_abs_error = grp['Abs_Diff'].sum()
            
            if total_actual > 0:
                wmape = total_abs_error / total_actual * 100
                accuracy = max(0, 100 - wmape)
            else:
                accuracy = 0
            
            hit_rate_15 = grp['Hit_15'].mean() * 100
            mae = grp['Abs_Diff'].mean()
            
            shift_vn = 'Ca Sáng' if shift == 'MORNING' else 'Ca Chiều'
            
            results.append({
                'Ca_Làm_Việc': shift_vn,
                'Loại_Ngày': day_type,
                'Số_Ca_Đã_So': n,
                'Accuracy_%': round(accuracy, 1),
                'Hit_Rate_±15_Khách_%': round(hit_rate_15, 1),
                'MAE_Khách': round(mae, 1),
                'TB_Dự_Đoán': round(grp['Final_Predicted_Guests'].mean(), 0),
                'TB_Thực_Tế': round(grp['Actual_Guest'].mean(), 0),
                'TB_Sai_Số': round(grp['Diff'].mean(), 1),
                'Max_Sai_Số': round(grp['Abs_Diff'].max(), 0),
                'Số_NH': grp['Restaurant_Code'].nunique(),
            })
        
        # ── Part B: Thêm breakdown theo từng thứ trong tuần ──
        results.append({
            'Ca_Làm_Việc': '', 'Loại_Ngày': '',
            'Số_Ca_Đã_So': '', 'Accuracy_%': '',
            'Hit_Rate_±15_Khách_%': '', 'MAE_Khách': '',
            'TB_Dự_Đoán': '', 'TB_Thực_Tế': '',
            'TB_Sai_Số': '', 'Max_Sai_Số': '', 'Số_NH': '',
        })
        results.append({
            'Ca_Làm_Việc': '=== CHI TIẾT THEO THỨ ===',
            'Loại_Ngày': '', 'Số_Ca_Đã_So': '',
            'Accuracy_%': '', 'Hit_Rate_±15_Khách_%': '',
            'MAE_Khách': '', 'TB_Dự_Đoán': '',
            'TB_Thực_Tế': '', 'TB_Sai_Số': '',
            'Max_Sai_Số': '', 'Số_NH': '',
        })
        
        weekday_order = [
            'Monday', 'Tuesday', 'Wednesday', 'Thursday',
            'Friday', 'Saturday', 'Sunday'
        ]
        weekday_vn = {
            'Monday': 'Thứ 2', 'Tuesday': 'Thứ 3',
            'Wednesday': 'Thứ 4', 'Thursday': 'Thứ 5',
            'Friday': 'Thứ 6', 'Saturday': 'Thứ 7',
            'Sunday': 'Chủ Nhật',
        }
        
        for (shift, weekday), grp in df_shift.groupby(['Shift', 'Weekday']):  # type: ignore[reportGeneralTypeIssues]
            total_actual = grp['Actual_Guest'].sum()
            total_abs_error = grp['Abs_Diff'].sum()
            
            if total_actual > 0:
                accuracy = max(0, 100 - total_abs_error / total_actual * 100)
            else:
                accuracy = 0
            
            hit_rate_15 = grp['Hit_15'].mean() * 100
            shift_vn = 'Ca Sáng' if shift == 'MORNING' else 'Ca Chiều'
            
            results.append({
                'Ca_Làm_Việc': shift_vn,
                'Loại_Ngày': weekday_vn.get(weekday, weekday),
                'Số_Ca_Đã_So': len(grp),
                'Accuracy_%': round(accuracy, 1),
                'Hit_Rate_±15_Khách_%': round(hit_rate_15, 1),
                'MAE_Khách': round(grp['Abs_Diff'].mean(), 1),
                'TB_Dự_Đoán': round(grp['Final_Predicted_Guests'].mean(), 0),
                'TB_Thực_Tế': round(grp['Actual_Guest'].mean(), 0),
                'TB_Sai_Số': round(grp['Diff'].mean(), 1),
                'Max_Sai_Số': round(grp['Abs_Diff'].max(), 0),
                'Số_NH': grp['Restaurant_Code'].nunique(),
            })
        
        df_result = pd.DataFrame(results)
        return df_result
    
    # ==========================================
    # KPI 3: ±15 Guest Tolerance + Error Tracking
    # ==========================================
    
    @staticmethod
    def calculate_kpi3_tolerance(
        df_shift: pd.DataFrame
    ) -> tuple:
        """
        KPI 3: Tỷ lệ dự đoán trong khoảng ±15 khách cho mỗi ca.
        + Danh sách nhà hàng có forecast sai (ngoài ±15 khách).
        
        Returns:
            (df_summary, df_wrong_restaurants, df_daily_error_count)
            
            df_summary: Overall ±15 hit rate by shift
            df_wrong_restaurants: DS nhà hàng sai từng ngày
            df_daily_error_count: Số NH sai theo ngày chạy Model
        """
        if df_shift.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # ── Summary: ±15 tolerance by shift ──
        summary_rows = []
        
        # Overall (all restaurants)
        total_hit = df_shift['Hit_15'].sum()
        total_n = len(df_shift)
        overall_rate = total_hit / total_n * 100 if total_n > 0 else 0
        
        summary_rows.append({
            'Phân_Loại': '📊 TỔNG QUAN',
            'Shift': 'Tất cả',
            'Tổng_Số_Ca': total_n,
            'Số_Ca_Đúng_±15': int(total_hit),
            'Số_Ca_Sai_>15': int(total_n - total_hit),
            'Tỷ_Lệ_Đúng_±15_%': round(overall_rate, 1),
            'Tỷ_Lệ_Sai_%': round(100 - overall_rate, 1),
            'MAE_Trung_Bình': round(df_shift['Abs_Diff'].mean(), 1),
        })
        
        # ⭐ v6: Overall EXCLUDING low-volume restaurants
        # Determines per-restaurant avg_actual → exclude those with avg < threshold
        res_avg_actual = df_shift.groupby('Restaurant_Code')['Actual_Guest'].mean()
        low_vol_res = set(res_avg_actual[res_avg_actual < LOW_VOLUME_THRESHOLD].index)  # type: ignore[reportAttributeAccessIssue, reportOperatorIssue]
        
        df_excl_low = df_shift[~df_shift['Restaurant_Code'].isin(low_vol_res)]  # type: ignore[reportArgumentType]
        if not df_excl_low.empty:
            excl_hit = df_excl_low['Hit_15'].sum()
            excl_n = len(df_excl_low)
            excl_rate = excl_hit / excl_n * 100 if excl_n > 0 else 0
            
            summary_rows.append({
                'Phân_Loại': '📊 TỔNG QUAN (excl. Low Vol)',
                'Shift': f'Tất cả (loại {len(low_vol_res)} NH low-vol)',
                'Tổng_Số_Ca': excl_n,
                'Số_Ca_Đúng_±15': int(excl_hit),
                'Số_Ca_Sai_>15': int(excl_n - excl_hit),
                'Tỷ_Lệ_Đúng_±15_%': round(excl_rate, 1),
                'Tỷ_Lệ_Sai_%': round(100 - excl_rate, 1),
                'MAE_Trung_Bình': round(df_excl_low['Abs_Diff'].mean(), 1),
            })
        
        # By shift
        for shift, grp in df_shift.groupby('Shift'):
            n = len(grp)
            hit = grp['Hit_15'].sum()
            rate = hit / n * 100 if n > 0 else 0
            shift_vn = 'Ca Sáng' if shift == 'MORNING' else 'Ca Chiều'
            
            summary_rows.append({
                'Phân_Loại': '',
                'Shift': shift_vn,
                'Tổng_Số_Ca': n,
                'Số_Ca_Đúng_±15': int(hit),
                'Số_Ca_Sai_>15': int(n - hit),
                'Tỷ_Lệ_Đúng_±15_%': round(rate, 1),
                'Tỷ_Lệ_Sai_%': round(100 - rate, 1),
                'MAE_Trung_Bình': round(grp['Abs_Diff'].mean(), 1),
            })
        
        # By shift × day type
        summary_rows.append({
            'Phân_Loại': '', 'Shift': '',
            'Tổng_Số_Ca': '', 'Số_Ca_Đúng_±15': '',
            'Số_Ca_Sai_>15': '', 'Tỷ_Lệ_Đúng_±15_%': '',
            'Tỷ_Lệ_Sai_%': '', 'MAE_Trung_Bình': '',
        })
        summary_rows.append({
            'Phân_Loại': '📅 NGÀY THƯỜNG vs CUỐI TUẦN',
            'Shift': '', 'Tổng_Số_Ca': '',
            'Số_Ca_Đúng_±15': '', 'Số_Ca_Sai_>15': '',
            'Tỷ_Lệ_Đúng_±15_%': '', 'Tỷ_Lệ_Sai_%': '',
            'MAE_Trung_Bình': '',
        })
        
        for (shift, day_type), grp in df_shift.groupby(['Shift', 'Day_Type']):  # type: ignore[reportGeneralTypeIssues]
            n = len(grp)
            hit = grp['Hit_15'].sum()
            rate = hit / n * 100 if n > 0 else 0
            shift_vn = 'Ca Sáng' if shift == 'MORNING' else 'Ca Chiều'
            
            summary_rows.append({
                'Phân_Loại': '',
                'Shift': f'{shift_vn} - {day_type}',
                'Tổng_Số_Ca': n,
                'Số_Ca_Đúng_±15': int(hit),
                'Số_Ca_Sai_>15': int(n - hit),
                'Tỷ_Lệ_Đúng_±15_%': round(rate, 1),
                'Tỷ_Lệ_Sai_%': round(100 - rate, 1),
                'MAE_Trung_Bình': round(grp['Abs_Diff'].mean(), 1),
            })
        
        df_summary = pd.DataFrame(summary_rows)
        
        # ── Wrong restaurants: DS nhà hàng sai (>±15) ──
        df_wrong = df_shift[df_shift['Hit_15'] == 0].copy()
        
        if not df_wrong.empty:
            df_wrong_restaurants = df_wrong[[
                'Forecast_Run_Date', 'Restaurant_Code', 'sap_code',
                'restaurant_name', 'Date', 'Weekday', 'Shift',
                'Final_Predicted_Guests', 'Actual_Guest', 'Diff',
                'Abs_Diff', 'Error_Pct',
            ]].copy()
            
            # Rename for clarity
            df_wrong_restaurants.columns = [  # type: ignore[reportAttributeAccessIssue]
                'Ngày_Chạy_Model', 'Mã_NH', 'SAP_Code',
                'Tên_NH', 'Ngày_Forecast', 'Thứ', 'Ca',
                'Dự_Đoán', 'Thực_Tế', 'Chênh_Lệch',
                'Sai_Số_Tuyệt_Đối', 'Sai_%',
            ]
            
            df_wrong_restaurants['Ca'] = df_wrong_restaurants['Ca'].map(  # type: ignore[reportAttributeAccessIssue]
                {'MORNING': 'Ca Sáng', 'EVENING': 'Ca Chiều'}  # type: ignore[reportArgumentType]
            )
            
            df_wrong_restaurants = df_wrong_restaurants.sort_values(  # type: ignore[reportAttributeAccessIssue, reportCallIssue]
                ['Ngày_Chạy_Model', 'Sai_Số_Tuyệt_Đối'],
                ascending=[False, False]
            ).reset_index(drop=True)
        else:
            df_wrong_restaurants = pd.DataFrame()
        
        # ── Daily error count: Số NH sai theo từng ngày chạy ──
        daily_error_rows = []
        
        if 'Forecast_Run_Date' in df_shift.columns:
            for run_date, grp in df_shift.groupby('Forecast_Run_Date'):
                n_total = len(grp)
                n_wrong = (grp['Hit_15'] == 0).sum()
                n_right = n_total - n_wrong
                
                # Unique restaurants with errors
                wrong_restaurants = grp[grp['Hit_15'] == 0]['Restaurant_Code'].unique()  # type: ignore[reportAttributeAccessIssue]
                total_restaurants = grp['Restaurant_Code'].nunique()
                
                daily_error_rows.append({
                    'Ngày_Chạy_Model': run_date,
                    'Tổng_Số_Ca': n_total,
                    'Số_Ca_Đúng_±15': n_right,
                    'Số_Ca_Sai_>15': n_wrong,
                    'Tỷ_Lệ_Đúng_%': round(n_right / n_total * 100, 1) if n_total > 0 else 0,
                    'Tổng_NH': total_restaurants,
                    'Số_NH_Có_Sai': len(wrong_restaurants),
                    'Tỷ_Lệ_NH_Sai_%': round(
                        len(wrong_restaurants) / total_restaurants * 100, 1
                    ) if total_restaurants > 0 else 0,
                    'DS_NH_Sai': ', '.join(sorted(str(x) for x in wrong_restaurants)[:20]),
                })
        
        df_daily_error = pd.DataFrame(daily_error_rows)
        if not df_daily_error.empty:
            df_daily_error = df_daily_error.sort_values(
                'Ngày_Chạy_Model', ascending=False
            ).reset_index(drop=True)
        
        return df_summary, df_wrong_restaurants, df_daily_error
    
    # ==========================================
    # KPI 4: SHIFT TOLERANCE ACCURACY CHART
    # ==========================================
    
    @staticmethod
    def calculate_shift_tolerance_accuracy(
        df_shift: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Tính tỷ lệ phân bổ sai số của từng ca làm việc theo 4 nhóm TÁCH BIỆT
        (mutually exclusive) cộng lại = 100%:
          - ±5 khách:  |error| ≤ 5
          - ±10 khách: 5 < |error| ≤ 10
          - ±15 khách: 10 < |error| ≤ 15
          - >15 khách: |error| > 15
        
        Returns: DataFrame với columns:
            Ca_Làm_Việc, Mức_Tolerance, Số_Ca, Tổng_Số_Ca, Tỷ_Lệ_%
        """
        if df_shift.empty:
            return pd.DataFrame()
        
        results = []
        
        shifts_to_check = ['MORNING', 'EVENING']
        shift_labels = {'MORNING': 'Ca Sáng', 'EVENING': 'Ca Chiều'}
        
        # Also compute for ALL shifts combined
        groups = {}
        for shift in shifts_to_check:
            sub = df_shift[df_shift['Shift'] == shift]
            if not sub.empty:
                groups[shift] = sub
        groups['ALL'] = df_shift
        shift_labels['ALL'] = 'Tổng Cộng'
        
        for shift_key, sub in groups.items():
            n = len(sub)
            if n == 0:
                continue
            
            label = shift_labels.get(shift_key, shift_key)
            abs_diff = sub['Abs_Diff']
            
            # Bin 1: ±5 khách → |error| ≤ 5
            count_5 = int((abs_diff <= 5).sum())
            # Bin 2: ±10 khách → 5 < |error| ≤ 10
            count_10 = int(((abs_diff > 5) & (abs_diff <= 10)).sum())
            # Bin 3: ±15 khách → 10 < |error| ≤ 15
            count_15 = int(((abs_diff > 10) & (abs_diff <= 15)).sum())
            # Bin 4: >15 khách → |error| > 15
            count_gt15 = int((abs_diff > 15).sum())
            
            for tol_label, count in [
                ('±5 khách', count_5),
                ('±10 khách', count_10),
                ('±15 khách', count_15),
                ('>15 khách', count_gt15),
            ]:
                results.append({
                    'Ca_Làm_Việc': label,
                    'Mức_Tolerance': tol_label,
                    'Số_Ca': count,
                    'Tổng_Số_Ca': n,
                    'Tỷ_Lệ_%': round(count / n * 100, 1),
                })
        
        return pd.DataFrame(results)
    
    # ==========================================
    # RUN HISTORY RECORD
    # ==========================================
    
    @staticmethod
    def generate_run_record(
        forecast_stats: dict,
        report: dict,
        forecast_mode: str = 'daily',
        run_duration_minutes: float = None,  # type: ignore[reportArgumentType]
    ) -> dict:
        """Tạo 1 record cho lần chạy hiện tại."""
        overall = report.get('overall', {}) if report else {}
        comparison = report.get('model_comparison', {}) if report else {}
        drift = report.get('drift', {}) if report else {}
        
        # Health grades
        mape = overall.get('MAPE')
        hit_rate = overall.get('Hit_Rate')
        mae = overall.get('MAE')
        
        record = {
            'Ngày_Chạy': str(CURRENT_DATE),
            'Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Mode': forecast_mode,
            'Thời_Gian_Chạy_Phút': (
                round(run_duration_minutes, 1) if run_duration_minutes else None
            ),
            
            # Pipeline
            'Tổng_NH': forecast_stats.get('total', 0),
            'Thành_Công': forecast_stats.get('success_ensemble', 0),
            'Thất_Bại': forecast_stats.get('failed', 0),
            'NH_Đóng_Cửa': forecast_stats.get('permanently_closed', 0),
            
            # Accuracy (overall)
            'MAE_Khách': mae,
            'MAPE_%': mape,
            'RMSE': overall.get('RMSE'),
            'Bias': overall.get('Bias'),
            'Hit_Rate_%': hit_rate,
            'N_Samples': overall.get('N_samples', 0),
            
            # Model comparison
            'Ensemble_MAPE': comparison.get('ensemble', {}).get('MAPE'),
            'AI_Raw_MAPE': comparison.get('ai_raw', {}).get('MAPE'),
            'Winner': comparison.get('winner', 'N/A'),
            
            # Drift
            'Drift': 'CÓ' if drift.get('has_drift') else 'KHÔNG',
            
            # Need retune
            'NH_Cần_Retune': len(
                report.get('problem_restaurants', {}).get('needs_retune', [])
            ) if report else 0,
            
            # Health
            'Đánh_Giá': PerformanceReportAgent._get_overall_health(mape, hit_rate, mae),
        }
        
        return record
    
    @staticmethod
    def _get_overall_health(mape, hit_rate, mae) -> str:
        """Đánh giá tổng thể sức khỏe model."""
        scores = []
        
        if mape is not None and not (isinstance(mape, float) and np.isnan(mape)):
            if mape <= 15: scores.append(5)
            elif mape <= 25: scores.append(4)
            elif mape <= 35: scores.append(3)
            elif mape <= 50: scores.append(2)
            else: scores.append(1)
        
        if hit_rate is not None and not (isinstance(hit_rate, float) and np.isnan(hit_rate)):
            if hit_rate >= 75: scores.append(5)
            elif hit_rate >= 60: scores.append(4)
            elif hit_rate >= 45: scores.append(3)
            elif hit_rate >= 30: scores.append(2)
            else: scores.append(1)
        
        if not scores:
            return '❓ Chưa đủ dữ liệu'
        
        avg = np.mean(scores)
        if avg >= 4.5:
            return '🟢 XUẤT SẮC'
        elif avg >= 3.5:
            return '🟢 TỐT'
        elif avg >= 2.5:
            return '🟡 TRUNG BÌNH'
        elif avg >= 1.5:
            return '🟠 YẾU'
        else:
            return '🔴 KÉM'
    
    # ==========================================
    # SAVE MODEL PERFORMANCE REPORT
    # ==========================================
    
    @staticmethod
    def save_model_performance_report(
        new_record: dict,
        report: dict = None,  # type: ignore[reportArgumentType]
        df_master: pd.DataFrame = None,  # type: ignore[reportArgumentType]
    ):
        """
        Lưu/cập nhật Model_Performance_Report.xlsx với 3 KPI chính.
        """
        perf_file = MODEL_PERFORMANCE_FILE
        
        # ── Prepare shift data ──
        df_shift = PerformanceReportAgent._prepare_shift_data(df_master)
        
        # ── Calculate 3 KPIs + Tolerance Chart ──
        df_kpi1 = PerformanceReportAgent.calculate_kpi1_restaurant_shift(df_shift)
        df_kpi2 = PerformanceReportAgent.calculate_kpi2_weekday_weekend(df_shift)
        df_kpi3_summary, df_kpi3_wrong, df_kpi3_daily = (
            PerformanceReportAgent.calculate_kpi3_tolerance(df_shift)
        )
        df_tolerance = PerformanceReportAgent.calculate_shift_tolerance_accuracy(df_shift)
        
        # ── Load/append Run History ──
        df_history = pd.DataFrame()
        if os.path.exists(perf_file):
            try:
                df_history = pd.read_excel(
                    perf_file, sheet_name='Run_History'
                )
            except Exception:
                df_history = pd.DataFrame()
        
        new_row = pd.DataFrame([new_record])
        df_history = pd.concat([df_history, new_row], ignore_index=True)
        
        if 'Ngày_Chạy' in df_history.columns:
            df_history = df_history.drop_duplicates(
                subset=['Ngày_Chạy'], keep='last'
            )
            df_history = df_history.sort_values('Ngày_Chạy').reset_index(drop=True)
        
        # ── Save Excel ──
        try:
            with pd.ExcelWriter(perf_file, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # ═══ Sheet 1: KPI_1_Restaurant_Shift ═══
                if not df_kpi1.empty:
                    df_kpi1.to_excel(
                        writer, sheet_name='KPI_1_Restaurant_Shift', index=False
                    )
                    PerformanceReportAgent._format_kpi1_sheet(
                        workbook,
                        writer.sheets['KPI_1_Restaurant_Shift'],
                        df_kpi1
                    )
                else:
                    pd.DataFrame([{
                        'Thông_báo': 'Chưa có dữ liệu actual để so sánh với forecast'
                    }]).to_excel(
                        writer, sheet_name='KPI_1_Restaurant_Shift', index=False
                    )
                
                # ═══ Sheet 2: KPI_2_Weekday_Weekend ═══
                if not df_kpi2.empty:
                    df_kpi2.to_excel(
                        writer, sheet_name='KPI_2_Weekday_Weekend', index=False
                    )
                    PerformanceReportAgent._format_kpi2_sheet(
                        workbook,
                        writer.sheets['KPI_2_Weekday_Weekend'],
                        df_kpi2
                    )
                else:
                    pd.DataFrame([{
                        'Thông_báo': 'Chưa có dữ liệu actual để so sánh'
                    }]).to_excel(
                        writer, sheet_name='KPI_2_Weekday_Weekend', index=False
                    )
                
                # ═══ Sheet 3: KPI_3_Tolerance_15 ═══
                if not df_kpi3_summary.empty:
                    df_kpi3_summary.to_excel(
                        writer, sheet_name='KPI_3_Tolerance_15', index=False
                    )
                    PerformanceReportAgent._format_kpi3_sheet(
                        workbook,
                        writer.sheets['KPI_3_Tolerance_15'],
                        df_kpi3_summary
                    )
                
                # ═══ Sheet 4: KPI_Tolerance_By_Shift (CHART) ═══
                if not df_tolerance.empty:
                    PerformanceReportAgent._write_tolerance_chart_sheet(
                        workbook, writer, df_tolerance
                    )
                
                # ═══ Sheet 5: Run_History ═══
                df_history.to_excel(
                    writer, sheet_name='Run_History', index=False
                )
                PerformanceReportAgent._format_run_history_sheet(
                    workbook, writer.sheets['Run_History'], df_history
                )
                
                # ═══ Sheet 5: Daily_Error_Tracking ═══
                if not df_kpi3_daily.empty:
                    df_kpi3_daily.to_excel(
                        writer, sheet_name='Daily_Error_Tracking', index=False
                    )
                    PerformanceReportAgent._format_daily_error_sheet(
                        workbook,
                        writer.sheets['Daily_Error_Tracking'],
                        df_kpi3_daily
                    )
                
                # ═══ Sheet 6: DS_NH_Forecast_Sai ═══
                if not df_kpi3_wrong.empty:
                    df_kpi3_wrong.to_excel(
                        writer, sheet_name='DS_NH_Forecast_Sai', index=False
                    )
                    PerformanceReportAgent._format_wrong_sheet(
                        workbook,
                        writer.sheets['DS_NH_Forecast_Sai'],
                        df_kpi3_wrong
                    )
            
            n_kpi1 = len(df_kpi1) if not df_kpi1.empty else 0
            n_wrong = len(df_kpi3_wrong) if not df_kpi3_wrong.empty else 0
            
            logger.info(
                f"📊 Model Performance Report saved: {perf_file}"
            )
            logger.info(
                f"   KPI 1: {n_kpi1} restaurant-shift combinations"
            )
            logger.info(
                f"   KPI 3: {n_wrong} forecast sai (>±15 khách)"
            )
            logger.info(
                f"   Run History: {len(df_history)} runs tracked"
            )
            
        except Exception as e:
            logger.error(f"Failed to save Model Performance Report: {e}")
            traceback.print_exc()
    
    # ==========================================
    # EXCEL FORMATTING
    # ==========================================
    
    @staticmethod
    def _format_kpi1_sheet(workbook, worksheet, df):
        """Format KPI 1: Restaurant × Shift accuracy."""
        header_fmt = workbook.add_format({
            'bold': True, 'bg_color': '#1F4E79',
            'font_color': 'white', 'border': 1,
            'text_wrap': True, 'valign': 'vcenter', 'font_size': 10,
        })
        
        for col_num, col_name in enumerate(df.columns):
            worksheet.write(0, col_num, col_name, header_fmt)
        
        # Column widths
        widths = {
            'Restaurant_Code': 16, 'sap_code': 10, 'restaurant_name': 35,
            'Shift': 10, 'Số_Ca_Đã_So': 12, 'Accuracy_%': 12,
            'Hit_Rate_±15_Khách_%': 18, 'MAE_Khách': 12,
            'TB_Dự_Đoán': 12, 'TB_Thực_Tế': 12, 'TB_Sai_Số': 12,
        }
        for col_num, col_name in enumerate(df.columns):
            w = widths.get(col_name, 14)
            worksheet.set_column(col_num, col_num, w)
        
        n = len(df)
        
        # Conditional formatting: Accuracy_%
        if 'Accuracy_%' in df.columns:
            col_idx = list(df.columns).index('Accuracy_%')
            good = workbook.add_format({
                'bg_color': '#C6EFCE', 'font_color': '#006100', 'border': 1
            })
            bad = workbook.add_format({
                'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'border': 1
            })
            worksheet.conditional_format(1, col_idx, n, col_idx, {
                'type': 'cell', 'criteria': '>=', 'value': 70, 'format': good
            })
            worksheet.conditional_format(1, col_idx, n, col_idx, {
                'type': 'cell', 'criteria': '<', 'value': 50, 'format': bad
            })
        
        # Conditional formatting: Hit_Rate
        if 'Hit_Rate_±15_Khách_%' in df.columns:
            col_idx = list(df.columns).index('Hit_Rate_±15_Khách_%')
            good = workbook.add_format({
                'bg_color': '#C6EFCE', 'font_color': '#006100', 'border': 1
            })
            bad = workbook.add_format({
                'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'border': 1
            })
            worksheet.conditional_format(1, col_idx, n, col_idx, {
                'type': 'cell', 'criteria': '>=', 'value': 70, 'format': good
            })
            worksheet.conditional_format(1, col_idx, n, col_idx, {
                'type': 'cell', 'criteria': '<', 'value': 40, 'format': bad
            })
        
        worksheet.freeze_panes(1, 3)
        worksheet.autofilter(0, 0, n, len(df.columns) - 1)
    
    @staticmethod
    def _format_kpi2_sheet(workbook, worksheet, df):
        """Format KPI 2: Weekday/Weekend × Shift."""
        header_fmt = workbook.add_format({
            'bold': True, 'bg_color': '#2F5496',
            'font_color': 'white', 'border': 1,
            'text_wrap': True, 'font_size': 10,
        })
        cat_fmt = workbook.add_format({
            'bold': True, 'bg_color': '#D6E4F0',
            'font_size': 10, 'border': 1,
        })
        
        for col_num, col_name in enumerate(df.columns):
            worksheet.write(0, col_num, col_name, header_fmt)
        
        # Highlight section headers
        for row_num in range(len(df)):
            val = str(df.iloc[row_num].get('Ca_Làm_Việc', ''))
            if val.startswith('==='):
                for col_num in range(len(df.columns)):
                    v = df.iloc[row_num, col_num]
                    worksheet.write(row_num + 1, col_num, v, cat_fmt)
        
        widths = [15, 15, 12, 12, 18, 12, 12, 12, 12, 12, 10]
        for i, w in enumerate(widths):
            if i < len(df.columns):
                worksheet.set_column(i, i, w)
        
        worksheet.freeze_panes(1, 0)
    
    @staticmethod
    def _format_kpi3_sheet(workbook, worksheet, df):
        """Format KPI 3: ±15 Tolerance."""
        header_fmt = workbook.add_format({
            'bold': True, 'bg_color': '#4472C4',
            'font_color': 'white', 'border': 1,
            'text_wrap': True, 'font_size': 10,
        })
        cat_fmt = workbook.add_format({
            'bold': True, 'bg_color': '#D6E4F0',
            'font_size': 10, 'border': 1,
        })
        
        for col_num, col_name in enumerate(df.columns):
            worksheet.write(0, col_num, col_name, header_fmt)
        
        # Highlight section headers
        for row_num in range(len(df)):
            val = str(df.iloc[row_num].get('Phân_Loại', ''))
            if val.startswith('📊') or val.startswith('📅'):
                for col_num in range(len(df.columns)):
                    v = df.iloc[row_num, col_num]
                    worksheet.write(row_num + 1, col_num, v, cat_fmt)
        
        widths = [30, 20, 12, 14, 12, 16, 12, 14]
        for i, w in enumerate(widths):
            if i < len(df.columns):
                worksheet.set_column(i, i, w)
        
        n = len(df)
        if 'Tỷ_Lệ_Đúng_±15_%' in df.columns:
            col_idx = list(df.columns).index('Tỷ_Lệ_Đúng_±15_%')
            good = workbook.add_format({
                'bg_color': '#C6EFCE', 'font_color': '#006100', 'border': 1
            })
            bad = workbook.add_format({
                'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'border': 1
            })
            worksheet.conditional_format(1, col_idx, n, col_idx, {
                'type': 'cell', 'criteria': '>=', 'value': 70, 'format': good
            })
            worksheet.conditional_format(1, col_idx, n, col_idx, {
                'type': 'cell', 'criteria': '<', 'value': 50, 'format': bad
            })
    
    @staticmethod
    def _format_run_history_sheet(workbook, worksheet, df):
        """Format Run_History sheet."""
        header_fmt = workbook.add_format({
            'bold': True, 'bg_color': '#1F4E79',
            'font_color': 'white', 'border': 1,
            'text_wrap': True, 'font_size': 10,
        })
        
        for col_num, col_name in enumerate(df.columns):
            worksheet.write(0, col_num, col_name, header_fmt)
        
        for col_num, col_name in enumerate(df.columns):
            if 'Đánh_Giá' in col_name:
                worksheet.set_column(col_num, col_num, 20)
            elif 'Timestamp' in col_name or 'Ngày' in col_name:
                worksheet.set_column(col_num, col_num, 18)
            else:
                worksheet.set_column(col_num, col_num, 14)
        
        worksheet.freeze_panes(1, 1)
    
    @staticmethod
    def _format_daily_error_sheet(workbook, worksheet, df):
        """Format Daily Error Tracking sheet."""
        header_fmt = workbook.add_format({
            'bold': True, 'bg_color': '#C00000',
            'font_color': 'white', 'border': 1,
            'text_wrap': True, 'font_size': 10,
        })
        
        for col_num, col_name in enumerate(df.columns):
            worksheet.write(0, col_num, col_name, header_fmt)
        
        widths = [18, 12, 14, 12, 14, 10, 12, 14, 40]
        for i, w in enumerate(widths):
            if i < len(df.columns):
                worksheet.set_column(i, i, w)
        
        n = len(df)
        if 'Tỷ_Lệ_NH_Sai_%' in df.columns:
            col_idx = list(df.columns).index('Tỷ_Lệ_NH_Sai_%')
            bad = workbook.add_format({
                'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'border': 1
            })
            worksheet.conditional_format(1, col_idx, n, col_idx, {
                'type': 'cell', 'criteria': '>', 'value': 50, 'format': bad
            })
        
        worksheet.freeze_panes(1, 0)
    
    @staticmethod
    def _format_wrong_sheet(workbook, worksheet, df):
        """Format DS_NH_Forecast_Sai sheet."""
        header_fmt = workbook.add_format({
            'bold': True, 'bg_color': '#C00000',
            'font_color': 'white', 'border': 1,
            'text_wrap': True, 'font_size': 10,
        })
        
        for col_num, col_name in enumerate(df.columns):
            worksheet.write(0, col_num, col_name, header_fmt)
        
        widths = [18, 10, 10, 35, 14, 10, 12, 12, 12, 12, 14, 10]
        for i, w in enumerate(widths):
            if i < len(df.columns):
                worksheet.set_column(i, i, w)
        
        n = len(df)
        if 'Sai_Số_Tuyệt_Đối' in df.columns:
            col_idx = list(df.columns).index('Sai_Số_Tuyệt_Đối')
            severe = workbook.add_format({
                'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'border': 1
            })
            worksheet.conditional_format(1, col_idx, n, col_idx, {
                'type': 'cell', 'criteria': '>', 'value': 30, 'format': severe
            })
        
        worksheet.freeze_panes(1, 4)
        worksheet.autofilter(0, 0, n, len(df.columns) - 1)
    
    # ==========================================
    # TOLERANCE BY SHIFT - CHART SHEET
    # ==========================================
    
    @staticmethod
    def _write_tolerance_chart_sheet(workbook, writer, df_tolerance):
        """
        Tạo sheet "KPI_Tolerance_By_Shift" với:
          - Bảng dữ liệu chéo (Ca Sáng / Ca Chiều / Tổng Cộng) × (±5, ±10, ±15, >15)
          - Các nhóm TÁCH BIỆT cộng lại = 100%
          - Biểu đồ cột xếp chồng 100% (Percent Stacked Bar Chart)
        """
        sheet_name = 'KPI_Tolerance_By_Shift'
        
        # ── Build pivot table: rows = Ca, cols = Tolerance level ──
        tolerance_labels = ['±5 khách', '±10 khách', '±15 khách', '>15 khách']
        
        shifts = list(df_tolerance['Ca_Làm_Việc'].unique())
        
        # Row: shift, Cols: tolerance levels → values = Tỷ_Lệ_%
        pivot_data = []
        for ca in shifts:
            row = {'Ca_Làm_Việc': ca}
            total_ca = 0
            for tol in tolerance_labels:
                match = df_tolerance[
                    (df_tolerance['Ca_Làm_Việc'] == ca) &
                    (df_tolerance['Mức_Tolerance'] == tol)
                ]
                if not match.empty:
                    row[tol] = match.iloc[0]['Tỷ_Lệ_%']
                    row[f'{tol}_count'] = match.iloc[0]['Số_Ca']
                    row[f'{tol}_total'] = match.iloc[0]['Tổng_Số_Ca']
                    total_ca = match.iloc[0]['Tổng_Số_Ca']
                else:
                    row[tol] = 0
                    row[f'{tol}_count'] = 0
                    row[f'{tol}_total'] = 0
            row['_total'] = total_ca
            pivot_data.append(row)
        
        # ── Write title + data table ──
        worksheet = workbook.add_worksheet(sheet_name)
        
        # Title
        title_fmt = workbook.add_format({
            'bold': True, 'font_size': 16,
            'font_color': '#1A1A2E', 'bottom': 2,
        })
        subtitle_fmt = workbook.add_format({
            'italic': True, 'font_color': '#666666', 'font_size': 10,
        })
        note_fmt = workbook.add_format({
            'italic': True, 'font_color': '#333333', 'font_size': 9,
            'text_wrap': True,
        })
        
        worksheet.merge_range('A1:G1',
            'PHÂN BỔ TỶ LỆ CHÍNH XÁC THEO MỨC DUNG SAI (= 100%)',
            title_fmt
        )
        worksheet.set_row(0, 30)
        
        worksheet.merge_range('A2:G2',
            '4 nhóm tách biệt: ±5 / ±10 / ±15 / >15 khách — Tổng = 100% cho mỗi ca',
            subtitle_fmt
        )
        worksheet.merge_range('A3:G3',
            '⚙ ±5 = sai ≤5 khách  |  ±10 = sai 6-10  |  ±15 = sai 11-15  |  >15 = sai >15 khách',
            note_fmt
        )
        
        # Header row
        header_fmt = workbook.add_format({
            'bold': True, 'bg_color': '#1F4E79',
            'font_color': 'white', 'border': 1,
            'text_wrap': True, 'valign': 'vcenter',
            'align': 'center', 'font_size': 10,
        })
        
        data_start_row = 5  # 0-indexed
        
        headers = [
            'Ca Làm Việc', '±5 khách (%)', '±10 khách (%)',
            '±15 khách (%)', '>15 khách (%)', 'Tổng (%)', 'Tổng Số Ca',
        ]
        for ci, h in enumerate(headers):
            worksheet.write(data_start_row, ci, h, header_fmt)
        
        # Cell formats
        label_fmt = workbook.add_format({
            'bold': True, 'border': 1, 'align': 'left', 'valign': 'vcenter',
            'font_size': 10, 'bg_color': '#E8EAF6',
        })
        green_fmt = workbook.add_format({
            'border': 1, 'align': 'center', 'font_size': 10,
            'num_format': '0.0', 'bg_color': '#C6EFCE', 'font_color': '#006100',
        })
        blue_fmt = workbook.add_format({
            'border': 1, 'align': 'center', 'font_size': 10,
            'num_format': '0.0', 'bg_color': '#BDD7EE', 'font_color': '#1F4E79',
        })
        yellow_fmt = workbook.add_format({
            'border': 1, 'align': 'center', 'font_size': 10,
            'num_format': '0.0', 'bg_color': '#FFEB9C', 'font_color': '#9C6500',
        })
        red_fmt = workbook.add_format({
            'border': 1, 'align': 'center', 'font_size': 10,
            'num_format': '0.0', 'bg_color': '#FFC7CE', 'font_color': '#9C0006',
        })
        total_fmt = workbook.add_format({
            'bold': True, 'border': 1, 'align': 'center', 'font_size': 10,
            'num_format': '0.0', 'bg_color': '#D9E2F3',
        })
        count_fmt = workbook.add_format({
            'border': 1, 'align': 'center', 'font_size': 10,
            'num_format': '#,##0', 'bg_color': '#F2F2F2',
        })
        
        # Fixed color per tolerance column
        tol_fmts = [green_fmt, blue_fmt, yellow_fmt, red_fmt]
        
        for ri, row in enumerate(pivot_data):
            r = data_start_row + 1 + ri
            worksheet.write(r, 0, row['Ca_Làm_Việc'], label_fmt)
            
            row_sum = 0
            for ci, tol in enumerate(tolerance_labels):
                val = row.get(tol, 0)
                row_sum += val
                worksheet.write(r, ci + 1, val, tol_fmts[ci])
            
            # Tổng % (should be ≈100)
            worksheet.write(r, 5, round(row_sum, 1), total_fmt)
            # Tổng số ca
            worksheet.write(r, 6, row.get('_total', 0), count_fmt)
        
        n_data_rows = len(pivot_data)
        
        # ── Detail count table below ──
        detail_row = data_start_row + n_data_rows + 3
        detail_header_fmt = workbook.add_format({
            'bold': True, 'bg_color': '#4472C4',
            'font_color': 'white', 'border': 1,
            'text_wrap': True, 'align': 'center', 'font_size': 10,
        })
        detail_headers = [
            'Ca Làm Việc',
            '±5 (số ca / tổng)',
            '±10 (số ca / tổng)',
            '±15 (số ca / tổng)',
            '>15 (số ca / tổng)',
        ]
        for ci, h in enumerate(detail_headers):
            worksheet.write(detail_row, ci, h, detail_header_fmt)
        
        detail_data_fmt = workbook.add_format({
            'border': 1, 'align': 'center', 'font_size': 10,
        })
        
        for ri, row in enumerate(pivot_data):
            r = detail_row + 1 + ri
            worksheet.write(r, 0, row['Ca_Làm_Việc'], label_fmt)
            for ci, tol in enumerate(tolerance_labels):
                count = row.get(f'{tol}_count', 0)
                total = row.get(f'{tol}_total', 0)
                worksheet.write(r, ci + 1, f"{count} / {total}", detail_data_fmt)
        
        # Column widths
        worksheet.set_column(0, 0, 16)
        worksheet.set_column(1, 4, 18)
        worksheet.set_column(5, 5, 12)
        worksheet.set_column(6, 6, 14)
        
        # ── Chart 1: Percent Stacked Bar (= 100%) ──
        chart_stacked = workbook.add_chart({'type': 'column', 'subtype': 'percent_stacked'})
        chart_stacked.set_title({'name': 'Phân Bổ Tỷ Lệ Chính Xác Theo Ca (= 100%)'})
        chart_stacked.set_style(10)
        chart_stacked.set_size({'width': 720, 'height': 480})
        
        # Colors: green (best) → blue → yellow → red (worst)
        colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
        
        for ci, (tol, color) in enumerate(zip(tolerance_labels, colors)):
            chart_stacked.add_series({
                'name': tol,
                'categories': [sheet_name, data_start_row + 1, 0,
                               data_start_row + n_data_rows, 0],
                'values': [sheet_name, data_start_row + 1, ci + 1,
                           data_start_row + n_data_rows, ci + 1],
                'fill': {'color': color},
                'border': {'color': '#FFFFFF', 'width': 0.5},
                'data_labels': {
                    'value': True,
                    'num_format': '0.0"%"',
                    'font': {'size': 9, 'color': '#FFFFFF', 'bold': True},
                },
            })
        
        chart_stacked.set_x_axis({
            'name': 'Ca Làm Việc',
            'name_font': {'bold': True, 'size': 11},
        })
        chart_stacked.set_y_axis({
            'name': 'Tỷ lệ phân bổ (%)',
            'name_font': {'bold': True, 'size': 11},
            'major_gridlines': {'visible': True, 'line': {'color': '#E0E0E0'}},
        })
        
        chart_stacked.set_legend({
            'position': 'bottom',
            'font': {'size': 10},
        })
        
        chart_stacked.set_plotarea({
            'border': {'color': '#D0D0D0', 'width': 1},
            'fill': {'color': '#FAFAFA'},
        })
        
        worksheet.insert_chart(f'H{data_start_row}', chart_stacked)
        
        # ── Chart 2: Grouped Bar (absolute values side-by-side) ──
        chart_grouped = workbook.add_chart({'type': 'column'})
        chart_grouped.set_title({'name': 'So Sánh % Theo Từng Mức Dung Sai & Ca Làm Việc'})
        chart_grouped.set_style(10)
        chart_grouped.set_size({'width': 720, 'height': 480})
        
        for ci, (tol, color) in enumerate(zip(tolerance_labels, colors)):
            chart_grouped.add_series({
                'name': tol,
                'categories': [sheet_name, data_start_row + 1, 0,
                               data_start_row + n_data_rows, 0],
                'values': [sheet_name, data_start_row + 1, ci + 1,
                           data_start_row + n_data_rows, ci + 1],
                'fill': {'color': color},
                'border': {'color': color},
                'gap': 150,
                'data_labels': {
                    'value': True,
                    'num_format': '0.0"%"',
                    'font': {'size': 8},
                },
            })
        
        chart_grouped.set_x_axis({
            'name': 'Ca Làm Việc',
            'name_font': {'bold': True, 'size': 11},
        })
        chart_grouped.set_y_axis({
            'name': 'Tỷ lệ (%)',
            'name_font': {'bold': True, 'size': 11},
            'min': 0,
            'major_gridlines': {'visible': True, 'line': {'color': '#E0E0E0'}},
        })
        
        chart_grouped.set_legend({
            'position': 'bottom',
            'font': {'size': 10},
        })
        
        chart_grouped.set_plotarea({
            'border': {'color': '#D0D0D0', 'width': 1},
            'fill': {'color': '#FAFAFA'},
        })
        
        # Place Chart 2 below Chart 1
        chart2_row = data_start_row + 26
        worksheet.insert_chart(f'H{chart2_row}', chart_grouped)
        
        logger.info(
            f"   📊 Tolerance chart sheet created with {n_data_rows} shift groups (bins sum to 100%)"
        )
    
    # ==========================================
    # SAVE ACCURACY REPORT (enhanced)
    # ==========================================
    
    @staticmethod
    def save_enhanced_accuracy_report(
        df_master: pd.DataFrame,
        report: dict,
        forecast_stats: dict = None,  # type: ignore[reportArgumentType]
    ):
        """
        Lưu Accuracy_Report.xlsx enhanced.
        """
        from forecast_system.agents.monitoring_agent import MonitoringAgent
        
        accuracy_file = ACCURACY_REPORT_FILE
        
        try:
            with pd.ExcelWriter(accuracy_file, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                overall = report.get('overall', {}) if report else {}
                rolling = report.get('rolling', {}) if report else {}
                comparison = report.get('model_comparison', {}) if report else {}
                
                # === Sheet 1: Summary ===
                summary_rows = [
                    {'Metric': '📅 Ngày chạy', 'Value': str(CURRENT_DATE)},
                    {'Metric': '⏰ Thời gian', 'Value': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                    {'Metric': '', 'Value': ''},
                    {'Metric': '=== ĐỘ CHÍNH XÁC TỔNG THỂ ===', 'Value': ''},
                    {'Metric': 'MAE (sai lệch TB, khách)', 'Value': overall.get('MAE')},
                    {'Metric': 'MAPE (sai lệch TB, %)', 'Value': overall.get('MAPE')},
                    {'Metric': 'RMSE', 'Value': overall.get('RMSE')},
                    {'Metric': 'Bias', 'Value': overall.get('Bias')},
                    {'Metric': 'Hit Rate (±15 khách, %)', 'Value': overall.get('Hit_Rate')},
                    {'Metric': 'Số mẫu', 'Value': overall.get('N_samples', 0)},
                    {'Metric': '', 'Value': ''},
                    {'Metric': '=== ĐÁNH GIÁ ===', 'Value': ''},
                    {
                        'Metric': '⭐ Đánh giá tổng thể',
                        'Value': PerformanceReportAgent._get_overall_health(
                            overall.get('MAPE'), overall.get('Hit_Rate'), overall.get('MAE')
                        )
                    },
                ]
                
                if rolling:
                    summary_rows.append({'Metric': '', 'Value': ''})
                    summary_rows.append({'Metric': '=== ROLLING ===', 'Value': ''})
                    for window, m in sorted(rolling.items()):
                        summary_rows.append({'Metric': f'MAPE ({window})', 'Value': m.get('MAPE')})
                        summary_rows.append({'Metric': f'Hit Rate ({window})', 'Value': m.get('Hit_Rate')})
                
                if comparison.get('ensemble'):
                    summary_rows.extend([
                        {'Metric': '', 'Value': ''},
                        {'Metric': '=== ML vs AI ===', 'Value': ''},
                        {'Metric': 'Ensemble MAPE', 'Value': comparison['ensemble'].get('MAPE')},
                        {'Metric': 'AI Raw MAPE', 'Value': comparison.get('ai_raw', {}).get('MAPE')},
                        {'Metric': 'Winner', 'Value': comparison.get('winner', 'N/A')},
                    ])
                
                if forecast_stats:
                    summary_rows.extend([
                        {'Metric': '', 'Value': ''},
                        {'Metric': '=== PIPELINE ===', 'Value': ''},
                        {'Metric': 'Tổng nhà hàng', 'Value': forecast_stats.get('total', 0)},
                        {'Metric': 'Thành công', 'Value': forecast_stats.get('success_ensemble', 0)},
                        {'Metric': 'Thất bại', 'Value': forecast_stats.get('failed', 0)},
                        {'Metric': 'Đóng cửa', 'Value': forecast_stats.get('permanently_closed', 0)},
                    ])
                
                df_summary = pd.DataFrame(summary_rows)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                ws = writer.sheets['Summary']
                hdr = workbook.add_format({
                    'bold': True, 'bg_color': '#2F5496',
                    'font_color': 'white', 'border': 1
                })
                ws.write(0, 0, 'Metric', hdr)
                ws.write(0, 1, 'Value', hdr)
                ws.set_column(0, 0, 35)
                ws.set_column(1, 1, 40)
                
                # === Sheet 2: Per Restaurant Daily ===
                per_res = MonitoringAgent.calculate_restaurant_daily_accuracy(df_master)
                if not per_res.empty:
                    per_res.to_excel(
                        writer, sheet_name='Per_Restaurant_Daily', index=False
                    )
                
                # === Sheet 3: Weekday ===
                wd_data = report.get('weekday', []) if report else []
                if wd_data:
                    pd.DataFrame(wd_data).to_excel(
                        writer, sheet_name='Weekday', index=False
                    )
                
                # === Sheet 4: Needs_Retune ===
                problems = report.get('problem_restaurants', {}) if report else {}
                retune = problems.get('needs_retune', [])
                if retune:
                    pd.DataFrame(retune).to_excel(
                        writer, sheet_name='Needs_Retune', index=False
                    )
                
                # === Sheet 5: Daily accuracy ===
                daily = MonitoringAgent.calculate_daily_accuracy(df_master)
                if not daily.empty:
                    daily.to_excel(writer, sheet_name='Daily', index=False)
            
            logger.info(f"📊 Accuracy Report saved: {accuracy_file}")
            
        except Exception as e:
            logger.error(f"Failed to save Accuracy Report: {e}")
            traceback.print_exc()
    
    # ==========================================
    # CONSOLE HEALTH SUMMARY
    # ==========================================
    
    @staticmethod
    def print_health_summary(record: dict, logger_func=None):
        """Print health summary to console."""
        log = logger_func or logger.info
        
        log("\n" + "=" * 65)
        log("📊 MODEL HEALTH CHECK")
        log("=" * 65)
        log(f"  Đánh giá: {record.get('Đánh_Giá', 'N/A')}")
        log(f"")
        log(f"  MAPE:      {record.get('MAPE_%', 'N/A')}%")
        log(f"  Hit Rate:  {record.get('Hit_Rate_%', 'N/A')}%")
        log(f"  MAE:       {record.get('MAE_Khách', 'N/A')} khách")
        log(f"  Bias:      {record.get('Bias', 'N/A')}")
        log(f"")
        log(f"  Pipeline:  {record.get('Thành_Công', 0)}/{record.get('Tổng_NH', 0)} thành công")
        log(f"  Thất bại:  {record.get('Thất_Bại', 0)}")
        log(f"  Đóng cửa:  {record.get('NH_Đóng_Cửa', 0)}")
        log(f"  Drift:     {record.get('Drift', 'KHÔNG')}")
        log(f"  Cần retune: {record.get('NH_Cần_Retune', 0)} nhà hàng")
        log("=" * 65)
        log(f"  📁 {MODEL_PERFORMANCE_FILE}")
        log(f"  📁 {ACCURACY_REPORT_FILE}")
        log("=" * 65)
