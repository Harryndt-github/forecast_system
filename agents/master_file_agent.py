"""
==============================================
MASTER FILE AGENT
==============================================
Trách nhiệm:
- Load/Save master forecast file (Excel + CSV backup)
- Update actuals từ DB data  
- Tính toán sai số (Error %)
- Data integrity protection

Refactored từ forecast_fb.py MasterFileAgent class.
Fixes: Better error handling, cleaner code.
"""

import pandas as pd
import numpy as np
import os
import shutil
import time
import datetime
import tempfile
import traceback

from forecast_system.config.settings import CURRENT_DATE, MASTER_FILE_NAME
from forecast_system.agents.data_agent import DataAgent
from forecast_system.utils.logger import get_logger

logger = get_logger('master_file_agent')


# ==========================================
# SAFE EXCEL SAVE
# ==========================================

def _file_mtime(path):
    try:
        return os.path.getmtime(path) if os.path.exists(path) else 0
    except Exception:
        return 0


def save_excel_safely(df, filename):
    """
    Save DataFrame to Excel với safety measures:
    1. Luôn lưu CSV backup trước
    2. Lưu Excel ra temp file trước
    3. Backup file Excel cũ
    4. Move temp → final
    
    Args:
        df: DataFrame to save
        filename: Target Excel filename
    """
    if df.empty:
        logger.warning("Attempted to save empty DataFrame. Aborting to protect data.")
        return
    
    temp_path = os.path.join(
        tempfile.gettempdir(),
        os.path.basename(filename) + f".{int(time.time())}.tmp.xlsx"
    )
    csv_backup = filename.replace('.xlsx', '.csv')
    bak_file = filename + ".bak"
    
    try:
        # 1. CSV backup TRƯỚC (data safety)
        df.to_csv(csv_backup, index=False)
        logger.info(f"Saved CSV backup: {csv_backup}")
        
        # 2. Lưu Excel tạm
        logger.info(f"💾 Saving {len(df):,} rows to Excel...")
        with pd.ExcelWriter(temp_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Forecast')
        
        # 3. Backup Excel cũ
        if os.path.exists(filename):
            try:
                shutil.copy2(filename, bak_file)
            except Exception:
                pass
        
        # 4. Move temp → final. If Excel is open/locked, keep CSV as the
        # source of truth and leave the temp xlsx for manual inspection.
        try:
            if os.path.exists(filename):
                os.remove(filename)
            shutil.move(temp_path, filename)
            logger.info(f"Final Excel saved: {filename}")
        except PermissionError as e:
            locked_copy = filename.replace('.xlsx', f".locked_{int(time.time())}.xlsx")
            try:
                shutil.move(temp_path, locked_copy)
            except Exception:
                locked_copy = temp_path
            logger.error(
                f"Excel file is locked and was not overwritten: {filename}. "
                f"CSV backup is current and will be preferred on next load: {csv_backup}. "
                f"Excel temp copy: {locked_copy}. Error: {e}"
            )
            return False
        
    except Exception as e:
        logger.error(f"Excel save failed: {e}")
        traceback.print_exc()
        # CSV đã được lưu ở bước 1 → data vẫn an toàn
        return False

    return True


class MasterFileAgent:
    """
    Agent quản lý Master Forecast file.
    """
    
    COLUMNS = [
        'Forecast_Run_Date', 'Restaurant_Code', 'sap_code', 'restaurant_name',
        'Date', 'Weekday', 'Hour', 'Shift', 'Final_Predicted_Guests', 'Actual_Guest',
        'Diff_Guest', 'Error_%', 'Is_Holiday', 'Is_Veg', 'AI_Raw_Daily_Forecast',
        'AI_Forecast_Available', 'System_Predicted_Before_Brain',
        'Booking_Guests', 'Forecast_Mode'
    ]
    
    @staticmethod
    def load_or_create(filename=None):
        """
        Load existing forecast file hoặc tạo mới.
        
        Priority:
        1. Newer CSV backup, if it is fresher than Excel
        2. Excel file (nếu valid)
        3. CSV backup (nếu Excel corrupt)
        3. Generate skeleton (nếu không có gì)
        
        Returns:
            pd.DataFrame
        """
        if filename is None:
            filename = MASTER_FILE_NAME
            
        cols = MasterFileAgent.COLUMNS
        csv_backup = filename.replace('.xlsx', '.csv')
        
        df = pd.DataFrame(columns=cols)  # type: ignore[reportArgumentType]
        
        prefer_csv = (
            os.path.exists(csv_backup) and
            _file_mtime(csv_backup) > _file_mtime(filename)
        )

        if prefer_csv:
            try:
                logger.warning(
                    f"CSV backup is newer than Excel. Loading CSV as source of truth: {csv_backup}"
                )
                df = pd.read_csv(csv_backup)
                logger.info(f"Loaded {len(df):,} rows from newer CSV")
            except Exception as e:
                logger.warning(f"Newer CSV load failed, falling back to Excel: {e}")

        if df.empty and os.path.exists(filename):
            try:
                logger.info(f"📂 Loading Excel: {filename}...")
                df_xl = pd.read_excel(filename, engine='openpyxl')
                
                # Data quality check
                important_cols = ['Restaurant_Code', 'Date', 'Final_Predicted_Guests']
                available_important = [c for c in important_cols if c in df_xl.columns]
                
                if available_important:
                    null_counts = df_xl[available_important].isnull().sum().sum()
                    
                    if len(df_xl) > 10 and null_counts < (len(df_xl) * 0.5):
                        df = df_xl
                        logger.info(f"Loaded {len(df):,} rows from Excel")
                    else:
                        logger.warning("Excel data quality check failed. Triggering recovery...")
                        raise ValueError("Excel file seems corrupted or mostly empty.")
                else:
                    df = df_xl
                    logger.info(f"Loaded {len(df):,} rows from Excel (columns may differ)")
                    
            except Exception as e:
                logger.warning(f"Excel load failed: {e}")
                if os.path.exists(csv_backup):
                    try:
                        logger.info(f"🆘 Emergency recovery from CSV: {csv_backup}...")
                        df = pd.read_csv(csv_backup)
                        logger.info(f"Recovered {len(df):,} rows from CSV")
                    except Exception as e2:
                        logger.error(f"Recovery failed: {e2}")
                        
        elif df.empty and os.path.exists(csv_backup):
            try:
                df = pd.read_csv(csv_backup)
                logger.info(f"Loaded {len(df):,} rows from CSV")
            except Exception as e:
                logger.error(f"CSV load failed: {e}")
        
        # Generate skeleton nếu empty
        if df.empty or len(df) < 10:
            logger.info("📄 Generating historical skeleton (Last 45 days)...")
            skeleton = []
            for d in range(45):
                dt = CURRENT_DATE - datetime.timedelta(days=d)
                for h in range(8, 23):
                    skeleton.append({'Date': dt, 'Hour': h})
            df = pd.DataFrame(skeleton)
        
        # Normalize (pandas 3.0 compat: drop NaT before .dt.date)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Date'] = df['Date'].dt.date
        
        if 'Restaurant_Code' in df.columns:
            df['Restaurant_Code'] = DataAgent.normalize_key(df['Restaurant_Code'].fillna('0'))
        
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        
        return df
    
    @staticmethod
    def update_actuals(df_history, df_train):
        """
        Cập nhật Actual_Guest bằng dữ liệu thực từ DB.
        Hỗ trợ cả hourly rows (Hour có giá trị) và daily-only rows (Hour=None).
        
        Args:
            df_history: Master forecast DataFrame
            df_train: Transaction data DataFrame
        
        Returns:
            pd.DataFrame với actuals updated
        """
        if df_history.empty or df_train.empty:
            return df_history
        
        # 1. Normalize date types (pandas 3.0 compat: drop NaT before .dt.date)
        df_history['Date'] = pd.to_datetime(df_history['Date'], errors='coerce')
        df_history = df_history.dropna(subset=['Date'])
        df_history['Date'] = df_history['Date'].dt.date
        df_history['Restaurant_Code'] = DataAgent.normalize_key(df_history['Restaurant_Code'])
        
        # 2. Separate into 3 row types: hourly (legacy), shift-based, daily-only
        has_shift = df_history.get('Shift', pd.Series(dtype='str')).notna() if 'Shift' in df_history.columns else pd.Series(False, index=df_history.index)
        has_hour = df_history['Hour'].notna() & ~has_shift
        is_daily_only = ~has_hour & ~has_shift
        
        df_hourly = df_history[has_hour].copy()
        df_shift_based = df_history[has_shift].copy()
        df_daily_only = df_history[is_daily_only].copy()
        
        # 3. Prepare hourly actuals (for legacy + shift mapping)
        actuals_hourly = df_train.groupby(
            ['restaurant_code', 'date', 'hour']
        )['guest_count'].sum().reset_index()
        
        actuals_hourly.rename(columns={
            'restaurant_code': 'Restaurant_Code',
            'date': 'Date',
            'hour': 'Hour',
            'guest_count': 'Real'
        }, inplace=True)
        
        actuals_hourly['Date'] = pd.to_datetime(actuals_hourly['Date'], errors='coerce')
        actuals_hourly = actuals_hourly.dropna(subset=['Date'])
        actuals_hourly['Date'] = actuals_hourly['Date'].dt.date
        actuals_hourly['Restaurant_Code'] = DataAgent.normalize_key(actuals_hourly['Restaurant_Code'])
        
        # Robust Hour conversion
        def clean_hour(val):
            if pd.isna(val):
                return np.nan
            if hasattr(val, 'hour'):
                return val.hour
            try:
                return int(float(str(val).replace('.0', '')))
            except (ValueError, TypeError):
                return 0
        
        # === Process HOURLY rows (legacy) ===
        if not df_hourly.empty:
            df_hourly['Hour'] = df_hourly['Hour'].apply(clean_hour)
            actuals_hourly_clean = actuals_hourly.copy()
            actuals_hourly_clean['Hour'] = actuals_hourly_clean['Hour'].apply(clean_hour)
            
            merged_hourly = pd.merge(
                df_hourly, actuals_hourly_clean,
                on=['Restaurant_Code', 'Date', 'Hour'],
                how='left'
            )
            
            mask_update = merged_hourly['Date'] <= CURRENT_DATE
            mask_final = mask_update & pd.notna(merged_hourly['Real'])
            merged_hourly.loc[mask_final, 'Actual_Guest'] = merged_hourly.loc[mask_final, 'Real']
            
            merged_hourly['Diff_Guest'] = merged_hourly['Final_Predicted_Guests'] - merged_hourly['Actual_Guest']
            
            nz = (merged_hourly['Actual_Guest'] > 0) & pd.notna(merged_hourly['Actual_Guest'])
            merged_hourly.loc[nz, 'Error_%'] = round(
                (merged_hourly.loc[nz, 'Diff_Guest'] / merged_hourly.loc[nz, 'Actual_Guest']) * 100,
                1
            )
            
            df_hourly = merged_hourly.drop(columns=['Real'], errors='ignore')
        
        # === Process SHIFT-BASED rows (Phase 8: MORNING/EVENING) ===
        if not df_shift_based.empty:
            # Aggregate actual data by shift: map each hour to its shift
            actuals_hourly_clean = actuals_hourly.copy()
            actuals_hourly_clean['Hour'] = actuals_hourly_clean['Hour'].apply(clean_hour)
            actuals_hourly_clean['Shift'] = actuals_hourly_clean['Hour'].apply(
                DataAgent.map_to_shift
            )
            
            # Group by Restaurant, Date, Shift → total actual guests per shift
            actuals_shift = actuals_hourly_clean.groupby(
                ['Restaurant_Code', 'Date', 'Shift']
            )['Real'].sum().reset_index()
            
            merged_shift = pd.merge(
                df_shift_based, actuals_shift,
                on=['Restaurant_Code', 'Date', 'Shift'],
                how='left'
            )
            
            mask_update = merged_shift['Date'] <= CURRENT_DATE
            mask_final = mask_update & pd.notna(merged_shift['Real'])
            merged_shift.loc[mask_final, 'Actual_Guest'] = merged_shift.loc[mask_final, 'Real']
            
            merged_shift['Diff_Guest'] = merged_shift['Final_Predicted_Guests'] - merged_shift['Actual_Guest']
            
            nz = (merged_shift['Actual_Guest'] > 0) & pd.notna(merged_shift['Actual_Guest'])
            merged_shift.loc[nz, 'Error_%'] = round(
                (merged_shift.loc[nz, 'Diff_Guest'] / merged_shift.loc[nz, 'Actual_Guest']) * 100,
                1
            )
            
            df_shift_based = merged_shift.drop(columns=['Real'], errors='ignore')
        
        # === Process DAILY-ONLY rows ===
        if not df_daily_only.empty:
            actuals_daily = df_train.groupby(
                ['restaurant_code', 'date']
            )['guest_count'].sum().reset_index()
            
            actuals_daily.rename(columns={
                'restaurant_code': 'Restaurant_Code',
                'date': 'Date',
                'guest_count': 'Real'
            }, inplace=True)
            
            actuals_daily['Date'] = pd.to_datetime(actuals_daily['Date'], errors='coerce')
            actuals_daily = actuals_daily.dropna(subset=['Date'])
            actuals_daily['Date'] = actuals_daily['Date'].dt.date
            actuals_daily['Restaurant_Code'] = DataAgent.normalize_key(actuals_daily['Restaurant_Code'])
            
            merged_daily = pd.merge(
                df_daily_only, actuals_daily,
                on=['Restaurant_Code', 'Date'],
                how='left'
            )
            
            mask_update = merged_daily['Date'] <= CURRENT_DATE
            mask_final = mask_update & pd.notna(merged_daily['Real'])
            merged_daily.loc[mask_final, 'Actual_Guest'] = merged_daily.loc[mask_final, 'Real']
            
            merged_daily['Diff_Guest'] = merged_daily['Final_Predicted_Guests'] - merged_daily['Actual_Guest']
            
            nz = (merged_daily['Actual_Guest'] > 0) & pd.notna(merged_daily['Actual_Guest'])
            merged_daily.loc[nz, 'Error_%'] = round(
                (merged_daily.loc[nz, 'Diff_Guest'] / merged_daily.loc[nz, 'Actual_Guest']) * 100,
                1
            )
            
            df_daily_only = merged_daily.drop(columns=['Real'], errors='ignore')
        
        # 4. Recombine all 3 types
        result = pd.concat([df_hourly, df_shift_based, df_daily_only], ignore_index=True)
        
        # Sort for consistency
        sort_cols = ['Restaurant_Code', 'Date']
        if 'Shift' in result.columns:
            sort_cols.append('Shift')
        if 'Hour' in result.columns:
            sort_cols.append('Hour')
        result = result.sort_values(sort_cols, na_position='last').reset_index(drop=True)
        
        has_real = result[pd.notna(result['Actual_Guest'])]
        if not has_real.empty:
            logger.info(f"Actuals updated up to {has_real['Date'].max()}")
        
        return result
    
    @staticmethod
    def save_with_booking_sheet(df_forecast, df_booking, filename=None):
        """
        Lưu Master file với 2 sheets:
        - Sheet 'Forecast': Dữ liệu forecast chính
        - Sheet 'Booking_Guests': Dữ liệu khách đặt booking
        
        Args:
            df_forecast: DataFrame forecast chính
            df_booking: DataFrame booking summary
            filename: File path (default: MASTER_FILE_NAME)
        """
        if filename is None:
            filename = MASTER_FILE_NAME
        
        if df_forecast.empty:
            logger.warning("Forecast data empty. Aborting save.")
            return
        
        temp_path = os.path.join(
            tempfile.gettempdir(),
            os.path.basename(filename) + f".{int(time.time())}.tmp.xlsx"
        )
        csv_backup = filename.replace('.xlsx', '.csv')
        bak_file = filename + ".bak"
        
        try:
            # 1. CSV backup (data safety)
            df_forecast.to_csv(csv_backup, index=False)
            logger.info(f"Saved CSV backup: {csv_backup}")
            
            # Booking CSV backup
            if df_booking is not None and not df_booking.empty:
                booking_csv = filename.replace('.xlsx', '_Booking.csv')
                df_booking.to_csv(booking_csv, index=False)
                logger.info(f"Saved Booking CSV backup: {booking_csv}")
            
            # 2. Save multi-sheet Excel
            logger.info(
                f"💾 Saving Master file with Booking sheet "
                f"({len(df_forecast):,} forecast + "
                f"{len(df_booking) if df_booking is not None else 0:,} booking rows)..."
            )
            
            with pd.ExcelWriter(temp_path, engine='xlsxwriter') as writer:
                # Sheet 1: Forecast (main data)
                df_forecast.to_excel(
                    writer, index=False, sheet_name='Forecast'
                )
                
                # Sheet 2: Booking Guests
                if df_booking is not None and not df_booking.empty:
                    df_booking.to_excel(
                        writer, index=False, sheet_name='Booking_Guests'
                    )
                    
                    # Format Booking sheet
                    workbook = writer.book
                    worksheet = writer.sheets['Booking_Guests']
                    
                    # Header format
                    header_fmt = workbook.add_format({
                        'bold': True,
                        'bg_color': '#4472C4',
                        'font_color': 'white',
                        'border': 1,
                        'text_wrap': True,
                        'valign': 'vcenter',
                    })
                    
                    # Write formatted headers
                    for col_num, col_name in enumerate(df_booking.columns):
                        worksheet.write(0, col_num, col_name, header_fmt)
                    
                    # Auto-fit column widths
                    for col_num, col_name in enumerate(df_booking.columns):
                        max_len = max(
                            df_booking[col_name].astype(str).str.len().max(),
                            len(col_name)
                        ) + 2
                        worksheet.set_column(col_num, col_num, min(max_len, 25))
                    
                    # Number format for guest columns
                    num_fmt = workbook.add_format({'num_format': '#,##0'})
                    guest_cols = [
                        i for i, c in enumerate(df_booking.columns)
                        if 'guest' in c.lower() or 'count' in c.lower()
                    ]
                    for col_idx in guest_cols:
                        col_name = df_booking.columns[col_idx]
                        col_width = max(
                            df_booking[col_name].astype(str).str.len().max(),
                            len(col_name)
                        ) + 2
                        worksheet.set_column(
                            col_idx, col_idx,
                            min(col_width, 25),
                            num_fmt
                        )
            
            # 3. Backup old file
            if os.path.exists(filename):
                try:
                    shutil.copy2(filename, bak_file)
                except Exception:
                    pass
            
            # 4. Move temp → final
            if os.path.exists(filename):
                os.remove(filename)
            shutil.move(temp_path, filename)
            logger.info(f"✅ Master file saved with Booking sheet: {filename}")
            
        except Exception as e:
            logger.error(f"Multi-sheet save failed: {e}")
            traceback.print_exc()
            # Fallback: save without booking sheet
            logger.info("Falling back to single-sheet save...")
            save_excel_safely(df_forecast, filename)
