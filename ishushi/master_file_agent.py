"""
==============================================
ISHUSHI MASTER FILE AGENT (v3 - Shift-Aware)
==============================================
Quản lý file Master cho Ishushi forecast:
- Lưu forecast + actual cùng 1 file (SHIFT-BASED)
- Cập nhật Actual khi có dữ liệu thực tế (theo SHIFT)
- Tính error: Error_% = (Predicted - Actual) / Actual × 100
- So sánh forecast vs actual để feedback loop hoạt động

Master File structure:
    Forecast_Run_Date | Restaurant_Code | Date | Weekday | Shift |
    Predicted_Guests | Actual_Guests | Diff_Guest | Error_% |
    Model_Used

Shift definitions:
    MORNING: hours 8-15 (8h – 15h30)
    EVENING: hours 16-23 (15h30 – 23h)
"""

import pandas as pd
import numpy as np
import os
import shutil
import time
import datetime
import tempfile
import traceback

from forecast_system.utils.logger import get_logger
from forecast_system.ishushi.config import ISHUSHI_CONFIG, ISHUSHI_HOUR_TO_SHIFT
from forecast_system.ishushi.data_agent import IshushiDataAgent

logger = get_logger('ishushi_master_file')


# ==========================================
# SAFE EXCEL SAVE
# ==========================================

def save_ishushi_excel(df, filename, extra_sheets=None):
    """
    Save DataFrame to Excel với safety measures.
    
    Args:
        df: Main DataFrame to save
        filename: Target Excel filename
        extra_sheets: Dict of {sheet_name: DataFrame} for additional sheets
    """
    if df.empty:
        logger.warning("Attempted to save empty DataFrame. Aborting.")
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
        
        # 2. Save Excel temp
        with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Forecast')
            
            if extra_sheets:
                for sheet_name, sheet_df in extra_sheets.items():
                    if sheet_df is not None and not sheet_df.empty:
                        sheet_df.to_excel(writer, index=False, sheet_name=sheet_name)
        
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
        logger.info(f"💾 Saved: {filename} ({len(df):,} rows)")
        
    except Exception as e:
        logger.error(f"Excel save failed: {e}")
        traceback.print_exc()


class IshushiMasterFileAgent:
    """
    Agent quản lý Master Forecast file cho Ishushi.
    
    Workflow:
    1. Load master file hiện tại (nếu có)
    2. Append forecast mới
    3. Cập nhật actual từ DB khi có dữ liệu thực tế
    4. Tính error metrics
    5. Save lại
    """
    
    GUEST_COLUMNS = [
        'Forecast_Run_Date', 'Restaurant_Code', 'Date', 'Weekday', 'Shift',
        'Predicted_Guests', 'Predicted_Transactions',
        'Actual_Guests', 'Actual_Transactions',
        'Diff_Guest', 'Error_%', 'Model_Used',
    ]
    
    ITEM_COLUMNS = [
        'Forecast_Run_Date', 'Restaurant_Code', 'Date', 'Weekday', 'Shift',
        'SAP_Code', 'Item_Name', 'Item_Group',
        'Predicted_Quantity', 'Actual_Quantity',
        'Diff_Quantity', 'Error_%', 'Model_Used',
    ]
    
    # File paths
    GUEST_MASTER_FILE = os.path.join(ISHUSHI_CONFIG['output_dir'], 'Ishushi_Master_Guests.xlsx')
    ITEM_MASTER_FILE = os.path.join(ISHUSHI_CONFIG['output_dir'], 'Ishushi_Master_Items.xlsx')
    
    # ==========================================
    # LOAD / CREATE
    # ==========================================
    
    @staticmethod
    def load_guest_master():
        """Load existing guest master file hoặc tạo mới."""
        return IshushiMasterFileAgent._load_or_create(
            IshushiMasterFileAgent.GUEST_MASTER_FILE,
            IshushiMasterFileAgent.GUEST_COLUMNS
        )
    
    @staticmethod
    def load_item_master():
        """Load existing item master file hoặc tạo mới."""
        return IshushiMasterFileAgent._load_or_create(
            IshushiMasterFileAgent.ITEM_MASTER_FILE,
            IshushiMasterFileAgent.ITEM_COLUMNS
        )
    
    @staticmethod
    def _load_or_create(filename, columns):
        """Load existing file or create empty DataFrame."""
        csv_backup = filename.replace('.xlsx', '.csv')
        
        if os.path.exists(filename):
            try:
                df = pd.read_excel(filename, engine='openpyxl')
                if len(df) > 0:
                    logger.info(f"📂 Loaded {len(df):,} rows from {os.path.basename(filename)}")
                    return df
            except Exception as e:
                logger.warning(f"Excel load failed: {e}")
                # Try CSV backup
                if os.path.exists(csv_backup):
                    try:
                        df = pd.read_csv(csv_backup)
                        logger.info(f"🆘 Recovered {len(df):,} rows from CSV backup")
                        return df
                    except Exception:
                        pass
        
        # Create empty
        df = pd.DataFrame(columns=columns)
        logger.info(f"📄 Created new master file: {os.path.basename(filename)}")
        return df
    
    # ==========================================
    # APPEND FORECAST RESULTS
    # ==========================================
    
    @staticmethod
    def append_guest_forecast(df_master, df_forecast, run_date=None):
        """
        Append forecast mới vào master file.
        Nếu đã có forecast cho cùng (restaurant, date) từ cùng run_date → overwrite.
        Nếu từ run_date khác → giữ cả 2 để so sánh.
        
        Args:
            df_master: Existing master DataFrame
            df_forecast: New forecast từ IshushiForecastModel.forecast_guests()
            run_date: Ngày chạy model (default: today)
        
        Returns:
            Updated master DataFrame
        """
        if df_forecast.empty:
            return df_master
        
        if run_date is None:
            run_date = datetime.date.today()
        
        # Build rows to append
        rows = []
        for _, row in df_forecast.iterrows():
            rows.append({
                'Forecast_Run_Date': run_date,
                'Restaurant_Code': str(row['restaurant_code']),
                'Date': row['date'],
                'Weekday': row.get('weekday', ''),
                'Shift': row.get('shift', ''),
                'Predicted_Guests': int(row['predicted_guests']),
                'Predicted_Transactions': int(row.get('predicted_transactions', 0)),
                'Actual_Guests': np.nan,
                'Actual_Transactions': np.nan,
                'Diff_Guest': np.nan,
                'Error_%': np.nan,
                'Model_Used': row.get('model_used', 'ensemble'),
            })
        
        df_new = pd.DataFrame(rows)
        
        if df_master.empty:
            result = df_new
        else:
            # Remove old forecast from same run_date for same (restaurant, date, shift)
            df_master['Forecast_Run_Date'] = pd.to_datetime(
                df_master['Forecast_Run_Date'], errors='coerce'
            ).dt.date
            df_master['Date'] = pd.to_datetime(
                df_master['Date'], errors='coerce'
            ).dt.date
            df_new['Date'] = pd.to_datetime(df_new['Date'], errors='coerce').dt.date
            
            # Ensure Shift column exists in master
            if 'Shift' not in df_master.columns:
                df_master['Shift'] = ''
            
            # Remove duplicates (same run_date + restaurant + date + shift)
            mask_keep = ~(
                (df_master['Forecast_Run_Date'] == run_date) &
                (df_master['Restaurant_Code'].isin(df_new['Restaurant_Code'])) &
                (df_master['Date'].isin(df_new['Date'])) &
                (df_master['Shift'].isin(df_new['Shift']))
            )
            df_master_kept = df_master[mask_keep]
            
            result = pd.concat([df_master_kept, df_new], ignore_index=True)
        
        # Sort
        result = result.sort_values(
            ['Restaurant_Code', 'Date', 'Shift', 'Forecast_Run_Date']
        ).reset_index(drop=True)
        
        n_new = len(rows)
        logger.info(
            f"📝 Appended {n_new:,} guest forecast rows "
            f"(run_date={run_date}, total={len(result):,})"
        )
        
        return result
    
    @staticmethod
    def append_item_forecast(df_master, df_forecast, run_date=None):
        """
        Append item forecast vào item master file.
        
        ⚡ 2026 Filter: Forecast mới chỉ từ run_date trở đi.
           Dữ liệu master cũ được giữ nguyên để so sánh actual.
        """
        if df_forecast.empty:
            return df_master
        
        if run_date is None:
            run_date = datetime.date.today()
        
        rows = []
        for _, row in df_forecast.iterrows():
            rows.append({
                'Forecast_Run_Date': run_date,
                'Restaurant_Code': str(row['restaurant_code']),
                'Date': row['date'],
                'Weekday': row.get('weekday', ''),
                'Shift': row.get('shift', ''),
                'SAP_Code': row.get('sap_code', ''),
                'Item_Name': row.get('item_name', ''),
                'Item_Group': row.get('item_group', ''),
                'Predicted_Quantity': int(row['predicted_quantity']),
                'Actual_Quantity': np.nan,
                'Diff_Quantity': np.nan,
                'Error_%': np.nan,
                'Model_Used': row.get('model_used', 'ensemble'),
            })
        
        df_new = pd.DataFrame(rows)
        
        # ⚡ Filter: Chỉ giữ dữ liệu năm 2026, bắt đầu từ ngày chạy model
        df_new['Date'] = pd.to_datetime(df_new['Date'], errors='coerce').dt.date
        df_new = df_new[
            df_new['Date'].apply(
                lambda d: d is not None and d.year == 2026 and d >= run_date
            )
        ]
        
        if df_new.empty:
            logger.info("   ⚠️ No item forecast rows in 2026 from run_date onwards")
            return df_master
        
        logger.info(
            f"   📅 Item forecast filtered to 2026 from {run_date}: "
            f"{len(df_new):,} rows kept"
        )
        
        if df_master.empty:
            result = df_new
        else:
            df_master['Forecast_Run_Date'] = pd.to_datetime(
                df_master['Forecast_Run_Date'], errors='coerce'
            ).dt.date
            df_master['Date'] = pd.to_datetime(
                df_master['Date'], errors='coerce'
            ).dt.date
            
            # Ensure Shift column exists in master
            if 'Shift' not in df_master.columns:
                df_master['Shift'] = ''
            
            # NOTE: Do NOT filter existing master by d >= run_date!
            # That would delete historical forecast data (10/4, 11/4, etc.)
            # which we need to keep for actual comparison.
            # Only filter out non-2026 data if any exists.
            df_master = df_master[
                df_master['Date'].apply(
                    lambda d: d is not None and d.year == 2026
                )
            ]
            
            # Remove same run_date duplicates (including shift)
            mask_keep = ~(
                (df_master['Forecast_Run_Date'] == run_date) &
                (df_master['Restaurant_Code'].isin(df_new['Restaurant_Code'])) &
                (df_master['Date'].isin(df_new['Date'])) &
                (df_master['Shift'].isin(df_new['Shift']))
            )
            df_master_kept = df_master[mask_keep]
            
            result = pd.concat([df_master_kept, df_new], ignore_index=True)
        
        result = result.sort_values(  # type: ignore[reportCallIssue]
            ['Restaurant_Code', 'Date', 'Shift', 'SAP_Code', 'Forecast_Run_Date']
        ).reset_index(drop=True)
        
        logger.info(
            f"📝 Appended {len(rows):,} item forecast rows "
            f"(run_date={run_date}, total={len(result):,})"
        )
        
        return result
    
    # ==========================================
    # UPDATE ACTUALS
    # ==========================================
    
    @staticmethod
    def update_guest_actuals(df_master, df_transactions):
        """
        Cập nhật Actual_Guests cho các ngày đã có dữ liệu thực tế.
        ⭐ v3: Shift-aware - tính actual THEO CA LÀM VIỆC.
        
        Args:
            df_master: Guest master DataFrame (phải có cột Shift)
            df_transactions: Raw transaction data từ IshushiDataAgent
        
        Returns:
            Updated master DataFrame with actuals filled in
        """
        if df_master.empty or df_transactions.empty:
            return df_master
        
        logger.info("📊 Updating guest actuals (shift-aware v3)...")
        
        # Build actual totals from transactions BY SHIFT
        df_trans = df_transactions.copy()
        df_trans['restaurant_code'] = IshushiDataAgent.normalize_key(df_trans['restaurant_code'])
        
        # Map hour → shift
        df_trans['shift'] = df_trans['hour'].map(ISHUSHI_HOUR_TO_SHIFT)
        df_trans = df_trans.dropna(subset=['shift'])
        
        # Aggregate per (restaurant, date, shift)
        actuals = df_trans.groupby(['restaurant_code', 'date', 'shift']).agg(
            actual_guests=('guest_count', 'sum'),
            actual_transactions=('transaction_id', 'nunique'),
        ).reset_index()
        
        actuals.rename(columns={
            'restaurant_code': 'Restaurant_Code',
            'date': 'Date',
            'shift': 'Shift',
        }, inplace=True)
        
        actuals['Date'] = pd.to_datetime(actuals['Date'], errors='coerce').dt.date
        
        # Normalize master dates
        df_master['Date'] = pd.to_datetime(df_master['Date'], errors='coerce').dt.date
        df_master['Restaurant_Code'] = df_master['Restaurant_Code'].astype(str)
        
        # Ensure Shift column exists and normalize
        if 'Shift' not in df_master.columns:
            df_master['Shift'] = ''
        df_master['Shift'] = df_master['Shift'].fillna('').astype(str)
        
        # Drop old temp columns to avoid conflicts
        df_master = df_master.drop(columns=['_actual_guests', '_actual_trans'], errors='ignore')
        
        # Merge actuals into master BY SHIFT
        merged = pd.merge(
            df_master,
            actuals.rename(columns={
                'actual_guests': '_actual_guests',
                'actual_transactions': '_actual_trans',
            }),
            on=['Restaurant_Code', 'Date', 'Shift'],
            how='left',
        )
        
        # Update only where we have actual data
        has_actual = merged['_actual_guests'].notna()
        merged.loc[has_actual, 'Actual_Guests'] = merged.loc[has_actual, '_actual_guests']
        merged.loc[has_actual, 'Actual_Transactions'] = merged.loc[has_actual, '_actual_trans']
        
        # Calculate errors - Error_% = (Pred - Actual) / Actual × 100
        has_both = (
            merged['Actual_Guests'].notna() &
            merged['Predicted_Guests'].notna() &
            (merged['Actual_Guests'] > 0)
        )
        
        merged.loc[has_both, 'Diff_Guest'] = (
            merged.loc[has_both, 'Predicted_Guests'] -
            merged.loc[has_both, 'Actual_Guests']
        )
        
        # ⭐ Error_% dùng Actual làm mẫu số (standard MAPE)
        merged.loc[has_both, 'Error_%'] = np.round(
            (merged.loc[has_both, 'Diff_Guest'] /
             merged.loc[has_both, 'Actual_Guests']) * 100,
            1
        )
        
        # Clean up temp columns
        merged = merged.drop(columns=['_actual_guests', '_actual_trans'], errors='ignore')
        
        n_updated = has_actual.sum()
        n_with_error = has_both.sum()
        logger.info(
            f"   ✅ Guest actuals updated: {n_updated:,} rows, "
            f"{n_with_error:,} with error calculated (shift-aware)"
        )
        
        return merged
    
    @staticmethod
    def update_item_actuals(df_master, df_transactions, df_items):
        """
        Cập nhật Actual_Quantity cho items đã có dữ liệu thực tế.
        ⭐ v3: Shift-aware - tính actual THEO CA LÀM VIỆC.
        
        Args:
            df_master: Item master DataFrame (phải có cột Shift)
            df_transactions: Transaction data (có date, restaurant_code, hour)
            df_items: Item detail data (có transaction_id, sap_code, quantity)
        
        Returns:
            Updated item master DataFrame
        """
        if df_master.empty or df_transactions.empty or df_items.empty:
            return df_master
        
        logger.info("📊 Updating item actuals (shift-aware v3)...")
        
        # Merge transactions with items, including hour for shift mapping
        df_trans = df_transactions.copy()
        df_trans['shift'] = df_trans['hour'].map(ISHUSHI_HOUR_TO_SHIFT)
        df_trans = df_trans.dropna(subset=['shift'])
        
        df_merged = pd.merge(
            df_trans[['transaction_id', 'restaurant_code', 'date', 'shift']],
            df_items[['transaction_id', 'sap_code', 'quantity']],
            on='transaction_id',
            how='inner',
        )
        
        if df_merged.empty:
            return df_master
        
        df_merged['restaurant_code'] = IshushiDataAgent.normalize_key(df_merged['restaurant_code'])
        
        # Aggregate per (restaurant, date, shift, sap_code)
        actuals = df_merged.groupby(
            ['restaurant_code', 'date', 'shift', 'sap_code']
        )['quantity'].sum().reset_index()
        
        actuals.rename(columns={
            'restaurant_code': 'Restaurant_Code',
            'date': 'Date',
            'shift': 'Shift',
            'sap_code': 'SAP_Code',
            'quantity': '_actual_qty',
        }, inplace=True)
        
        actuals['Date'] = pd.to_datetime(actuals['Date'], errors='coerce').dt.date
        
        # Normalize master
        df_master['Date'] = pd.to_datetime(df_master['Date'], errors='coerce').dt.date
        df_master['Restaurant_Code'] = df_master['Restaurant_Code'].astype(str)
        
        # Ensure Shift column exists and normalize
        if 'Shift' not in df_master.columns:
            df_master['Shift'] = ''
        df_master['Shift'] = df_master['Shift'].fillna('').astype(str)
        
        # Ensure SAP_Code types match
        actuals['SAP_Code'] = actuals['SAP_Code'].astype(str)
        df_master['SAP_Code'] = df_master['SAP_Code'].astype(str)
        
        # Merge BY SHIFT
        merged = pd.merge(
            df_master.drop(columns=['_actual_qty'], errors='ignore'),
            actuals,
            on=['Restaurant_Code', 'Date', 'Shift', 'SAP_Code'],
            how='left',
        )
        
        has_actual = merged['_actual_qty'].notna()
        merged.loc[has_actual, 'Actual_Quantity'] = merged.loc[has_actual, '_actual_qty']
        
        # Calculate errors - Error_% = (Pred - Actual) / Actual × 100
        has_both = (
            merged['Actual_Quantity'].notna() &
            merged['Predicted_Quantity'].notna() &
            (merged['Actual_Quantity'] > 0)
        )
        
        merged.loc[has_both, 'Diff_Quantity'] = (
            merged.loc[has_both, 'Predicted_Quantity'] -
            merged.loc[has_both, 'Actual_Quantity']
        )
        
        # ⭐ Error_% dùng Actual làm mẫu số (standard MAPE)
        merged.loc[has_both, 'Error_%'] = np.round(
            (merged.loc[has_both, 'Diff_Quantity'] /
             merged.loc[has_both, 'Actual_Quantity']) * 100,
            1
        )
        
        merged = merged.drop(columns=['_actual_qty'], errors='ignore')
        
        n_updated = has_actual.sum()
        n_with_error = has_both.sum()
        logger.info(
            f"   ✅ Item actuals updated: {n_updated:,} rows, "
            f"{n_with_error:,} with error calculated (shift-aware)"
        )
        
        return merged
    
    # ==========================================
    # ACCURACY METRICS
    # ==========================================
    
    @staticmethod
    def calculate_accuracy(df_master, target='guests'):
        """
        Tính accuracy metrics cho guest hoặc item forecast.
        
        Args:
            df_master: Master DataFrame
            target: 'guests' | 'items'
        
        Returns:
            Dict: Overall + per-restaurant metrics
        """
        if target == 'guests':
            pred_col = 'Predicted_Guests'
            actual_col = 'Actual_Guests'
        else:
            pred_col = 'Predicted_Quantity'
            actual_col = 'Actual_Quantity'
        
        # Filter valid rows
        mask = (
            pd.notna(df_master[pred_col]) &
            pd.notna(df_master[actual_col]) &
            (df_master[pred_col] >= 0) &
            (df_master[actual_col] >= 0)
        )
        df_valid = df_master[mask].copy()
        
        if df_valid.empty:
            return {'overall': {}, 'per_restaurant': {}}
        
        pred = df_valid[pred_col].values.astype(float)
        actual = df_valid[actual_col].values.astype(float)
        
        def _calc_metrics(p, a):
            if len(p) == 0:
                return {}
            errors = p - a
            abs_errors = np.abs(errors)
            nonzero = a > 0
            
            mae = float(np.mean(abs_errors))
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            bias = float(np.mean(errors))
            
            if nonzero.sum() > 0:
                mape = float(np.mean(abs_errors[nonzero] / a[nonzero]) * 100)
            else:
                mape = np.nan
            
            # Hit rate: error ≤ 15 (absolute threshold)
            hit_rate = float(np.mean(abs_errors <= 15) * 100)
            
            return {
                'MAE': round(mae, 2),
                'MAPE': round(mape, 1) if not np.isnan(mape) else None,
                'RMSE': round(rmse, 2),
                'Bias': round(bias, 2),
                'Hit_Rate': round(hit_rate, 1),
                'N_samples': int(len(p)),
            }
        
        # Overall
        overall = _calc_metrics(pred, actual)
        
        # Per restaurant
        per_restaurant = {}
        for res_code in df_valid['Restaurant_Code'].unique():
            mask_res = df_valid['Restaurant_Code'] == res_code
            p_res = pred[mask_res]
            a_res = actual[mask_res]
            per_restaurant[str(res_code)] = _calc_metrics(p_res, a_res)
        
        # Per shift
        per_shift = {}
        if 'Shift' in df_valid.columns:
            for shift in df_valid['Shift'].unique():
                mask_shift = df_valid['Shift'] == shift
                p_shift = pred[mask_shift]
                a_shift = actual[mask_shift]
                per_shift[str(shift)] = _calc_metrics(p_shift, a_shift)
        
        return {
            'overall': overall,
            'per_restaurant': per_restaurant,
            'per_shift': per_shift,
        }
    
    # ==========================================
    # SAVE
    # ==========================================
    
    @staticmethod
    def _clean_shift_rows(df):
        """
        ⭐ Loại bỏ rows không có Shift (data cũ hoặc daily total thừa).
        Chỉ giữ rows có Shift = 'MORNING' hoặc 'EVENING'.
        """
        if df.empty:
            return df
        
        if 'Shift' not in df.columns:
            return df
        
        valid_shifts = {'MORNING', 'EVENING'}
        before = len(df)
        df = df[df['Shift'].isin(valid_shifts)].copy()
        removed = before - len(df)
        
        if removed > 0:
            logger.info(f"   🧹 Cleaned {removed:,} non-shift rows (kept {len(df):,})")
        
        return df
    
    @staticmethod
    def save_guest_master(df_master, accuracy=None):
        """Save guest master file with optional accuracy sheet.
        ⭐ v3: Tự động dọn dẹp rows không có Shift."""
        os.makedirs(os.path.dirname(IshushiMasterFileAgent.GUEST_MASTER_FILE), exist_ok=True)
        
        # Clean: chỉ giữ shift rows
        df_master = IshushiMasterFileAgent._clean_shift_rows(df_master)
        
        extra_sheets = {}
        if accuracy and accuracy.get('per_restaurant'):
            acc_rows = []
            for rc, m in accuracy['per_restaurant'].items():
                acc_rows.append({'Restaurant_Code': rc, **m})
            if acc_rows:
                extra_sheets['Accuracy_Per_Restaurant'] = pd.DataFrame(acc_rows)
        
        if accuracy and accuracy.get('overall'):
            extra_sheets['Overall_Accuracy'] = pd.DataFrame([accuracy['overall']])
        
        # ⭐ Per-shift accuracy sheet
        if accuracy and accuracy.get('per_shift'):
            shift_rows = []
            for shift, m in accuracy['per_shift'].items():
                shift_rows.append({'Shift': shift, **m})
            if shift_rows:
                extra_sheets['Accuracy_Per_Shift'] = pd.DataFrame(shift_rows)
        
        save_ishushi_excel(
            df_master, IshushiMasterFileAgent.GUEST_MASTER_FILE,
            extra_sheets=extra_sheets
        )
    
    @staticmethod
    def save_item_master(df_master, accuracy=None):
        """Save item master file.
        ⭐ v3: Tự động dọn dẹp rows không có Shift."""
        os.makedirs(os.path.dirname(IshushiMasterFileAgent.ITEM_MASTER_FILE), exist_ok=True)
        
        # Clean: chỉ giữ shift rows
        df_master = IshushiMasterFileAgent._clean_shift_rows(df_master)
        
        extra_sheets = {}
        if accuracy and accuracy.get('overall'):
            extra_sheets['Overall_Accuracy'] = pd.DataFrame([accuracy['overall']])
        
        save_ishushi_excel(
            df_master, IshushiMasterFileAgent.ITEM_MASTER_FILE,
            extra_sheets=extra_sheets
        )
