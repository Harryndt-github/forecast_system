"""
==============================================
DATA AGENT
==============================================
Trách nhiệm:
- Load dữ liệu từ Database (parameterized queries)
- Clean & normalize data
- Tính hourly ratios, shift mapping
- Thống kê window statistics

Refactored từ forecast_fb.py DataAgent class.
Fixes: SQL injection, bare except, hardcoded values.
"""

import pandas as pd
import numpy as np
import datetime
import traceback

from forecast_system.config.settings import (
    CURRENT_DATE, DATA_LOOKBACK_DAYS, MASTER_FILE_NAME,
    SHIFT_DEFINITIONS, ALL_OPERATING_HOURS
)
from forecast_system.utils.db_utils import (
    fetch_with_chunks, execute_parameterized_query,
    QUERY_PAYMENT_HUB, QUERY_RK_DC, QUERY_RESTAURANT_INFO
)
from forecast_system.utils.logger import get_logger

logger = get_logger('data_agent')


class DataAgent:
    """
    Agent xử lý dữ liệu: load, clean, normalize, aggregate.
    """
    
    # ==========================================
    # DATA CLEANING
    # ==========================================
    
    @staticmethod
    def clean_id(series):
        """Clean transaction IDs: uppercase, strip, remove brackets"""
        return (series.astype(str)
                .str.upper()
                .str.strip()
                .str.replace(r'[{}]', '', regex=True))

    @staticmethod
    def normalize_key(series):
        """
        Chuẩn hóa restaurant_code để merge chính xác:
        - Uppercase, strip whitespace
        - Remove .0 suffix (Excel float issue)
        - Remove leading zeros
        """
        s = series.astype(str).str.upper().str.strip()
        s = s.str.replace(r'\.0$', '', regex=True)
        s = s.str.replace(r'^0+', '', regex=True)
        return s.replace('', '0')

    # ==========================================
    # DATA LOADING (PARAMETERIZED QUERIES)
    # ==========================================
    
    @staticmethod
    def load_restaurant_info(engine):
        """
        Load thông tin nhà hàng từ DB.
        Sử dụng parameterized query.
        
        Returns:
            pd.DataFrame với columns: Restaurant_Code, sap_code, restaurant_name, merge_key
        """
        logger.info("🏢 Loading Restaurant Info...")
        try:
            df = execute_parameterized_query(engine, QUERY_RESTAURANT_INFO)
            
            if df.empty:
                logger.warning("No restaurant info found in DB")
                return pd.DataFrame()
            
            # Tạo merge key chuẩn hóa
            df['merge_key'] = DataAgent.normalize_key(df['restaurant_code'])
            df = df.rename(columns={'restaurant_code': 'Restaurant_Code'})
            df = df.drop_duplicates(subset=['merge_key'])
            
            logger.info(f"Loaded {len(df)} restaurants info")
            return df
            
        except Exception as e:
            logger.error(f"Error loading restaurant info: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    @staticmethod
    def load_recent_data(engine, lookback_days=None):
        """
        Load transaction data từ 2 bảng DB, merge & clean.
        Sử dụng parameterized queries + chunked loading.
        
        Args:
            engine: SQLAlchemy engine
            lookback_days: Số ngày lookback (default từ config)
        
        Returns:
            pd.DataFrame với columns: restaurant_code, transaction_id, guest_count, 
                                      open_time, date, hour, weekday
        """
        if lookback_days is None:
            lookback_days = DATA_LOOKBACK_DAYS
            
        logger.info(f"⏳ Loading Recent Data (Last {lookback_days} Days)...")
        
        start_dt = CURRENT_DATE - datetime.timedelta(days=lookback_days)
        end_dt = CURRENT_DATE + datetime.timedelta(days=1)
        
        params = {
            'start_date': start_dt.strftime('%Y-%m-%d'),
            'end_date': end_dt.strftime('%Y-%m-%d'),
        }
        
        try:
            # Fetch từ 2 bảng sử dụng parameterized queries + chunks
            d1 = fetch_with_chunks(
                engine, QUERY_PAYMENT_HUB, params=params,
                name="Payment Hub"
            )
            d2 = fetch_with_chunks(
                engine, QUERY_RK_DC, params=params,
                name="RK DC"
            )
            
            # Clean transaction IDs
            if not d1.empty:
                d1['transaction_id'] = DataAgent.clean_id(d1['transaction_id'])
            if not d2.empty:
                d2['transaction_id'] = DataAgent.clean_id(d2['transaction_id'])
            
            # Merge & deduplicate
            # Sort by guest_count DESC before dedup so that when the same
            # transaction exists in both tables, we keep the record with the
            # HIGHER guest_count (fix: payment_hub may have guest_count=0
            # while rk_dc has the correct value, or vice versa)
            df = pd.concat([d1, d2], ignore_index=True)
            df['guest_count'] = pd.to_numeric(df['guest_count'], errors='coerce').fillna(0)  # type: ignore[reportAttributeAccessIssue]
            df = df.sort_values('guest_count', ascending=False)
            df = df.drop_duplicates(subset=['transaction_id'], keep='first')
            
            # Normalize
            df['restaurant_code'] = DataAgent.normalize_key(df['restaurant_code'])
            df['open_time'] = pd.to_datetime(df['open_time'], errors='coerce')
            df = df.dropna(subset=['open_time'])
            
            # Extract time features
            df['date'] = df['open_time'].dt.date
            df['hour'] = df['open_time'].dt.hour
            df['weekday'] = df['open_time'].dt.day_name()
            
            logger.info(f"Loaded {len(df):,} transactions (by open_time hour)")
            return df
            
        except Exception as e:
            logger.error(f"DB Error loading recent data: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    @staticmethod
    def load_date_range(engine, start_date, end_date):
        """
        Load transaction data cho một khoảng ngày cụ thể.
        Dùng cho việc load historical data (VD: Tết năm trước).
        
        Args:
            engine: SQLAlchemy engine
            start_date: datetime.date - ngày bắt đầu
            end_date: datetime.date - ngày kết thúc (inclusive)
            
        Returns:
            pd.DataFrame giống format load_recent_data
        """
        logger.info(f"⏳ Loading Date Range: {start_date} → {end_date}...")
        
        params = {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': (end_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d'),
        }
        
        try:
            d1 = fetch_with_chunks(
                engine, QUERY_PAYMENT_HUB, params=params,
                name=f"Payment Hub ({start_date}→{end_date})"
            )
            d2 = fetch_with_chunks(
                engine, QUERY_RK_DC, params=params,
                name=f"RK DC ({start_date}→{end_date})"
            )
            
            if not d1.empty:
                d1['transaction_id'] = DataAgent.clean_id(d1['transaction_id'])
            if not d2.empty:
                d2['transaction_id'] = DataAgent.clean_id(d2['transaction_id'])
            
            df = pd.concat([d1, d2], ignore_index=True)
            df['guest_count'] = pd.to_numeric(df['guest_count'], errors='coerce').fillna(0)  # type: ignore[reportAttributeAccessIssue]
            df = df.sort_values('guest_count', ascending=False)
            df = df.drop_duplicates(subset=['transaction_id'], keep='first')
            
            df['restaurant_code'] = DataAgent.normalize_key(df['restaurant_code'])
            df['open_time'] = pd.to_datetime(df['open_time'], errors='coerce')
            df = df.dropna(subset=['open_time'])
            
            df['date'] = df['open_time'].dt.date
            df['hour'] = df['open_time'].dt.hour
            df['weekday'] = df['open_time'].dt.day_name()
            
            logger.info(f"Loaded {len(df):,} transactions for {start_date} → {end_date}")
            return df
            
        except Exception as e:
            logger.error(f"DB Error loading date range: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    # ==========================================
    # HOURLY RATIOS & SHIFT MAPPING
    # ==========================================
    
    @staticmethod
    def get_hourly_ratios(df_res):
        """
        Tính tỉ lệ phân bổ khách theo giờ, riêng cho Ngày thường và Cuối tuần.
        Bỏ giờ inactive (ratio < 1% hoặc avg < 0.5 guest).
        
        Returns:
            (weekday_ratios, weekend_ratios): Tuple of dicts {hour: ratio}
        """
        if df_res.empty:
            return {}, {}
        
        df_res = df_res.copy()
        df_res['is_weekend'] = pd.to_datetime(df_res['date']).dt.dayofweek.isin([5, 6])
        
        def calculate_ratios(sub_df):
            if sub_df.empty:
                return {}
            hourly_sum = sub_df.groupby('hour')['guest_count'].sum()
            hourly_avg = sub_df.groupby('hour')['guest_count'].mean()
            total = hourly_sum.sum()
            if total == 0:
                return {}
            ratios = {}
            for h, c in hourly_sum.items():
                r = c / total
                avg = hourly_avg.get(h, 0)
                if r > 0.01 and avg >= 0.5:  # Skip inactive hours
                    ratios[int(h)] = r
            s = sum(ratios.values())
            return {h: v / s for h, v in ratios.items()} if s > 0 else {}

        wd_ratios = calculate_ratios(df_res[~df_res['is_weekend']])
        we_ratios = calculate_ratios(df_res[df_res['is_weekend']])
        
        # Fallback
        if not wd_ratios:
            wd_ratios = we_ratios
        if not we_ratios:
            we_ratios = wd_ratios
        
        return wd_ratios, we_ratios

    @staticmethod
    def get_weekday_hourly_ratios(df_res, recency_days=30):  # [FIX #2a] 14 → 30 ngày
        """
        Tính tỉ lệ phân bổ khách theo giờ CHO TỪNG ngày trong tuần.
        Data gần đây (< recency_days) được weight 3x so với data cũ.
        
        Lọc:
        - Bỏ giờ có tỉ lệ < 1% (negligible)
        - Bỏ giờ có trung bình < 0.5 khách/lần xuất hiện (inactive hour)
        
        Returns:
            Dict[str, Dict[int, float]]: 
            {weekday_name: {hour: ratio}}
            VD: {'Monday': {10: 0.05, 11: 0.08, ...}, ...}
        """
        if df_res.empty:
            return {}
        
        df = df_res.copy()
        df['_date'] = pd.to_datetime(df['date'], errors='coerce')
        df['_weekday'] = df['_date'].dt.day_name()
        
        # Add recency weight
        max_date = df['_date'].max()
        df['_days_ago'] = (max_date - df['_date']).dt.days
        df['_weight'] = df['_days_ago'].apply(
            lambda d: 3.0 if d <= recency_days else 1.0
        )
        df['_weighted_guests'] = df['guest_count'] * df['_weight']
        
        result = {}
        for wd_name in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                        'Friday', 'Saturday', 'Sunday']:
            sub = df[df['_weekday'] == wd_name]
            if sub.empty:
                continue
            
            hourly = sub.groupby('hour')['_weighted_guests'].sum()
            total = hourly.sum()
            if total <= 0:
                continue
            
            # Also compute actual average per hour to filter inactive hours
            hourly_avg = sub.groupby('hour')['guest_count'].mean()
            
            ratios = {}
            for h, val in hourly.items():
                r = val / total
                avg_guests = hourly_avg.get(h, 0)
                # Filter: ratio > 1% AND average >= 0.5 guests per occurrence
                if r > 0.01 and avg_guests >= 0.5:
                    ratios[int(h)] = r
            
            # Normalize
            s = sum(ratios.values())
            if s > 0:
                result[wd_name] = {h: v / s for h, v in ratios.items()}
        
        return result

    @staticmethod
    def map_to_shift(hour):
        """
        Map hour → shift name (Phase 8: 2-shift system).
        
        Shifts:
            08-15: MORNING (Ca Sáng) - kết thúc 15:30
            16-23: EVENING (Ca Tối)  - bắt đầu 15:30, kết thúc 23h
        """
        for shift_key, shift_def in SHIFT_DEFINITIONS.items():
            if hour in shift_def['hours']:
                return shift_key
        return "OTHER"

    @staticmethod
    def get_shift_ratios(df_res):
        """
        Tính tỉ lệ phân bổ khách theo CA (MORNING/EVENING),
        riêng cho Ngày thường và Cuối tuần.
        
        Returns:
            (weekday_ratios, weekend_ratios): Tuple of dicts {shift_key: ratio}
        """
        if df_res.empty:
            return {}, {}
        
        df = df_res.copy()
        df['is_weekend'] = pd.to_datetime(df['date']).dt.dayofweek.isin([5, 6])
        df['shift'] = df['hour'].apply(DataAgent.map_to_shift)
        df = df[df['shift'] != 'OTHER']  # Exclude non-operating hours
        
        def calc_ratios(sub_df):
            if sub_df.empty:
                return {}
            shift_sum = sub_df.groupby('shift')['guest_count'].sum()
            total = shift_sum.sum()
            if total == 0:
                return {}
            return {s: v / total for s, v in shift_sum.items()}
        
        wd_ratios = calc_ratios(df[~df['is_weekend']])
        we_ratios = calc_ratios(df[df['is_weekend']])
        
        if not wd_ratios:
            wd_ratios = we_ratios
        if not we_ratios:
            we_ratios = wd_ratios
        
        return wd_ratios, we_ratios

    @staticmethod
    def get_weekday_shift_ratios(df_res, recency_days=30):  # [FIX #2a] 14 → 30 ngày
        """
        Tính tỉ lệ phân bổ khách theo CA CHO TỪNG ngày trong tuần.
        Data gần đây (< recency_days) được weight 3x.
        
        Returns:
            Dict[str, Dict[str, float]]: 
            {weekday_name: {shift_key: ratio}}
            VD: {'Monday': {'MORNING': 0.45, 'EVENING': 0.55}, ...}
        """
        if df_res.empty:
            return {}
        
        df = df_res.copy()
        df['_date'] = pd.to_datetime(df['date'], errors='coerce')
        df['_weekday'] = df['_date'].dt.day_name()
        df['shift'] = df['hour'].apply(DataAgent.map_to_shift)
        df = df[df['shift'] != 'OTHER']
        
        # Add recency weight
        max_date = df['_date'].max()
        df['_days_ago'] = (max_date - df['_date']).dt.days
        df['_weight'] = df['_days_ago'].apply(
            lambda d: 3.0 if d <= recency_days else 1.0
        )
        df['_weighted_guests'] = df['guest_count'] * df['_weight']
        
        result = {}
        for wd_name in ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                        'Friday', 'Saturday', 'Sunday']:
            sub = df[df['_weekday'] == wd_name]
            if sub.empty:
                continue
            
            shift_totals = sub.groupby('shift')['_weighted_guests'].sum()
            total = shift_totals.sum()
            if total <= 0:
                continue
            
            ratios = {s: v / total for s, v in shift_totals.items()}
            result[wd_name] = ratios
        
        return result

    @staticmethod
    def aggregate_shifts(df_master):
        """
        Tổng hợp kết quả forecast theo ca (shift) cho báo cáo.
        Phase 8: Dùng trực tiếp cột Shift thay vì map từ Hour.
        
        Returns:
            pd.DataFrame grouped by Restaurant, Date, Shift
        """
        if df_master.empty:
            return pd.DataFrame()
        
        df = df_master.copy()
        
        # Phase 8: Shift-based forecast có cột Shift trực tiếp
        if 'Shift' in df.columns:
            df = df[df['Shift'].notna()].copy()
        elif 'Hour' in df.columns:
            # Legacy: map từ Hour
            df = df[df['Hour'].notna()].copy()
            if df.empty:
                return pd.DataFrame()
            df['Hour'] = df['Hour'].astype(int)
            df['Shift'] = df['Hour'].apply(DataAgent.map_to_shift)
        else:
            return pd.DataFrame()
        
        if df.empty:
            return pd.DataFrame()
        
        # Group by columns that exist
        group_cols = ['Restaurant_Code', 'Date', 'Weekday', 'Shift']
        for opt_col in ['sap_code', 'restaurant_name']:
            if opt_col in df.columns:
                group_cols.insert(1, opt_col)
        
        agg = df.groupby(group_cols).agg({
            'Final_Predicted_Guests': 'sum',
            'Is_Holiday': 'first',
            'Is_Veg': 'first'
        }).reset_index()
        
        return agg

    # ==========================================
    # WINDOW STATISTICS
    # ==========================================
    
    @staticmethod
    def get_window_statistics(df_daily, window_days, target_weekday, ref_date):
        """
        Lấy thống kê guest_count trung bình cho weekday cụ thể trong time window.
        
        Args:
            df_daily: DataFrame đã group theo ngày
            window_days: Số ngày lookback
            target_weekday: Tên thứ (e.g. 'Monday')
            ref_date: Ngày tham chiếu
        
        Returns:
            float hoặc None
        """
        start_date = ref_date - datetime.timedelta(days=window_days)
        df_window = df_daily[
            (df_daily['date'] >= start_date) & (df_daily['date'] < ref_date)
        ]
        
        if df_window.empty:
            return None
        
        df_weekday = df_window[df_window['weekday'] == target_weekday]
        return df_weekday['guest_count'].mean() if not df_weekday.empty else None

    # ==========================================
    # DATA QUALITY CHECKS
    # ==========================================
    
    @staticmethod
    def get_active_restaurants(df_train, inactive_threshold=30):
        """
        Lọc danh sách nhà hàng active (có data trong N ngày gần nhất).
        
        Returns:
            pd.Index: Danh sách restaurant_code active
        """
        active = df_train.groupby('restaurant_code')['date'].max()
        cutoff = CURRENT_DATE - datetime.timedelta(days=inactive_threshold)
        active_codes = active[active >= cutoff].index
        
        logger.info(f"Found {len(active_codes)} active restaurants "
                    f"(threshold: {inactive_threshold} days)")
        return active_codes

    @staticmethod
    def get_daily_summary(df_res):
        """
        Tạo daily summary cho 1 nhà hàng.
        
        Returns:
            pd.DataFrame với columns: date, weekday, guest_count
        """
        if df_res.empty:
            return pd.DataFrame(columns=['date', 'weekday', 'guest_count'])  # type: ignore[reportArgumentType]
        
        daily = df_res.groupby(['date', 'weekday'])['guest_count'].sum().reset_index()
        return daily.sort_values('date')

    @staticmethod
    def load_lunar_ny_closures(closure_file: str, df_info: pd.DataFrame) -> dict:
        """
        Load lịch đóng cửa Tết Nguyên Đán từ file Excel.
        
        File format: 
            Columns: Khu vực, SAP, Location, Nhãn hàng, NH Hệ thống,
                     14/02/2026, 15/02/2026, ... (date columns)
            Values:  Mở, Đóng, Bán nửa ngày, Bán sáng, Bán chiều, etc.
        
        Args:
            closure_file: Đường dẫn tới file Excel (Close_lunar_NY_2026.xlsx)
            df_info: DataFrame từ load_restaurant_info() chứa sap_code → Restaurant_Code mapping
            
        Returns:
            Dict[str, Dict[datetime.date, str]]:
            {restaurant_code: {date: status}}
            status: 'CLOSED', 'HALF_DAY', 'OPEN'
            
        Example:
            {'766': {date(2026,2,16): 'CLOSED', date(2026,2,17): 'CLOSED', ...}}
        """
        try:
            df_close = pd.read_excel(closure_file, engine='openpyxl')
        except FileNotFoundError:
            logger.warning(f"Closure file not found: {closure_file}")
            return {}
        except Exception as e:
            logger.warning(f"Error reading closure file: {e}")
            return {}
        
        if df_close.empty or 'SAP' not in df_close.columns:
            logger.warning("Closure file is empty or missing 'SAP' column")
            return {}
        
        # Build SAP → Restaurant_Code mapping
        sap_to_res = {}
        if not df_info.empty and 'sap_code' in df_info.columns:
            for _, row in df_info.iterrows():
                sap_str = str(row['sap_code']).strip()
                res_code = str(row.get('Restaurant_Code', '')).strip()
                # Remove .0 suffix from Restaurant_Code
                if res_code.endswith('.0'):
                    res_code = res_code[:-2]
                if sap_str and res_code and res_code != 'nan':
                    sap_to_res[sap_str] = res_code
        
        # Identify date columns (format: dd/mm/yyyy or datetime)
        date_columns = []
        for col in df_close.columns:
            col_str = str(col).strip()
            # Try parsing dd/mm/yyyy
            for fmt in ['%d/%m/%Y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                try:
                    parsed_date = datetime.datetime.strptime(col_str, fmt).date()
                    date_columns.append((col, parsed_date))
                    break
                except ValueError:
                    continue
        
        if not date_columns:
            logger.warning("No date columns found in closure file")
            return {}
        
        logger.info(
            f"📋 Lunar NY closures: {len(df_close)} restaurants, "
            f"{len(date_columns)} days "
            f"({date_columns[0][1]} → {date_columns[-1][1]})"
        )
        
        # Parse closure status
        def classify_status(value: str) -> str:
            """Phân loại trạng thái: CLOSED, HALF_DAY, OPEN"""
            if pd.isna(value):
                return 'OPEN'
            v = str(value).strip().lower()
            
            if v in ('đóng', 'đóng '):
                return 'CLOSED'
            elif v in ('convert',):
                return 'CLOSED'  # Convert = chuyển đổi → coi như đóng
            elif any(kw in v for kw in ['nửa ngày', 'bán sáng', 'bán chiều', 
                                         'đóng 0.5', 'mở0.5', 'trực']):
                return 'HALF_DAY'
            else:
                return 'OPEN'
        
        # Build result dict
        result = {}
        matched = 0
        unmatched_saps = []
        
        for _, row in df_close.iterrows():
            sap_str = str(row['SAP']).strip()
            # Remove .0 suffix
            if sap_str.endswith('.0'):
                sap_str = sap_str[:-2]
            
            res_code = sap_to_res.get(sap_str)
            if not res_code:
                unmatched_saps.append(sap_str)
                continue
            
            matched += 1
            res_closures = {}
            
            for col, date_obj in date_columns:
                status = classify_status(row.get(col, ''))  # type: ignore[reportArgumentType]
                if status != 'OPEN':
                    res_closures[date_obj] = status
            
            if res_closures:
                result[res_code] = res_closures
        
        # Summary
        total_closed_days = sum(
            1 for closures in result.values() 
            for s in closures.values() if s == 'CLOSED'
        )
        total_half_days = sum(
            1 for closures in result.values() 
            for s in closures.values() if s == 'HALF_DAY'
        )
        
        logger.info(
            f"   Matched: {matched}/{len(df_close)} restaurants (SAP→Code)"
        )
        logger.info(
            f"   Closures: {len(result)} restaurants with closure dates"
        )
        logger.info(
            f"   Total: {total_closed_days} closed-days, "
            f"{total_half_days} half-days"
        )
        
        if unmatched_saps:
            logger.debug(
                f"   Unmatched SAP codes ({len(unmatched_saps)}): "
                f"{unmatched_saps[:10]}..."
            )
        
        return result
    
    # ==========================================
    # OPEN/CLOSE STATUS (Permanent closure trigger)
    # ==========================================
    
    @staticmethod
    def load_open_close_status(open_close_file, df_info):
        """
        Load trạng thái Open/Close từ file Open_Close.xlsx — sheet 'Master'.

        Cấu trúc file (header tại row 5 = index 4):
            Col 'Code Budget'      → SAP code đầy đủ (VD: 30KK4298)
            Col 'Code Report (New)'→ Code mới (giống Code Budget nếu không đổi)
            Col 'Status'           → ACTIVE | CLOSED | (các trạng thái khác)
            Col 'Closing date'     → Ngày đóng cửa (datetime hoặc NaT)
            Col 'Re-Opening Date'  → Ngày mở lại (datetime hoặc NaT)
            Col 'Opening date'     → Ngày khai trương

        Logic mới (v2):
            STATUS = ACTIVE  → Nhà hàng đang hoạt động.
                               Nếu có Closing date + Re-Opening Date → đã từng đóng
                               tạm thời. Pipeline sẽ loại dữ liệu trong khoảng thời
                               gian đóng cửa (closure_windows) ra khỏi training data
                               để không làm sai tỷ lệ forecast.

            STATUS = CLOSED  → Nhà hàng đã đóng cửa vĩnh viễn (hoặc chưa mở lại).
                               Nếu Re-Opening Date > today → chưa đến ngày mở, vẫn
                               coi là CLOSED, đặt future_reopen_date.
                               Nếu Re-Opening Date <= today → đã mở lại nhưng Status
                               chưa cập nhật, coi là ACTIVE (re-opened).

        Returns:
            Dict:
              closed_restaurants   → {res_code: {sap_code, code_report, closing_date,
                                                  future_reopen_date, status}}
              reopened_restaurants → {res_code: {sap_code, code_report, reopen_date}}
              active_with_closure  → {res_code: {closure_windows: [(start, end), ...]}}
              future_open_restaurants → {key: {opening_date, ...}}
              all_statuses         → {res_code: {current_status, closing_date,
                                                  reopen_date, closure_windows}}
        """
        import os as _os
        from forecast_system.config.settings import CURRENT_DATE

        result = {
            'closed_restaurants':    {},
            'reopened_restaurants':  {},
            'active_with_closure':   {},   # ← NEW: temp-closed active restaurants
            'future_open_restaurants': {},
            'all_statuses':          {},
        }

        if not open_close_file or not _os.path.exists(open_close_file):
            logger.warning(f"Open/Close file not found: {open_close_file}")
            return result

        # ── Build SAP → Restaurant_Code mapping ──
        sap_to_res = {}
        if not df_info.empty and 'sap_code' in df_info.columns:
            for _, row in df_info.iterrows():
                sap_str  = str(row['sap_code']).strip()
                res_code = str(row.get('Restaurant_Code', '')).strip()
                if res_code.endswith('.0'):
                    res_code = res_code[:-2]
                if sap_str and res_code and res_code != 'nan':
                    sap_to_res[sap_str] = res_code

        # ── Read Master sheet. The business file has changed layout more
        # than once, so scan the first rows and select the header row that
        # actually contains Status + Code Budget instead of assuming row 5.
        try:
            raw_master = pd.read_excel(
                open_close_file,
                sheet_name='Master',
                header=None,
                dtype=str,
            )

            def _norm_header_cell(v):
                return str(v).strip().lower() if pd.notna(v) else ''

            header_idx = None
            for idx in range(min(15, len(raw_master))):
                row_vals = [_norm_header_cell(v) for v in raw_master.iloc[idx].tolist()]
                has_status = any(v == 'status' for v in row_vals)
                has_code = any(v in ('code budget', 'code report (new)', 'sap code') for v in row_vals)
                if has_status and has_code:
                    header_idx = idx
                    break

            if header_idx is None:
                header_idx = 0

            df_master = raw_master.iloc[header_idx + 1:].copy()
            df_master.columns = [
                str(c).strip() if pd.notna(c) else f'Unnamed: {i}'
                for i, c in enumerate(raw_master.iloc[header_idx].tolist())
            ]
            df_master.columns = [str(c).strip() for c in df_master.columns]
            df_master = df_master.dropna(how='all')
        except Exception as e:
            logger.error(f"Error reading Open/Close Master sheet: {e}")
            return result

        # ── Helper: parse date cell ──
        def _parse_date(raw):
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                return None
            if isinstance(raw, (datetime.date, datetime.datetime)):
                return raw.date() if isinstance(raw, datetime.datetime) else raw
            s = str(raw).strip()
            if s in ('', 'nan', 'NaT', 'None'):
                return None
            # Excel often gives 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD'
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
                try:
                    return datetime.datetime.strptime(s[:19], fmt).date()
                except ValueError:
                    continue
            return None

        # ── Column name aliases (case-insensitive, stripped) ──
        def _canon_col(s):
            return ''.join(ch for ch in str(s).lower().strip() if ch.isalnum())

        col_map = {c.lower().strip(): c for c in df_master.columns}
        canon_col_map = {_canon_col(c): c for c in df_master.columns}

        def _col(name):
            """Get actual column name by case-insensitive key."""
            direct = col_map.get(name.lower().strip())
            if direct:
                return direct
            return canon_col_map.get(_canon_col(name))

        status_col       = _col('status')
        code_budget_col  = _col('code budget') or _col('sap code') or _col('code')
        code_report_col  = _col('code report (new)')
        closing_col      = _col('closing date')
        reopen_col       = _col('re-opening date')
        opening_col      = _col('opening date')
        store_col        = _col('store')
        brand_col        = _col('br')

        if not status_col or not code_budget_col:
            logger.error(
                f"Open/Close Master sheet missing required columns. "
                f"Found: {df_master.columns.tolist()[:10]}"
            )
            return result

        closed_count  = 0
        reopened_count = 0
        temp_closed_count = 0
        future_count  = 0
        matched_count = 0
        total_rows    = 0

        for _, row in df_master.iterrows():
            code_budget = str(row.get(code_budget_col, '')).strip()
            if not code_budget or code_budget in ('nan', ''):
                continue

            status_raw = str(row.get(status_col, '')).strip().upper()
            # Only process ACTIVE / CLOSED; skip 'Not yet open', 'Double code', etc.
            if status_raw not in ('ACTIVE', 'CLOSED'):
                if status_raw in ('NOT YET OPEN', 'WAITING FOR OPEN'):
                    # Track as future-open
                    opening_date = _parse_date(row.get(opening_col))
                    if opening_date and opening_date > CURRENT_DATE:
                        sap_code = code_budget[4:] if len(code_budget) > 4 and code_budget[2:4].isalpha() else code_budget[-4:]
                        res_code = sap_to_res.get(code_budget) or sap_to_res.get(sap_code)
                        future_key = res_code or code_budget
                        result['future_open_restaurants'][future_key] = {
                            'code_report': str(row.get(code_report_col, code_budget)).strip(),
                            'sap_code': sap_code,
                            'opening_date': opening_date,
                            'store_name': str(row.get(store_col, '')).strip() if store_col else '',
                            'brand': str(row.get(brand_col, '')).strip() if brand_col else '',
                            'res_code': res_code,
                        }
                        future_count += 1
                continue

            total_rows += 1
            closing_date  = _parse_date(row.get(closing_col)  if closing_col  else None)
            reopen_date   = _parse_date(row.get(reopen_col)   if reopen_col   else None)
            opening_date  = _parse_date(row.get(opening_col)  if opening_col  else None)
            store_name    = str(row.get(store_col, '')).strip()  if store_col  else ''
            brand_name    = str(row.get(brand_col, '')).strip()  if brand_col  else ''
            code_report   = str(row.get(code_report_col, code_budget)).strip() if code_report_col else code_budget
            sap_code      = code_budget[4:] if len(code_budget) > 4 and code_budget[2:4].isalpha() else code_budget[-4:]

            # ── Map to internal res_code ──
            res_code = (
                sap_to_res.get(code_budget) or
                sap_to_res.get(code_report) or
                sap_to_res.get(sap_code)
            )
            # Fuzzy: match by stripping leading zeros to avoid false suffix match (e.g. 474101 vs 4101)
            if not res_code:
                for sap_key, res_val in sap_to_res.items():
                    if sap_key.strip().lstrip('0') == sap_code.strip().lstrip('0'):
                        res_code = res_val
                        break

            # Future opening (not yet active)
            if opening_date and opening_date > CURRENT_DATE:
                future_key = res_code or code_budget
                result['future_open_restaurants'][future_key] = {
                    'code_report': code_report,
                    'sap_code': sap_code,
                    'opening_date': opening_date,
                    'store_name': store_name,
                    'brand': brand_name,
                    'res_code': res_code,
                }
                future_count += 1
                if not res_code:
                    continue

            if not res_code:
                continue

            matched_count += 1

            # ═══════════════════════════════════════════════
            # CASE 1: STATUS = ACTIVE
            #   → Nhà hàng đang hoạt động.
            #   → Nếu có Closing + Reopen date → đã đóng tạm thời rồi mở lại.
            #     Pipeline cần loại data trong khoảng [closing_date, reopen_date)
            #     ra khỏi training để tránh làm sai accuracy model.
            # ═══════════════════════════════════════════════
            if status_raw == 'ACTIVE':
                closure_windows = []

                if closing_date:
                    end_of_closure = reopen_date or CURRENT_DATE
                    if closing_date < end_of_closure:
                        closure_windows.append((closing_date, end_of_closure))

                base_info = {
                    'current_status': 'ACTIVE',
                    'closing_date':   closing_date,
                    'reopen_date':    reopen_date,
                    'closure_windows': closure_windows,
                }
                result['all_statuses'][res_code] = base_info

                if closure_windows:
                    result['active_with_closure'][res_code] = {
                        'sap_code':       sap_code,
                        'code_report':    code_report,
                        'closure_windows': closure_windows,
                        'store_name':     store_name,
                    }
                    # If reopen already happened → track as reopened
                    if reopen_date and reopen_date <= CURRENT_DATE:
                        result['reopened_restaurants'][res_code] = {
                            'sap_code':    sap_code,
                            'code_report': code_report,
                            'reopen_date': reopen_date,
                            'status':      'ACTIVE',
                        }
                        reopened_count += 1
                    temp_closed_count += 1

            # ═══════════════════════════════════════════════
            # CASE 2: STATUS = CLOSED
            #   → Nhà hàng đã đóng cửa.
            #   → Nếu Re-Opening Date đã qua → thực ra đã mở lại (data chưa update),
            #     coi là ACTIVE + add vào reopened_restaurants.
            #   → Nếu Re-Opening Date trong tương lai → coi là CLOSED nhưng sẽ
            #     mở lại sau (future_reopen_date).
            #   → Nếu không có Re-Opening Date → đóng cửa vĩnh viễn.
            # ═══════════════════════════════════════════════
            elif status_raw == 'CLOSED':
                future_reopen = None

                # Check if effectively re-opened (reopen_date in the past)
                if reopen_date and reopen_date <= CURRENT_DATE:
                    # Status not yet updated in file, but restaurant is open
                    result['reopened_restaurants'][res_code] = {
                        'sap_code':    sap_code,
                        'code_report': code_report,
                        'reopen_date': reopen_date,
                        'status':      'ACTIVE (re-opened, file not updated)',
                    }
                    result['all_statuses'][res_code] = {
                        'current_status': 'ACTIVE',
                        'closing_date':   closing_date,
                        'reopen_date':    reopen_date,
                        'closure_windows': [(closing_date, reopen_date)] if closing_date else [],
                    }
                    # Add closure window so training data excludes closed period
                    if closing_date and closing_date < reopen_date:
                        result['active_with_closure'][res_code] = {
                            'sap_code':       sap_code,
                            'code_report':    code_report,
                            'closure_windows': [(closing_date, reopen_date)],
                            'store_name':     store_name,
                        }
                    reopened_count += 1

                else:
                    # Truly closed (or future reopen)
                    if reopen_date and reopen_date > CURRENT_DATE:
                        future_reopen = reopen_date

                    result['closed_restaurants'][res_code] = {
                        'sap_code':          sap_code,
                        'code_report':       code_report,
                        'closing_date':      closing_date,
                        'future_reopen_date': future_reopen,
                        'status':            'CLOSED',
                    }
                    result['all_statuses'][res_code] = {
                        'current_status':    'CLOSED',
                        'closing_date':      closing_date,
                        'reopen_date':       future_reopen,
                        'closure_windows':   [],
                    }
                    closed_count += 1

        # ── Summary logging ──
        logger.info(
            f"📋 Open/Close Status (Master sheet): "
            f"{total_rows} ACTIVE/CLOSED rows, {matched_count} mapped to pipeline"
        )
        logger.info(
            f"   🔴 CLOSED (no forecast): {closed_count} restaurants"
        )
        if temp_closed_count > 0:
            logger.info(
                f"   🟡 ACTIVE with temp-closure window: {temp_closed_count} restaurants "
                f"(training data will exclude closure period)"
            )
        if reopened_count > 0:
            logger.info(
                f"   🟢 Re-Opened: {reopened_count} restaurants (forecasting normally)"
            )
        if future_count > 0:
            logger.info(
                f"   🆕 Future opening: {future_count} restaurants"
            )

        # Log closed list
        for res, info in result['closed_restaurants'].items():
            fr = info.get('future_reopen_date')
            fr_str = f" → reopens {fr}" if fr else ""
            logger.info(
                f"     ❌ {res} (SAP: {info['sap_code']}) "
                f"— CLOSED since {info.get('closing_date', 'N/A')}{fr_str}"
            )

        # Log temp-closed windows
        for res, info in result['active_with_closure'].items():
            for (s, e) in info['closure_windows']:
                logger.info(
                    f"     🟡 {res} (SAP: {info['sap_code']}) "
                    f"— exclude training data {s} → {e}"
                )

        return result

