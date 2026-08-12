"""
==============================================
ISHUSHI DATA AGENT
==============================================
Load dữ liệu từ 2 bảng:
- v_fact_db_rk_dc_transactions      → Số khách (guest_count)
- v_fact_db_rk_dc_transaction_details → Chi tiết món ăn (sap_code, Quantity)

Join bằng transaction_id.
Lọc theo danh mục SAP codes Ishushi.
"""

import pandas as pd
import numpy as np
import datetime
import traceback

from forecast_system.utils.db_utils import (
    fetch_with_chunks, execute_parameterized_query
)
from forecast_system.utils.logger import get_logger
from forecast_system.ishushi.config import (
    ISHUSHI_SAP_CODES, ISHUSHI_SAP_CATALOG, ISHUSHI_SAP_GROUPS, ISHUSHI_CONFIG,
    ISHUSHI_HOUR_TO_SHIFT
)

logger = get_logger('ishushi_data_agent')


# ==========================================
# SQL QUERIES
# ==========================================

# Bảng 1: Lấy thông tin giao dịch (khách) - CÓ FILTER restaurant_code
# {restaurant_codes_placeholder} sẽ được format() trước khi chạy
QUERY_TRANSACTIONS_FILTERED = """
    SELECT 
        restaurant_code,
        transaction_id,
        guest_count,
        open_time,
        shiftdate as date
    FROM v_fact_db_rk_dc_transactions
    WHERE shiftdate >= :start_date 
      AND shiftdate < :end_date
      AND restaurant_code IN ({restaurant_codes_placeholder})
"""

# Bảng 2: Lấy chi tiết món ăn - FILTER cả restaurant_code và sap_code
QUERY_TRANSACTION_DETAILS_FILTERED = """
    SELECT 
        d.transaction_id,
        d.sap_code,
        d.Quantity as quantity
    FROM v_fact_db_rk_dc_transaction_details d
    INNER JOIN v_fact_db_rk_dc_transactions t 
        ON d.transaction_id = t.transaction_id
    WHERE t.shiftdate >= :start_date 
      AND t.shiftdate < :end_date
      AND t.restaurant_code IN ({restaurant_codes_placeholder})
      AND d.sap_code IN ({sap_codes_placeholder})
"""

# Query lấy nhà hàng Ishushi
QUERY_ISHUSHI_RESTAURANTS = """
    SELECT DISTINCT restaurant_code, restaurant_name
    FROM v_dim_restaurant_address
    WHERE restaurant_name LIKE '%ishushi%'
       OR restaurant_name LIKE '%Ishushi%'
       OR restaurant_name LIKE '%ISHUSHI%'
       OR restaurant_name LIKE '%isushi%'
       OR restaurant_name LIKE '%Isushi%'
"""


class IshushiDataAgent:
    """
    Agent load và xử lý dữ liệu cho các nhà hàng Ishushi.
    Kết hợp transactions (số khách) + transaction_details (món ăn).
    """
    
    @staticmethod
    def clean_id(series):
        """Chuẩn hóa transaction IDs"""
        return (series.astype(str)
                .str.upper()
                .str.strip()
                .str.replace(r'[{}]', '', regex=True))
    
    @staticmethod
    def normalize_key(series):
        """Chuẩn hóa restaurant_code"""
        s = series.astype(str).str.upper().str.strip()
        s = s.str.replace(r'\.0$', '', regex=True)
        s = s.str.replace(r'^0+', '', regex=True)
        return s.replace('', '0')
    
    # ==========================================
    # LOAD ISHUSHI RESTAURANTS
    # ==========================================
    
    @staticmethod
    def load_ishushi_restaurants(engine):
        """
        Tìm tất cả nhà hàng Ishushi trong hệ thống.
        
        Returns:
            set: Danh sách restaurant_code của các nhà hàng Ishushi
        """
        logger.info("🍣 Finding Ishushi restaurants...")
        
        try:
            df = execute_parameterized_query(engine, QUERY_ISHUSHI_RESTAURANTS)
            
            if df.empty:
                logger.warning("No Ishushi restaurants found via name search")
                return set()
            
            # Normalize
            df['restaurant_code'] = IshushiDataAgent.normalize_key(df['restaurant_code'])
            restaurants = set(df['restaurant_code'].unique())
            
            logger.info(f"Found {len(restaurants)} Ishushi restaurants:")
            for _, row in df.iterrows():
                logger.info(f"  - {row['restaurant_code']}: {row['restaurant_name']}")
            
            return restaurants
            
        except Exception as e:
            logger.error(f"Error finding Ishushi restaurants: {e}")
            traceback.print_exc()
            return set()
    
    # ==========================================
    # LOAD GUEST DATA (SỐ KHÁCH)
    # ==========================================
    
    @staticmethod
    def load_guest_data(engine, ishushi_codes, start_date=None, end_date=None):
        """
        Load dữ liệu khách từ v_fact_db_rk_dc_transactions.
        Filter restaurant_code trực tiếp trong SQL để tối ưu performance.
        
        Args:
            engine: SQLAlchemy engine
            ishushi_codes: Set of restaurant_code cho Ishushi
            start_date: Ngày bắt đầu (default: 2024-01-01)
            end_date: Ngày kết thúc (default: today)
            
        Returns:
            pd.DataFrame: Dữ liệu khách theo ngày cho từng nhà hàng Ishushi
        """
        if start_date is None:
            start_date = ISHUSHI_CONFIG['data_start_date']
        if end_date is None:
            end_date = datetime.date.today() + datetime.timedelta(days=1)
        
        logger.info(f"📊 Loading guest data: {start_date} → {end_date}")
        logger.info(f"   Filtering {len(ishushi_codes)} Ishushi restaurants in SQL")
        
        # Build restaurant_code IN clause (filter ngay tại SQL)
        res_codes_str = ', '.join([f"'{c}'" for c in ishushi_codes])
        query = QUERY_TRANSACTIONS_FILTERED.format(
            restaurant_codes_placeholder=res_codes_str
        )
        
        params = {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
        }
        
        try:
            df = fetch_with_chunks(
                engine, query, params=params,
                name="Ishushi Transactions"
            )
            
            if df.empty:
                logger.warning("No transaction data loaded")
                return pd.DataFrame()
            
            # Clean & normalize
            df['transaction_id'] = IshushiDataAgent.clean_id(df['transaction_id'])
            df['restaurant_code'] = IshushiDataAgent.normalize_key(df['restaurant_code'])
            df['guest_count'] = pd.to_numeric(df['guest_count'], errors='coerce').fillna(0)  # type: ignore[reportAttributeAccessIssue]
            
            # Parse datetime
            df['open_time'] = pd.to_datetime(df['open_time'], errors='coerce')
            df = df.dropna(subset=['open_time'])
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
            df['hour'] = df['open_time'].dt.hour
            df['weekday'] = df['open_time'].dt.day_name()
            
            # Deduplicate (keep highest guest_count)
            df = df.sort_values('guest_count', ascending=False)
            df = df.drop_duplicates(subset=['transaction_id'], keep='first')
            
            logger.info(f"✅ Loaded {len(df):,} Ishushi transactions")
            logger.info(f"   Restaurants: {df['restaurant_code'].nunique()}")
            logger.info(f"   Date range: {df['date'].min()} → {df['date'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading guest data: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    # ==========================================
    # LOAD ITEM DATA (MÓN ĂN THEO SAP CODE)
    # ==========================================
    
    @staticmethod
    def load_item_data(engine, ishushi_codes, start_date=None, end_date=None):
        """
        Load dữ liệu chi tiết món ăn từ v_fact_db_rk_dc_transaction_details.
        Filter cả restaurant_code VÀ sap_code trực tiếp trong SQL.
        
        Args:
            engine: SQLAlchemy engine
            ishushi_codes: Set of restaurant_code cho Ishushi
            start_date: Ngày bắt đầu
            end_date: Ngày kết thúc
            
        Returns:
            pd.DataFrame: Chi tiết món ăn (transaction_id, sap_code, quantity)
        """
        if start_date is None:
            start_date = ISHUSHI_CONFIG['data_start_date']
        if end_date is None:
            end_date = datetime.date.today() + datetime.timedelta(days=1)
        
        logger.info(f"🍱 Loading item details for {len(ISHUSHI_SAP_CODES)} SAP codes...")
        logger.info(f"   Filtering {len(ishushi_codes)} Ishushi restaurants in SQL")
        
        # Build placeholders for SQL IN clauses
        sap_codes_str = ', '.join([str(code) for code in ISHUSHI_SAP_CODES])
        res_codes_str = ', '.join([f"'{c}'" for c in ishushi_codes])
        
        # Filter cả restaurant_code + sap_code ngay tại SQL
        query = QUERY_TRANSACTION_DETAILS_FILTERED.format(
            restaurant_codes_placeholder=res_codes_str,
            sap_codes_placeholder=sap_codes_str
        )
        
        params = {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
        }
        
        try:
            df = fetch_with_chunks(
                engine, query, params=params,
                name="Ishushi Item Details"
            )
            
            if df.empty:
                logger.warning("No item detail data loaded")
                return pd.DataFrame()
            
            # Clean
            df['transaction_id'] = IshushiDataAgent.clean_id(df['transaction_id'])
            df['sap_code'] = pd.to_numeric(df['sap_code'], errors='coerce').astype('Int64')  # type: ignore[reportAttributeAccessIssue]
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)  # type: ignore[reportAttributeAccessIssue]
            
            # Add item name
            df['item_name'] = df['sap_code'].map(ISHUSHI_SAP_CATALOG)  # type: ignore[reportArgumentType]
            
            # Add group name
            sap_to_group = {}
            for group, codes in ISHUSHI_SAP_GROUPS.items():
                for code in codes:
                    sap_to_group[code] = group
            df['item_group'] = df['sap_code'].map(sap_to_group)  # type: ignore[reportArgumentType]
            
            logger.info(f"✅ Loaded {len(df):,} item records")
            logger.info(f"   Unique SAP codes found: {df['sap_code'].nunique()}")
            logger.info(f"   Total quantity: {df['quantity'].sum():,.0f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading item data: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    # ==========================================
    # MERGE & AGGREGATE
    # ==========================================
    
    @staticmethod
    def build_daily_guest_summary(df_transactions):
        """
        Tạo bảng tổng hợp số khách theo ngày cho mỗi nhà hàng Ishushi.
        
        Cách tính:
        - Đếm số transaction_id unique = số lượt khách
        - Tổng guest_count = tổng số khách
        
        Returns:
            pd.DataFrame: [restaurant_code, date, weekday, num_transactions, total_guests]
        """
        if df_transactions.empty:
            return pd.DataFrame()
        
        df = df_transactions.copy()
        
        daily = df.groupby(['restaurant_code', 'date']).agg(
            num_transactions=('transaction_id', 'nunique'),
            total_guests=('guest_count', 'sum'),
            weekday=('weekday', 'first'),
        ).reset_index()
        
        daily = daily.sort_values(['restaurant_code', 'date'])
        
        logger.info(f"📋 Daily guest summary: {len(daily)} records")
        logger.info(f"   Restaurants: {daily['restaurant_code'].nunique()}")
        
        return daily
    
    @staticmethod
    def build_daily_guest_summary_by_shift(df_transactions):
        """
        Tạo bảng tổng hợp số khách theo ngày VÀ CA LÀM VIỆC cho mỗi nhà hàng.
        
        Shift mapping:
        - MORNING (Ca Sáng): hours 8-15 (8h – 15h30)
        - EVENING (Ca Tối): hours 16-23 (15h30 – 23h)
        
        Returns:
            pd.DataFrame: [restaurant_code, date, shift, weekday, 
                          num_transactions, total_guests]
        """
        if df_transactions.empty:
            return pd.DataFrame()
        
        df = df_transactions.copy()
        
        # Map hour → shift
        df['shift'] = df['hour'].map(ISHUSHI_HOUR_TO_SHIFT)
        # Drop rows outside shift hours (e.g., before 8am)
        df = df.dropna(subset=['shift'])
        
        daily_shift = df.groupby(['restaurant_code', 'date', 'shift']).agg(
            num_transactions=('transaction_id', 'nunique'),
            total_guests=('guest_count', 'sum'),
            weekday=('weekday', 'first'),
        ).reset_index()
        
        daily_shift = daily_shift.sort_values(['restaurant_code', 'date', 'shift'])
        
        logger.info(f"📋 Daily guest summary by shift: {len(daily_shift)} records")
        logger.info(f"   Restaurants: {daily_shift['restaurant_code'].nunique()}")
        logger.info(f"   Shift distribution:")
        for shift, cnt in daily_shift.groupby('shift').size().items():
            logger.info(f"     {shift}: {cnt:,} records")
        
        return daily_shift
    
    @staticmethod
    def build_daily_item_summary(df_transactions, df_items):
        """
        Tạo bảng tổng hợp số lượng món ăn theo ngày, theo từng sap_code,
        cho mỗi nhà hàng Ishushi.
        
        Join: transactions (có date, restaurant_code) + items (có sap_code, quantity)
        Join key: transaction_id
        
        Returns:
            pd.DataFrame: [restaurant_code, date, weekday, sap_code, item_name, 
                          item_group, total_quantity]
        """
        if df_transactions.empty or df_items.empty:
            return pd.DataFrame()
        
        # Merge transactions with items on transaction_id
        df_merged = pd.merge(
            df_transactions[['transaction_id', 'restaurant_code', 'date', 'weekday']],
            df_items[['transaction_id', 'sap_code', 'quantity', 'item_name', 'item_group']],
            on='transaction_id',
            how='inner'
        )
        
        if df_merged.empty:
            logger.warning("No matching transactions found after merge")
            return pd.DataFrame()
        
        # Aggregate by restaurant + date + sap_code
        daily_items = df_merged.groupby(
            ['restaurant_code', 'date', 'weekday', 'sap_code', 'item_name', 'item_group']
        ).agg(
            total_quantity=('quantity', 'sum'),
            num_orders=('transaction_id', 'nunique'),  # Bao nhiêu đơn order món này
        ).reset_index()
        
        daily_items = daily_items.sort_values(['restaurant_code', 'date', 'sap_code'])
        
        logger.info(f"📋 Daily item summary: {len(daily_items)} records")
        logger.info(f"   Unique items: {daily_items['sap_code'].nunique()}")
        
        return daily_items
    
    @staticmethod
    def build_daily_item_summary_by_shift(df_transactions, df_items):
        """
        Tạo bảng tổng hợp số lượng món ăn theo ngày, ca làm việc, và sap_code.
        
        Returns:
            pd.DataFrame: [restaurant_code, date, shift, weekday, sap_code, 
                          item_name, item_group, total_quantity, num_orders]
        """
        if df_transactions.empty or df_items.empty:
            return pd.DataFrame()
        
        # Add shift to transactions
        df_trans = df_transactions.copy()
        df_trans['shift'] = df_trans['hour'].map(ISHUSHI_HOUR_TO_SHIFT)
        df_trans = df_trans.dropna(subset=['shift'])
        
        # Merge transactions with items
        df_merged = pd.merge(
            df_trans[['transaction_id', 'restaurant_code', 'date', 'weekday', 'shift']],
            df_items[['transaction_id', 'sap_code', 'quantity', 'item_name', 'item_group']],
            on='transaction_id',
            how='inner'
        )
        
        if df_merged.empty:
            logger.warning("No matching transactions found after merge (shift-based)")
            return pd.DataFrame()
        
        # Aggregate by restaurant + date + shift + sap_code
        daily_items = df_merged.groupby(
            ['restaurant_code', 'date', 'shift', 'weekday', 'sap_code', 'item_name', 'item_group']
        ).agg(
            total_quantity=('quantity', 'sum'),
            num_orders=('transaction_id', 'nunique'),
        ).reset_index()
        
        daily_items = daily_items.sort_values(
            ['restaurant_code', 'date', 'shift', 'sap_code']
        )
        
        logger.info(f"📋 Daily item summary by shift: {len(daily_items)} records")
        logger.info(f"   Unique items: {daily_items['sap_code'].nunique()}")
        
        return daily_items
    
    @staticmethod
    def build_daily_group_summary(df_daily_items):
        """
        Tổng hợp theo nhóm món ăn (item_group) thay vì từng sap_code.
        VD: Tất cả "California tôm tempura" variants gộp lại.
        
        Returns:
            pd.DataFrame: [restaurant_code, date, weekday, item_group, 
                          total_quantity, num_orders]
        """
        if df_daily_items.empty:
            return pd.DataFrame()
        
        group_summary = df_daily_items.groupby(
            ['restaurant_code', 'date', 'weekday', 'item_group']
        ).agg(
            total_quantity=('total_quantity', 'sum'),
            num_orders=('num_orders', 'sum'),
        ).reset_index()
        
        return group_summary.sort_values(['restaurant_code', 'date', 'item_group'])
    
    # ==========================================
    # FEATURE ENGINEERING
    # ==========================================
    
    @staticmethod
    def add_time_features(df):
        """
        Thêm các features thời gian cho model.
        ⭐ v4: Thêm is_friday, is_saturday (thay hardcoded weekend multiplier)
        
        Args:
            df: DataFrame có cột 'date'
            
        Returns:
            DataFrame với thêm các cột features
        """
        df = df.copy()
        df['date_dt'] = pd.to_datetime(df['date'])
        
        # Calendar features
        df['day_of_week'] = df['date_dt'].dt.dayofweek      # 0=Mon, 6=Sun
        df['day_of_month'] = df['date_dt'].dt.day
        df['month'] = df['date_dt'].dt.month
        df['year'] = df['date_dt'].dt.year
        df['week_of_year'] = df['date_dt'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['quarter'] = df['date_dt'].dt.quarter
        
        # ⭐ v4: Explicit day-of-week flags (fix weekend bias)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_saturday'] = (df['day_of_week'] == 5).astype(int)
        df['is_sunday'] = (df['day_of_week'] == 6).astype(int)
        
        # Cyclical encoding of day_of_week and month
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Is beginning/end of month
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        
        # Days since start
        min_date = df['date_dt'].min()
        df['days_since_start'] = (df['date_dt'] - min_date).dt.days
        
        return df
    
    @staticmethod
    def add_lag_features(df, target_col, lag_days=None, group_cols=None):
        """
        Thêm lag features cho target column.
        
        Args:
            df: DataFrame sorted by date
            target_col: Cột cần tạo lag (VD: 'total_guests' hoặc 'total_quantity')
            lag_days: List of lag days [1, 7, 14, 28]
            group_cols: Columns to group by (VD: ['restaurant_code'] hoặc 
                       ['restaurant_code', 'sap_code'])
        """
        if lag_days is None:
            lag_days = ISHUSHI_CONFIG.get('lag_days', [1, 7, 14, 28])
        if group_cols is None:
            group_cols = ['restaurant_code']
        
        df = df.copy()
        df = df.sort_values(group_cols + ['date_dt'])
        
        for lag in lag_days:
            col_name = f'{target_col}_lag{lag}'
            df[col_name] = df.groupby(group_cols)[target_col].shift(lag)
        
        return df
    
    @staticmethod
    def add_rolling_features(df, target_col, windows=None, group_cols=None):
        """
        Thêm rolling average features.
        
        Args:
            df: DataFrame sorted by date
            target_col: Target column
            windows: List of rolling window sizes [7, 14, 28]
            group_cols: Group columns
        """
        if windows is None:
            windows = ISHUSHI_CONFIG.get('rolling_windows', [7, 14, 28])
        if group_cols is None:
            group_cols = ['restaurant_code']
        
        df = df.copy()
        df = df.sort_values(group_cols + ['date_dt'])
        
        for w in windows:
            col_mean = f'{target_col}_rolling_mean_{w}'
            col_std = f'{target_col}_rolling_std_{w}'
            
            df[col_mean] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[col_std] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.rolling(w, min_periods=1).std()
            )
        
        return df
    
    @staticmethod
    def add_yoy_features(df, target_col, group_cols=None):
        """
        Thêm Year-over-Year features (so sánh cùng kỳ năm trước).
        Dùng 364-day lag (nearest same weekday).
        """
        if group_cols is None:
            group_cols = ['restaurant_code']
        
        df = df.copy()
        df = df.sort_values(group_cols + ['date_dt'])
        
        # YoY lag (364 days = 52 weeks = same day-of-week)
        df[f'{target_col}_yoy'] = df.groupby(group_cols)[target_col].shift(364)
        
        # YoY growth rate
        df[f'{target_col}_yoy_growth'] = (
            (df[target_col] - df[f'{target_col}_yoy']) / 
            df[f'{target_col}_yoy'].replace(0, np.nan)
        )
        
        return df
    
    @staticmethod
    def add_trend_features(df, target_col, group_cols=None):
        """
        ⭐ v4: Thêm trend features.
        - trend_7d = rolling_mean_7d / rolling_mean_14d (>1 = uptrend)
        - momentum = rolling_mean_3d / rolling_mean_7d (>1 = accelerating)
        
        Requires rolling features đã được tính trước.
        """
        if group_cols is None:
            group_cols = ['restaurant_code']
        
        df = df.copy()
        df = df.sort_values(group_cols + ['date_dt'])
        
        # Tính rolling 3d nếu chưa có
        col_3d = f'{target_col}_rolling_mean_3'
        if col_3d not in df.columns:
            df[col_3d] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
        
        # Trend: 7d / 14d
        col_7d = f'{target_col}_rolling_mean_7'
        col_14d = f'{target_col}_rolling_mean_14'
        
        if col_7d in df.columns and col_14d in df.columns:
            df[f'{target_col}_trend_7d'] = (
                df[col_7d] / df[col_14d].replace(0, np.nan)
            ).fillna(1.0)
        
        # Momentum: 3d / 7d
        if col_3d in df.columns and col_7d in df.columns:
            df[f'{target_col}_momentum'] = (
                df[col_3d] / df[col_7d].replace(0, np.nan)
            ).fillna(1.0)
        
        return df
    
    @staticmethod
    def add_booking_features(df, df_bookings=None, group_cols=None):
        """
        ⭐ v4: Inject booking data as features.
        - booking_count: số khách đặt trước
        - booking_ratio: booking / avg_daily_guests  
        - booking_flag: 1 if booking_ratio > threshold
        
        Args:
            df: Main DataFrame with (restaurant_code, date, total_guests)
            df_bookings: Booking DataFrame with (restaurant_code, date, booking_guests)
            group_cols: Group columns
        """
        if group_cols is None:
            group_cols = ['restaurant_code']
        
        df = df.copy()
        threshold = ISHUSHI_CONFIG.get('booking_threshold_ratio', 0.3)
        
        if df_bookings is not None and not df_bookings.empty:
            # Merge booking data
            book_cols = ['restaurant_code', 'date', 'booking_guests']
            avail_cols = [c for c in book_cols if c in df_bookings.columns]
            
            if 'booking_guests' in df_bookings.columns:
                book_agg = df_bookings.groupby(
                    ['restaurant_code', 'date']
                )['booking_guests'].sum().reset_index()
                
                book_agg['date'] = pd.to_datetime(book_agg['date']).dt.date
                df['date_merge'] = pd.to_datetime(df['date']).dt.date if df['date'].dtype != 'object' else df['date']
                
                df = pd.merge(
                    df, 
                    book_agg.rename(columns={'booking_guests': 'booking_count'}),
                    left_on=['restaurant_code', 'date_merge'],
                    right_on=['restaurant_code', 'date'],
                    how='left',
                    suffixes=('', '_book'),
                )
                df.drop(columns=['date_merge', 'date_book'], errors='ignore', inplace=True)
            else:
                df['booking_count'] = 0
        else:
            df['booking_count'] = 0
        
        df['booking_count'] = df['booking_count'].fillna(0)
        
        # Compute avg daily guests per restaurant
        target_col = 'total_guests' if 'total_guests' in df.columns else None
        if target_col:
            avg_daily = df.groupby('restaurant_code')[target_col].transform('mean')
            avg_daily = avg_daily.replace(0, 1)  # avoid division by zero
            df['booking_ratio'] = df['booking_count'] / avg_daily
            df['booking_flag'] = (df['booking_ratio'] > threshold).astype(int)
        else:
            df['booking_ratio'] = 0.0
            df['booking_flag'] = 0
        
        return df
    
    # ==========================================
    # FULL PIPELINE
    # ==========================================
    
    @staticmethod
    def load_all_data(engine):
        """
        Pipeline đầy đủ: Load tất cả dữ liệu Ishushi.
        
        Returns:
            dict: {
                'ishushi_restaurants': set,
                'df_transactions': pd.DataFrame,
                'df_items': pd.DataFrame,
                'df_daily_guests': pd.DataFrame,
                'df_daily_items': pd.DataFrame,
                'df_daily_groups': pd.DataFrame,
            }
        """
        result = {
            'ishushi_restaurants': set(),
            'df_transactions': pd.DataFrame(),
            'df_items': pd.DataFrame(),
            'df_daily_guests': pd.DataFrame(),
            'df_daily_items': pd.DataFrame(),
            'df_daily_groups': pd.DataFrame(),
        }
        
        # Step 1: Tìm nhà hàng Ishushi
        ishushi_codes = IshushiDataAgent.load_ishushi_restaurants(engine)
        if not ishushi_codes:
            logger.error("No Ishushi restaurants found!")
            return result
        result['ishushi_restaurants'] = ishushi_codes
        
        # Step 2: Load transactions (khách)
        df_trans = IshushiDataAgent.load_guest_data(engine, ishushi_codes)
        if df_trans.empty:
            logger.error("No transaction data for Ishushi!")
            return result
        result['df_transactions'] = df_trans
        
        # Step 3: Load item details (món ăn)
        df_items = IshushiDataAgent.load_item_data(engine, ishushi_codes)
        result['df_items'] = df_items
        
        # Step 4: Build daily summaries
        df_daily_guests = IshushiDataAgent.build_daily_guest_summary(df_trans)
        result['df_daily_guests'] = df_daily_guests
        
        if not df_items.empty:
            df_daily_items = IshushiDataAgent.build_daily_item_summary(df_trans, df_items)
            result['df_daily_items'] = df_daily_items
            
            df_daily_groups = IshushiDataAgent.build_daily_group_summary(df_daily_items)
            result['df_daily_groups'] = df_daily_groups
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info(f"🍣 ISHUSHI DATA SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"   Restaurants: {len(ishushi_codes)}")
        logger.info(f"   Transactions: {len(df_trans):,}")
        logger.info(f"   Item records: {len(df_items):,}")
        logger.info(f"   Daily guest records: {len(df_daily_guests):,}")
        logger.info(f"   Daily item records: {len(result.get('df_daily_items', pd.DataFrame())):,}")
        logger.info(f"{'='*50}")
        
        return result
