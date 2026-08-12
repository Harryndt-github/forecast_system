"""
==============================================
BOOKING AGENT
==============================================
Trách nhiệm:
- Load dữ liệu booking từ bảng v_fact_db_booking_booking_info
- Loại trừ booking đã cancel (cancelled_reason có giá trị)
- Phân bổ khách booking theo nhà hàng, ngày, giờ bắt đầu
- Tổng hợp thành DataFrame sẵn sàng merge vào Master file

Logic:
    - restaurant: mã nhà hàng
    - total_guest: tổng số khách đặt trước
    - shift_date: ngày khách sẽ tới
    - Start (start_time): giờ bắt đầu (để sắp xếp khuch giờ)
    - cancelled_reason: nếu có giá trị → đã cancel → loại trừ
"""

import datetime
import pandas as pd
import numpy as np
import traceback

from forecast_system.utils.logger import get_logger
from forecast_system.utils.db_utils import (
    fetch_with_chunks,
    execute_parameterized_query,
    QUERY_BOOKING,
)
from forecast_system.agents.data_agent import DataAgent

logger = get_logger('booking_agent')


class BookingAgent:
    """
    Agent chuyên xử lý dữ liệu booking (đặt bàn trước).
    
    Lấy dữ liệu từ DB → lọc cancel → tổng hợp theo nhà hàng / ngày / giờ.
    Kết quả được ghi vào sheet riêng "Booking_Guests" trong Master file.
    """
    
    @staticmethod
    def load_booking_data(engine, start_date=None, end_date=None):
        """
        Load dữ liệu booking từ DB cho khoảng thời gian tương lai.
        
        Args:
            engine: SQLAlchemy engine
            start_date: Ngày bắt đầu (default: hôm nay)
            end_date: Ngày kết thúc (default: hôm nay + 90 ngày)
        
        Returns:
            pd.DataFrame raw booking data (đã loại cancel)
        """
        if start_date is None:
            start_date = datetime.date.today()
        if end_date is None:
            end_date = start_date + datetime.timedelta(days=90)
        
        params = {
            'start_date': str(start_date),
            'end_date': str(end_date),
        }
        
        logger.info(f"📅 Loading booking data: {start_date} → {end_date}")
        
        try:
            df = fetch_with_chunks(
                engine,
                QUERY_BOOKING,
                params=params,
                chunksize=50000,
                name="booking_data",
            )
            
            if df.empty:
                logger.warning("No booking data found in date range")
                return pd.DataFrame()
            
            logger.info(f"   Raw booking records: {len(df):,}")
            
            # ── STEP 1: Loại trừ booking đã cancel ──
            # Nếu cancelled_reason có giá trị (not null, not empty) → đã cancel
            before_cancel = len(df)
            
            if 'cancelled_reason' in df.columns:
                # Giữ lại rows mà cancelled_reason là NULL hoặc rỗng
                cancelled_mask = (
                    df['cancelled_reason'].notna() & 
                    (df['cancelled_reason'].astype(str).str.strip() != '') &
                    (df['cancelled_reason'].astype(str).str.lower() != 'nan')
                )
                df_cancelled = df[cancelled_mask]
                df = df[~cancelled_mask].copy()
                
                n_cancelled = len(df_cancelled)
                cancelled_guests = df_cancelled['total_guest'].sum() if not df_cancelled.empty else 0
                logger.info(
                    f"   ❌ Cancelled bookings removed: {n_cancelled:,} "
                    f"({cancelled_guests:,.0f} guests)"
                )
            
            logger.info(f"   ✅ Active bookings: {len(df):,}")
            
            # ── STEP 2: Clean & normalize ──
            df = BookingAgent._clean_booking_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load booking data: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    @staticmethod
    def _clean_booking_data(df):
        """
        Clean và normalize booking data.
        
        - Normalize restaurant code
        - Parse shift_date → date
        - Extract hour từ start_time
        - Ensure total_guest là numeric
        """
        if df.empty:
            return df
        
        # Normalize restaurant code (cùng logic với DataAgent)
        if 'restaurant' in df.columns:
            df['restaurant_code'] = DataAgent.normalize_key(
                df['restaurant'].astype(str)
            )
        else:
            logger.warning("Column 'restaurant' not found in booking data")
            return pd.DataFrame()
        
        # Parse shift_date
        df['shift_date'] = pd.to_datetime(df['shift_date'], errors='coerce')
        df = df.dropna(subset=['shift_date'])
        df['date'] = df['shift_date'].dt.date
        df['weekday'] = df['shift_date'].dt.day_name()
        
        # Parse start_time → extract hour
        df['hour'] = BookingAgent._extract_hour(df['start_time'])
        
        # Ensure total_guest is numeric
        df['total_guest'] = pd.to_numeric(df['total_guest'], errors='coerce').fillna(0)  # type: ignore[reportAttributeAccessIssue]
        df['total_guest'] = df['total_guest'].astype(int)
        
        # Drop records with 0 guests
        df = df[df['total_guest'] > 0].copy()
        
        return df
    
    @staticmethod
    def _extract_hour(series):
        """
        Extract hour từ cột start_time.
        
        Hỗ trợ nhiều format:
        - datetime/timestamp: lấy .hour
        - string "HH:MM", "HH:MM:SS": parse hour
        - numeric: lấy giá trị int
        """
        def parse_hour(val):
            if pd.isna(val):
                return None
            
            # Nếu là datetime/timestamp
            if hasattr(val, 'hour'):
                return val.hour
            
            # Nếu là string
            val_str = str(val).strip()
            
            # Format "HH:MM" hoặc "HH:MM:SS"
            if ':' in val_str:
                try:
                    parts = val_str.split(':')
                    h = int(parts[0])
                    if 0 <= h <= 23:
                        return h
                except (ValueError, IndexError):
                    pass
            
            # Format numeric (e.g., 11.0, 18)
            try:
                h = int(float(val_str))
                if 0 <= h <= 23:
                    return h
            except (ValueError, TypeError):
                pass
            
            return None
        
        return series.apply(parse_hour)
    
    @staticmethod
    def aggregate_booking_summary(df_booking, df_info=None):
        """
        Tổng hợp booking data thành bảng summary.
        
        Output columns:
        - Restaurant_Code: mã nhà hàng
        - sap_code, restaurant_name (nếu có df_info)
        - Date: ngày khách sẽ tới
        - Weekday: thứ trong tuần
        - Hour: giờ bắt đầu (khung giờ)
        - Booking_Guests: tổng số khách đặt trước
        - Booking_Count: số lượt booking
        
        Args:
            df_booking: DataFrame từ load_booking_data()
            df_info: DataFrame thông tin nhà hàng (optional)
        
        Returns:
            pd.DataFrame summary
        """
        if df_booking.empty:
            logger.warning("No booking data to aggregate")
            return pd.DataFrame()
        
        # ── Aggregate theo Restaurant + Date + Hour ──
        # Group by restaurant, date, hour → sum guests, count bookings
        agg_hourly = df_booking.groupby(
            ['restaurant_code', 'date', 'weekday', 'hour'],
            dropna=False
        ).agg(
            Booking_Guests=('total_guest', 'sum'),
            Booking_Count=('total_guest', 'count'),
        ).reset_index()
        
        # Rename để consistent với Master file
        agg_hourly.rename(columns={
            'restaurant_code': 'Restaurant_Code',
            'date': 'Date',
            'weekday': 'Weekday',
            'hour': 'Hour',
        }, inplace=True)
        
        # Sort by date, hour
        agg_hourly = agg_hourly.sort_values(
            ['Restaurant_Code', 'Date', 'Hour'],
            na_position='last'
        ).reset_index(drop=True)
        
        # ── Tổng hợp daily summary (không theo giờ) ──
        agg_daily = df_booking.groupby(
            ['restaurant_code', 'date', 'weekday'],
        ).agg(
            Booking_Guests_Daily=('total_guest', 'sum'),
            Booking_Count_Daily=('total_guest', 'count'),
        ).reset_index()
        
        agg_daily.rename(columns={
            'restaurant_code': 'Restaurant_Code',
            'date': 'Date',
            'weekday': 'Weekday',
        }, inplace=True)
        
        # Merge daily total vào hourly
        agg_hourly = pd.merge(
            agg_hourly,
            agg_daily[['Restaurant_Code', 'Date', 'Booking_Guests_Daily', 'Booking_Count_Daily']],
            on=['Restaurant_Code', 'Date'],
            how='left',
        )
        
        # ── Merge restaurant info ──
        if df_info is not None and not df_info.empty:
            agg_hourly['merge_key'] = DataAgent.normalize_key(
                agg_hourly['Restaurant_Code']
            )
            agg_hourly = pd.merge(
                agg_hourly,
                df_info[['merge_key', 'sap_code', 'restaurant_name']],
                on='merge_key',
                how='left',
            )
            agg_hourly.drop(columns=['merge_key'], inplace=True)
        
        # ── Reorder columns ──
        col_order = [
            'Restaurant_Code', 'sap_code', 'restaurant_name',
            'Date', 'Weekday', 'Hour',
            'Booking_Guests', 'Booking_Count',
            'Booking_Guests_Daily', 'Booking_Count_Daily',
        ]
        col_order = [c for c in col_order if c in agg_hourly.columns]
        agg_hourly = agg_hourly[col_order]
        
        logger.info(
            f"📊 Booking Summary: {len(agg_hourly):,} rows | "
            f"{agg_hourly['Restaurant_Code'].nunique()} restaurants | "  # type: ignore[reportAttributeAccessIssue]
            f"{agg_hourly['Date'].nunique()} dates | "  # type: ignore[reportAttributeAccessIssue]
            f"Total guests: {agg_hourly['Booking_Guests'].sum():,.0f}"
        )
        
        return agg_hourly
    
    @staticmethod
    def get_daily_booking_totals(df_booking):
        """
        Tạo bảng tóm tắt daily totals (không phân theo giờ).
        Dùng để merge nhanh vào Master Forecast file.
        
        Returns:
            pd.DataFrame: Restaurant_Code, Date, Booking_Guests_Total
        """
        if df_booking.empty:
            return pd.DataFrame()
        
        daily = df_booking.groupby(
            ['restaurant_code', 'date']
        ).agg(
            Booking_Guests_Total=('total_guest', 'sum'),
        ).reset_index()
        
        daily.rename(columns={
            'restaurant_code': 'Restaurant_Code',
            'date': 'Date',
        }, inplace=True)
        
        return daily
    
    @staticmethod
    def print_booking_summary(df_summary, logger_func=None):
        """
        In tóm tắt booking đẹp cho pipeline log.
        """
        if logger_func is None:
            logger_func = logger.info
        
        if df_summary.empty:
            logger_func("   No booking data available")
            return
        
        total_guests = df_summary['Booking_Guests'].sum()
        total_bookings = df_summary['Booking_Count'].sum()
        n_restaurants = df_summary['Restaurant_Code'].nunique()
        n_dates = df_summary['Date'].nunique()
        min_date = df_summary['Date'].min()
        max_date = df_summary['Date'].max()
        
        logger_func(f"   📅 Booking Period: {min_date} → {max_date}")
        logger_func(f"   🏪 Restaurants with bookings: {n_restaurants}")
        logger_func(f"   📆 Dates with bookings: {n_dates}")
        logger_func(f"   👥 Total booked guests: {total_guests:,.0f}")
        logger_func(f"   🎫 Total booking count: {total_bookings:,.0f}")
        
        # Top 5 restaurants by booking guests
        top_res = (
            df_summary.groupby('Restaurant_Code')['Booking_Guests']
            .sum()
            .nlargest(5)
        )
        if not top_res.empty:
            logger_func(f"\n   🏆 Top 5 Restaurants by Booking Guests:")
            for res, guests in top_res.items():
                # Get restaurant name if available
                name = ''
                if 'restaurant_name' in df_summary.columns:
                    name_rows = df_summary[
                        df_summary['Restaurant_Code'] == res
                    ]['restaurant_name'].dropna()
                    if not name_rows.empty:
                        name = f" ({name_rows.iloc[0]})"
                logger_func(f"      {res}{name}: {guests:,.0f} guests")
        
        # Top 5 dates by booking guests
        top_dates = (
            df_summary.groupby('Date')['Booking_Guests']
            .sum()
            .nlargest(5)
        )
        if not top_dates.empty:
            logger_func(f"\n   📆 Top 5 Dates by Booking Guests:")
            for dt, guests in top_dates.items():
                weekday = ''
                wd_rows = df_summary[df_summary['Date'] == dt]['Weekday'].dropna()
                if not wd_rows.empty:
                    weekday = f" ({wd_rows.iloc[0][:3]})"
                logger_func(f"      {dt}{weekday}: {guests:,.0f} guests")
