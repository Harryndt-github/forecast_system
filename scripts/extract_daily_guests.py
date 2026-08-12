"""
==============================================
EXTRACT DAILY GUEST DATA FOR MACRO ANALYSIS
==============================================
Truy xuất tổng lượt khách theo ngày từ DB,
xuất ra CSV để phân tích tương quan với yếu tố kinh tế vĩ mô.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import datetime
from forecast_system.utils.db_utils import create_db_engine, fetch_with_chunks
from forecast_system.utils.logger import get_logger

logger = get_logger('extract_daily_guests')

QUERY_ALL_GUESTS = """
    SELECT 
        DATE(COALESCE(pos_open_transaction_time, shift_date)) as business_date,
        restaurant_code,
        SUM(guest_count) as total_guests,
        COUNT(DISTINCT order_guid) as total_transactions
    FROM v_fact_db_payment_hub_transactions 
    WHERE shift_date >= :start_date AND shift_date < :end_date
    GROUP BY DATE(COALESCE(pos_open_transaction_time, shift_date)), restaurant_code
    
    UNION ALL
    
    SELECT 
        DATE(COALESCE(open_time, shiftdate)) as business_date,
        restaurant_code,
        SUM(guest_count) as total_guests,
        COUNT(DISTINCT transaction_id) as total_transactions
    FROM v_fact_db_rk_dc_transactions 
    WHERE shiftdate >= :start_date AND shiftdate < :end_date
    GROUP BY DATE(COALESCE(open_time, shiftdate)), restaurant_code
"""

def extract_daily_guests(start_date='2024-01-01', end_date=None):
    """Truy xuất tổng lượt khách theo ngày từ tất cả nhà hàng."""
    
    if end_date is None:
        end_date = datetime.date.today().strftime('%Y-%m-%d')
    
    engine = create_db_engine()
    if engine is None:
        logger.error("Cannot connect to DB")
        return None
    
    params = {
        'start_date': start_date,
        'end_date': end_date
    }
    
    logger.info(f"Extracting daily guests from {start_date} to {end_date}...")
    
    df = fetch_with_chunks(
        engine, QUERY_ALL_GUESTS, params=params,
        name="Daily Guests"
    )
    
    if df.empty:
        logger.warning("No data returned from DB")
        return None
    
    # Aggregate across sources (remove duplicates by summing per date)
    df['business_date'] = pd.to_datetime(df['business_date'])
    daily = df.groupby('business_date').agg({
        'total_guests': 'sum',
        'total_transactions': 'sum',
        'restaurant_code': 'nunique'
    }).reset_index()
    
    daily.rename(columns={
        'restaurant_code': 'active_restaurants'
    }, inplace=True)
    
    daily = daily.sort_values('business_date').reset_index(drop=True)
    
    # Add time features
    daily['weekday'] = daily['business_date'].dt.day_name()
    daily['month'] = daily['business_date'].dt.month
    daily['year'] = daily['business_date'].dt.year
    daily['day_of_week'] = daily['business_date'].dt.dayofweek
    
    # Save CSV
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'daily_guests_data.csv'
    )
    daily.to_csv(output_path, index=False)
    logger.info(f"Saved {len(daily)} rows to {output_path}")
    
    print(f"\n{'='*60}")
    print(f"DAILY GUEST DATA EXTRACTED")
    print(f"{'='*60}")
    print(f"Period: {daily['business_date'].min().date()} → {daily['business_date'].max().date()}")
    print(f"Total days: {len(daily)}")
    print(f"Total guests: {daily['total_guests'].sum():,.0f}")
    print(f"Avg daily guests: {daily['total_guests'].mean():,.0f}")
    print(f"Output: {output_path}")
    
    return daily


if __name__ == '__main__':
    extract_daily_guests()
