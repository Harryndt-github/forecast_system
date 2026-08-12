"""
==============================================
DATABASE UTILITIES
==============================================
- Parameterized queries (fix SQL injection)
- Connection management với retry logic
- Chunked data loading cho large datasets
"""

import pandas as pd
from sqlalchemy import create_engine, text
import time
import traceback

from forecast_system.utils.logger import get_logger
from forecast_system.config.settings import get_connection_string, ENGINE_OPTIONS

logger = get_logger('db_utils')


def create_db_engine(max_retries=3, retry_delay=5):
    """
    Tạo SQLAlchemy engine với retry logic.
    
    Returns:
        sqlalchemy.Engine hoặc None nếu thất bại
    """
    for attempt in range(1, max_retries + 1):
        try:
            conn_str = get_connection_string()
            engine = create_engine(conn_str, **ENGINE_OPTIONS)
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info(f"Database connected (attempt {attempt}/{max_retries})")
            return engine
            
        except Exception as e:
            logger.warning(f"DB connection attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                logger.error(f"All {max_retries} DB connection attempts failed")
                traceback.print_exc()
                return None


def execute_parameterized_query(engine, query_template, params=None, chunksize=None):
    """
    Execute parameterized SQL query (SAFE - no SQL injection).
    
    Args:
        engine: SQLAlchemy engine
        query_template: SQL string with :param_name placeholders
        params: Dict of parameter values
        chunksize: If set, returns chunks iterator
    
    Returns:
        pd.DataFrame
    
    Example:
        df = execute_parameterized_query(
            engine,
            "SELECT * FROM table WHERE date >= :start AND date < :end",
            {'start': '2024-01-01', 'end': '2024-12-31'}
        )
    """
    try:
        query = text(query_template)
        if chunksize:
            return pd.read_sql(query, engine, params=params or {}, chunksize=chunksize)
        else:
            return pd.read_sql(query, engine, params=params or {})
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        traceback.print_exc()
        return pd.DataFrame()


def fetch_with_chunks(engine, query_template, params=None, chunksize=100000, name="data"):
    """
    Load dữ liệu lớn theo chunks để tránh timeout/memory overflow.
    
    Args:
        engine: SQLAlchemy engine
        query_template: SQL string with :param_name placeholders
        params: Dict of parameter values
        chunksize: Số rows mỗi chunk
        name: Tên hiển thị cho logging
    
    Returns:
        pd.DataFrame (concatenated từ tất cả chunks)
    """
    chunks = []
    total = 0
    
    try:
        logger.info(f"📥 Fetching {name} in chunks (size={chunksize})...")
        query = text(query_template)
        
        for chunk in pd.read_sql(query, engine, params=params or {}, chunksize=chunksize):
            chunks.append(chunk)
            total += len(chunk)
            print(f"     > {name}: {total:,} rows received...", end="\r")
        
        print()  # New line after carriage return
        logger.info(f"{name} loaded: {total:,} rows total")
        
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Chunked fetch failed for {name}: {e}")
        traceback.print_exc()
        return pd.DataFrame()


# ==========================================
# PARAMETERIZED QUERY TEMPLATES
# ==========================================

QUERY_PAYMENT_HUB = """
    SELECT restaurant_code, 
           order_guid as transaction_id, 
           guest_count, 
           pos_open_transaction_time as open_time, 
           shift_date as date 
    FROM v_fact_db_payment_hub_transactions 
    WHERE shift_date >= :start_date AND shift_date < :end_date
"""

QUERY_RK_DC = """
    SELECT restaurant_code, 
           transaction_id, 
           guest_count, 
           open_time, 
           shiftdate as date 
    FROM v_fact_db_rk_dc_transactions 
    WHERE shiftdate >= :start_date AND shiftdate < :end_date
"""

QUERY_PAYMENT_HUB_REVENUE = """
    SELECT restaurant_code, 
           order_guid as transaction_id, 
           guest_count, 
           total_pay_sum as revenue,
           pos_open_transaction_time as open_time, 
           shift_date as date 
    FROM v_fact_db_payment_hub_transactions 
    WHERE shift_date >= :start_date AND shift_date < :end_date
"""

QUERY_RK_DC_REVENUE = """
    SELECT restaurant_code, 
           transaction_id, 
           guest_count, 
           paysum as revenue,
           open_time, 
           shiftdate as date 
    FROM v_fact_db_rk_dc_transactions 
    WHERE shiftdate >= :start_date AND shiftdate < :end_date
"""

QUERY_BOOKING = """
    SELECT restaurant, 
           total_guest, 
           shift_date, 
           Start as start_time, 
           cancelled_reason
    FROM v_fact_db_booking_booking_info 
    WHERE shift_date >= :start_date 
      AND shift_date < :end_date
"""

QUERY_RESTAURANT_INFO = """
    SELECT restaurant_code, sap_code, restaurant_name 
    FROM v_dim_restaurant_address
"""
