"""
===============================================================
🔄 QUICK UPDATE RESULTS - Không cần chạy lại forecast
===============================================================
Script này chỉ chạy lại các bước cuối của pipeline:
  1. Kết nối DB
  2. Load data mới (actuals + booking)  
  3. Cập nhật Actuals trong Master File
  4. Merge Booking data
  5. Regenerate Shift Summary
  6. Monitoring & Brain insights

KHÔNG chạy lại forecast loop (Step 6) → tiết kiệm 6-8 tiếng!

Usage:
    python -m forecast_system.update_results
    python -m forecast_system.update_results --skip-booking   # Bỏ qua booking nếu DB lỗi
===============================================================
"""

import sys
import os
import datetime
import warnings
import traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from forecast_system.config.settings import (
    CURRENT_DATE, FORECAST_START_DATE, FORECAST_HORIZON,
    FORECAST_MONTHS_AHEAD, get_forecast_end_date,
    MASTER_FILE_NAME, SHIFT_FILE_NAME, LOG_DIR,
    DAILY_FORECAST_DAYS,
)
from forecast_system.utils.logger import setup_logger, get_logger
from forecast_system.utils.db_utils import create_db_engine
from forecast_system.agents.data_agent import DataAgent
from forecast_system.agents.master_file_agent import MasterFileAgent, save_excel_safely
from forecast_system.agents.forecast_brain import ForecastBrain
from forecast_system.agents.correction_validator import CorrectionValidator
from forecast_system.agents.booking_agent import BookingAgent


def update_results(skip_booking=False):
    """
    Cập nhật kết quả mà KHÔNG cần chạy lại forecast.
    
    Workflow:
      1. Load Master File hiện tại (đã có forecast)
      2. Kết nối DB → load actuals mới nhất
      3. Update actuals vào Master File
      4. Load booking data → merge vào Master File (optional)
      5. Regenerate Shift Summary
      6. Chạy Monitoring + Brain insights
    """
    
    logger = setup_logger('forecast_system', log_dir=LOG_DIR)
    
    logger.info("=" * 60)
    logger.info(f"🔄 QUICK UPDATE RESULTS | Date: {CURRENT_DATE}")
    logger.info(f"   Mode: Update actuals + booking (NO re-forecast)")
    logger.info(f"   Skip Booking: {skip_booking}")
    logger.info("=" * 60)
    
    # ==========================================
    # STEP 1: LOAD EXISTING MASTER FILE
    # ==========================================
    logger.info("\n📂 STEP 1: Loading existing Master File...")
    
    try:
        df_master = MasterFileAgent.load_or_create(MASTER_FILE_NAME)
        if df_master.empty:
            logger.error("❌ Master File trống! Cần chạy full pipeline trước.")
            logger.error("   python -m forecast_system.main --mode daily")
            return
        
        n_rows = len(df_master)
        n_restaurants = df_master['Restaurant_Code'].nunique() if 'Restaurant_Code' in df_master.columns else 0
        logger.info(f"   ✅ Loaded: {n_rows:,} rows, {n_restaurants} restaurants")
        
        # Show date range
        if 'Date' in df_master.columns:
            df_master['Date'] = pd.to_datetime(df_master['Date'], errors='coerce')
            date_min = df_master['Date'].min()
            date_max = df_master['Date'].max()
            logger.info(f"   📅 Date range: {date_min.date() if pd.notna(date_min) else '?'} → {date_max.date() if pd.notna(date_max) else '?'}")  # type: ignore[reportGeneralTypeIssues]
        
        # Show existing forecast run dates
        if 'Forecast_Run_Date' in df_master.columns:
            frd = pd.to_datetime(df_master['Forecast_Run_Date'], errors='coerce')
            run_dates = frd.dropna().dt.date.unique()
            if len(run_dates) > 0:
                logger.info(f"   🔮 Forecast runs: {sorted(run_dates)[-3:]}")
    except Exception as e:
        logger.error(f"❌ Cannot load Master File: {e}")
        traceback.print_exc()
        return
    
    # ==========================================
    # STEP 2: CONNECT TO DB & LOAD FRESH DATA
    # ==========================================
    logger.info("\n📡 STEP 2: Connecting to Database...")
    engine = None
    df_train = pd.DataFrame()
    df_info = pd.DataFrame()
    
    try:
        engine = create_db_engine(max_retries=3, retry_delay=5)
        if engine is None:
            logger.warning("⚠️ DB connection failed. Will update with existing data only.")
        else:
            logger.info("\n📥 Loading fresh data from DB...")
            df_train = DataAgent.load_recent_data(engine)
            df_info = DataAgent.load_restaurant_info(engine)
            logger.info(f"   ✅ Loaded {len(df_train):,} training rows")
    except Exception as e:
        logger.warning(f"⚠️ DB loading failed (will continue with existing data): {e}")
    
    # ==========================================
    # STEP 3: UPDATE ACTUALS
    # ==========================================
    if not df_train.empty:
        logger.info("\n📊 STEP 3: Updating Actuals in Master File...")
        try:
            df_master = MasterFileAgent.update_actuals(df_master, df_train)
            
            # Count how many rows got actuals
            if 'Actual_Guests' in df_master.columns:
                n_actuals = df_master['Actual_Guests'].notna().sum()
                logger.info(f"   ✅ Actuals updated: {n_actuals:,} rows have actual data")
            
            # Save intermediate
            save_excel_safely(df_master, MASTER_FILE_NAME)
            logger.info(f"   💾 Master file saved (with updated actuals)")
        except Exception as e:
            logger.warning(f"⚠️ Actuals update failed: {e}")
            traceback.print_exc()
    else:
        logger.info("\n📊 STEP 3: Skipped (no fresh training data)")
    
    # ==========================================
    # STEP 4: LOAD & MERGE BOOKING DATA
    # ==========================================
    df_booking_summary = pd.DataFrame()
    
    if not skip_booking and engine is not None:
        logger.info("\n🎫 STEP 4: Loading Booking Data...")
        
        # Calculate forecast end
        forecast_end = CURRENT_DATE + datetime.timedelta(days=DAILY_FORECAST_DAYS)
        
        try:
            df_booking_raw = BookingAgent.load_booking_data(
                engine,
                start_date=CURRENT_DATE,
                end_date=forecast_end,
            )
            
            if not df_booking_raw.empty:
                df_booking_summary = BookingAgent.aggregate_booking_summary(
                    df_booking_raw, df_info=df_info
                )
                BookingAgent.print_booking_summary(
                    df_booking_summary, logger_func=logger.info
                )
                
                # Get daily totals for merge
                df_booking_daily = BookingAgent.get_daily_booking_totals(df_booking_raw)
                
                if not df_booking_daily.empty:
                    # Merge booking into master file
                    logger.info("\n   📋 Merging booking data into Master File...")
                    
                    df_bk = df_booking_daily.copy()
                    df_bk['Restaurant_Code'] = DataAgent.normalize_key(df_bk['Restaurant_Code'])
                    df_bk['Date'] = pd.to_datetime(df_bk['Date'], errors='coerce')
                    df_bk = df_bk.dropna(subset=['Date'])
                    df_bk['Date'] = df_bk['Date'].dt.date
                    
                    df_master['Date'] = pd.to_datetime(df_master['Date'], errors='coerce')
                    df_master = df_master.dropna(subset=['Date'])
                    df_master['Date'] = df_master['Date'].dt.date
                    
                    # Drop old booking column
                    df_master.drop(columns=['Booking_Guests'], errors='ignore', inplace=True)
                    
                    # Merge
                    df_master = pd.merge(
                        df_master,
                        df_bk[['Restaurant_Code', 'Date', 'Booking_Guests_Total']].rename(  # type: ignore[reportCallIssue]
                            columns={'Booking_Guests_Total': 'Booking_Guests'}
                        ),
                        on=['Restaurant_Code', 'Date'],
                        how='left'
                    )
                    
                    n_matched = df_master['Booking_Guests'].notna().sum()
                    logger.info(f"   ✅ Booking merged: {n_matched:,} rows matched")
            else:
                logger.info("   No future booking data found")
                
        except Exception as e:
            logger.warning(f"⚠️ Booking data failed: {e}")
            traceback.print_exc()
    elif skip_booking:
        logger.info("\n🎫 STEP 4: Skipped (--skip-booking flag)")
    else:
        logger.info("\n🎫 STEP 4: Skipped (no DB connection)")
    
    # ==========================================
    # STEP 5: REGENERATE SHIFT SUMMARY
    # ==========================================
    logger.info("\n📊 STEP 5: Generating Shift Summary...")
    
    try:
        if 'Hour' in df_master.columns:
            hour_filter = df_master['Hour'].notna()
            if 'Forecast_Mode' in df_master.columns:
                hour_filter = hour_filter & (df_master['Forecast_Mode'] != 'daily_only')
            df_hourly_only = df_master[hour_filter].copy()
        else:
            df_hourly_only = df_master.copy()
        
        df_shift = DataAgent.aggregate_shifts(df_hourly_only)
        if not df_shift.empty:
            save_excel_safely(df_shift, SHIFT_FILE_NAME)
            logger.info(f"   ✅ Shift Summary saved: {len(df_shift):,} rows → {SHIFT_FILE_NAME}")
        else:
            logger.info("   ℹ️ No hourly data for shift summary")
    except Exception as e:
        logger.warning(f"⚠️ Shift summary failed: {e}")
        traceback.print_exc()
    
    # ==========================================
    # STEP 6: SAVE FINAL MASTER FILE
    # ==========================================
    logger.info("\n💾 STEP 6: Saving Final Master File...")
    
    try:
        # Merge restaurant info if available
        if not df_info.empty:
            df_master.drop(columns=['sap_code', 'restaurant_name'], errors='ignore', inplace=True)
            df_master['merge_key'] = DataAgent.normalize_key(df_master['Restaurant_Code'])
            df_master = pd.merge(
                df_master,
                df_info[['merge_key', 'sap_code', 'restaurant_name']],
                on='merge_key', how='left'
            )
            df_master.drop(columns=['merge_key'], inplace=True)
        
        # Reorder columns
        cols = MasterFileAgent.COLUMNS
        df_master = df_master[[c for c in cols if c in df_master.columns]]
        
        # Save with booking sheet if available
        if not df_booking_summary.empty:
            MasterFileAgent.save_with_booking_sheet(
                df_master, df_booking_summary, MASTER_FILE_NAME
            )
        else:
            save_excel_safely(df_master, MASTER_FILE_NAME)
        
        logger.info(f"   ✅ Master file saved: {len(df_master):,} rows → {MASTER_FILE_NAME}")
    except Exception as e:
        logger.error(f"❌ Failed to save Master File: {e}")
        traceback.print_exc()
    
    # ==========================================
    # STEP 7: MONITORING & ACCURACY
    # ==========================================
    logger.info("\n📊 STEP 7: Accuracy Monitoring...")
    
    report = None
    try:
        from forecast_system.agents.monitoring_agent import MonitoringAgent
        
        report = MonitoringAgent.generate_full_report(df_master)  # type: ignore[reportArgumentType]
        MonitoringAgent.print_report(report, logger_func=logger.info)
        MonitoringAgent.save_report_excel(df_master, report)  # type: ignore[reportArgumentType]
        
        logger.info("   ✅ Monitoring complete")
    except Exception as e:
        logger.warning(f"⚠️ Monitoring failed (non-critical): {e}")
    
    # ==========================================
    # STEP 8: BRAIN LEARNING
    # ==========================================
    logger.info("\n🧠 STEP 8: Brain Learning & Insights...")
    
    try:
        learn_result = ForecastBrain.learn_from_errors(df_master)  # type: ignore[reportArgumentType]
        # Memento: Validate pending corrections
        try:
            _brain_mem = ForecastBrain.load_memory()
            _rollbacks = CorrectionValidator.validate_all_pending(df_master, _brain_mem)
            if _rollbacks:
                ForecastBrain.save_memory(_brain_mem)
        except Exception:
            pass
        logger.info(f"   Restaurants learned: {learn_result.get('restaurants_learned', 0)}")
        logger.info(f"   Issues found: {learn_result.get('issues_found', 0)}")
        
        if report:
            absorb_result = ForecastBrain.absorb_monitoring_report(report)
            logger.info(f"   Metrics updated: {absorb_result.get('metrics_updated', 0)}")
            logger.info(f"   Drift adjustments: {absorb_result.get('drift_adjustments', 0)}")
        
        ForecastBrain.print_insights(logger_func=logger.info)
        logger.info("   ✅ Brain insights complete")
    except Exception as e:
        logger.warning(f"⚠️ Brain learning failed (non-critical): {e}")
    
    # ==========================================
    # SUMMARY
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("✅ QUICK UPDATE COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"  Master File: {MASTER_FILE_NAME} ({len(df_master):,} rows)")
    logger.info(f"  Actuals updated: {'Yes' if not df_train.empty else 'No (no DB)'}")
    logger.info(f"  Booking merged: {'Yes' if not df_booking_summary.empty else 'Skipped'}")
    logger.info(f"  Shift Summary: {SHIFT_FILE_NAME}")
    logger.info(f"  Monitoring: {'Done' if report else 'Skipped'}")
    logger.info("=" * 60)
    logger.info("💡 Tip: Forecast data từ lần chạy trước được GIỮ NGUYÊN.")
    logger.info("   Chỉ Actuals + Booking + Monitoring được cập nhật mới.")
    logger.info("   Để chạy lại forecast đầy đủ: python -m forecast_system.main")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Quick Update Results - Không cần chạy lại forecast',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m forecast_system.update_results              # Full update (actuals + booking)
  python -m forecast_system.update_results --skip-booking  # Bỏ qua booking (nếu DB lỗi mạng)
        """
    )
    parser.add_argument(
        '--skip-booking', action='store_true',
        help='Bỏ qua bước load booking data (dùng khi network lỗi)'
    )
    args = parser.parse_args()
    update_results(skip_booking=args.skip_booking)
