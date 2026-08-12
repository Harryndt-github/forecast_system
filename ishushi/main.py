"""
===============================================================
🍣 ISHUSHI FORECAST SYSTEM - MAIN ORCHESTRATOR (v2)
===============================================================
Hệ thống dự đoán cho chuỗi nhà hàng Ishushi với FEEDBACK LOOP:
1. Dự đoán số khách (guest count) theo ngày
2. Dự đoán số lượng món ăn theo sap_code
3. ⭐ Cập nhật dữ liệu thực tế (actual) vào forecast đã dự đoán
4. ⭐ So sánh forecast vs actual → Brain tự học sửa lỗi
5. ⭐ Áp dụng corrections cho forecast tương lai

Data Sources:
- v_fact_db_rk_dc_transactions       → Giao dịch (khách)
- v_fact_db_rk_dc_transaction_details → Chi tiết món (sap_code, quantity)

Join key: transaction_id

Modes:
    --mode forecast   : Chạy forecast mới (default)
    --mode update     : Chỉ cập nhật actuals + brain learn (KHÔNG re-forecast)
    --mode full       : Cả forecast + update + brain learn

Usage:
    python -m forecast_system.ishushi.main
    python -m forecast_system.ishushi.main --mode update
    python -m forecast_system.ishushi.main --mode full --days 60
    python -m forecast_system.ishushi.main --backtest
===============================================================
"""

import sys
import os
import datetime
import time
import argparse
import warnings
import traceback
import pandas as pd

warnings.filterwarnings('ignore')

# Đảm bảo import path chính xác
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from forecast_system.utils.logger import setup_logger
from forecast_system.utils.db_utils import create_db_engine
from forecast_system.ishushi.data_agent import IshushiDataAgent
from forecast_system.ishushi.forecast_model import IshushiForecastModel
from forecast_system.ishushi.master_file_agent import IshushiMasterFileAgent
from forecast_system.ishushi.brain import IshushiBrain
from forecast_system.ishushi.config import ISHUSHI_CONFIG, ISHUSHI_SAP_CATALOG, ISHUSHI_SAP_GROUPS


def main(forecast_days=30, run_backtest=True, mode='forecast'):
    """
    Main pipeline cho Ishushi Forecast.
    
    Steps:
    1. Connect Database
    2. Find Ishushi Restaurants
    3. Load Guest Data (transactions)
    4. Load Item Data (transaction_details)
    5. Build Daily Summaries
    6. ⭐ Update Actuals (cập nhật dữ liệu thực tế vào Master File)
    7. ⭐ Brain Learning (học từ errors để sửa forecast tương lai)
    8. Backtest (optional)
    9. Forecast Guests (+ Brain corrections)
    10. Forecast Items (+ Brain corrections)
    11. Save to Master File
    12. ⭐ Accuracy Report
    
    Args:
        forecast_days: Số ngày dự đoán (default: 30)
        run_backtest: Có chạy backtest hay không (default: True)
        mode: 'forecast' | 'update' | 'full'
    """
    run_start = time.time()
    logger = setup_logger('ishushi_main', log_dir=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'logs'
    ))
    
    logger.info("="* 60)
    logger.info("🍣 ISHUSHI FORECAST SYSTEM v4 (Direct Shift ML + Meta-Learner)")
    logger.info(f"   Date: {datetime.date.today()}")
    logger.info(f"   Mode: {mode.upper()}")
    logger.info(f"   Forecast Days: {forecast_days}")
    logger.info(f"   Backtest: {'Yes' if run_backtest else 'No'}")
    logger.info(f"   SAP Codes: {len(ISHUSHI_SAP_CATALOG)}")
    logger.info(f"   Item Groups: {len(ISHUSHI_SAP_GROUPS)}")
    logger.info("=" * 60)
    
    # ==========================================
    # STEP 1: DATABASE CONNECTION
    # ==========================================
    logger.info("\n📡 STEP 1: Connecting to Database...")
    engine = create_db_engine(max_retries=3, retry_delay=5)
    
    if engine is None:
        logger.error("Failed to connect to Database. Aborting.")
        return
    
    # ==========================================
    # STEP 2: FIND ISHUSHI RESTAURANTS
    # ==========================================
    logger.info("\n🍣 STEP 2: Finding Ishushi Restaurants...")
    ishushi_codes = IshushiDataAgent.load_ishushi_restaurants(engine)
    
    if not ishushi_codes:
        logger.error("No Ishushi restaurants found! Aborting.")
        logger.info("TIP: Check if restaurant names contain 'ishushi' or 'Ishushi'")
        return
    
    logger.info(f"   Found {len(ishushi_codes)} Ishushi restaurants")
    
    # ==========================================
    # STEP 3: LOAD GUEST DATA
    # ==========================================
    logger.info("\n📊 STEP 3: Loading Guest Data...")
    
    df_transactions = IshushiDataAgent.load_guest_data(engine, ishushi_codes)
    
    if df_transactions.empty:
        logger.error("No transaction data loaded for Ishushi. Aborting.")
        return
    
    # ==========================================
    # STEP 4: LOAD ITEM DATA
    # ==========================================
    logger.info("\n🍱 STEP 4: Loading Item Data (SAP codes)...")
    
    df_items = IshushiDataAgent.load_item_data(engine, ishushi_codes)
    
    if df_items.empty:
        logger.warning("No item data loaded. Will only forecast guests.")
    else:
        # Show SAP code distribution
        logger.info("\n   📋 SAP Code Distribution:")
        sap_dist = df_items.groupby(['sap_code', 'item_name'])['quantity'].sum()
        for (sap, name), total_qty in sap_dist.sort_values(ascending=False).items():  # type: ignore[reportCallIssue, reportGeneralTypeIssues]
            logger.info(f"      {sap} | {name}: {total_qty:,.0f} total qty")
    
    # ==========================================
    # STEP 5: BUILD DAILY SUMMARIES
    # ==========================================
    logger.info("\n📋 STEP 5: Building Daily Summaries...")
    
    # Guest summary (daily total)
    df_daily_guests = IshushiDataAgent.build_daily_guest_summary(df_transactions)
    logger.info(f"   Daily guest records: {len(df_daily_guests):,}")
    
    # ⭐ Guest summary BY SHIFT (MORNING/EVENING)
    df_daily_guests_shift = IshushiDataAgent.build_daily_guest_summary_by_shift(
        df_transactions
    )
    logger.info(f"   Daily guest records (by shift): {len(df_daily_guests_shift):,}")
    
    # Item summary
    df_daily_items = pd.DataFrame()
    df_daily_items_shift = pd.DataFrame()
    df_daily_groups = pd.DataFrame()
    
    if not df_items.empty:
        df_daily_items = IshushiDataAgent.build_daily_item_summary(
            df_transactions, df_items
        )
        logger.info(f"   Daily item records: {len(df_daily_items):,}")
        
        # ⭐ Item summary BY SHIFT
        df_daily_items_shift = IshushiDataAgent.build_daily_item_summary_by_shift(
            df_transactions, df_items
        )
        logger.info(f"   Daily item records (by shift): {len(df_daily_items_shift):,}")
        
        df_daily_groups = IshushiDataAgent.build_daily_group_summary(df_daily_items)
        logger.info(f"   Daily group records: {len(df_daily_groups):,}")
    
    # ==========================================
    # STEP 6: UPDATE ACTUALS IN MASTER FILE ⭐
    # ==========================================
    logger.info("\n📊 STEP 6: Updating Actuals in Master File...")
    
    df_guest_master = IshushiMasterFileAgent.load_guest_master()
    df_item_master = IshushiMasterFileAgent.load_item_master()
    
    n_before_guest = len(df_guest_master[df_guest_master['Actual_Guests'].notna()]) if not df_guest_master.empty and 'Actual_Guests' in df_guest_master.columns else 0
    n_before_item = len(df_item_master[df_item_master['Actual_Quantity'].notna()]) if not df_item_master.empty and 'Actual_Quantity' in df_item_master.columns else 0
    
    # Update guest actuals
    if not df_guest_master.empty:
        df_guest_master = IshushiMasterFileAgent.update_guest_actuals(
            df_guest_master, df_transactions
        )
        n_after = len(df_guest_master[df_guest_master['Actual_Guests'].notna()])
        logger.info(f"   Guest actuals: {n_before_guest} → {n_after} rows with actual data")
    
    # Update item actuals
    if not df_item_master.empty and not df_items.empty:
        df_item_master = IshushiMasterFileAgent.update_item_actuals(
            df_item_master, df_transactions, df_items
        )
        n_after_item = len(df_item_master[df_item_master['Actual_Quantity'].notna()])
        logger.info(f"   Item actuals: {n_before_item} → {n_after_item} rows with actual data")
    
    # ==========================================
    # STEP 7: BRAIN LEARNING ⭐
    # ==========================================
    logger.info("\n🧠 STEP 7: Brain Learning from Errors...")
    
    guest_learn = {}
    item_learn = {}
    
    if not df_guest_master.empty:
        guest_learn = IshushiBrain.learn_from_guest_errors(df_guest_master)
        logger.info(f"   Guest learning: {guest_learn.get('status', 'N/A')}")
    
    if not df_item_master.empty:
        item_learn = IshushiBrain.learn_from_item_errors(df_item_master)
        logger.info(f"   Item learning: {item_learn.get('status', 'N/A')}")
    
    # Print brain insights
    IshushiBrain.print_insights(logger_func=logger.info)
    
    # If mode is 'update', stop here (no re-forecast)
    if mode == 'update':
        # Save updated masters
        guest_accuracy = IshushiMasterFileAgent.calculate_accuracy(
            df_guest_master, target='guests'
        ) if not df_guest_master.empty else None
        IshushiMasterFileAgent.save_guest_master(df_guest_master, accuracy=guest_accuracy)
        
        if not df_item_master.empty:
            item_accuracy = IshushiMasterFileAgent.calculate_accuracy(
                df_item_master, target='items'
            )
            IshushiMasterFileAgent.save_item_master(df_item_master, accuracy=item_accuracy)
        
        elapsed = time.time() - run_start
        logger.info(f"\n{'='*60}")
        logger.info(f"🍣 ISHUSHI UPDATE COMPLETE (no re-forecast)")
        logger.info(f"   ⏱️ Duration: {elapsed:.1f} seconds")
        if guest_accuracy and guest_accuracy.get('overall'):
            ov = guest_accuracy['overall']
            logger.info(f"   📊 Guest MAPE: {ov.get('MAPE', 'N/A')}%")
            logger.info(f"   📊 Guest Hit Rate: {ov.get('Hit_Rate', 'N/A')}%")
        logger.info(f"{'='*60}")
        return
    
    # ==========================================
    # STEP 8: DATA EXPLORATION
    # ==========================================
    logger.info("\n📈 STEP 8: Data Exploration...")
    
    for res_code in sorted(ishushi_codes, key=lambda x: str(x)):
        df_res = df_daily_guests[df_daily_guests['restaurant_code'] == res_code]
        if df_res.empty:
            continue
        
        avg_guests = df_res['total_guests'].mean()
        avg_txns = df_res['num_transactions'].mean()
        date_range = f"{df_res['date'].min()} → {df_res['date'].max()}"
        n_days = len(df_res)
        
        logger.info(
            f"   🏪 {res_code}: {n_days} days, "
            f"avg {avg_guests:.1f} guests/day, "
            f"{avg_txns:.1f} orders/day, "
            f"range: {date_range}"
        )
        
        # Show item breakdown for this restaurant
        if not df_daily_groups.empty:
            df_res_items = df_daily_groups[df_daily_groups['restaurant_code'] == res_code]
            if not df_res_items.empty:
                for group in df_res_items['item_group'].unique():  # type: ignore[reportAttributeAccessIssue]
                    avg_qty = df_res_items[
                        df_res_items['item_group'] == group
                    ]['total_quantity'].mean()
                    logger.info(f"      🍱 {group}: avg {avg_qty:.1f} qty/day")
    
    # ==========================================
    # STEP 9: BACKTEST (Optional)
    # ==========================================
    backtest_results = None
    if run_backtest:
        logger.info("\n📊 STEP 9: Running Backtest...")
        backtest_results = IshushiForecastModel.backtest_guest_model(df_daily_guests)
        
        if backtest_results:
            avg_mape = sum(r['mape'] for r in backtest_results.values()) / len(backtest_results)
            avg_r2 = sum(r['r2'] for r in backtest_results.values()) / len(backtest_results)
            logger.info(f"\n   📊 Overall Backtest: MAPE={avg_mape:.1f}%, R²={avg_r2:.3f}")
    else:
        logger.info("\n⏭️ STEP 9: Backtest skipped")
    
    # ==========================================
    # STEP 10: FORECAST GUESTS + BRAIN CORRECTIONS ⭐
    # ==========================================
    logger.info(f"\n🔮 STEP 10: Forecasting Guests BY SHIFT ({forecast_days} days)...")
    
    df_guest_forecast = IshushiForecastModel.forecast_guests_by_shift(
        df_daily_guests, df_daily_guests_shift,
        forecast_days=forecast_days, df_bookings=None
    )
    
    # ⭐ Apply Brain corrections
    if not df_guest_forecast.empty:
        logger.info("   🧠 Applying Brain corrections to guest forecast...")
        df_guest_forecast = IshushiBrain.apply_guest_corrections(df_guest_forecast)
        
        logger.info(f"\n   Guest Forecast Summary (by shift, after Brain corrections):")
        for res_code in df_guest_forecast['restaurant_code'].unique():
            df_res = df_guest_forecast[df_guest_forecast['restaurant_code'] == res_code]
            for shift in ['MORNING', 'EVENING']:
                df_shift = df_res[df_res['shift'] == shift]
                if not df_shift.empty:  # type: ignore[reportAttributeAccessIssue]
                    logger.info(
                        f"   🏠 {res_code} [{shift}]: "
                        f"avg {df_shift['predicted_guests'].mean():.0f} guests/day, "
                        f"range [{df_shift['predicted_guests'].min()} - {df_shift['predicted_guests'].max()}]"
                    )
    
    # ==========================================
    # STEP 11: FORECAST ITEMS + BRAIN CORRECTIONS ⭐
    # ==========================================
    df_item_forecast = pd.DataFrame()
    
    if not df_daily_items.empty:
        logger.info(f"\n🍱 STEP 11: Forecasting Items BY SHIFT ({forecast_days} days)...")
        
        df_item_forecast = IshushiForecastModel.forecast_items_by_shift(
            df_daily_items, df_daily_items_shift,
            df_daily_guests, forecast_days=forecast_days
        )
        
        # ⭐ Apply Brain corrections
        if not df_item_forecast.empty:
            logger.info("   🧠 Applying Brain corrections to item forecast...")
            df_item_forecast = IshushiBrain.apply_item_corrections(df_item_forecast)
            
            logger.info(f"\n   Item Forecast Summary by Shift and Group:")
            for shift in ['MORNING', 'EVENING']:
                df_shift = df_item_forecast[df_item_forecast['shift'] == shift]
                if df_shift.empty:
                    continue
                group_summary = df_shift.groupby('item_group')['predicted_quantity'].agg(
                    ['mean', 'sum']
                )
                logger.info(f"   [{shift}]:")
                for group, row in group_summary.iterrows():
                    logger.info(
                        f"   🍱 {group}: "
                        f"avg {row['mean']:.1f} qty/day, "
                        f"total {row['sum']:,.0f} over {forecast_days} days"
                    )
    else:
        logger.info("\n⏭️ STEP 11: Item forecast skipped (no item data)")
    
    # ==========================================
    # STEP 12: SAVE TO MASTER FILE ⭐
    # ==========================================
    logger.info("\n💾 STEP 12: Saving to Master File...")
    
    run_date = datetime.date.today()
    
    # Save legacy output files (backward compatible)
    IshushiForecastModel.save_results(
        df_guest_forecast=df_guest_forecast,
        df_item_forecast=df_item_forecast,
        backtest_results=backtest_results,
    )
    
    # ⭐ Save to Master File (for feedback loop)
    if not df_guest_forecast.empty:
        df_guest_master = IshushiMasterFileAgent.append_guest_forecast(
            df_guest_master, df_guest_forecast, run_date=run_date
        )
        
        # Calculate accuracy
        guest_accuracy = IshushiMasterFileAgent.calculate_accuracy(
            df_guest_master, target='guests'
        )
        
        IshushiMasterFileAgent.save_guest_master(df_guest_master, accuracy=guest_accuracy)
        
        if guest_accuracy.get('overall'):
            ov = guest_accuracy['overall']
            logger.info(f"   📊 Guest Accuracy: MAPE={ov.get('MAPE', 'N/A')}%, "
                       f"MAE={ov.get('MAE', 'N/A')}, "
                       f"Hit Rate={ov.get('Hit_Rate', 'N/A')}%")
        
        # ⭐ Per-shift accuracy
        if guest_accuracy.get('per_shift'):
            for shift, sm in guest_accuracy['per_shift'].items():
                logger.info(f"      [{shift}]: MAPE={sm.get('MAPE', 'N/A')}%, "
                           f"MAE={sm.get('MAE', 'N/A')}, "
                           f"N={sm.get('N_samples', 0)}")
    
    if not df_item_forecast.empty:
        df_item_master = IshushiMasterFileAgent.append_item_forecast(
            df_item_master, df_item_forecast, run_date=run_date
        )
        
        item_accuracy = IshushiMasterFileAgent.calculate_accuracy(
            df_item_master, target='items'
        )
        
        IshushiMasterFileAgent.save_item_master(df_item_master, accuracy=item_accuracy)
        
        if item_accuracy.get('overall'):
            ov = item_accuracy['overall']
            logger.info(f"   📊 Item Accuracy: MAPE={ov.get('MAPE', 'N/A')}%, "
                       f"MAE={ov.get('MAE', 'N/A')}")
        
        # ⭐ Per-shift accuracy for items
        if item_accuracy.get('per_shift'):
            for shift, sm in item_accuracy['per_shift'].items():
                logger.info(f"      [{shift}]: MAPE={sm.get('MAPE', 'N/A')}%, "
                           f"N={sm.get('N_samples', 0)}")
    
    # ==========================================
    # SUMMARY
    # ==========================================
    elapsed = time.time() - run_start
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🍣 ISHUSHI FORECAST COMPLETE (v4 Direct Shift + Meta-Learner)")
    logger.info(f"{'='*60}")
    logger.info(f"   ⏱️ Duration: {elapsed:.1f} seconds")
    logger.info(f"   🏠 Restaurants: {len(ishushi_codes)}")
    logger.info(f"   📅 Forecast days: {forecast_days}")
    logger.info(f"   🔄 Shifts: MORNING (8-15h30) + EVENING (15h30-23h)")
    
    if not df_guest_forecast.empty:
        logger.info(f"   👥 Guest forecasts: {len(df_guest_forecast):,} records")
        for shift in ['MORNING', 'EVENING']:
            df_s = df_guest_forecast[df_guest_forecast['shift'] == shift] if 'shift' in df_guest_forecast.columns else pd.DataFrame()
            if not df_s.empty:
                logger.info(f"      [{shift}]: avg {df_s['predicted_guests'].mean():.0f} guests")
    if not df_item_forecast.empty:
        logger.info(f"   🍱 Item forecasts: {len(df_item_forecast):,} records")
    if backtest_results:
        avg_mape = sum(r['mape'] for r in backtest_results.values()) / len(backtest_results)
        logger.info(f"   📊 Backtest MAPE: {avg_mape:.1f}%")
    
    # Brain status
    logger.info(f"\n   🧠 Brain Status:")
    logger.info(f"      Guest learning: {guest_learn.get('status', 'N/A')} "
               f"({guest_learn.get('restaurants_learned', 0)} restaurants)")
    logger.info(f"      Item learning: {item_learn.get('status', 'N/A')} "
               f"({item_learn.get('items_learned', 0)} items)")
    
    logger.info(f"\n   📁 Output: {ISHUSHI_CONFIG['output_dir']}")
    logger.info(f"   📁 Master Guest: {IshushiMasterFileAgent.GUEST_MASTER_FILE}")
    logger.info(f"   📁 Master Item: {IshushiMasterFileAgent.ITEM_MASTER_FILE}")
    logger.info(f"   📅 Item Master: 2026 data preserved (new rows from {run_date} onwards)")
    logger.info(f"{'='*60}")
    
    return {
        'guest_forecast': df_guest_forecast,
        'item_forecast': df_item_forecast,
        'backtest': backtest_results,
        'guest_accuracy': guest_accuracy if not df_guest_forecast.empty else None,  # type: ignore[reportPossiblyUnboundVariable]
        'item_accuracy': item_accuracy if not df_item_forecast.empty else None,  # type: ignore[reportPossiblyUnboundVariable]
        'shift_mode': True,  # v2 shift-based
    }




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ishushi Forecast System v4')
    parser.add_argument(
        '--days', type=int, default=30,
        help='Số ngày forecast (default: 30)'
    )
    parser.add_argument(
        '--mode', type=str, default='forecast',
        choices=['forecast', 'update', 'full'],
        help='Mode: forecast (default) | update (chỉ cập nhật actuals) | full (forecast + update)'
    )
    parser.add_argument(
        '--backtest', action='store_true', default=True,
        help='Chạy backtest (default: True)'
    )
    parser.add_argument(
        '--no-backtest', action='store_true',
        help='Bỏ qua backtest'
    )
    
    args = parser.parse_args()
    
    run_bt = not args.no_backtest
    main(forecast_days=args.days, run_backtest=run_bt, mode=args.mode)
