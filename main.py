"""
===============================================================
🚀 AI FORECAST SYSTEM - MAIN ORCHESTRATOR (PHASE 3: ENSEMBLE)
===============================================================
Hệ thống dự đoán lượt khách cho 500+ nhà hàng trên toàn quốc.

Architecture:
    1. DataAgent            → Load & Clean dữ liệu từ DB
    2. AnalysisAgent        → Phân tích trends, gaps, outliers
    3. MLForecastAgent      → Feature engineering
    4. EnsembleForecastAgent→ Multi-model stacking + AI ensemble
    5. AIForecastAgent      → LLM-based prediction (LM Studio)
    6. MasterFileAgent      → Quản lý file kết quả

Flow:
    Load Data → Analysis → Filter/Clean → Feature Engineering
    → Stacking ML (XGBoost+CatBoost+LightGBM+RF)
    → Prophet Daily Trend
    → AI Forecast (LM Studio)
    → Ensemble Combine (Strategy-based weights)
    → Save Results → Update Actuals

Usage:
    python -m forecast_system.main
    
    hoặc:
    from forecast_system.main import main
    main()
===============================================================
"""

import sys
import os
import copy
import datetime
import warnings
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm  # type: ignore[import-untyped]

# Đảm bảo import path chính xác
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

from forecast_system.config.settings import (
    CURRENT_DATE, FORECAST_START_DATE, FORECAST_HORIZON,
    FORECAST_MONTHS_AHEAD, get_forecast_end_date,
    INACTIVE_THRESHOLD, MASTER_FILE_NAME, SHIFT_FILE_NAME,
    LOG_DIR, STRATEGY_WEIGHTS, DATA_LOOKBACK_DAYS, ANALYSIS_CONFIG,
    LUNAR_NY_CLOSURE_FILE, OPEN_CLOSE_FILE, PARALLEL_CONFIG,
    SHORT_HORIZON_DAYS, DAILY_FORECAST_DAYS,
    HOLIDAY_CLOSURE_RATE_THRESHOLD,  # ⭐ v7
    AI_INFERENCE_THRESHOLD, AI_TIMEOUT_PER_RESTAURANT,  # ⭐ v9 performance
)
from forecast_system.utils.logger import setup_logger, get_logger
from forecast_system.utils.db_utils import create_db_engine
from forecast_system.utils.date_utils import get_vn_holidays, build_forecast_days
from forecast_system.agents.data_agent import DataAgent
from forecast_system.agents.analysis_agent import AnalysisAgent
from forecast_system.agents.ml_forecast_agent import MLForecastAgent
from forecast_system.agents.ensemble_agent import EnsembleForecastAgent
from forecast_system.agents.master_file_agent import MasterFileAgent, save_excel_safely
from forecast_system.agents.forecast_brain import ForecastBrain
from forecast_system.agents.new_restaurant_agent import NewRestaurantAgent
from forecast_system.agents.booking_agent import BookingAgent
from forecast_system.agents.correction_validator import CorrectionValidator

# Phase 6: RAG Self-Learning Agent (replaces LM Studio)
try:
    from forecast_system.config.settings import USE_RAG_AGENT
    if USE_RAG_AGENT:
        from forecast_system.agents.rag_forecast_agent import RAGForecastAgent
        from forecast_system.agents.knowledge_store import KnowledgeStore
        AIAgent = RAGForecastAgent  # Use RAG agent
        logger_name = 'RAG (Self-Learning)'
    else:
        from forecast_system.agents.ai_forecast_agent import AIForecastAgent
        AIAgent = AIForecastAgent  # Use LM Studio
        logger_name = 'LM Studio'
except ImportError:
    from forecast_system.agents.ai_forecast_agent import AIForecastAgent
    AIAgent = AIForecastAgent
    USE_RAG_AGENT = False
    logger_name = 'LM Studio (fallback)'

def _analyze_active_restaurants(df_train, open_close_status, permanently_closed, logger):
    logger.info("\n📊 STEP 3: Analyzing Restaurants...")
    
    active_restaurants = DataAgent.get_active_restaurants(
        df_train, inactive_threshold=INACTIVE_THRESHOLD
    )
    
    if len(active_restaurants) == 0:
        logger.error("No active restaurants found. Aborting.")
        return [], {}, [], [], {}, {}
    
    analysis_reports = AnalysisAgent.analyze_all_restaurants(
        df_train, active_restaurants
    )
    
    restaurants_to_forecast = []
    restaurants_excluded = []
    restaurants_closed_permanently = []
    
    for res_code in active_restaurants:
        res_code_str = str(res_code)
        report = analysis_reports.get(res_code, {})
        
        if res_code_str in permanently_closed:
            closed_info = open_close_status['closed_restaurants'][res_code_str]
            closing_date = closed_info.get('closing_date', 'N/A')
            restaurants_closed_permanently.append({
                'code': res_code,
                'reason': f"PERMANENTLY CLOSED (Open_Close.xlsx) since {closing_date}",
                'sap_code': closed_info.get('sap_code', '?'),
            })
            continue
        
        if report.get('should_exclude', False):
            restaurants_excluded.append({
                'code': res_code,
                'reason': report.get('exclude_reason', 'Unknown')
            })
        else:
            restaurants_to_forecast.append(res_code)
    
    logger.info(f"\n📋 ANALYSIS RESULTS:")
    logger.info(f"   Active restaurants: {len(active_restaurants)}")
    if restaurants_closed_permanently:
        logger.info(f"   🔴 Permanently CLOSED: {len(restaurants_closed_permanently)}")
    logger.info(f"   Excluded (gaps/noise): {len(restaurants_excluded)}")
    logger.info(f"   To forecast: {len(restaurants_to_forecast)}")
    
    if restaurants_closed_permanently:
        logger.info(f"\n   🔴 Permanently CLOSED restaurants:")
        for cl in restaurants_closed_permanently:
            logger.info(f"      ❌ {cl['code']} (SAP: {cl['sap_code']}): {cl['reason']}")
    
    if restaurants_excluded:
        logger.info(f"\n   🚫 Excluded restaurants:")
        for ex in restaurants_excluded[:15]:
            logger.info(f"      - {ex['code']}: {ex['reason']}")
        if len(restaurants_excluded) > 15:
            logger.info(f"      ... and {len(restaurants_excluded) - 15} more")
    
    categories = {}
    strategies = {}
    for res_code in restaurants_to_forecast:
        report = analysis_reports.get(res_code, {})
        cat = report.get('category', 'UNKNOWN')
        strat = report.get('strategy', 'UNKNOWN')
        categories[cat] = categories.get(cat, 0) + 1
        strategies[strat] = strategies.get(strat, 0) + 1
    
    logger.info(f"\n   📊 Category Distribution:")
    for cat, count in sorted(categories.items()):
        logger.info(f"      {cat}: {count} restaurants")
    
    logger.info(f"\n   🎯 Strategy Distribution:")
    for strat, count in sorted(strategies.items()):
        w = STRATEGY_WEIGHTS.get(strat, {})
        logger.info(f"      {strat}: {count} restaurants "
                   f"(ML:{w.get('ml', '?')}, AI:{w.get('ai', '?')})")
    return active_restaurants, analysis_reports, restaurants_to_forecast, restaurants_closed_permanently, categories, strategies


def _run_learning_and_updates(df_train, analysis_reports, logger):
    logger.info("\n📂 STEP 4: Updating Master File...")
    df_hist = None
    try:
        df_hist = MasterFileAgent.load_or_create(MASTER_FILE_NAME)
        df_hist = MasterFileAgent.update_actuals(df_hist, df_train)
        save_excel_safely(df_hist, MASTER_FILE_NAME)
    except Exception as e:
        logger.error(f"Error updating master file: {e}")
        traceback.print_exc()
    
    logger.info("\n🧠 STEP 4.5: Brain Learning from Historical Errors...")
    brain_overrides = {}
    try:
        df_for_learning = MasterFileAgent.load_or_create(MASTER_FILE_NAME)
        if not df_for_learning.empty:
            learn_result = ForecastBrain.learn_from_errors(df_for_learning)
            logger.info(f"   Status: {learn_result.get('status', 'unknown')}")
            logger.info(f"   Restaurants learned: {learn_result.get('restaurants_learned', 0)}")
            logger.info(f"   Issues found: {learn_result.get('issues_found', 0)}")
            brain_overrides = ForecastBrain.get_all_strategy_overrides()
            # Memento: Validate pending corrections daily
            try:
                _brain_mem = ForecastBrain.load_memory()
                _rollbacks = CorrectionValidator.validate_all_pending(df_for_learning, _brain_mem)
                if _rollbacks:
                    ForecastBrain.save_memory(_brain_mem)
                    logger.warning(f"   Rolled back corrections for {len(_rollbacks)} restaurants: {_rollbacks[:5]}")
                _vs = CorrectionValidator.get_stats()
                logger.info(f"   Validation stats: {_vs['pending']} pending, {_vs['confirmed']} confirmed, {_vs['rolled_back']} rolled back")
                _audit_path = CorrectionValidator.generate_daily_learning_audit(df_for_learning)
                if _audit_path:
                    logger.info(f"   Daily learning audit: {_audit_path}")
            except Exception as _ve:
                logger.warning(f"CorrectionValidator error: {_ve}")
            if brain_overrides:
                logger.info(f"   Strategy overrides: {len(brain_overrides)} restaurants")
        else:
            logger.info("   No historical data for brain learning.")
    except Exception as e:
        logger.warning(f"Brain learning step failed (non-critical): {e}")
        traceback.print_exc()
    
    transfer_clusters = None
    neural_corrector_ready = False
    try:
        from forecast_system.agents.transfer_learning import TransferLearningAgent
        logger.info(f"\n🔗 STEP 4.7a: Building Restaurant Clusters (Transfer Learning)...")
        transfer_clusters = TransferLearningAgent.build_clusters(df_train, analysis_reports)
    except Exception as e:
        logger.warning(f"Transfer Learning setup failed (non-critical): {e}")
    
    try:
        from forecast_system.agents.neural_corrector import NeuralCorrector
        logger.info(f"\n🧠 STEP 4.7b: Training Neural Corrector...")
        df_for_neural = MasterFileAgent.load_or_create(MASTER_FILE_NAME)
        if not df_for_neural.empty:
            nc_result = NeuralCorrector.train(df_for_neural)
            neural_corrector_ready = nc_result is not None
            if neural_corrector_ready:
                logger.info("   ✅ Neural Corrector trained and ready")
            else:
                logger.info("   ℹ️ Not enough data or R² too low, using rule-based Brain")
    except Exception as e:
        logger.warning(f"Neural Corrector training failed (non-critical): {e}")
    
    # ⭐ V8 Task 2: Train Shift Residual Corrector
    try:
        from forecast_system.agents.shift_residual_corrector import ShiftResidualCorrector
        logger.info(f"\n🔧 STEP 4.7c: Training Shift Residual Corrector...")
        df_for_shift_rc = MasterFileAgent.load_or_create(MASTER_FILE_NAME)
        if not df_for_shift_rc.empty:
            shift_rc = ShiftResidualCorrector()
            rc_result = shift_rc.train(df_for_shift_rc)
            if rc_result:
                logger.info(
                    f"   ✅ Shift Residual Corrector ready "
                    f"(WE improvement: {shift_rc.weekend_evening_improvement:.1f}%)"
                )
            else:
                logger.info("   ℹ️ Shift Residual Corrector not activated (gate conditions not met)")
    except Exception as e:
        logger.warning(f"Shift Residual Corrector training failed (non-critical): {e}")
    
    rag_agent = None
    brain_memory_dict = None
    if USE_RAG_AGENT:
        logger.info(f"\n🧠 STEP 4.6: RAG Knowledge Update (Self-Learning)...")
        try:
            from forecast_system.agents.rag_forecast_agent import RAGForecastAgent
            rag_agent = RAGForecastAgent()
            brain_memory_dict = ForecastBrain.load_memory()
            df_for_rag = MasterFileAgent.load_or_create(MASTER_FILE_NAME)
            rag_agent.update_knowledge(
                brain_memory=brain_memory_dict or {}, # type: ignore[bad-argument-type]
                df_master=df_for_rag,
                df_train=df_train,
            )
            stats = rag_agent.knowledge_stats
            logger.info(f"   📚 Knowledge Store: {stats}")
            logger.info(f"   🤖 AI Backend: {logger_name}")
        except Exception as e:
            logger.warning(f"RAG knowledge update failed (non-critical): {e}")
            traceback.print_exc()
    
    finetune_status = 'SKIPPED'
    try:
        from forecast_system.config.settings import USE_FINETUNED_LLM, FINETUNE_CONFIG
        if USE_FINETUNED_LLM:
            from forecast_system.agents.llm_finetuner import LLMFineTuner, TrainingDataGenerator
            logger.info(f"\n🎯 STEP 4.8: LLM Fine-Tuning Check...")
            needs_retrain = LLMFineTuner.should_retrain(
                max_age_days=int(FINETUNE_CONFIG.get('retrain_interval_days', 7))  # type: ignore[bad-argument-type]
            )
            if needs_retrain and FINETUNE_CONFIG.get('auto_retrain', False):
                logger.info("   🔄 Auto-retraining enabled, generating training data...")
                df_for_ft = MasterFileAgent.load_or_create(MASTER_FILE_NAME)
                training_data = TrainingDataGenerator.generate_from_master_file(
                    df_master=df_for_ft,
                    df_train=df_train,
                    brain_memory=brain_memory_dict or {},
                    lookback_days=int(FINETUNE_CONFIG.get('lookback_days', 60)),  # type: ignore[bad-argument-type]
                    max_pairs=int(FINETUNE_CONFIG.get('max_training_pairs', 5000)),  # type: ignore[bad-argument-type]
                )
                if len(training_data) >= int(FINETUNE_CONFIG.get('min_training_pairs', 100)):  # type: ignore[bad-argument-type]
                    TrainingDataGenerator.save_training_data(training_data)
                    finetuner = LLMFineTuner(config=FINETUNE_CONFIG)
                    ft_result = finetuner.finetune(training_data)
                    finetune_status = ft_result.get('status', 'UNKNOWN')
                    logger.info(f"   Fine-tuning result: {finetune_status}")
                    if finetune_status == 'SUCCESS':
                        logger.info(f"   📈 Training loss: {ft_result.get('training_loss', 'N/A')}")
                        logger.info(f"   💾 Adapter: {ft_result.get('adapter_path', 'N/A')}")
                else:
                    logger.info(f"   Not enough training data: {len(training_data)} "
                                f"(need ≥ {FINETUNE_CONFIG.get('min_training_pairs', 100)})")
                    finetune_status = 'INSUFFICIENT_DATA'
            elif needs_retrain:
                logger.info("   ⚠️ Adapter needs retraining but auto_retrain=false")
                logger.info("   To fine-tune manually, run:")
                logger.info("     python -m forecast_system.agents.llm_finetuner")
                finetune_status = 'NEEDS_RETRAIN'
            else:
                adapter_path = LLMFineTuner.get_latest_adapter()
                if adapter_path:
                    logger.info(f"   ✅ Fine-tuned adapter up-to-date")
                    finetune_status = 'UP_TO_DATE'
                else:
                    logger.info("   ℹ️ No adapter found, using base GGUF model")
                    finetune_status = 'NO_ADAPTER'
        else:
            logger.info(f"\n🎯 STEP 4.8: LLM Fine-Tuning (disabled via USE_FINETUNED_LLM=false)")
    except ImportError as e:
        logger.info(f"\n🎯 STEP 4.8: LLM Fine-Tuning (dependencies not installed: {e})")
        logger.info("   To enable: pip install peft transformers datasets trl accelerate")
    except Exception as e:
        logger.warning(f"LLM Fine-Tuning check failed (non-critical): {e}")
        traceback.print_exc()
        
    return df_hist, brain_overrides, transfer_clusters, neural_corrector_ready, rag_agent, brain_memory_dict, finetune_status

# pyright: ignore[reportGeneralTypeIssues]
def main(forecast_mode='daily'):
    """
    Main pipeline orchestrator.
    
    Args:
        forecast_mode: 'daily' hoặc 'full'
            - 'daily': Chỉ forecast 30 ngày tiếp theo (nhanh, cho job hằng ngày)
            - 'full': Forecast 3 tháng tiếp theo (đầy đủ, chạy khi yêu cầu)
    
    Steps:
    1. Initialize (DB, Logger, Holidays)
    2. Load Data
    3. Analysis (Growth, Gaps, Outliers, Classification)
    4. Update Master File Actuals
    5. Prepare Forecast Parameters
    6. Forecast Loop (Ensemble: ML Stacking + Prophet + AI per restaurant)
    7. Save Results
    8. Summary Report
    """
    import time as _time
    _run_start_time = _time.time()
    
    # ==========================================
    # LOCK FILE: Prevent multiple concurrent runs
    # (Two processes competing for GPU VRAM = extreme slowdown)
    # ==========================================
    import atexit, signal as _signal
    import os as _os
    _lock_file = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), '.forecast_lock')
    _current_pid = _os.getpid()
    
    def _release_lock():
        try:
            if _os.path.exists(_lock_file):
                _os.remove(_lock_file)
        except Exception:
            pass
    
    if _os.path.exists(_lock_file):
        try:
            with open(_lock_file, 'r', encoding='utf-8') as f:
                lock_info = f.read().strip()
            lock_pid = int(lock_info.split('|')[0])
            lock_ts = None
            try:
                lock_ts = datetime.datetime.fromisoformat(lock_info.split('|', 1)[1])
            except Exception:
                lock_ts = None
            # Check if the process is still alive (Windows-compatible)
            import psutil as _psutil
            _lock_process_alive = False
            try:
                if lock_pid == _current_pid:
                    _lock_process_alive = False
                elif _psutil.pid_exists(lock_pid):
                    _p = _psutil.Process(lock_pid)
                    cmdline = " ".join(_p.cmdline()).lower()
                    is_forecast = "forecast_system.main" in cmdline
                    if _p.is_running() and _p.status() != _psutil.STATUS_ZOMBIE and is_forecast:
                        _lock_process_alive = True
            except (_psutil.NoSuchProcess, _psutil.AccessDenied):
                _lock_process_alive = False

            if lock_ts is not None:
                lock_age_hours = (datetime.datetime.now() - lock_ts).total_seconds() / 3600
                if lock_age_hours > 6:
                    _lock_process_alive = False
            
            if _lock_process_alive:
                # Process is still running → abort
                print(
                    f"\n⚠️  FORECAST ALREADY RUNNING (PID {lock_pid})\n"
                    f"   Lock file: {_lock_file}\n"
                    f"   Lock info: {lock_info}\n"
                    f"   To force restart: delete the lock file and retry.\n"
                    f"   Aborting to avoid GPU VRAM contention."
                )
                return
            else:
                # Process is dead → stale lock, clean up
                print(f"🔓 Stale lock found (PID {lock_pid} no longer running). Cleaning up...")
                _os.remove(_lock_file)
        except Exception:
            # Corrupted lock file → remove it
            try:
                _os.remove(_lock_file)
            except Exception:
                pass
    
    # Create lock file with current PID and timestamp
    with open(_lock_file, 'w', encoding='utf-8') as f:
        f.write(f"{_os.getpid()}|{datetime.datetime.now().isoformat()}")
    atexit.register(_release_lock)
    
    # Validate forecast_mode
    if forecast_mode not in ('daily', 'full'):
        forecast_mode = 'daily'
    
    is_full_mode = (forecast_mode == 'full')
    
    # ==========================================
    # STEP 0: INITIALIZATION
    # ==========================================
    logger = setup_logger('forecast_system', log_dir=LOG_DIR)
    
    if is_full_mode:
        forecast_end = get_forecast_end_date()
        mode_label = f"FULL ({FORECAST_MONTHS_AHEAD} tháng → {forecast_end})"
    else:
        forecast_end = CURRENT_DATE + datetime.timedelta(days=DAILY_FORECAST_DAYS)
        mode_label = f"DAILY ({DAILY_FORECAST_DAYS} ngày → {forecast_end})"
    
    logger.info("=" * 60)
    logger.info(f"🚀 AI FORECAST SYSTEM (ENSEMBLE) | Date: {CURRENT_DATE}")
    logger.info(f"   Forecast Start: {FORECAST_START_DATE}")
    logger.info(f"   Forecast End: {forecast_end}")
    logger.info(f"   Forecast Mode: {mode_label}")
    logger.info(f"   Pipeline: Multi-Model Stacking + AI Ensemble")
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
    # STEP 2: LOAD DATA
    # ==========================================
    logger.info("\n📥 STEP 2: Loading Data...")
    
    try:
        df_train = DataAgent.load_recent_data(engine)
        df_info = DataAgent.load_restaurant_info(engine)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        traceback.print_exc()
        return
    
    if df_train.empty:
        logger.error("No training data loaded. Aborting.")
        return
    
    # ── Load Open/Close status (permanent closure trigger) ──
    logger.info("\n📋 Loading Open/Close Status...")
    try:
        open_close_status = DataAgent.load_open_close_status(
            OPEN_CLOSE_FILE, df_info
        )
    except Exception as e:
        logger.warning(f"Open/Close status load failed (non-critical): {e}")
        open_close_status = {'closed_restaurants': {}, 'reopened_restaurants': {}, 'all_statuses': {}}
    
    permanently_closed = set(open_close_status.get('closed_restaurants', {}).keys())

    # ── Filter training data: remove temp-closure windows for ACTIVE restaurants ──
    active_with_closure = open_close_status.get('active_with_closure', {})
    if active_with_closure and not df_train.empty:
        before = len(df_train)
        df_train = df_train.copy()
        df_train['_date_dt'] = pd.to_datetime(df_train['date'], errors='coerce').dt.date
        exclusion_mask = pd.Series(False, index=df_train.index)
        for res_code_cl, cl_info in active_with_closure.items():
            res_mask = df_train['restaurant_code'].astype(str) == str(res_code_cl)
            for (close_start, close_end) in cl_info.get('closure_windows', []):
                window_mask = (
                    res_mask &
                    (df_train['_date_dt'] >= close_start) &
                    (df_train['_date_dt'] < close_end)
                )
                exclusion_mask = exclusion_mask | window_mask
        df_train = df_train[~exclusion_mask].drop(columns=['_date_dt'])
        excluded = before - len(df_train)
        if excluded > 0:
            logger.info(
                f"   🟡 Excluded {excluded:,} training rows from temp-closure windows "
                f"({len(active_with_closure)} restaurants)"
            )


    # ==========================================
    # STEP 3: ANALYSIS
    # ==========================================
    active_restaurants, analysis_reports, restaurants_to_forecast, restaurants_closed_permanently, categories, strategies = _analyze_active_restaurants(
        df_train, open_close_status, permanently_closed, logger
    )
    if len(active_restaurants) == 0:
        return
    
    (
        df_hist,
        brain_overrides, 
        transfer_clusters, 
        neural_corrector_ready, 
        rag_agent, 
        brain_memory_dict, 
        finetune_status
    ) = _run_learning_and_updates(df_train, analysis_reports, logger)
    
    # ==========================================
    # STEP 5: PREPARE FORECAST
    # ==========================================
    logger.info("\n🗓️ STEP 5: Preparing Forecast Parameters...")
    
    # Calculate forecast end date based on mode
    # forecast_end đã được tính ở STEP 0
    
    # Ensure vn_holidays covers all years in forecast range
    forecast_years = list(set([
        FORECAST_START_DATE.year,
        CURRENT_DATE.year,
        forecast_end.year,
        forecast_end.year + 1,  # buffer for post-holiday detection
    ]))
    vn_holidays = get_vn_holidays(forecast_years)
    
    # Calculate total days from FORECAST_START_DATE to forecast_end (inclusive)
    days_from_start = (forecast_end - FORECAST_START_DATE).days + 1
    if days_from_start < 1:
        # Fallback: at least forecast from today to forecast_end
        days_from_start = (forecast_end - CURRENT_DATE).days + 1
        all_forecast_days = build_forecast_days(CURRENT_DATE, days_from_start, vn_holidays)
    else:
        all_forecast_days = build_forecast_days(FORECAST_START_DATE, days_from_start, vn_holidays)
    
    # ── FILTER: Chỉ forecast từ hôm nay trở đi, KHÔNG chạy lại ngày đã qua ──
    next_days = [d for d in all_forecast_days if d['date'] >= CURRENT_DATE]
    skipped_past_days = len(all_forecast_days) - len(next_days)
    
    if skipped_past_days > 0:
        logger.info(f"   ⏭️ Skipped {skipped_past_days} past days (before {CURRENT_DATE}) — data preserved in Master File")
    
    # ── SPLIT: Short-term (hourly+shift) vs Long-term (daily-only) ──
    # Luôn định nghĩa short_term_cutoff (dùng trong forecast loop per-restaurant)
    short_term_cutoff = CURRENT_DATE + datetime.timedelta(days=SHORT_HORIZON_DAYS)
    
    if is_full_mode:
        # FULL MODE: 30 ngày đầu hourly + phần còn lại daily-only
        short_term_days = [d for d in next_days if d['date'] < short_term_cutoff]
        long_term_days = [d for d in next_days if d['date'] >= short_term_cutoff]
        
        logger.info(f"   📅 Forecast mode: FULL (DUAL-HORIZON)")
        logger.info(f"   ┌─ SHORT-TERM ({SHORT_HORIZON_DAYS} ngày đầu): {len(short_term_days)} ngày → theo ngày + giờ + ca")
        if short_term_days:
            logger.info(f"   │  Period: {short_term_days[0]['date']} → {short_term_days[-1]['date']}")
        logger.info(f"   └─ LONG-TERM ({FORECAST_MONTHS_AHEAD} tháng): {len(long_term_days)} ngày → theo nhà hàng + ngày (daily total)")
        if long_term_days:
            logger.info(f"      Period: {long_term_days[0]['date']} → {long_term_days[-1]['date']}")
    else:
        # DAILY MODE: Chỉ forecast short-term (30 ngày), KHÔNG có long-term
        short_term_days = next_days  # Tất cả đều là short-term
        long_term_days = []          # Không có long-term
        
        logger.info(f"   📅 Forecast mode: DAILY ({DAILY_FORECAST_DAYS} ngày)")
        logger.info(f"   ─── {len(short_term_days)} ngày → theo ngày + giờ + ca")
        if short_term_days:
            logger.info(f"       Period: {short_term_days[0]['date']} → {short_term_days[-1]['date']}")
        logger.info(f"   ℹ️  Để forecast 3 tháng đầy đủ, chạy: python -m forecast_system.main --mode full")
    
    logger.info(f"   Total forecast days: {len(next_days)} (active, from today onward)")
    
    # Log holiday types found in forecast period
    holiday_days = [d for d in next_days if d['is_holiday']]
    pre_holiday_days = [d for d in next_days if d.get('is_pre_holiday')]
    post_holiday_days = [d for d in next_days if d.get('is_post_holiday')]
    
    if holiday_days:
        logger.info(f"   🎌 Holidays in forecast period:")
        for d in holiday_days:
            logger.info(
                f"      {d['date']} ({d['weekday']}): "
                f"{d.get('holiday_name', 'Holiday')} [{d.get('holiday_type', '?')}] "
                f"impact={d.get('holiday_impact', 1.0):.0%}"
                f"{' ⚠️CLOSED_LIKELY' if d.get('closed_likely') else ''}"
            )
    if pre_holiday_days:
        logger.info(f"   📅 Pre-holiday days: {len(pre_holiday_days)}")
    if post_holiday_days:
        logger.info(f"   📅 Post-holiday days: {len(post_holiday_days)}")
    
    # Log special events in forecast period
    special_event_days = [d for d in next_days if d.get('is_special_event')]
    if special_event_days:
        logger.info(f"   🎉 Special Events in forecast period:")
        for d in special_event_days:
            logger.info(
                f"      {d['date']} ({d['weekday']}): "
                f"{d.get('event_name', 'Event')} [{d.get('event_type', '?')}] "
                f"impact={d.get('event_impact', 1.0):.0%}"
            )
    
    # ==========================================
    # STEP 5.5: DETECT HOLIDAY CLOSURES + LOAD HISTORICAL DATA
    # ==========================================
    from forecast_system.utils.date_utils import (
        detect_holiday_closures, get_holiday_periods, HOLIDAY_TYPES
    )
    
    holiday_closures = {}  # {holiday_type: set of res_codes}
    closed_likely_types = set(
        d.get('holiday_type') for d in next_days
        if d.get('closed_likely') and d.get('holiday_type')
    )
    
    # [FIX #3] Also collect holiday types that need historical calibration data
    # (LIBERATION_DAY 30/4, LABOR_DAY 1/5, NATIONAL_DAY 2/9 have needs_historical_calibration=True)
    calibration_needed_types = set(
        d.get('holiday_type') for d in next_days
        if d.get('holiday_type') and
        HOLIDAY_TYPES.get(d.get('holiday_type'), {}).get('needs_historical_calibration', False)
    )
    
    # Combine: load historical data for closure detection AND calibration needs
    all_types_needing_history = closed_likely_types | calibration_needed_types
    
    df_closure = None
    if all_types_needing_history and not df_train.empty:
        logger.info(f"\n🏪 STEP 5.5: Detecting Holiday Closures + Loading Historical Data...")
        if closed_likely_types:
            logger.info(f"   Closed-likely types: {closed_likely_types}")
        if calibration_needed_types:
            logger.info(f"   [FIX #3] Calibration-needed types: {calibration_needed_types} (30/4, 1/5, ...)")
        
        try:
            # === KEY FIX: df_train chỉ có 120 ngày gần đây ===
            # Dịp Tết 2025 (28/01/2025) cách nay ~375 ngày → KHÔNG CÓ trong df_train!
            # Cần load thêm data từ các dịp lễ năm trước
            
            # Extend vn_holidays to cover prior year
            prior_year = CURRENT_DATE.year - 1
            vn_holidays_extended = get_vn_holidays([
                prior_year, CURRENT_DATE.year, CURRENT_DATE.year + 1
            ])
            
            df_closure = df_train
            df_dates = set(pd.to_datetime(df_train['date'], errors='coerce').dt.date)
            
            # Find ALL prior-year holiday periods that match closed_likely types
            prior_periods = get_holiday_periods(prior_year, vn_holidays_extended)
            missing_periods = []
            
            for period in prior_periods:
                # [FIX #3] Check if this holiday type is in either closed_likely OR calibration_needed
                if period['type'] in all_types_needing_history:
                    # Check if data for this period is in df_train
                    if period['start'] not in df_dates and period['end'] not in df_dates:
                        missing_periods.append(period)
                        logger.info(
                            f"   ⚠️ {period['type']} {prior_year} "
                            f"({period['start']} → {period['end']}) not in df_train — will load"
                        )
                    else:
                        logger.info(
                            f"   ✅ {period['type']} {prior_year} already in df_train"
                        )
            
            # Load data for missing periods
            if missing_periods:
                logger.info(f"   📥 Loading historical data for {len(missing_periods)} holiday period(s)...")
                
                # Calculate date ranges to load (with buffer for normal avg comparison)
                # Merge overlapping ranges for efficiency
                load_ranges = []
                for p in missing_periods:
                    load_start = p['start'] - datetime.timedelta(days=21)
                    load_end = p['end'] + datetime.timedelta(days=14)
                    load_ranges.append((load_start, load_end))
                
                # Merge overlapping ranges
                load_ranges.sort()
                merged = [load_ranges[0]]
                for start, end in load_ranges[1:]:
                    if start <= merged[-1][1] + datetime.timedelta(days=1):
                        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                    else:
                        merged.append((start, end))
                
                extra_dfs = []
                for load_start, load_end in merged:
                    try:
                        df_extra = DataAgent.load_date_range(
                            engine, load_start, load_end
                        )
                        if not df_extra.empty:
                            extra_dfs.append(df_extra)
                            logger.info(
                                f"   ✅ Loaded {len(df_extra):,} rows "
                                f"({load_start} → {load_end})"
                            )
                        else:
                            logger.warning(
                                f"   No data found for {load_start} → {load_end}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"   Could not load {load_start} → {load_end}: {e}"
                        )
                
                if extra_dfs:
                    df_closure = pd.concat(
                        [df_train] + extra_dfs, ignore_index=True
                    )
                    df_closure = df_closure.drop_duplicates(
                        subset=['restaurant_code', 'date', 'hour'], keep='first'
                    )
                    logger.info(
                        f"   📊 Combined dataset: {len(df_closure):,} rows "
                        f"(train: {len(df_train):,} + historical: "
                        f"{sum(len(d) for d in extra_dfs):,})"
                    )
            
            holiday_closures = detect_holiday_closures(
                df_closure, vn_holidays_extended,
                threshold_ratio=0.10,  # < 10% of normal = closed
            )
            
            # Log results
            total_closed = 0
            for h_type, closed_set in holiday_closures.items():
                if closed_set:
                    total_closed += len(closed_set)
                    logger.info(
                        f"   {h_type}: {len(closed_set)} restaurants detected as closed"
                    )
            
            if total_closed == 0:
                logger.info("   No historical closures detected")
                
        except Exception as e:
            logger.warning(f"Holiday closure detection failed (non-critical): {e}")
            traceback.print_exc()
    
    # ==========================================
    # STEP 5.6: LOAD ACTUAL LUNAR NY CLOSURE SCHEDULE
    # ==========================================
    # Override pattern-based detection with ACTUAL closure data from operations
    lunar_ny_closures = {}  # {res_code: {date: 'CLOSED'|'HALF_DAY'}}
    
    try:
        import os
        if os.path.exists(LUNAR_NY_CLOSURE_FILE):
            logger.info(f"\n📅 STEP 5.6: Loading Lunar NY Closure Schedule...")
            lunar_ny_closures = DataAgent.load_lunar_ny_closures(
                LUNAR_NY_CLOSURE_FILE, df_info
            )
            
            if lunar_ny_closures:
                # Log per-date summary
                from collections import Counter
                date_summary = Counter()
                half_summary = Counter()
                for res_closures in lunar_ny_closures.values():
                    for d, status in res_closures.items():
                        if status == 'CLOSED':
                            date_summary[d] += 1
                        elif status == 'HALF_DAY':
                            half_summary[d] += 1
                
                for d in sorted(set(list(date_summary.keys()) + list(half_summary.keys()))):
                    closed_n = date_summary.get(d, 0)
                    half_n = half_summary.get(d, 0)
                    wd = d.strftime('%a')
                    logger.info(
                        f"   {d} ({wd}): "
                        f"{closed_n} đóng cửa, {half_n} bán nửa ngày"
                    )
        else:
            logger.info(f"   No closure file found at {LUNAR_NY_CLOSURE_FILE}")
    except Exception as e:
        logger.warning(f"Loading closure schedule failed (non-critical): {e}")
        traceback.print_exc()
    
    # ==========================================
    # STEP 5.7: HOLIDAY IMPACT CALIBRATION (DATA-DRIVEN)
    # ==========================================
    calibration = None
    try:
        from forecast_system.agents.holiday_calibrator import HolidayCalibrator
        from forecast_system.utils.date_utils import invalidate_calibration_cache
        
        # [FIX] Run calibration if ANY holiday types need it — not just closed-likely ones
        # Previously: only ran when `closed_likely_types` (TET etc) was non-empty
        # Now: also runs when calibration_needed_types (30/4, 1/5, 2/9) is non-empty
        types_needing_calibration = closed_likely_types | calibration_needed_types
        
        if types_needing_calibration and not df_train.empty:
            logger.info(f"\n📐 STEP 5.7: Calibrating Holiday Impact Factors...")
            logger.info(f"   Types: {types_needing_calibration}")
            logger.info(f"   Using historical data to replace hardcoded impact factors")
            
            # Use df_closure which includes historical holiday data from Step 5.5
            try:
                cal_data = df_closure if df_closure is not None and not df_closure.empty else df_train
            except NameError:
                cal_data = df_train
            
            # Extend holidays to cover prior years for calibration
            cal_years = list(set(
                [CURRENT_DATE.year - 1, CURRENT_DATE.year, CURRENT_DATE.year + 1]
            ))
            cal_holidays = get_vn_holidays(cal_years)
            
            calibration = HolidayCalibrator.calibrate(
                df_data=cal_data,
                vn_holidays=cal_holidays,
                engine=engine,
            )
            
            if calibration and calibration.get('holiday_types'):
                # Invalidate cache so date_utils picks up new calibration
                invalidate_calibration_cache()
                
                # Update next_days with calibrated impacts
                next_days = HolidayCalibrator.apply_to_forecast_days(
                    next_days, calibration
                )
                
                # Log comparison: old vs new impacts
                calibrated_days = [
                    d for d in next_days
                    if d.get('holiday_impact_source') == 'calibrated'
                ]
                if calibrated_days:
                    logger.info(f"\n   📊 Calibrated Impact vs Default:")
                    for d in calibrated_days[:20]:
                        old = d.get('holiday_impact_default', d.get('holiday_impact'))
                        new = d.get('holiday_impact')
                        diff_pct = (new - old) * 100 if old != new else 0
                        logger.info(
                            f"      {d['date']} ({d['weekday'][:3]}) "
                            f"[{d.get('pre_post_type', d.get('holiday_type', ''))}]: "
                            f"{old:.3f} → {new:.3f} "
                            f"({'↑' if diff_pct > 0 else '↓'}{abs(diff_pct):+.1f}pp)"
                        )
            else:
                logger.info("   No holiday periods found in data for calibration")
        else:
            logger.info(f"\n📐 STEP 5.7: Skipped (no holidays needing calibration in forecast period)")
    except Exception as e:
        logger.warning(f"Holiday calibration failed (non-critical): {e}")
        traceback.print_exc()
    
    # ==========================================
    # STEP 5.8: SPECIAL EVENT CALIBRATION (DATA-DRIVEN)
    # ==========================================
    event_calibration = None
    try:
        from forecast_system.agents.event_calibrator import EventCalibrator
        
        if not df_train.empty:
            logger.info(f"\n🎉 STEP 5.8: Calibrating Special Event Impact Factors...")
            logger.info(f"   Using historical data to replace estimated event impacts")
            
            event_calibration = EventCalibrator.calibrate(
                df_data=df_train,
                current_date=CURRENT_DATE,
            )
            
            if event_calibration and event_calibration.get('n_events_calibrated', 0) > 0:
                # NOTE: DO NOT apply aggregate calibration to shared `next_days` here!
                # Per-restaurant calibration is now applied inside the forecast loop
                # (each restaurant gets its own copy + its own calibration).
                
                # Log aggregate comparison
                for evt_type, cal_info in event_calibration.get('events', {}).items():
                    logger.info(
                        f"   {evt_type}: "
                        f"default={cal_info['default_factor']:.2f} → "
                        f"calibrated={cal_info['calibrated_factor']:.2f} "
                        f"(n={cal_info['n_restaurants']}, "
                        f"confidence={cal_info['confidence']})"
                    )
                
                n_per_res = event_calibration.get('n_restaurants_calibrated', 0)
                if n_per_res > 0:
                    logger.info(
                        f"   📊 Per-restaurant calibration available for {n_per_res} restaurants"
                    )
            else:
                logger.info("   No special events found in historical data for calibration")
    except Exception as e:
        logger.warning(f"Event calibration failed (non-critical): {e}")
        traceback.print_exc()
    
    # ==========================================
    # STEP 5.9: NEURALPROPHET GLOBAL MODEL (Train Once)
    # ==========================================
    neuralprophet_global_model = None
    
    try:
        from forecast_system.agents.neuralprophet_agent import NeuralProphetAgent, HAS_NEURALPROPHET
        
        if HAS_NEURALPROPHET and not df_train.empty:
            logger.info(f"\n🧠 STEP 5.9: Training Global NeuralProphet Model (once for ALL restaurants)...")
            logger.info(f"   ⚡ This replaces per-restaurant training (was causing 9h+ hangs)")
            
            # Prepare restaurant data dict: {res_code: df_for_that_restaurant}
            all_restaurant_data = {}
            for res_code in restaurants_to_forecast:
                df_res = df_train[df_train['restaurant_code'] == res_code]
                if not df_res.empty:
                    all_restaurant_data[str(res_code)] = df_res
            
            if all_restaurant_data:
                neuralprophet_global_model = NeuralProphetAgent.train_global_model_safe(
                    all_restaurant_data, vn_holidays
                )
                
                if neuralprophet_global_model is not None:
                    logger.info(f"   ✅ Global NeuralProphet ready ({len(all_restaurant_data)} restaurants)")
                else:
                    logger.warning(f"   ⚠️ Global NeuralProphet failed/timeout — will use per-restaurant fallback with timeout")
        else:
            logger.info(f"\n🧠 STEP 5.9: NeuralProphet skipped (not available or no data)")
    except ImportError:
        logger.info(f"\n🧠 STEP 5.9: NeuralProphet not installed — skipping")
    except Exception as e:
        logger.warning(f"NeuralProphet global training failed (non-critical): {e}")
        traceback.print_exc()
    # ==========================================
    # STEP 5.95: ⭐ V8 LOAD BOOKING DATA (Pre-Forecast)
    # ==========================================
    # Moved from Step 6.5 → here so booking features are available to ML models
    df_booking_summary = pd.DataFrame()
    df_booking_daily = pd.DataFrame()
    booking_lookup = {}  # (restaurant_code, date_str) → guest_count
    try:
        logger.info(f"\n🎫 STEP 5.95: Loading Booking Data (pre-forecast integration)...")
        logger.info(f"   Source: v_fact_db_booking_booking_info")
        logger.info(f"   Period: {CURRENT_DATE} → {forecast_end}")
        
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
            
            df_booking_daily = BookingAgent.get_daily_booking_totals(df_booking_raw)
            if not df_booking_daily.empty:
                # Build lookup for fast feature injection
                for _, row in df_booking_daily.iterrows():
                    key = (str(row['Restaurant_Code']), str(row['Date']))
                    bk_val = row['Booking_Guests_Total'] if 'Booking_Guests_Total' in row.index else 0
                    booking_lookup[key] = int(bk_val)  # type: ignore[reportArgumentType]
                logger.info(
                    f"   📊 Booking lookup built: {len(booking_lookup):,} entries "
                    f"(will inject into ML features)"
                )
        else:
            logger.info("   No future booking data found")
    except Exception as e:
        logger.warning(f"Booking data loading failed (non-critical): {e}")
        traceback.print_exc()
    
    # ==========================================
    # STEP 6: ENSEMBLE FORECAST LOOP
    # ==========================================
    new_rows = []
    new_rows_daily_only = []  # Separate list for daily-only (long-term) results
    forecast_stats = {
        'total': len(restaurants_to_forecast),
        'success_ensemble': 0,
        'success_daily_only': 0,
        'success_ai': 0,
        'success_prophet': 0,
        'failed': 0,
        'skipped_closed': 0,
        'skipped_half_day': 0,
        'permanently_closed': len(restaurants_closed_permanently),
        'success_new_restaurant': 0,
        'success_future_opening': 0,   # [FIX #4b] Track future-opening forecasts
        'errors': [],
        'models_usage': {},       # Track which models were used
        'avg_confidence': 0,
        'confidence_scores': [],
    }
    
    # ──────────────────────────────────────────────────────────
    # STEP 6.0: [FIX #4b] FORECAST NHÀ HÀNG SẮP KHAI TRƯƠNG
    # ──────────────────────────────────────────────────────────
    # Đọc future_open_restaurants từ Open_Close.xlsx
    # → Nếu opening_date trong forecast horizon → forecast từ ngày đó
    # → Dùng NewRestaurantAgent chain-average blend (không có data riêng)
    future_open_restaurants = open_close_status.get('future_open_restaurants', {})

    # ── [FIX] Reconcile: nếu nhà hàng đã có data thực trong df_train
    # (đã chuyển ACTIVE và có số thực tế) → KHÔNG dùng chain-blend,
    # thay vào đó đưa vào restaurants_to_forecast để chạy ensemble chuẩn.
    if future_open_restaurants and not df_train.empty:
        train_codes_with_data = set(df_train['restaurant_code'].astype(str).unique())
        graduated = {}   # Các nhà hàng đã tốt nghiệp khỏi future-open
        still_future = {}

        for key, fo_info in future_open_restaurants.items():
            res_code_fo = str(fo_info.get('res_code') or key)
            # Kiểm tra: đã có data thực (≥1 ngày) trong df_train?
            has_real_data = res_code_fo in train_codes_with_data
            if not has_real_data:
                # Thử thêm bằng code_budget / key
                has_real_data = str(key) in train_codes_with_data

            if has_real_data:
                graduated[key] = fo_info
                # Đảm bảo restaurant này nằm trong forecast list chính
                if res_code_fo not in restaurants_to_forecast:
                    restaurants_to_forecast.append(res_code_fo)
                    logger.info(
                        f"   ✅ [FIX] {res_code_fo} đã có data thực → "
                        f"chuyển sang Ensemble Pipeline (bỏ future-open mode)"
                    )
            else:
                still_future[key] = fo_info

        if graduated:
            logger.info(
                f"   🔄 {len(graduated)} nhà hàng đã ACTIVE với data thực "
                f"→ chuyển sang Ensemble Pipeline: "
                f"{[str(v.get('res_code') or k) for k, v in graduated.items()]}"
            )
        future_open_restaurants = still_future  # Chỉ giữ lại nhà hàng thực sự chưa mở

    if future_open_restaurants:
        forecast_horizon_end = forecast_end  # Ngày cuối của forecast window
        
        logger.info(
            f"\n🆕 STEP 6.0: Forecasting {len(future_open_restaurants)} "
            f"future-opening restaurants..."
        )

        
        for key, fo_info in future_open_restaurants.items():
            try:
                opening_date = fo_info['opening_date']
                store_name = fo_info.get('store_name', fo_info.get('code_report', key))
                brand = fo_info.get('brand', '')
                code_report = fo_info.get('code_report', key)
                
                # Chỉ forecast nếu opening_date trong forecast horizon
                if opening_date > forecast_horizon_end:
                    logger.debug(
                        f"   ⏭️ {code_report} opens {opening_date} > "
                        f"forecast horizon {forecast_horizon_end} → skip"
                    )
                    continue
                
                # Các ngày forecast CHỈ từ opening_date trở đi
                active_fo_days = [
                    d for d in next_days
                    if d['date'] >= opening_date
                ]
                
                if not active_fo_days:
                    continue
                
                logger.info(
                    f"   📅 {code_report} [{brand}] '{store_name[:35]}' "
                    f"opens {opening_date} → forecasting {len(active_fo_days)} days"
                )
                
                # Dùng code_report làm res_code (chưa có DB code)
                fo_res_code = fo_info.get('res_code') or code_report
                fo_res_name = f"{brand} {store_name}".strip()
                
                # Không có data riêng → df_res rỗng hoàn toàn
                df_fo_res = pd.DataFrame()
                
                nr_predictions, nr_meta = NewRestaurantAgent.generate_new_restaurant_forecast(
                    res_code=fo_res_code,
                    res_name=fo_res_name,
                    df_res=df_fo_res,
                    df_train=df_train,
                    df_info=df_info,
                    next_days=active_fo_days,
                    vn_holidays=vn_holidays,
                    brand_code=brand,   # [FIX #4] Pass brand code (SS, WY, TCH...) for chain lookup
                )
                
                if nr_predictions:
                    forecast_stats['success_future_opening'] += 1
                    forecast_stats['success_new_restaurant'] += 1
                    
                    for p in nr_predictions:
                        # [FIX] Map hour → shift (Phase 8 alignment, same as New Restaurant block)
                        _fo_hour = p.get('hour')
                        if _fo_hour is not None:
                            from forecast_system.agents.data_agent import DataAgent as _DA
                            _fo_shift = _DA.map_to_shift(int(_fo_hour))
                            _fo_shift = _fo_shift if _fo_shift != 'OTHER' else None
                        else:
                            _fo_shift = p.get('shift')
                        
                        new_rows.append({
                            'Forecast_Run_Date': CURRENT_DATE,
                            'Restaurant_Code': str(fo_res_code),
                            'Date': p['date'],
                            'Weekday': p['weekday'],
                            'Hour': _fo_hour,
                            'Shift': _fo_shift,  # [FIX] MORNING/EVENING thay vì None
                            'Final_Predicted_Guests': p['predicted_guests'],
                            'Actual_Guest': np.nan,
                            'Diff_Guest': np.nan,
                            'Error_%': np.nan,
                            'Is_Holiday': any(
                                d['date'] == p['date'] and d.get('is_holiday')
                                for d in active_fo_days
                            ),
                            'Is_Veg': False,
                            'AI_Raw_Daily_Forecast': np.nan,
                            'AI_Forecast_Available': False,
                            'Forecast_Mode': 'new_restaurant',
                            'Chain': nr_meta.get('chain', ''),
                            'Opening_Date': opening_date,
                        })
                    
                    logger.info(
                        f"      ✅ {fo_res_code}: {sum(p['predicted_guests'] for p in nr_predictions)} "
                        f"guests over {len(active_fo_days)}d "
                        f"(chain={nr_meta.get('chain')}, siblings={nr_meta.get('siblings_found', 0)})"
                    )
                else:
                    logger.warning(
                        f"   ⚠️ {code_report}: No forecast generated "
                        f"(chain not detected or no siblings)"
                    )
            
            except Exception as e:
                logger.warning(f"   Future opening forecast failed for {key}: {e}")
        
        if forecast_stats['success_future_opening'] > 0:
            logger.info(
                f"\n   ✅ Future opening forecasts generated: "
                f"{forecast_stats['success_future_opening']} restaurants"
            )

    if is_full_mode:
        logger.info(f"\n🧠 STEP 6: Running FULL DUAL-HORIZON ENSEMBLE Forecast for "
                   f"{len(restaurants_to_forecast)} restaurants...")
        logger.info(f"   Pipeline: ML Stacking (XGB+CAT+LGBM+RF) + Prophet + AI (LLM)")
        logger.info(f"   Mode A (Short-term {SHORT_HORIZON_DAYS}d): hourly + shift forecast")
        logger.info(f"   Mode B (Long-term {FORECAST_MONTHS_AHEAD}mo): daily-only forecast")
    else:
        logger.info(f"\n🧠 STEP 6: Running DAILY ENSEMBLE Forecast for "
                   f"{len(restaurants_to_forecast)} restaurants...")
        logger.info(f"   Pipeline: ML Stacking (XGB+CAT+LGBM+RF) + Prophet + AI (LLM)")
        logger.info(f"   Forecast: {DAILY_FORECAST_DAYS} ngày tiếp theo (hourly + shift)")
    
    # ── PARALLEL PROCESSING OPTION ──
    use_parallel = PARALLEL_CONFIG.get('max_workers', 0) != 0 and len(restaurants_to_forecast) > 10
    parallel_success = False
    
    if use_parallel:
        try:
            from forecast_system.parallel_engine import (
                ParallelForecastEngine, RestaurantTask
            )
            logger.info(f"   ⚡ Parallel mode enabled")
            
            engine_p = ParallelForecastEngine()
            n_workers = engine_p.get_optimal_workers()
            logger.info(f"   Workers: {n_workers}")
            
            # Create tasks
            tasks = ParallelForecastEngine.create_tasks_from_data(
                df_train=df_train,
                restaurant_list=df_info,
                analysis_reports=analysis_reports,
                next_days_info=next_days,
                vn_holidays=vn_holidays,
            )
            
            if tasks:
                results = engine_p.run_parallel(
                    tasks,
                    max_workers=n_workers,
                    use_threads=bool(PARALLEL_CONFIG.get('use_threads', True)),  # type: ignore[bad-argument-type]
                )
                
                # Aggregate results
                for r in results:
                    if r.success and r.predictions:
                        new_rows.extend(r.predictions)
                        forecast_stats['success_ensemble'] += 1
                        forecast_stats['confidence_scores'].append(r.confidence)
                    else:
                        forecast_stats['failed'] += 1
                        if r.error:
                            forecast_stats['errors'].append(f"{r.res_code}: {r.error}")
                
                parallel_success = True
                logger.info(f"   ⚡ Parallel complete: {forecast_stats['success_ensemble']} success, "
                           f"{forecast_stats['failed']} failed")
        except Exception as e:
            logger.warning(f"   Parallel processing failed, falling back to sequential: {e}")
            traceback.print_exc()
    
    if not parallel_success:
        # ── SEQUENTIAL PROCESSING (fallback or default) ──
        for i, res_code in enumerate(tqdm(restaurants_to_forecast, desc="Ensemble Forecasting")):
            try:
                # Get analysis report
                report = analysis_reports.get(res_code, {})
                strategy = report.get('strategy', 'ENSEMBLE_EQUAL')
                
                # 🧠 Brain strategy override
                if res_code in brain_overrides:
                    brain_strategy = brain_overrides[res_code]
                    if brain_strategy != strategy:
                        logger.debug(
                            f"🧠 {res_code}: Brain override {strategy} → {brain_strategy}"
                        )
                        strategy = brain_strategy
                        report['strategy'] = strategy  # Update for downstream
                
                weights = STRATEGY_WEIGHTS.get(strategy, {'ml': 0.5, 'ai': 0.5})
                
                # 6.0 Holiday Closure Check
                # ═══════════════════════════════════════
                # Priority 1: Actual closure schedule (from Excel file)
                # Priority 2: Pattern-based detection (historical data)
                # ═══════════════════════════════════════
                
                closed_dates = set()      # Dates with forecast = 0
                half_day_dates = set()    # Dates with forecast × 0.5
                
                # --- Priority 0: Future permanent closure (Open_Close.xlsx) ---
                res_code_str = str(res_code)
                res_oc_status = open_close_status.get('all_statuses', {}).get(res_code_str, {})
                future_close_date = res_oc_status.get('future_close_date')
                
                if future_close_date:
                    for d in next_days:
                        if d['date'] >= future_close_date:
                            closed_dates.add(d['date'])
                
                # --- Priority 1: Actual closure schedule ---
                res_lunar_closures = lunar_ny_closures.get(res_code_str, {})
                
                if res_lunar_closures:
                    for d_date, status in res_lunar_closures.items():
                        if status == 'CLOSED':
                            closed_dates.add(d_date)
                        elif status == 'HALF_DAY':
                            half_day_dates.add(d_date)
                
                # --- Priority 2: Pattern-based detection (only for dates NOT in closure file) ---
                closure_file_dates = set(res_lunar_closures.keys()) if res_lunar_closures else set()
                for h_type, closed_set in holiday_closures.items():
                    if res_code_str in closed_set:
                        for d in next_days:
                            if (d.get('holiday_type') == h_type and 
                                d['date'] not in closure_file_dates):
                                # Only use pattern-based if closure file doesn't cover this date
                                closed_dates.add(d['date'])
                
                # --- ⭐ v7 Priority 3: Historical closure rate-based force close ---
                # If restaurant was closed >80% of times for this holiday → force close
                # Prevents massive forecast errors for restaurants that consistently close
                if res_code in analysis_reports:
                    res_profile = analysis_reports[res_code].get('profile', {})
                    holiday_closure_history = res_profile.get('holiday_closure_rates', {})
                    
                    for d in next_days:
                        h_type = d.get('holiday_type')
                        if h_type and d['date'] not in closed_dates and d['date'] not in closure_file_dates:
                            closure_rate = holiday_closure_history.get(h_type, 0.0)
                            if closure_rate > HOLIDAY_CLOSURE_RATE_THRESHOLD:
                                closed_dates.add(d['date'])
                                logger.info(
                                    f"🔒 v7: Force close {res_code_str} on {d['date']} "
                                    f"({h_type}): historical closure rate={closure_rate:.0%} "
                                    f"> {HOLIDAY_CLOSURE_RATE_THRESHOLD:.0%}"
                                )
                
                # Generate rows for closed dates (forecast = 0)
                has_closure = bool(closed_dates or half_day_dates)
                if closed_dates:
                    forecast_stats['skipped_closed'] += 1
                    for d_info in next_days:
                        if d_info['date'] in closed_dates:
                            # Phase 8: Generate 2 shift rows per closed day (not 15 hourly)
                            for shift_key in ['MORNING', 'EVENING']:
                                new_rows.append({
                                    'Forecast_Run_Date': CURRENT_DATE,
                                    'Restaurant_Code': res_code_str,
                                    'Date': d_info['date'],
                                    'Weekday': d_info['weekday'],
                                    'Hour': None,  # Shift-based, no hour
                                    'Shift': shift_key,
                                    'Final_Predicted_Guests': 0,
                                    'Actual_Guest': np.nan,
                                    'Diff_Guest': np.nan,
                                    'Error_%': np.nan,
                                    'Is_Holiday': True,
                                    'Is_Veg': d_info.get('is_veg', False),
                                    'AI_Raw_Daily_Forecast': np.nan,
                                    'AI_Forecast_Available': False,
                                })
                
                # Filter: exclude CLOSED dates, keep HALF_DAY (will be adjusted later)
                active_next_days = [
                    copy.deepcopy(d) for d in next_days 
                    if d['date'] not in closed_dates
                ]
                if not active_next_days:
                    # All forecast days are closed → skip this restaurant entirely
                    forecast_stats['success_ensemble'] += 1
                    continue
                
                # Apply per-restaurant event calibration (priority: per-res > aggregate > default)
                if event_calibration:
                    from forecast_system.agents.event_calibrator import EventCalibrator
                    active_next_days = EventCalibrator.apply_calibration_to_forecast_days(
                        active_next_days, event_calibration, res_code=str(res_code)
                    )
                
                # Split active days into short-term (hourly) and long-term (daily-only)
                active_short_term = [d for d in active_next_days if d['date'] < short_term_cutoff]
                active_long_term = [d for d in active_next_days if d['date'] >= short_term_cutoff]
                
                # Get restaurant data
                df_res = pd.DataFrame(df_train[df_train['restaurant_code'] == res_code])
                
                # ───────────────────────────────────────
                # 6.0.5 NEW RESTAURANT CHECK (Chain-Average Blend)
                # ───────────────────────────────────────
                category = report.get('category', 'STANDARD')
                active_days = len(df_res['date'].unique()) if not df_res.empty else 0
                new_threshold = ANALYSIS_CONFIG.get('new_restaurant_forecast_days', 14)
                
                if category == 'NEW' or active_days < new_threshold:
                    # Nhà hàng mới → dùng chain-average blend
                    res_name = ''
                    if not df_info.empty:
                        info_row = df_info[df_info['merge_key'] == str(res_code)]
                        if not info_row.empty:
                            res_name = str(info_row.iloc[0].get('restaurant_name', ''))
                    
                    nr_predictions, nr_meta = NewRestaurantAgent.generate_new_restaurant_forecast(
                        res_code=res_code,
                        res_name=res_name,
                        df_res=df_res,
                        df_train=df_train,
                        df_info=df_info,
                        next_days=active_next_days,
                        vn_holidays=vn_holidays,
                    )
                    
                    if nr_predictions:
                        forecast_stats['success_new_restaurant'] += 1
                        
                        for p in nr_predictions:
                            # [FIX] Map hour → shift (Phase 8 alignment)
                            _hour = p.get('hour')
                            if _hour is not None:
                                from forecast_system.agents.data_agent import DataAgent as _DA
                                _shift = _DA.map_to_shift(int(_hour))
                                _shift = _shift if _shift != 'OTHER' else None
                            else:
                                _shift = p.get('shift')
                            
                            new_rows.append({
                                'Forecast_Run_Date': CURRENT_DATE,
                                'Restaurant_Code': str(res_code),
                                'Date': p['date'],
                                'Weekday': p['weekday'],
                                'Hour': _hour,
                                'Shift': _shift,  # [FIX] Thêm Shift column
                                'Final_Predicted_Guests': p['predicted_guests'],
                                'Actual_Guest': np.nan,
                                'Diff_Guest': np.nan,
                                'Error_%': np.nan,
                                'Is_Holiday': any(
                                    d['date'] == p['date'] and d.get('is_holiday')
                                    for d in active_next_days
                                ),
                                'Is_Veg': any(
                                    d['date'] == p['date'] and d.get('is_veg')
                                    for d in active_next_days
                                ),
                                'AI_Raw_Daily_Forecast': np.nan,
                                'AI_Forecast_Available': False,
                                'Forecast_Mode': 'new_restaurant',  # [FIX] Tag rõ mode
                            })
                        
                        continue  # Skip normal ensemble pipeline
                    # If chain-blend fails, fall through to normal pipeline
                
                # ───────────────────────────────────────
                # ⭐ v6: VOLUME SEGMENTATION ROUTER
                # ───────────────────────────────────────
                volume_segment = report.get('profile', {}).get('volume_segment', 'MEDIUM_VOLUME')
                
                if strategy == 'BASELINE_ONLY' and volume_segment == 'LOW_VOLUME':
                    # LOW VOLUME: Use simple baseline median (no ML, no AI, no Prophet)
                    _, df_res_cleaned = AnalysisAgent.detect_outliers(df_res)
                    
                    baseline_predictions, baseline_info = EnsembleForecastAgent.run_baseline_forecast(
                        res_code=res_code,
                        df_res_cleaned=df_res_cleaned,
                        next_days_info=active_next_days,
                        analysis_report=report,
                    )
                    
                    if baseline_predictions:
                        forecast_stats['success_ensemble'] += 1
                        
                        for p in baseline_predictions:
                            new_rows.append({
                                'Forecast_Run_Date': CURRENT_DATE,
                                'Restaurant_Code': str(res_code),
                                'Date': p['date'],
                                'Weekday': p['weekday'],
                                'Hour': None,
                                'Shift': p.get('shift'),
                                'Final_Predicted_Guests': p['forecast'],
                                'Actual_Guest': np.nan,
                                'Diff_Guest': np.nan,
                                'Error_%': np.nan,
                                'Is_Holiday': p.get('is_holiday', False),
                                'Is_Veg': p.get('is_veg', False),
                                'AI_Raw_Daily_Forecast': np.nan,
                                'AI_Forecast_Available': False,
                                'Forecast_Mode': 'baseline',
                                'Volume_Segment': 'LOW_VOLUME',
                            })
                        
                        continue  # Skip normal ensemble pipeline
                
                # 6.1 Outlier Cleaning
                _, df_res_cleaned = AnalysisAgent.detect_outliers(df_res)
                
                # 6.2 Feature Engineering
                df_processed = MLForecastAgent.prepare_data(df_res_cleaned, vn_holidays)
                if df_processed.empty:
                    continue
                
                # 6.3 AI FORECAST (RAG Self-Learning / LM Studio)
                ai_daily_map = {}
                if weights['ai'] > AI_INFERENCE_THRESHOLD:
                    upcoming_days = [d for d in active_next_days if d['date'] >= CURRENT_DATE]
                    if upcoming_days:
                        hist_text = AIAgent.prepare_enhanced_prompt(
                            df_res_cleaned, CURRENT_DATE, vn_holidays,
                            analysis_report=report
                        )
                        
                        # Use RAG agent (with knowledge context) or LM Studio
                        import time as _ai_time
                        _ai_start = _ai_time.time()
                        try:
                            if USE_RAG_AGENT and rag_agent is not None:
                                ai_resp = rag_agent.generate_forecast(
                                    res_code, hist_text, upcoming_days,
                                    analysis_report=report,
                                    brain_memory=brain_memory_dict,  # type: ignore[bad-argument-type]
                                )
                            else:
                                ai_resp = AIAgent.generate_forecast(  # type: ignore[bad-argument-type]
                                    res_code, hist_text, upcoming_days,  # type: ignore[bad-argument-type]
                                    analysis_report=report
                                )
                        except Exception as ai_err:
                            logger.warning(f"AI inference failed for {res_code}: {ai_err}")
                            ai_resp = None
                        
                        _ai_elapsed = _ai_time.time() - _ai_start
                        if _ai_elapsed > AI_TIMEOUT_PER_RESTAURANT:
                            logger.warning(
                                f"⚠️ AI inference for {res_code} took {_ai_elapsed:.0f}s "
                                f"(timeout={AI_TIMEOUT_PER_RESTAURANT}s)"
                            )
                        
                        ai_data = AIAgent.parse_response(ai_resp) if ai_resp else None
                        
                        if ai_data:
                            for item in ai_data:
                                ai_daily_map[str(item.get('date'))] = item.get('forecast', 0)
                            forecast_stats['success_ai'] += 1
                elif weights['ai'] > 0:
                    # AI weight too low to justify LLM inference cost (~175s)
                    # ML-only forecast (weight will be redistributed in ensemble)
                    forecast_stats.setdefault('skipped_ai_low_weight', 0)
                    forecast_stats['skipped_ai_low_weight'] += 1
                
                # ═══════════════════════════════════════
                # 6.4A SHORT-TERM: HOURLY ENSEMBLE FORECAST
                # ═══════════════════════════════════════
                predictions = []
                ensemble_info = {'strategy': strategy, 'weights': weights, 'models_used': [], 'metrics': {}}
                
                if active_short_term:
                    predictions_short, ensemble_info = EnsembleForecastAgent.run_ensemble_forecast(
                        res_code=res_code,
                        df_res_cleaned=df_res_cleaned,
                        df_processed=df_processed,
                        next_days_info=active_short_term,
                        vn_holidays=vn_holidays,
                        analysis_report=report,
                        ai_daily_map=ai_daily_map,
                        neuralprophet_model=neuralprophet_global_model,
                        booking_lookup=booking_lookup,  # ⭐ V8 Task 1
                    )
                    predictions = predictions_short
                
                # ═══════════════════════════════════════
                # 6.4B LONG-TERM: DAILY-ONLY ENSEMBLE FORECAST
                # ═══════════════════════════════════════
                predictions_long = []
                if active_long_term:
                    predictions_long, ensemble_info_long = EnsembleForecastAgent.run_ensemble_forecast_daily_only(
                        res_code=res_code,
                        df_res_cleaned=df_res_cleaned,
                        df_processed=df_processed,
                        next_days_info=active_long_term,
                        vn_holidays=vn_holidays,
                        analysis_report=report,
                        ai_daily_map=ai_daily_map,
                        neuralprophet_model=neuralprophet_global_model,
                    )
                    # Merge info
                    if not ensemble_info.get('models_used'):
                        ensemble_info = ensemble_info_long
                
                if predictions or predictions_long:
                    forecast_stats['success_ensemble'] += 1
                    if predictions_long:
                        forecast_stats['success_daily_only'] = (
                            int(forecast_stats.get('success_daily_only', 0)) + 1
                        )
                    
                    # Track model usage
                    for model_name in ensemble_info.get('models_used', []):
                        forecast_stats['models_usage'][model_name] = (
                            int(forecast_stats['models_usage'].get(model_name, 0)) + 1  # type: ignore[union-attr]
                        )
                    
                    if 'prophet' in ensemble_info.get('models_used', []):
                        forecast_stats['success_prophet'] += 1
                    
                    # Confidence score
                    confidence = EnsembleForecastAgent.calculate_confidence(
                        ensemble_info, report
                    )
                    forecast_stats['confidence_scores'].append(confidence)
                
                # 6.5 🧠 Brain Correction (Rule-Based + Neural + Transfer)
                # Apply corrections to BOTH short-term and long-term predictions
                all_predictions = predictions + predictions_long
                all_predictions = ForecastBrain.apply_corrections(
                    all_predictions, res_code, analysis_report=report
                )
                
                # [NEURAL CORRECTOR] Apply neural corrections if available
                if neural_corrector_ready:
                    try:
                        from forecast_system.agents.neural_corrector import NeuralCorrector
                        all_predictions = NeuralCorrector.apply_corrections(all_predictions, res_code)
                    except Exception:
                        pass
                
                # [TRANSFER LEARNING] Apply transfer corrections for new/young restaurants
                if transfer_clusters and category in ('NEW', 'YOUNG', 'VOLATILE'):
                    try:
                        from forecast_system.agents.transfer_learning import TransferLearningAgent
                        transfer = TransferLearningAgent.get_transfer_corrections(res_code)
                        if transfer:
                            transfer_cf = transfer.get('correction_factor', 1.0)
                            if abs(transfer_cf - 1.0) >= 0.03:
                                for p in all_predictions:
                                    if p.get('forecast', 0) > 0:
                                        p['forecast'] = max(0, int(round(
                                            p['forecast'] * transfer_cf
                                        )))
                                        p['transfer_correction'] = transfer_cf
                    except Exception:
                        pass
                
                # ⭐ V8 Task 2: Shift Residual Corrector (after brain/neural/transfer, before cap/floor)
                try:
                    from forecast_system.agents.shift_residual_corrector import ShiftResidualCorrector
                    all_predictions = ShiftResidualCorrector.apply_corrections(
                        all_predictions, res_code
                    )
                except Exception:
                    pass
                
                # 6.55 📊 FINAL SMART MAX CAP + FLOOR (safety net)
                # [FIX #1] Sau Brain/Neural/Transfer corrections:
                #   CAP:   daily_total ≤ max(actual) trong 90 ngày gần
                #   FLOOR: daily_total ≥ min(actual) trong 30 ngày gần (ngày thường only)
                # Smart: ngày thường dùng max 3 tháng, ngày lễ dùng max cùng loại lễ
                profile = report.get('profile', {})
                hist_max_normal = profile.get('max_daily_normal') or profile.get('max_daily')
                hist_max_by_holiday = profile.get('max_daily_by_holiday', {})
                
                # [FIX #1] Floor values from profile (computed in compute_smart_max_daily)
                hist_floor_normal = profile.get('min_daily_normal', 0)
                hist_floor_weekday = profile.get('min_daily_normal_weekday', hist_floor_normal)
                hist_floor_weekend = profile.get('min_daily_normal_weekend', hist_floor_normal)
                
                if hist_max_normal:
                    n_max_capped = 0
                    n_floor_lifted = 0
                    # Group predictions by date to check daily total
                    from collections import defaultdict
                    date_forecasts = defaultdict(list)
                    for p in all_predictions:
                        date_forecasts[str(p['date'])].append(p)
                    
                    for d_str, preds_day in date_forecasts.items():
                        daily_total = sum(p.get('forecast', 0) for p in preds_day)
                        
                        # Determine effective cap for this day
                        sample_pred = preds_day[0]
                        is_holiday = sample_pred.get('is_holiday', False)
                        holiday_type = sample_pred.get('holiday_type')
                        is_special_event = sample_pred.get('is_special_event', False)
                        event_type = sample_pred.get('event_type')
                        is_weekend = sample_pred.get('weekday', '') in ('Saturday', 'Sunday')
                        high_traffic_holidays = {
                            'LIBERATION_DAY', 'LABOR_DAY', 'NATIONAL_DAY',
                            'HUNG_KINGS', 'TET_DUONG_LICH'
                        }
                        is_high_traffic_holiday = (
                            is_holiday and holiday_type in high_traffic_holidays
                        )
                        
                        effective_cap = None
                        if is_holiday and holiday_type and holiday_type in hist_max_by_holiday:
                            effective_cap = hist_max_by_holiday[holiday_type]
                        elif is_special_event and event_type and event_type in hist_max_by_holiday:
                            effective_cap = hist_max_by_holiday[event_type]
                        else:
                            effective_cap = hist_max_normal

                        if (
                            is_high_traffic_holiday
                            and effective_cap
                            and holiday_type not in hist_max_by_holiday
                        ):
                            holiday_factor = max(
                                1.0,
                                float(sample_pred.get('holiday_impact', 1.0) or 1.0),
                            )
                            effective_cap = float(effective_cap) * holiday_factor
                        
                        if effective_cap and daily_total > effective_cap:
                            # Scale down proportionally
                            scale = effective_cap / daily_total
                            for p in preds_day:
                                old_val = p['forecast']
                                p['forecast'] = max(0, int(round(old_val * scale)))
                                if p.get('combined_daily'):
                                    p['combined_daily'] = max(0, int(round(
                                        p['combined_daily'] * scale
                                    )))
                            n_max_capped += 1
                        
                        # [FIX #1] FLOOR enforcement — chỉ ngày bình thường (không phải lễ)
                        # Forecast không được thấp hơn min thực tế 30 ngày gần nhất
                        if not is_holiday and not is_special_event and daily_total > 0:
                            effective_floor = (
                                hist_floor_weekend if is_weekend else hist_floor_weekday
                            )
                            if effective_floor and daily_total < effective_floor:
                                # Scale up proportionally to meet floor
                                scale = effective_floor / max(daily_total, 1)
                                for p in preds_day:
                                    old_val = p['forecast']
                                    p['forecast'] = max(0, int(round(old_val * scale)))
                                    if p.get('combined_daily'):
                                        p['combined_daily'] = max(0, int(round(
                                            p['combined_daily'] * scale
                                        )))
                                n_floor_lifted += 1
                    
                    if n_max_capped > 0:
                        logger.debug(
                            f"  📊 Cap: {n_max_capped} days capped for {res_code}"
                        )
                    if n_floor_lifted > 0:
                        logger.debug(
                            f"  📊 [FIX #1] Floor: {n_floor_lifted} days lifted for {res_code}"
                        )
                
                # 6.56 ⚡ ACCURACY GUARDRAIL (Volume-based deviation clamp)
                # High-volume restaurants must not use a fixed ±10 guest clamp.
                # That rule was tighter than the operational KPI (±10%) and
                # suppressed valid demand increases, creating under-forecast.
                avg_daily_guests = profile.get('avg_daily', 0)
                weekday_patterns = report.get('weekday_patterns', {})
                
                if avg_daily_guests > 0 and weekday_patterns:
                    is_high_volume_guard = avg_daily_guests >= 100
                    n_guardrail_clamped = 0
                    
                    for d_str, preds_day in date_forecasts.items():
                        sample_pred = preds_day[0]
                        weekday_name = sample_pred.get('weekday', '')
                        wd_stats = weekday_patterns.get(weekday_name, {})
                        wd_avg = wd_stats.get('avg', avg_daily_guests)
                        
                        # Skip holidays/events — guardrails only for normal days
                        if sample_pred.get('is_holiday') or sample_pred.get('is_special_event'):
                            continue
                        
                        daily_total = sum(p.get('forecast', 0) for p in preds_day)
                        
                        if daily_total <= 0 or wd_avg <= 0:
                            continue
                        
                        # Calculate allowed range.
                        if is_high_volume_guard:
                            # ≥100 avg: use percentage band. Wider upper band
                            # allows recovery when recent bias is under.
                            lower_bound = max(0, wd_avg * 0.80)
                            upper_bound = wd_avg * 1.35
                        else:
                            # <100 avg: keep conservative but not production KPI-tight.
                            lower_bound = max(0, wd_avg * 0.85)
                            upper_bound = wd_avg * 1.20
                        
                        # Clamp if outside bounds
                        if daily_total > upper_bound or daily_total < lower_bound:
                            # Do not pull an under-forecasting high-volume
                            # restaurant down. Let validation decide later.
                            if is_high_volume_guard and daily_total > wd_avg:
                                pass
                            else:
                                clamped_total = max(lower_bound, min(daily_total, upper_bound))
                                if daily_total > 0:
                                    scale = clamped_total / daily_total
                                    for p in preds_day:
                                        old_val = p['forecast']
                                        p['forecast'] = max(0, int(round(old_val * scale)))
                                        if p.get('combined_daily'):
                                            p['combined_daily'] = max(0, int(round(
                                                p['combined_daily'] * scale
                                            )))
                                    n_guardrail_clamped += 1
                    
                    if n_guardrail_clamped > 0:
                        guard_type = "80%-135%" if is_high_volume_guard else "85%-120%"
                        logger.info(
                            f"  ⚡ Guardrail ({guard_type}): {n_guardrail_clamped} days "
                            f"clamped for {res_code} (avg={avg_daily_guests:.0f}/day)"
                        )
                
                # 6.6 Collect Rows (split by forecast mode)
                n_half_adjusted = 0
                for p in all_predictions:
                    ai_available = str(p['date']) in ai_daily_map
                    ai_val = ai_daily_map.get(str(p['date']), np.nan)
                    
                    # Apply HALF_DAY reduction from closure schedule
                    forecast_val = p['forecast']
                    is_half = p['date'] in half_day_dates
                    if is_half and forecast_val > 0:
                        forecast_val = max(0, int(round(forecast_val * 0.5)))
                        n_half_adjusted += 1
                    
                    is_daily_only = p.get('forecast_mode') == 'daily_only' or p.get('hour') is None and p.get('shift') is None
                    
                    row_data = {
                        'Forecast_Run_Date': CURRENT_DATE,
                        'Restaurant_Code': str(res_code),
                        'Date': p['date'],
                        'Weekday': p['weekday'],
                        'Hour': p.get('hour'),  # None for shift-based mode
                        'Shift': p.get('shift'),  # Phase 8: MORNING/EVENING
                        'Final_Predicted_Guests': forecast_val,
                        'Actual_Guest': np.nan,
                        'Diff_Guest': np.nan,
                        'Error_%': np.nan,
                        'Is_Holiday': p['is_holiday'],
                        'Is_Veg': p.get('is_veg', False),
                        'AI_Raw_Daily_Forecast': ai_val,
                        'AI_Forecast_Available': ai_available,
                        'Forecast_Mode': 'daily_only' if is_daily_only else 'shift',
                    }
                    
                    # Store brain correction info
                    if p.get('forecast_before_correction') is not None:
                        row_data['System_Predicted_Before_Brain'] = p['forecast_before_correction']
                        row_data['ML_Predicted'] = p['forecast_before_correction']
                        row_data['Brain_Correction'] = p.get('correction_applied', 0)
                    
                    new_rows.append(row_data)
                
                if n_half_adjusted > 0:
                    forecast_stats['skipped_half_day'] = (
                        int(forecast_stats.get('skipped_half_day', 0)) + 1
                    )
                    logger.debug(
                        f"📋 {res_code}: {n_half_adjusted} hours adjusted ×0.5 "
                        f"for half-day closure ({len(half_day_dates)} dates)"
                    )
                    
            except Exception as e:
                forecast_stats['failed'] += 1
                forecast_stats['errors'].append({
                    'restaurant': res_code,
                    'error': str(e)
                })
                logger.warning(f"Error forecasting for {res_code}: {e}")
                traceback.print_exc()
                continue
    
    # ==========================================
    # STEP 6.5: BOOKING DATA (Already loaded at Step 5.95)
    # ==========================================
    # ⭐ V8: Booking data was moved to Step 5.95 (pre-forecast) for ML feature injection.
    # df_booking_summary and df_booking_daily are already populated.
    if not df_booking_daily.empty:
        logger.info(f"\n🎫 STEP 6.5: Booking data ready ({len(df_booking_daily):,} rows, loaded at Step 5.95)")
    else:
        logger.info(f"\n🎫 STEP 6.5: No booking data available")
    
    # ==========================================
    # STEP 7: SAVE RESULTS
    # ==========================================
    logger.info(f"\n💾 STEP 7: Saving Results...")
    
    if new_rows:
        
        # Protect historical data (pandas 3.0 compat: drop NaT before comparison)
        if df_hist is None:
            df_hist = pd.DataFrame()
        df_hist['Date'] = pd.to_datetime(df_hist['Date'], errors='coerce')
        df_hist = df_hist.dropna(subset=['Date'])
        df_hist['Date'] = df_hist['Date'].dt.date  # type: ignore[union-attr]
        
        # Parse Forecast_Run_Date for proper filtering
        df_hist['Forecast_Run_Date'] = pd.to_datetime(
            df_hist['Forecast_Run_Date'], errors='coerce'
        )
        
        df_new = pd.DataFrame(new_rows)
        df_new = df_new[df_new['Date'] >= CURRENT_DATE].copy()
        
        # ── PRESERVE HISTORICAL RUN DATA ──
        # Strategy: Keep ALL rows from PREVIOUS runs (Forecast_Run_Date < today)
        # Remove only rows from the CURRENT run date (will be replaced by df_new)
        # Also remove "orphan" future rows that have no Forecast_Run_Date
        has_frd = df_hist['Forecast_Run_Date'].notna()
        
        # Previous runs: keep everything (even their future-date forecasts)
        prev_runs = df_hist[
            has_frd & (df_hist['Forecast_Run_Date'].dt.date < CURRENT_DATE)  # type: ignore[union-attr]
        ].copy()
        
        # Current run old data: discard (will be replaced by df_new)
        # Orphan rows (no FRD) with past dates: keep them
        orphan_past = df_hist[
            ~has_frd & (df_hist['Date'] < CURRENT_DATE)
        ].copy()
        
        # ── [FIX] Remove stale prev_run rows for dates covered by df_new ──
        # When the current run produces new forecasts for a (Restaurant_Code, Date),
        # the old forecasts from previous runs for that same combo must be removed.
        # Without this fix: old hourly rows (Shift=NaN) and new shift rows (Shift=MORNING/EVENING)
        # both survive dedup because their _shift_dedup keys differ ('__NONE__' vs 'MORNING').
        if not prev_runs.empty and not df_new.empty:
            new_keys = set(
                zip(
                    df_new['Restaurant_Code'].astype(str),
                    df_new['Date'].astype(str),
                )
            )
            prev_runs['_res_date_key'] = (
                prev_runs['Restaurant_Code'].astype(str).str.cat(  # type: ignore[union-attr]
                    prev_runs['Date'].astype(str), sep='|'
                )
            )
            stale_mask = prev_runs['_res_date_key'].isin(
                {f'{r}|{d}' for r, d in new_keys}
            )
            n_stale = stale_mask.sum()
            if n_stale > 0:
                logger.info(
                    f"   🧹 [FIX] Removed {n_stale:,} stale prev_run rows "
                    f"superseded by current run (mode-change or re-forecast)"
                )
            prev_runs = prev_runs[~stale_mask].drop(columns=['_res_date_key'])

        df_final = pd.concat(
            [prev_runs, orphan_past, df_new], ignore_index=True
        )

        
        # ── FIX: Remove stale daily_only rows when current run has hourly rows ──
        # When short-term horizon moves forward between runs, dates that were
        # daily_only (from older run) now have hourly data (from current run).
        # Without cleanup: both co-exist → double-counting (e.g. 106K + 144K = 250K)
        if 'Forecast_Mode' in df_final.columns:
            # Find (Restaurant_Code, Date) combos that have BOTH modes
            mode_counts = df_final.groupby(
                ['Restaurant_Code', 'Date']
            )['Forecast_Mode'].nunique()
            dual_mode_keys = list(mode_counts[mode_counts > 1].index) # type: ignore[unsupported-operation]
            
            if len(dual_mode_keys) > 0:
                # For each dual-mode combo, keep only the mode from the LATEST run
                rows_to_drop = []
                for res_code, date_val in dual_mode_keys: # type: ignore[not-iterable]
                    mask = (
                        (df_final['Restaurant_Code'] == res_code) & 
                        (df_final['Date'] == date_val)
                    )
                    subset = df_final[mask]
                    
                    # Get latest Forecast_Run_Date
                    latest_frd = pd.to_datetime(
                        subset['Forecast_Run_Date'], errors='coerce'
                    ).max()
                    
                    # Get the mode from the latest run
                    latest_rows = subset[
                        pd.to_datetime(subset['Forecast_Run_Date'], errors='coerce') == latest_frd
                    ]
                    latest_mode = latest_rows['Forecast_Mode'].mode() # type: ignore[union-attr]
                    if not latest_mode.empty:
                        keep_mode = latest_mode.iloc[0]
                        # Drop rows with the OTHER mode
                        drop_mask = mask & (df_final['Forecast_Mode'] != keep_mode)
                        rows_to_drop.extend(df_final[drop_mask].index.tolist()) # type: ignore[union-attr]
                
                if rows_to_drop:
                    logger.info(
                        f"   🧹 Removed {len(rows_to_drop):,} stale dual-mode rows "
                        f"(daily_only superseded by hourly or vice versa)"
                    )
                    df_final = df_final.drop(index=rows_to_drop).reset_index(drop=True)
        
        # Remove duplicates: if same (Restaurant_Code, Date, Hour) exists in both 
        # old and new, keep the NEWEST forecast (from df_new / latest Forecast_Run_Date)
        sort_cols = ['Forecast_Run_Date']
        if 'Forecast_Run_Date' in df_final.columns:
            df_final['_frd_sort'] = pd.to_datetime(
                df_final['Forecast_Run_Date'], errors='coerce'
            ).fillna(pd.Timestamp.min)
            df_final = df_final.sort_values(by=['_frd_sort'], ascending=False) # type: ignore[bad-argument-type]
            
            dedup_cols = ['Restaurant_Code', 'Date']
            if 'Hour' in df_final.columns:
                dedup_cols.append('Hour')
            
            # Fill NaN hours with sentinel for proper dedup (NaN != NaN in pandas)
            # Phase 8: Include 'Shift' column to distinguish MORNING/EVENING rows
            # (both have Hour=None, so without Shift they'd be treated as duplicates)
            if 'Hour' in df_final.columns:
                df_final['_hour_dedup'] = df_final['Hour'].fillna(-999)
                dedup_cols_fixed = ['Restaurant_Code', 'Date', '_hour_dedup']
            else:
                dedup_cols_fixed = dedup_cols
            
            # Add Shift to dedup key for Phase 8 shift-based rows
            if 'Shift' in df_final.columns:
                df_final['_shift_dedup'] = df_final['Shift'].fillna('__NONE__')
                dedup_cols_fixed.append('_shift_dedup')
            
            df_final = df_final.drop_duplicates(
                subset=dedup_cols_fixed, keep='first'
            )
            cleanup_cols = ['_frd_sort']
            if '_hour_dedup' in df_final.columns:
                cleanup_cols.append('_hour_dedup')
            if '_shift_dedup' in df_final.columns:
                cleanup_cols.append('_shift_dedup')
            df_final = df_final.drop(columns=cleanup_cols)
        
        logger.info(
            f"   Historical runs preserved: "
            f"{prev_runs['Forecast_Run_Date'].dt.date.nunique()} previous run dates, "  # type: ignore[union-attr]
            f"{len(orphan_past):,} orphan rows"
        )
        
        # Merge restaurant info
        if not df_info.empty:
            df_final.drop(
                columns=['sap_code', 'restaurant_name'],
                errors='ignore', inplace=True
            )
            df_final['merge_key'] = DataAgent.normalize_key(
                df_final['Restaurant_Code']
            )
            
            df_final = pd.merge(
                df_final,
                df_info[['merge_key', 'sap_code', 'restaurant_name']],
                on='merge_key', how='left'
            )
            df_final.drop(columns=['merge_key'], inplace=True)
        
        # ── MERGE BOOKING DATA vào Forecast sheet ──
        # Thêm cột Booking_Guests (khách đặt bàn trước) để hiển thị cạnh forecast
        if not df_booking_daily.empty:
            try:
                # Normalize keys cho merge
                df_bk = df_booking_daily.copy()
                df_bk['Restaurant_Code'] = DataAgent.normalize_key(df_bk['Restaurant_Code'])
                df_bk['Date'] = pd.to_datetime(df_bk['Date'], errors='coerce')
                df_bk = df_bk.dropna(subset=['Date'])
                df_bk['Date'] = df_bk['Date'].dt.date  # type: ignore[union-attr]
                
                # Đảm bảo df_final.Date cũng là date object
                df_final['Date'] = pd.to_datetime(df_final['Date'], errors='coerce')
                df_final = df_final.dropna(subset=['Date']) # type: ignore[bad-argument-type]
                df_final['Date'] = df_final['Date'].dt.date  # type: ignore[union-attr]
                
                # Drop cột cũ nếu tồn tại (tránh duplicate khi re-run)
                df_final.drop(columns=['Booking_Guests'], errors='ignore', inplace=True)
                
                # Left join: mỗi row forecast sẽ có Booking_Guests của ngày đó
                df_final = pd.merge(
                    df_final,
                    df_bk[['Restaurant_Code', 'Date', 'Booking_Guests_Total']].rename( # type: ignore[bad-argument-type]
                        columns={'Booking_Guests_Total': 'Booking_Guests'}
                    ),
                    on=['Restaurant_Code', 'Date'],
                    how='left'
                )
                
                n_matched = df_final['Booking_Guests'].notna().sum()
                logger.info(
                    f"   🎫 Booking data merged into Forecast: "
                    f"{n_matched:,} rows matched"
                )
            except Exception as e:
                logger.warning(f"Booking merge failed (non-critical): {e}")
                traceback.print_exc()
        
        # Reorder columns
        cols = MasterFileAgent.COLUMNS
        df_final = df_final[[c for c in cols if c in df_final.columns]]
        
        # Update actuals one last time
        df_final = MasterFileAgent.update_actuals(df_final, df_train)
        
        # Generate Shift Summary (only for SHORT-TERM hourly data)
        logger.info("\n📊 Generating Shift-based Summary (short-term only)...")
        if 'Hour' in df_final.columns:
            hour_filter = df_final['Hour'].notna()
            if 'Forecast_Mode' in df_final.columns:
                hour_filter = hour_filter & (df_final['Forecast_Mode'] != 'daily_only')
            df_hourly_only = df_final[hour_filter].copy()
        else:
            df_hourly_only = df_final.copy()
        df_shift = DataAgent.aggregate_shifts(df_hourly_only)
        if not df_shift.empty:
            save_excel_safely(df_shift, SHIFT_FILE_NAME)
        
        # Save master file (with Booking sheet if available)
        if not df_booking_summary.empty:
            MasterFileAgent.save_with_booking_sheet(
                df_final, df_booking_summary, MASTER_FILE_NAME
            )
        else:
            save_excel_safely(df_final, MASTER_FILE_NAME)
        
        # ==========================================
        # STEP 8: SUMMARY REPORT
        # ==========================================
        avg_conf = (
            np.mean(forecast_stats['confidence_scores'])
            if forecast_stats['confidence_scores'] else 0
        )
        
        logger.info("\n" + "=" * 65)
        logger.info("📊 ENSEMBLE FORECAST SUMMARY")
        logger.info("=" * 65)
        logger.info(f"  Total restaurants analyzed:    {forecast_stats['total']}")
        logger.info(f"  Ensemble forecast success:     {forecast_stats['success_ensemble']}")
        hourly_rows = len(df_new[df_new['Forecast_Mode'] != 'daily_only']) if 'Forecast_Mode' in df_new.columns else len(df_new)
        daily_only_rows = len(df_new) - hourly_rows
        logger.info(f"  ├─ Short-term (hourly) rows:   {hourly_rows:,}")
        logger.info(f"  └─ Long-term (daily-only) rows: {daily_only_rows:,}")
        logger.info(f"  AI (LLM) inference run:        {forecast_stats['success_ai']}")
        if int(forecast_stats.get('skipped_ai_low_weight', 0)) > 0:
            logger.info(f"  ⚡ AI skipped (low weight):     {forecast_stats['skipped_ai_low_weight']} (threshold={AI_INFERENCE_THRESHOLD})")
        logger.info(f"  Prophet success:               {forecast_stats['success_prophet']}")
        logger.info(f"  Failed:                        {forecast_stats['failed']}")
        if int(forecast_stats.get('permanently_closed', 0)) > 0:
            logger.info(f"  🔴 Permanently CLOSED:         {forecast_stats['permanently_closed']}")
        if forecast_stats['skipped_closed'] > 0:
            logger.info(f"  Skipped (holiday closed):      {forecast_stats['skipped_closed']}")
        if int(forecast_stats.get('skipped_half_day', 0)) > 0:
            logger.info(f"  Half-day adjusted (×0.5):      {forecast_stats['skipped_half_day']}")
        if forecast_stats['success_new_restaurant'] > 0:
            logger.info(f"  New restaurant (chain-blend):  {forecast_stats['success_new_restaurant']}")
        logger.info(f"  Total rows generated:          {len(df_new):,}")
        logger.info(f"  Master file total rows:        {len(df_final):,}")
        logger.info(f"  Average confidence:            {avg_conf:.2f}")
        
        logger.info(f"\n  🤖 Model Usage Across Restaurants:")
        for model, count in sorted(
            forecast_stats['models_usage'].items(),
            key=lambda x: x[1], reverse=True
        ):
            pct = count / max(forecast_stats['success_ensemble'], 1) * 100
            logger.info(f"     {model:15s}: {count:4d} ({pct:.0f}%)")
        
        logger.info(f"\n  📊 Category Breakdown:")
        for cat, count in sorted(categories.items()):
            logger.info(f"     {cat:15s}: {count:4d} restaurants")
        
        logger.info(f"\n  🎯 Strategy Breakdown:")
        for strat, count in sorted(strategies.items()):
            logger.info(f"     {strat:30s}: {count:4d} restaurants")
        
        logger.info("=" * 65)
        logger.info("✅ DONE! Ensemble AI Forecast completed.")
        
        if forecast_stats['errors']:
            logger.info(f"\n⚠️ Errors ({len(forecast_stats['errors'])}):") 
            for err in forecast_stats['errors'][:10]:
                logger.info(f"   - {err['restaurant']}: {err['error']}")
            if len(forecast_stats['errors']) > 10:
                logger.info(
                    f"   ... and {len(forecast_stats['errors']) - 10} more"
                )
        
        # ==========================================
        # STEP 9: MONITORING & ACCURACY TRACKING
        # ==========================================
        logger.info("\n" + "=" * 65)
        logger.info("📊 STEP 9: Accuracy Monitoring")
        logger.info("=" * 65)
        
        try:
            from forecast_system.agents.monitoring_agent import MonitoringAgent
            
            report = MonitoringAgent.generate_full_report(df_final)
            MonitoringAgent.print_report(report, logger_func=logger.info)
            
            # Log critical drift alerts
            drift = report.get('drift', {})
            if drift.get('has_drift'):
                for alert in drift.get('alerts', []):
                    if alert.get('level') == 'CRITICAL':
                        logger.warning(
                            f"🚨 CRITICAL: {alert.get('message', '')}"
                        )
            
            logger.info("✅ Monitoring complete.")
        except Exception as e:
            logger.warning(f"Monitoring step failed (non-critical): {e}")
            traceback.print_exc()
            report = None
        
        # ==========================================
        # STEP 9.1: PERFORMANCE REPORTS (Excel)
        # ==========================================
        logger.info("\n" + "=" * 65)
        logger.info("📊 STEP 9.1: Generating Performance Reports")
        logger.info("=" * 65)
        
        try:
            from forecast_system.agents.performance_report_agent import PerformanceReportAgent
            
            _run_duration = (_time.time() - _run_start_time) / 60
            
            # Generate run record
            run_record = PerformanceReportAgent.generate_run_record(
                forecast_stats=forecast_stats,
                report=report or {},
                forecast_mode=forecast_mode,
                run_duration_minutes=_run_duration,
            )
            
            # Save Model_Performance_Report.xlsx (cumulative)
            PerformanceReportAgent.save_model_performance_report(
                new_record=run_record,
                report=report or {},
                df_master=df_final,
            )
            
            # Save enhanced Accuracy_Report.xlsx
            PerformanceReportAgent.save_enhanced_accuracy_report(
                df_master=df_final,
                report=report or {},
                forecast_stats=forecast_stats,
            )
            
            # Print health summary
            PerformanceReportAgent.print_health_summary(
                run_record, logger_func=logger.info
            )
            
            logger.info("✅ Performance reports saved.")
            
            # ═══ Generate Model Evaluation By Shift & DayType Report ═══
            try:
                logger.info("\n" + "=" * 65)
                logger.info("📊 STEP 9.2: Generating Model Evaluation Report (By Shift & DayType)")
                logger.info("=" * 65)
                
                import subprocess
                import sys
                
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                script_path = os.path.join(root_dir, "generate_model_evaluation_report.py")
                
                if os.path.exists(script_path):
                    logger.info(f"   Running script: {script_path}")
                    result = subprocess.run(
                        [sys.executable, script_path],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    logger.info("✅ Model Evaluation By Shift & DayType Report generated successfully.")
                    logger.debug(result.stdout)
                else:
                    logger.warning(f"⚠️ generate_model_evaluation_report.py not found at: {script_path}")
            except subprocess.CalledProcessError as sub_err:
                logger.warning(f"❌ Failed to run generate_model_evaluation_report.py: {sub_err}")
                logger.warning(f"   Stdout: {sub_err.stdout}")
                logger.warning(f"   Stderr: {sub_err.stderr}")
            except Exception as eval_err:
                logger.warning(f"❌ Error generating Model Evaluation report: {eval_err}")
                traceback.print_exc()
        except Exception as e:
            logger.warning(f"Performance report step failed (non-critical): {e}")
            traceback.print_exc()
        
        # ==========================================
        # STEP 9.5: BRAIN ABSORB MONITORING RESULTS
        # ==========================================
        if report:
            logger.info("\n" + "=" * 65)
            logger.info("🧠 STEP 9.5: Brain Absorbing Monitoring Results (Closed-Loop)")
            logger.info("=" * 65)
            
            try:
                absorb_result = ForecastBrain.absorb_monitoring_report(report)
                logger.info(f"   Metrics updated: {absorb_result.get('metrics_updated', 0)}")
                logger.info(f"   Drift adjustments: {absorb_result.get('drift_adjustments', 0)}")
                logger.info(f"   Retune flagged: {absorb_result.get('retune_flagged', 0)}")
                
                for insight in absorb_result.get('history_insights', []):
                    logger.info(f"   📈 {insight}")
                
                logger.info("✅ Brain absorption complete.")
            except Exception as e:
                logger.warning(f"Brain absorption failed (non-critical): {e}")
                traceback.print_exc()
        
        # ==========================================
        # STEP 10: BRAIN INSIGHTS
        # ==========================================
        logger.info("\n" + "=" * 65)
        logger.info("🧠 STEP 10: Brain Self-Learning Insights")
        logger.info("=" * 65)
        
        try:
            ForecastBrain.print_insights(logger_func=logger.info)
            
            # Log high-MAPE restaurants with brain diagnosis
            insights = ForecastBrain.generate_insights()
            high_mape = insights.get('high_mape_restaurants', [])
            if high_mape:
                logger.info(f"\n   🔍 Top 5 restaurants cần Brain attention:")
                for r in high_mape[:5]:
                    diag = ForecastBrain.diagnose_restaurant(r['code'])
                    logger.info(f"     {r['code']}: {diag['diagnosis']}")
                    for cause in diag.get('root_causes', [])[:2]:
                        logger.info(f"       → {cause}")
            
            logger.info("✅ Brain insights complete.")
        except Exception as e:
            logger.warning(f"Brain insights failed (non-critical): {e}")
    else:
        logger.warning("⚠️ No forecast data generated.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='AI Forecast System - Main Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m forecast_system.main                # Daily mode (30 ngày, nhanh)
  python -m forecast_system.main --mode daily    # Daily mode (tương tự trên)
  python -m forecast_system.main --mode full     # Full mode (3 tháng, đầy đủ)
        """
    )
    parser.add_argument(
        '--mode', type=str, default='daily',
        choices=['daily', 'full'],
        help='Forecast mode: daily (30 ngày, nhanh) hoặc full (3 tháng, đầy đủ). Default: daily'
    )
    args = parser.parse_args()
    main(forecast_mode=args.mode)
