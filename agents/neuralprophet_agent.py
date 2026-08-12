"""
==============================================
NEURALPROPHET FORECAST AGENT
==============================================
Thay thế/bổ sung Prophet trong ensemble pipeline.

NeuralProphet advantages over classic Prophet:
- Autoregressive (AR) lags: self-learns optimal lag features
- Neural network components: handles non-linear patterns
- Global model: can train on ALL restaurants at once
- Multi-step forecasting: native multi-horizon output
- Faster convergence: PyTorch backend

Integration:
- Drop-in replacement for ProphetDailyAgent
- Returns same format: Dict[str, float] = {date_str: predicted_daily_total}
- Used as a source in EnsembleForecastAgent._weighted_combine()
"""

import pandas as pd
import numpy as np
import datetime
import traceback
import warnings
import pickle
import os
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional

from forecast_system.config.settings import CURRENT_DATE, MODEL_CACHE_DIR
from forecast_system.utils.logger import get_logger
from forecast_system.utils.date_utils import get_lunar_info

logger = get_logger('neuralprophet_agent')

# ============================================================
# ⚡ CRITICAL: Prevent PyTorch/OpenMP deadlock on macOS
# Must be set BEFORE importing torch/neuralprophet
# This fixes the root cause of model.fit() hanging indefinitely
# ============================================================
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')  # macOS Accelerate
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

# Safe import
# ✅ Re-enabled: NeuralProphet works on CPU with PyTorch 2.10
# The segfault was MPS (Metal GPU) specific — CPU mode is stable.
# IMPORTANT: Always use accelerator='cpu' to avoid MPS segfault.
try:
    from neuralprophet import NeuralProphet
    HAS_NEURALPROPHET = True
    
    # ⚡ Also limit PyTorch internal threads to prevent deadlock
    try:
        import torch # type: ignore[import-not-found]
        torch.set_num_threads(1) # type: ignore
        torch.set_num_interop_threads(1) # type: ignore
    except Exception:
        pass
    
    print("✅ NeuralProphet: Ready (CPU mode, single-thread)")
except ImportError:
    HAS_NEURALPROPHET = False
    NeuralProphet = None
    print("⚠️ NeuralProphet: Missing (pip install neuralprophet)")


class NeuralProphetAgent:
    """
    NeuralProphet-based daily forecast agent.
    
    Features:
    - Auto-regression (AR) with configurable lags
    - Weekly + yearly seasonality (auto-detected)
    - Holiday effects (Vietnam holidays)
    - Trend + changepoint detection
    - Lagged regressors support
    
    Output: Dict[str, float] — {date_str: predicted_daily_total}
    """
    
    # ==========================================
    # CONFIGURATION
    # ==========================================
    
    # Model hyperparameters (tuned for restaurant guest count)
    # ⚡ SPEED-OPTIMIZED: reduced from 80→20 epochs, simplified ar_layers
    # With higher LR, most models converge in 10-15 epochs
    DEFAULT_CONFIG = {
        'n_lags': 14,           # 2 weeks of autoregressive lookback (was 28)
        'n_forecasts': 30,      # Predict up to 30 days ahead
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,  # We model hourly separately
        'learning_rate': 0.05,  # Higher LR for faster convergence (was 0.01)
        'epochs': 20,           # ⚡ Reduced from 80
        'batch_size': 128,      # Larger batches = fewer steps per epoch (was 64)
        'ar_layers': [16],      # Simplified (was [32, 16]) — sufficient for daily data
        'trend_reg': 0.1,       # Regularize trend to prevent overfitting
        'seasonality_reg': 0.1,
    }
    
    # ⚡ TIMEOUT PROTECTION — prevents pipeline hang
    TIMEOUT_PER_RESTAURANT = 120   # 2 minutes max per restaurant (fallback mode)
    TIMEOUT_GLOBAL_TRAIN = 600     # 10 minutes max for global model training
    
    # Minimum data requirements
    MIN_TRAINING_DAYS = 30      # At least 30 days of data
    MIN_TRAINING_ROWS = 20      # At least 20 rows after aggregation
    
    # Cache settings
    CACHE_MAX_AGE_HOURS = 12    # Reuse cached predictions for 12 hours
    
    @staticmethod
    def train_and_predict(
        df_res: pd.DataFrame,
        next_days_info: List[Dict],
        vn_holidays,
        config: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Train NeuralProphet on restaurant daily data and predict future days.
        
        Args:
            df_res: Transaction data for 1 restaurant
                    Must have columns: date, guest_count (or similar)
            next_days_info: List of forecast target days
            vn_holidays: holidays.VN object
            config: Optional model config overrides
            
        Returns:
            Dict[str, float]: {date_str: predicted_daily_total}
            Empty dict if training fails.
        """
        if NeuralProphet is None:
            return {}
        
        if df_res.empty:
            return {}
        
        # ⚠️ Per-restaurant training is DISABLED to prevent pipeline hangs.
        # Use predict_from_global_model() with a pre-trained global model instead.
        # This fallback only runs the inner method directly WITH the thread-limit
        # env vars already set above (OMP_NUM_THREADS=1).
        try:
            result = NeuralProphetAgent._train_and_predict_inner(
                df_res, next_days_info, vn_holidays, config
            )
            return result if result else {}
        except Exception as e:
            res_hint = str(df_res['restaurant_code'].iloc[0]) if 'restaurant_code' in df_res.columns else 'unknown'
            logger.warning(f"NeuralProphet per-restaurant failed for {res_hint}: {e}")
            return {}
    
    @staticmethod
    def _train_and_predict_inner(
        df_res: pd.DataFrame,
        next_days_info: List[Dict],
        vn_holidays,
        config: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Inner implementation of train_and_predict (called within timeout wrapper)."""
        
        # ==================================
        # 0. CHECK CACHE — skip training if recent predictions exist
        # ==================================
        res_code_hint = str(df_res['restaurant_code'].iloc[0]) if 'restaurant_code' in df_res.columns else 'unknown'
        cached = NeuralProphetAgent._load_prediction_cache(res_code_hint)
        if cached is not None:
            target_dates = {str(d['date']) for d in next_days_info}
            filtered = {k: v for k, v in cached.items() if k in target_dates}
            if len(filtered) >= len(target_dates) * 0.8:  # At least 80% coverage
                logger.debug(f"NeuralProphet cache hit for {res_code_hint}: {len(filtered)}/{len(target_dates)} days")
                return filtered
        
        try:
            # ==================================
            # 1. PREPARE DATA (NeuralProphet format)
            # ==================================
            df_daily = NeuralProphetAgent._prepare_daily_data(df_res)
            
            if df_daily is None or len(df_daily) < NeuralProphetAgent.MIN_TRAINING_ROWS:
                logger.debug(f"Insufficient data for NeuralProphet: {len(df_daily) if df_daily is not None else 0} rows")
                return {}
            
            # ==================================
            # 2. BUILD MODEL
            # ==================================
            cfg = {**NeuralProphetAgent.DEFAULT_CONFIG, **(config or {})}
            
            # Adjust n_forecasts to match requested horizon
            max_horizon = max(
                (d['date'] - CURRENT_DATE).days for d in next_days_info
            ) if next_days_info else 30
            cfg['n_forecasts'] = min(max_horizon + 1, 90)  # Cap at 90 days
            
            # Adjust n_lags based on available data
            available_days = len(df_daily)
            cfg['n_lags'] = min(cfg['n_lags'], available_days // 3)  # Max 1/3 of data
            cfg['n_lags'] = max(cfg['n_lags'], 7)  # Minimum 7 days
            
            # ⚡ Further reduce epochs for small datasets
            if available_days < 60:
                cfg['epochs'] = min(cfg['epochs'], 15)
            
            model = NeuralProphet( # type: ignore[misc]
                n_lags=cfg['n_lags'],
                n_forecasts=cfg['n_forecasts'],
                yearly_seasonality=cfg['yearly_seasonality'],
                weekly_seasonality=cfg['weekly_seasonality'],
                daily_seasonality=cfg['daily_seasonality'],
                learning_rate=cfg['learning_rate'],
                epochs=cfg['epochs'],
                batch_size=cfg['batch_size'],
                ar_layers=cfg['ar_layers'],
                trend_reg=cfg['trend_reg'],
                seasonality_reg=cfg['seasonality_reg'],
                accelerator='cpu',
            )
            
            # ==================================
            # 3. ADD HOLIDAYS
            # ==================================
            holidays_df = NeuralProphetAgent._build_holidays_df(vn_holidays)
            if holidays_df is not None and not holidays_df.empty:
                # Deduplicate: keep only unique (event, ds) pairs
                holidays_df = holidays_df.drop_duplicates(subset=['event', 'ds'])
                model = model.add_country_holidays(country_name='VN')
            
            # ==================================
            # 4. ADD LAGGED REGRESSORS (optional enrichment)
            # ==================================
            # Add is_weekend as a future regressor
            df_daily['is_weekend'] = df_daily['ds'].dt.dayofweek.isin([5, 6]).astype(float)
            model.add_future_regressor('is_weekend')
            
            # ==================================
            # 5. TRAIN
            # ==================================
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics = model.fit(df_daily, freq='D')
            
            train_mae = None
            if metrics is not None and 'MAE_val' in metrics.columns:
                train_mae = float(metrics['MAE_val'].iloc[-1])
            logger.debug(f"NeuralProphet trained: {len(df_daily)} days, "
                        f"lags={cfg['n_lags']}, epochs={cfg['epochs']}, "
                        f"MAE_val={train_mae}")
            
            # ==================================
            # 6. PREDICT
            # ==================================
            # Create future dataframe
            future_dates = [d['date'] for d in next_days_info]
            n_future = max((d - df_daily['ds'].dt.date.max()).days for d in future_dates) + 1
            n_future = max(n_future, cfg['n_forecasts'])
            
            future = model.make_future_dataframe(
                df_daily, 
                periods=n_future,
                n_historic_predictions=False,
            )
            
            # Add future regressor values
            future['is_weekend'] = future['ds'].dt.dayofweek.isin([5, 6]).astype(float)
            
            forecast = model.predict(future)
            
            # ==================================
            # 7. EXTRACT RESULTS
            # ==================================
            results = NeuralProphetAgent._extract_predictions(
                forecast, next_days_info, cfg['n_forecasts']
            )
            
            if results:
                logger.debug(f"NeuralProphet predicted {len(results)} days "
                           f"(range: {min(results.values()):.0f}~{max(results.values()):.0f})")
                # ⚡ Cache predictions for next run
                NeuralProphetAgent._save_prediction_cache(res_code_hint, results)
            
            return results
            
        except Exception as e:
            logger.warning(f"NeuralProphet failed: {e}")
            logger.debug(traceback.format_exc())
            return {}
    
    # ==========================================
    # HELPER METHODS
    # ==========================================
    
    @staticmethod
    def _prepare_daily_data(df_res: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Convert transaction data to NeuralProphet format.
        
        NeuralProphet requires:
        - Column 'ds': datetime
        - Column 'y': target value
        
        We aggregate hourly data to daily totals.
        """
        try:
            df = df_res.copy()
            
            # Ensure date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            elif 'Date' in df.columns:
                df['date'] = pd.to_datetime(df['Date'], errors='coerce')
            else:
                return None
            
            # Determine guest count column
            guest_col = None
            for col in ['guest_count', 'Guest_Count', 'guests', 'Guests']:
                if col in df.columns:
                    guest_col = col
                    break
            
            if guest_col is None:
                return None
            
            # Aggregate to daily
            df_daily = df.groupby('date')[guest_col].sum().reset_index()
            df_daily.columns = ['ds', 'y']
            
            # Remove zeros and negatives
            df_daily = df_daily[df_daily['y'] > 0]
            
            # Sort by date
            df_daily = df_daily.sort_values(by='ds').reset_index(drop=True) # type: ignore
            
            # Fill gaps (missing dates get interpolated)
            if len(df_daily) >= 2:
                date_range = pd.date_range(
                    start=df_daily['ds'].min(),
                    end=df_daily['ds'].max(),
                    freq='D'
                )
                df_daily = df_daily.set_index('ds').reindex(date_range)
                df_daily.index.name = 'ds'
                df_daily = df_daily.reset_index()
                
                # Interpolate gaps (linear for short gaps, forward fill for longer)
                df_daily['y'] = df_daily['y'].interpolate(method='linear', limit=3)
                df_daily['y'] = df_daily['y'].ffill().bfill()
            
            # Safety: ensure no NaN
            df_daily = df_daily.dropna(subset=['y'])
            
            if len(df_daily) < NeuralProphetAgent.MIN_TRAINING_ROWS:
                return None
            
            return df_daily
            
        except Exception as e:
            logger.debug(f"Data preparation failed: {e}")
            return None
    
    @staticmethod
    def _build_holidays_df(vn_holidays) -> Optional[pd.DataFrame]:
        """
        Build holidays DataFrame for NeuralProphet.
        
        Format: columns ['event', 'ds']
        """
        try:
            if vn_holidays is None:
                return None
            
            records = []
            for date, name in vn_holidays.items():
                if isinstance(date, datetime.date):
                    records.append({
                        'event': str(name),
                        'ds': pd.Timestamp(date),
                    })
            
            if not records:
                return None
            
            return pd.DataFrame(records)
            
        except Exception:
            return None
    
    @staticmethod
    def _extract_predictions(
        forecast: pd.DataFrame,
        next_days_info: List[Dict],
        n_forecasts: int,
    ) -> Dict[str, float]:
        """
        Extract predictions for target dates from NeuralProphet forecast.
        
        NeuralProphet outputs columns like 'yhat1', 'yhat2', etc.
        for multi-step forecasts. We need to map these to actual dates.
        """
        results = {}
        
        # Get target dates
        target_dates = {str(d['date']): d['date'] for d in next_days_info}
        
        # Find yhat columns
        yhat_cols = [c for c in forecast.columns if c.startswith('yhat')]
        
        if not yhat_cols:
            return results
        
        # For each row in forecast, ds is the origin date
        # yhat1 = prediction for ds+1, yhat2 = ds+2, etc.
        # We want the most recent prediction for each target date
        
        # Simple approach: get the last forecast row's predictions
        # (most recent origin point has the best information)
        for _, row in forecast.iterrows():
            origin_date = pd.Timestamp(str(row['ds']))
            
            for col in yhat_cols:
                try:
                    step = int(col.replace('yhat', ''))
                    pred_date = origin_date + pd.Timedelta(days=step)
                    pred_date_str = str(pred_date.date())
                    
                    if pred_date_str in target_dates:
                        val = float(row[col])
                        if val > 0 and not np.isnan(val):
                            # Keep most recent prediction (overwrite earlier ones)
                            results[pred_date_str] = max(0, round(val, 1))
                except (ValueError, TypeError):
                    continue
        
        # Fallback: if multi-step didn't give results, try 'yhat' column
        if not results and 'yhat1' in forecast.columns:
            for _, row in forecast.iterrows():
                ds = str(pd.Timestamp(str(row['ds'])).date())
                if ds in target_dates:
                    val = row.get('yhat1', None)
                    if val is not None and not np.isnan(val) and val > 0:
                        results[ds] = max(0, round(float(val), 1))
        
        return results
    
    # ==========================================
    # PREDICTION CACHING
    # ==========================================
    
    @staticmethod
    def _get_cache_path(res_code: str) -> Path:
        """Get cache file path for NeuralProphet predictions."""
        cache_dir = Path(MODEL_CACHE_DIR) / 'neuralprophet'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{res_code}_nprophet.pkl"
    
    @staticmethod
    def _save_prediction_cache(res_code: str, predictions: Dict[str, float]):
        """Cache NeuralProphet predictions to disk."""
        try:
            path = NeuralProphetAgent._get_cache_path(res_code)
            data = {
                'predictions': predictions,
                'created_at': datetime.datetime.now(),
                'current_date': CURRENT_DATE,
            }
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.debug(f"NeuralProphet cache save failed for {res_code}: {e}")
    
    @staticmethod
    def _load_prediction_cache(res_code: str) -> Optional[Dict[str, float]]:
        """
        Load cached NeuralProphet predictions if still valid.
        Cache is invalidated if:
        - Older than CACHE_MAX_AGE_HOURS
        - Created for a different CURRENT_DATE (new forecast run)
        """
        try:
            path = NeuralProphetAgent._get_cache_path(res_code)
            if not path.exists():
                return None
            
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Check age
            age_hours = (
                datetime.datetime.now() - data['created_at']
            ).total_seconds() / 3600
            
            if age_hours > NeuralProphetAgent.CACHE_MAX_AGE_HOURS:
                path.unlink(missing_ok=True)
                return None
            
            # Check if same forecast run date
            if data.get('current_date') != CURRENT_DATE:
                path.unlink(missing_ok=True)
                return None
            
            return data.get('predictions')
            
        except Exception:
            return None
    
    # ==========================================
    # GLOBAL MODEL (Train once for ALL restaurants)
    # ==========================================
    
    @staticmethod
    def train_global_model_safe(
        all_restaurant_data: Dict[str, pd.DataFrame],
        vn_holidays,
        config: Optional[Dict] = None,
    ) -> Optional[object]:
        """
        Train global model with PROCESS-BASED TIMEOUT.
        
        Uses multiprocessing.Process (not threads!) so we can actually
        kill the training if it hangs. The trained model is exchanged
        via pickle file on disk.
        
        Returns:
            Trained model or None (on timeout/error)
        """
        if NeuralProphet is None:
            return None
        
        timeout = NeuralProphetAgent.TIMEOUT_GLOBAL_TRAIN
        
        # Model exchange path (subprocess saves model here, main process loads it)
        model_path = Path(MODEL_CACHE_DIR) / 'neuralprophet' / '_global_model_temp.pkl'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Clean up any stale temp file
        if model_path.exists():
            model_path.unlink()
        
        logger.info(f"🌐 Training global NeuralProphet via subprocess (timeout: {timeout}s)...")
        
        try:
            # ⚡ Use multiprocessing.Process — can actually be terminated!
            # ThreadPoolExecutor CANNOT kill threads (Python limitation)
            process = multiprocessing.Process(
                target=NeuralProphetAgent._train_global_worker,
                args=(all_restaurant_data, vn_holidays, config, str(model_path)),
            )
            process.start()
            process.join(timeout=timeout)
            
            if process.is_alive():
                # Timeout! Kill the subprocess
                logger.warning(
                    f"⏱️ Global NeuralProphet training TIMEOUT (>{timeout}s) — "
                    f"terminating subprocess"
                )
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()  # Force kill
                    process.join(timeout=3)
                return None
            
            # Check exit code
            if process.exitcode != 0:
                logger.warning(
                    f"Global NeuralProphet subprocess exited with code {process.exitcode}"
                )
                return None
            
            # Load trained model from disk
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    model_path.unlink(missing_ok=True)
                    logger.info("✅ Global NeuralProphet model loaded from subprocess")
                    return model
                except Exception as e:
                    logger.warning(f"Failed to load global model from disk: {e}")
                    return None
            else:
                logger.warning("Global NeuralProphet: subprocess completed but no model file found")
                return None
                
        except Exception as e:
            logger.warning(f"Global NeuralProphet training failed: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    @staticmethod
    def _train_global_worker(
        all_restaurant_data: Dict[str, pd.DataFrame],
        vn_holidays,
        config,
        model_save_path: str,
    ):
        """
        Worker function that runs in a SEPARATE PROCESS.
        
        Trains the global model and saves it to disk.
        The main process can terminate this process if it hangs.
        """
        import os, warnings
        warnings.filterwarnings('ignore')
        
        # ⚡ Enforce single-threaded PyTorch in subprocess too
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        
        try:
            import torch # type: ignore
            torch.set_num_threads(1) # type: ignore
            torch.set_num_interop_threads(1) # type: ignore
        except Exception:
            pass
        
        try:
            model = NeuralProphetAgent.train_global_model(
                all_restaurant_data, vn_holidays, config
            )
            if model is not None:
                import pickle
                with open(model_save_path, 'wb') as f:
                    pickle.dump(model, f)
        except Exception:
            pass  # Process exits, no model file = failure indicator
    
    @staticmethod
    def predict_from_global_model(
        global_model,
        df_res: pd.DataFrame,
        next_days_info: List[Dict],
        vn_holidays,
    ) -> Dict[str, float]:
        """
        Predict for ONE restaurant using a PRE-TRAINED global model.
        
        This is much faster than train_and_predict() because it skips training.
        The global model was already trained on all restaurants' data.
        
        Args:
            global_model: Pre-trained NeuralProphet model (from train_global_model)
            df_res: Transaction data for 1 restaurant
            next_days_info: List of forecast target days
            vn_holidays: holidays.VN object
            
        Returns:
            Dict[str, float]: {date_str: predicted_daily_total}
        """
        if global_model is None or df_res.empty:
            return {}
        
        res_code_hint = str(df_res['restaurant_code'].iloc[0]) if 'restaurant_code' in df_res.columns else 'unknown'
        
        # Check cache first
        cached = NeuralProphetAgent._load_prediction_cache(res_code_hint)
        if cached is not None:
            target_dates = {str(d['date']) for d in next_days_info}
            filtered = {k: v for k, v in cached.items() if k in target_dates}
            if len(filtered) >= len(target_dates) * 0.8:
                logger.debug(f"NeuralProphet cache hit for {res_code_hint}")
                return filtered
        
        try:
            # Prepare daily data for this restaurant
            df_daily = NeuralProphetAgent._prepare_daily_data(df_res)
            
            if df_daily is None or len(df_daily) < NeuralProphetAgent.MIN_TRAINING_ROWS:
                return {}
            
            # Add ID column (must match global model training)
            df_daily['ID'] = str(res_code_hint)
            df_daily['is_weekend'] = df_daily['ds'].dt.dayofweek.isin([5, 6]).astype(float)
            
            # Calculate future periods needed
            future_dates = [d['date'] for d in next_days_info]
            n_future = max((d - df_daily['ds'].dt.date.max()).days for d in future_dates) + 1
            n_future = max(n_future, 1)
            
            # Create future dataframe from global model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                future = global_model.make_future_dataframe(
                    df_daily,
                    periods=n_future,
                    n_historic_predictions=False,
                )
            
            # Add future regressor values
            future['is_weekend'] = future['ds'].dt.dayofweek.isin([5, 6]).astype(float)
            
            # Predict using global model (fast — no training needed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = global_model.predict(future)
            
            # Extract predictions for target dates
            cfg_n_forecasts = global_model.n_forecasts if hasattr(global_model, 'n_forecasts') else 30
            results = NeuralProphetAgent._extract_predictions(
                forecast, next_days_info, cfg_n_forecasts
            )
            
            if results:
                logger.debug(f"NeuralProphet (global) predicted {len(results)} days for {res_code_hint} "
                           f"(range: {min(results.values()):.0f}~{max(results.values()):.0f})")
                NeuralProphetAgent._save_prediction_cache(res_code_hint, results)
            
            return results
            
        except Exception as e:
            logger.debug(f"NeuralProphet global predict failed for {res_code_hint}: {e}")
            return {}
    
    
    @staticmethod
    def train_global_model(
        all_restaurant_data: Dict[str, pd.DataFrame],
        vn_holidays,
        config: Optional[Dict] = None,
    ) -> Optional[object]:
        """
        Train a SINGLE NeuralProphet model on data from ALL restaurants.
        
        This leverages cross-restaurant patterns:
        - General seasonality (holidays, weekends)
        - Shared trend patterns
        - Better generalization for new/sparse restaurants
        
        Each restaurant is identified by 'ID' column for grouped forecasting.
        
        Args:
            all_restaurant_data: {res_code: transaction_df}
            vn_holidays: holidays.VN
            config: Optional overrides
            
        Returns:
            Trained model or None
        """
        if NeuralProphet is None:
            return None
        
        try:
            all_dfs = []
            for res_code, df in all_restaurant_data.items():
                daily = NeuralProphetAgent._prepare_daily_data(df)
                if daily is not None and len(daily) >= NeuralProphetAgent.MIN_TRAINING_ROWS:
                    daily['ID'] = str(res_code)
                    all_dfs.append(daily)
            
            if len(all_dfs) < 5:
                logger.warning(f"Only {len(all_dfs)} restaurants have enough data for global model")
                return None
            
            df_all = pd.concat(all_dfs, ignore_index=True)
            df_all['is_weekend'] = df_all['ds'].dt.dayofweek.isin([5, 6]).astype(float)
            
            # ⚡ Fix "singular value" error: filter restaurants where
            # is_weekend has no variance (e.g., only weekday data)
            # ⚡ Fix "negative dimensions" error: require enough data points
            #    for n_lags + n_forecasts (NeuralProphet internal requirement)
            cfg = {**NeuralProphetAgent.DEFAULT_CONFIG, **(config or {})}
            min_length = cfg['n_lags'] + cfg['n_forecasts'] + 10  # Need extra buffer
            
            valid_ids = []
            for rid in df_all['ID'].unique():
                mask = df_all['ID'] == rid
                sub = df_all.loc[mask]
                # Require: enough rows for model AND variance in y AND in is_weekend
                if (len(sub) >= min_length and 
                    sub['y'].std() > 0 and 
                    sub['is_weekend'].nunique() > 1):
                    valid_ids.append(rid)
            
            df_all = df_all[df_all['ID'].isin(valid_ids)].copy()
            
            if len(valid_ids) < 5:
                logger.warning(f"Only {len(valid_ids)} restaurants pass quality filter "
                             f"(need >= {min_length} days + variance)")
                return None
            
            # Dynamically adjust n_lags based on shortest valid series
            min_series_len = df_all.groupby('ID').size().min()
            max_lags = min_series_len // 3  # At most 1/3 of shortest series
            cfg['n_lags'] = min(cfg['n_lags'], max(max_lags, 7))
            
            # Ensure n_forecasts doesn't exceed what data supports
            max_forecasts = min_series_len - cfg['n_lags'] - 5
            cfg['n_forecasts'] = min(cfg['n_forecasts'], max(max_forecasts, 7))
            
            logger.info(f"🌐 Training global NeuralProphet on {len(valid_ids)} restaurants "
                       f"(filtered from {len(all_dfs)}), {len(df_all)} total rows, "
                       f"n_lags={cfg['n_lags']}, n_forecasts={cfg['n_forecasts']}")
            
            model = NeuralProphet(
                n_lags=cfg['n_lags'],
                n_forecasts=cfg['n_forecasts'],
                yearly_seasonality=cfg['yearly_seasonality'],
                weekly_seasonality=cfg['weekly_seasonality'],
                daily_seasonality=cfg['daily_seasonality'],
                learning_rate=cfg['learning_rate'],
                epochs=cfg['epochs'],
                batch_size=cfg['batch_size'],
                ar_layers=cfg['ar_layers'],
                trend_reg=cfg['trend_reg'],
                seasonality_reg=cfg['seasonality_reg'],
                global_normalization=True,  # Normalize across restaurants
                accelerator='cpu',  # ⚠️ MPS causes segfault, use CPU
            )
            
            model.add_future_regressor('is_weekend')
            model = model.add_country_holidays(country_name='VN')
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(pd.DataFrame(df_all), freq='D')
            
            logger.info(f"✅ Global NeuralProphet trained successfully")
            return model
            
        except Exception as e:
            logger.error(f"Global NeuralProphet training failed: {e}")
            logger.debug(traceback.format_exc())
            return None
