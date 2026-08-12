"""
==============================================
AUTO-TUNER AGENT (PHASE 5)
==============================================
Trách nhiệm:
- Tự động tối ưu hyperparameters cho ML models
- Adaptive ensemble weights dựa trên accuracy history
- Restaurant-specific tuning: restaurants với accuracy kém → retune
- Time-series cross-validation (walk-forward)
- Optuna-based hyperparameter search (nếu có)
- Fallback: Grid/Random search nếu không có Optuna

Architecture:
    MonitoringAgent detects problems
        → AutoTuner identifies target restaurants
        → Walk-forward CV tìm hyperparams tối ưu
        → Update model configs
        → Re-train with new params

Tuning Strategies:
    1. QUICK:    Random search, 20 trials, ~2 min/restaurant
    2. STANDARD: Optuna/Random, 50 trials, ~5 min/restaurant  
    3. THOROUGH: Optuna, 100 trials, ~15 min/restaurant
"""

import numpy as np
import pandas as pd
import datetime
import time
import json
import os
import traceback
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

from forecast_system.config.settings import (
    CURRENT_DATE, PROJECT_ROOT, AUTOTUNER_CONFIG,
)
from forecast_system.utils.logger import get_logger

logger = get_logger('auto_tuner')

# Optional: Optuna
try:
    import optuna  # type: ignore[import-not-found]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None

# ML imports
try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    TimeSeriesSplit = None
    mean_absolute_error = None


class AutoTuner:
    """
    Tự động tối ưu hyperparameters cho hệ thống forecast.
    
    Features:
    1. Walk-forward cross-validation (time-series aware)
    2. Per-model hyperparameter search spaces
    3. Optuna integration (optional, fallback to random search)
    4. Adaptive ensemble weight tuning
    5. Restaurant-specific tuning
    6. Config persistence (save/load best params)
    """
    
    TUNED_PARAMS_FILE = str(PROJECT_ROOT / "tuned_params.json")
    
    # ==========================================
    # SEARCH SPACES PER MODEL
    # ==========================================
    
    SEARCH_SPACES = {
        'xgboost': {
            'n_estimators': {'type': 'int', 'range': [50, 300]},
            'learning_rate': {'type': 'float', 'range': [0.01, 0.2]},
            'max_depth': {'type': 'int', 'range': [3, 10]},
            'min_child_weight': {'type': 'int', 'range': [1, 10]},
            'subsample': {'type': 'float', 'range': [0.6, 1.0]},
            'colsample_bytree': {'type': 'float', 'range': [0.6, 1.0]},
            'reg_alpha': {'type': 'float', 'range': [0.0, 1.0]},
            'reg_lambda': {'type': 'float', 'range': [0.5, 3.0]},
        },
        'catboost': {
            'iterations': {'type': 'int', 'range': [50, 300]},
            'learning_rate': {'type': 'float', 'range': [0.01, 0.2]},
            'depth': {'type': 'int', 'range': [3, 10]},
            'l2_leaf_reg': {'type': 'float', 'range': [1.0, 10.0]},
        },
        'lightgbm': {
            'n_estimators': {'type': 'int', 'range': [50, 300]},
            'learning_rate': {'type': 'float', 'range': [0.01, 0.2]},
            'max_depth': {'type': 'int', 'range': [3, 10]},
            'num_leaves': {'type': 'int', 'range': [15, 63]},
            'min_child_samples': {'type': 'int', 'range': [5, 30]},
            'subsample': {'type': 'float', 'range': [0.6, 1.0]},
            'colsample_bytree': {'type': 'float', 'range': [0.6, 1.0]},
            'reg_alpha': {'type': 'float', 'range': [0.0, 1.0]},
            'reg_lambda': {'type': 'float', 'range': [0.5, 3.0]},
        },
        'random_forest': {
            'n_estimators': {'type': 'int', 'range': [50, 200]},
            'max_depth': {'type': 'int', 'range': [5, 20]},
            'min_samples_split': {'type': 'int', 'range': [2, 15]},
            'min_samples_leaf': {'type': 'int', 'range': [1, 10]},
        },
    }
    
    # ==========================================
    # WALK-FORWARD CROSS-VALIDATION
    # ==========================================
    
    @staticmethod
    def walk_forward_cv(
        model_class,
        params: Dict,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 3,
    ) -> float:
        """
        Walk-forward cross-validation cho time series.
        
        Khác với standard CV:
        - Train set luôn ở TRƯỚC test set (thời gian)
        - Không bao giờ "peek into future"
        - Mỗi fold tăng kích thước train set
        
        Returns:
            float: Average MAE across folds
        """
        if TimeSeriesSplit is None or mean_absolute_error is None:
            return float('inf')
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        maes = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            try:
                model = model_class(**params)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                preds = np.maximum(preds, 0)
                mae = mean_absolute_error(y_test, preds) # type: ignore[reportArgumentType]
                maes.append(mae)
            except Exception:
                maes.append(float('inf'))
        
        return float(np.mean(maes)) if maes else float('inf')
    
    # ==========================================
    # OPTUNA TUNING
    # ==========================================
    
    @staticmethod
    def tune_with_optuna(
        model_name: str,
        model_class,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        n_cv_splits: int = 3,
    ) -> Tuple[Dict, float]:
        """
        Tune hyperparameters dùng Optuna (Bayesian optimization).
        
        Returns:
            (best_params, best_mae)
        """
        if optuna is None:
            logger.debug("Optuna not available, falling back to random search")
            return AutoTuner.tune_random_search(
                model_name, model_class, X, y, n_trials, n_cv_splits
            )
        
        search_space = AutoTuner.SEARCH_SPACES.get(model_name, {})
        if not search_space:
            return {}, float('inf')
        
        def objective(trial):
            params = {}
            for param_name, config in search_space.items():
                if config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, config['range'][0], config['range'][1]
                    )
                elif config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, config['range'][0], config['range'][1]
                    )
            
            # Fixed params
            if model_name == 'xgboost':
                params['objective'] = 'reg:squarederror'
                params['verbosity'] = 0
                params['random_state'] = 42
            elif model_name == 'catboost':
                params['verbose'] = 0
                params['random_seed'] = 42
            elif model_name == 'lightgbm':
                params['verbose'] = -1
                params['random_state'] = 42
            elif model_name == 'random_forest':
                params['random_state'] = 42
                params['n_jobs'] = -1
            
            mae = AutoTuner.walk_forward_cv(
                model_class, params, X, y, n_cv_splits
            )
            return mae
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_params = study.best_params
        best_mae = study.best_value
        
        # Add fixed params back
        if model_name == 'xgboost':
            best_params['objective'] = 'reg:squarederror'
            best_params['verbosity'] = 0
            best_params['random_state'] = 42
        elif model_name == 'catboost':
            best_params['verbose'] = 0
            best_params['random_seed'] = 42
        elif model_name == 'lightgbm':
            best_params['verbose'] = -1
            best_params['random_state'] = 42
        elif model_name == 'random_forest':
            best_params['random_state'] = 42
            best_params['n_jobs'] = -1
        
        return best_params, best_mae
    
    # ==========================================
    # RANDOM SEARCH (FALLBACK)
    # ==========================================
    
    @staticmethod
    def tune_random_search(
        model_name: str,
        model_class,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 30,
        n_cv_splits: int = 3,
    ) -> Tuple[Dict, float]:
        """
        Random search hyperparameter tuning (fallback khi không có Optuna).
        
        Returns:
            (best_params, best_mae)
        """
        search_space = AutoTuner.SEARCH_SPACES.get(model_name, {})
        if not search_space:
            return {}, float('inf')
        
        best_params = {}
        best_mae = float('inf')
        
        for trial_i in range(n_trials):
            # Sample random params
            params = {}
            for param_name, config in search_space.items():
                if config['type'] == 'int':
                    params[param_name] = np.random.randint(
                        config['range'][0], config['range'][1] + 1
                    ) # type: ignore[reportCallIssue]
                elif config['type'] == 'float':
                    params[param_name] = np.random.uniform(
                        config['range'][0], config['range'][1]
                    )
            
            # Fixed params
            if model_name == 'xgboost':
                params['objective'] = 'reg:squarederror'
                params['verbosity'] = 0
                params['random_state'] = 42
            elif model_name == 'catboost':
                params['verbose'] = 0
                params['random_seed'] = 42
            elif model_name == 'lightgbm':
                params['verbose'] = -1
                params['random_state'] = 42
            elif model_name == 'random_forest':
                params['random_state'] = 42
                params['n_jobs'] = -1
            
            mae = AutoTuner.walk_forward_cv(
                model_class, params, X, y, n_cv_splits
            )
            
            if mae < best_mae:
                best_mae = mae
                best_params = params.copy()
        
        return best_params, best_mae
    
    # ==========================================
    # FULL TUNING PIPELINE
    # ==========================================
    
    @staticmethod
    def tune_all_models(
        X: pd.DataFrame,
        y: pd.Series,
        strategy: str = 'STANDARD',
    ) -> Dict:
        """
        Tune tất cả available models.
        
        Args:
            X: Feature matrix
            y: Target variable
            strategy: 'QUICK', 'STANDARD', 'THOROUGH'
        
        Returns:
            Dict: {model_name: {'params': best_params, 'mae': best_mae}}
        """
        from forecast_system.agents.ensemble_agent import EnsembleMLAgent
        
        n_trials_map = {
            'QUICK': AUTOTUNER_CONFIG.get('quick_trials', 20),
            'STANDARD': AUTOTUNER_CONFIG.get('standard_trials', 50),
            'THOROUGH': AUTOTUNER_CONFIG.get('thorough_trials', 100),
        }
        n_trials = n_trials_map.get(strategy, 50)
        n_cv_splits = 3 if strategy != 'THOROUGH' else 5
        
        available = EnsembleMLAgent.get_available_models()
        results = {}
        
        logger.info(f"🔧 Auto-tuning {len(available)} models "
                    f"(strategy={strategy}, trials={n_trials})")
        
        for model_name, config in available.items():
            start = time.time()
            logger.info(f"  Tuning {model_name}...")
            
            try:
                if HAS_OPTUNA:
                    best_params, best_mae = AutoTuner.tune_with_optuna(
                        model_name, config['class'], X, y,
                        n_trials=n_trials, n_cv_splits=n_cv_splits
                    )
                else:
                    best_params, best_mae = AutoTuner.tune_random_search(
                        model_name, config['class'], X, y,
                        n_trials=n_trials, n_cv_splits=n_cv_splits
                    )
                
                elapsed = time.time() - start
                
                # Compare with default
                default_mae = AutoTuner.walk_forward_cv(
                    config['class'], config['params'], X, y, n_cv_splits
                )
                
                improvement = (
                    (default_mae - best_mae) / default_mae * 100
                    if default_mae > 0 else 0
                )
                
                results[model_name] = {
                    'params': best_params,
                    'mae': round(best_mae, 3),
                    'default_mae': round(default_mae, 3),
                    'improvement_pct': round(improvement, 1),
                    'tuning_time_seconds': round(elapsed, 1),
                }
                
                better = '✅' if improvement > 0 else '⚠️'
                logger.info(
                    f"    {better} {model_name}: "
                    f"MAE {default_mae:.3f} → {best_mae:.3f} "
                    f"({improvement:+.1f}%) | {elapsed:.1f}s"
                )
                
            except Exception as e:
                logger.warning(f"    ❌ {model_name} tuning failed: {e}")
                results[model_name] = {
                    'params': config['params'],
                    'mae': float('inf'),
                    'error': str(e),
                }
        
        return results
    
    # ==========================================
    # ADAPTIVE ENSEMBLE WEIGHT TUNING
    # ==========================================
    
    @staticmethod
    def tune_ensemble_weights(
        df_master: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """
        Tự động điều chỉnh ensemble weights dựa trên accuracy history.
        
        Logic:
        1. Tính accuracy riêng cho ML vs AI
        2. Model nào tốt hơn → tăng weight
        3. Update STRATEGY_WEIGHTS dictionary
        
        Returns:
            Dict: Updated strategy weights
        """
        from forecast_system.agents.monitoring_agent import MonitoringAgent
        
        # Get ML vs AI comparison
        comparison = MonitoringAgent.compare_ml_vs_ai(df_master)
        
        if not comparison.get('ensemble') or not comparison.get('ai_raw'):
            logger.info("Not enough data to tune ensemble weights")
            return {}
        
        ens_mape = comparison['ensemble'].get('MAPE', 50)
        ai_mape = comparison['ai_raw'].get('MAPE', 50)
        
        # Calculate optimal ML weight based on error ratio
        # Lower error → higher weight
        total_error = ens_mape + ai_mape
        if total_error > 0:
            # Inverse error weighting
            ml_optimal = ai_mape / total_error  # Higher AI error → more ML weight
            ml_optimal = max(0.2, min(0.8, ml_optimal))  # Clamp 20-80%
        else:
            ml_optimal = 0.5
        
        ai_optimal = 1.0 - ml_optimal
        
        # Build updated weights
        updated_weights = {
            'ENSEMBLE_EQUAL': {'ml': round(ml_optimal, 2), 'ai': round(ai_optimal, 2)},
            'ENSEMBLE_WEIGHTED': {'ml': round(ml_optimal, 2), 'ai': round(ai_optimal, 2)},
            'AI_ONLY': {'ml': 0.0, 'ai': 1.0},  # Keep as-is
            'AI_PRIMARY_ML_SECONDARY': {
                'ml': round(min(0.4, ml_optimal), 2),
                'ai': round(max(0.6, ai_optimal), 2),
            },
            'ML_PRIMARY_AI_VALIDATE': {
                'ml': round(max(0.6, ml_optimal), 2),
                'ai': round(min(0.4, ai_optimal), 2),
            },
        }
        
        logger.info(f"🎯 Optimal weights: ML={ml_optimal:.0%}, AI={ai_optimal:.0%}")
        logger.info(f"   Based on: Ensemble MAPE={ens_mape:.1f}%, AI MAPE={ai_mape:.1f}%")
        
        return updated_weights
    
    # ==========================================
    # PERSISTENCE (SAVE/LOAD)
    # ==========================================
    
    @staticmethod
    def save_tuned_params(results: Dict):
        """Lưu tuned params vào JSON file."""
        save_data = {
            'tuned_date': str(CURRENT_DATE),
            'timestamp': datetime.datetime.now().isoformat(),
            'models': {},
        }
        
        for model_name, info in results.items():
            # Convert numpy types to Python types for JSON
            params = {}
            for k, v in info.get('params', {}).items():
                if isinstance(v, (np.integer,)):
                    params[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    params[k] = float(v)
                else:
                    params[k] = v
            
            save_data['models'][model_name] = {
                'params': params,
                'mae': info.get('mae'),
                'improvement_pct': info.get('improvement_pct', 0),
            }
        
        try:
            with open(AutoTuner.TUNED_PARAMS_FILE, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, default=str)
            logger.info(f"💾 Tuned params saved: {AutoTuner.TUNED_PARAMS_FILE}")
        except Exception as e:
            logger.warning(f"Failed to save tuned params: {e}")
    
    @staticmethod
    def load_tuned_params() -> Optional[Dict]:
        """
        Load previously tuned parameters.
        
        Returns:
            Dict hoặc None nếu no saved params / expired
        """
        if not os.path.exists(AutoTuner.TUNED_PARAMS_FILE):
            return None
        
        try:
            with open(AutoTuner.TUNED_PARAMS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check expiry
            tuned_date = datetime.date.fromisoformat(data.get('tuned_date', '2000-01-01'))
            max_age = AUTOTUNER_CONFIG.get('params_max_age_days', 7)
            
            if (CURRENT_DATE - tuned_date).days > max_age:
                logger.info(f"Tuned params expired ({tuned_date}), need re-tuning")
                return None
            
            logger.info(f"📂 Loaded tuned params from {tuned_date}")
            return data.get('models', {})
            
        except Exception as e:
            logger.warning(f"Failed to load tuned params: {e}")
            return None
    
    @staticmethod
    def get_best_params(model_name: str) -> Optional[Dict]:
        """
        Lấy best params cho 1 model cụ thể.
        Ưu tiên: tuned params → default params.
        """
        saved = AutoTuner.load_tuned_params()
        if saved and model_name in saved:
            return saved[model_name].get('params')
        return None
    
    # ==========================================
    # FULL AUTO-TUNE PIPELINE
    # ==========================================
    
    @staticmethod
    def run_auto_tune(
        df_train: pd.DataFrame,
        vn_holidays,
        df_master: Optional[pd.DataFrame] = None,
        strategy: str = 'STANDARD',
    ) -> Dict:
        """
        Run complete auto-tuning pipeline.
        
        Steps:
        1. Prepare combined training data
        2. Feature engineering
        3. Tune all ML models
        4. Tune ensemble weights (if master data available)
        5. Save results
        
        Args:
            df_train: Transaction training data
            vn_holidays: Vietnam holidays
            df_master: Master file (for ensemble weight tuning)
            strategy: 'QUICK', 'STANDARD', 'THOROUGH'
        
        Returns:
            Dict: {
                'model_params': {model: params_info},
                'ensemble_weights': updated_weights,
                'total_time': seconds,
            }
        """
        start = time.time()
        logger.info("=" * 60)
        logger.info(f"🔧 AUTO-TUNER: Starting ({strategy} mode)")
        logger.info("=" * 60)
        
        result = {
            'model_params': {},
            'ensemble_weights': {},
            'total_time': 0,
        }
        
        try:
            # 1. Prepare data
            from forecast_system.agents.ml_forecast_agent import MLForecastAgent
            
            logger.info("\n📊 Step 1: Preparing training data...")
            df_processed = MLForecastAgent.prepare_data(df_train, vn_holidays)
            
            if df_processed.empty or len(df_processed) < 50:
                logger.warning("Not enough data for tuning")
                return result
            
            feature_columns = [
                'shift_id', 'weekday', 'month', 'day_of_month',
                'is_weekend', 'is_holiday',
                'is_friday', 'is_saturday', 'is_sunday',          # v4
                'is_tet', 'is_pre_holiday', 'is_post_holiday', 'holiday_impact',
                'days_to_holiday', 'is_holiday_window',            # v7
                'is_T_minus_1', 'is_T_minus_2', 'is_T_plus_1',    # v7
                'lunar_day', 'lunar_month', 'is_veg',
                'lag_365d', 'same_weekday_last_year', 'yoy_growth_rate',
                'rolling_7d_mean',
                'trend_7d', 'momentum', 'trend_short',             # v4/v5
                'weighted_lag', 'delta_7d', 'trend_signal',        # v5
                'booking_count', 'booking_ratio', 'booking_flag',  # v4
            ]
            available_features = [
                f for f in feature_columns if f in df_processed.columns
            ]
            
            X = pd.DataFrame(df_processed[available_features])
            y = pd.Series(df_processed['guest_count'])
            
            logger.info(f"   Data: {len(X)} samples, {len(available_features)} features")
            
            # 2. Tune all models
            logger.info("\n🔧 Step 2: Tuning ML models...")
            model_results = AutoTuner.tune_all_models(X, y, strategy=strategy)
            result['model_params'] = model_results
            
            # Save tuned params
            AutoTuner.save_tuned_params(model_results)
            
            # 3. Tune ensemble weights
            if df_master is not None and not df_master.empty:
                logger.info("\n⚖️ Step 3: Tuning ensemble weights...")
                weights = AutoTuner.tune_ensemble_weights(df_master)
                result['ensemble_weights'] = weights
            
        except Exception as e:
            logger.error(f"Auto-tuning failed: {e}")
            traceback.print_exc()
        
        result['total_time'] = round(time.time() - start, 1)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info(f"🔧 AUTO-TUNE COMPLETE ({result['total_time']:.0f}s)")
        
        for model, info in result.get('model_params', {}).items():
            imp = info.get('improvement_pct', 0)
            icon = '✅' if imp > 0 else '➡️'
            logger.info(
                f"   {icon} {model}: {imp:+.1f}% improvement"
            )
        
        logger.info("=" * 60)
        
        return result
