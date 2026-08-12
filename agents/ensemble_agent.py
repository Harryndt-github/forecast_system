"""
==============================================
ENSEMBLE FORECAST AGENT (MỚI - PHASE 3)
==============================================
Trách nhiệm:
- Kết hợp kết quả từ NHIỀU ML models (Stacking Ensemble)
- Kết hợp ML predictions + AI predictions theo strategy weights
- Confidence scoring cho mỗi prediction
- Adaptive weighting dựa trên historical accuracy
- Hourly distribution từ AI daily forecast

Đây là "brain" của hệ thống:
    ML Models (XGBoost + CatBoost + LightGBM + Prophet)
       → Stacking → ML_combined
    AI (LM Studio)
       → AI_prediction
    ML_combined + AI_prediction
       → Strategy-based Ensemble → Final Forecast
"""

import numpy as np
import pandas as pd
import datetime
import traceback
from typing import Dict, List, Tuple, Optional

from forecast_system.config.settings import (
    CURRENT_DATE, STRATEGY_WEIGHTS, ANALYSIS_CONFIG,
    META_LEARNER_ENABLED,
    TREND_SPIKE_THRESHOLD, TREND_DROP_THRESHOLD,
    TREND_ADJUST_SPIKE_MIN, TREND_ADJUST_SPIKE_MAX,
    TREND_ADJUST_DROP_MIN, TREND_ADJUST_DROP_MAX,
    ML_STACKING_SHARE, NP_SHARE, PROPHET_SHARE,
    LOW_VOLUME_DAILY_THRESHOLD, LOW_VOLUME_ROUND_THRESHOLD,
    HOLIDAY_CURVE_HIGH_VOLUME, HOLIDAY_CURVE_MEDIUM_VOLUME, HOLIDAY_CURVE_LOW_VOLUME,
    HOLIDAY_BOOKING_OVERRIDE_THRESHOLD, MEDIUM_VOLUME_DAILY_THRESHOLD,
    BOOKING_THRESHOLD_RATIO,
    # ⭐ V8: Weekend×EVENING optimization
    SHIFT_ALPHA_DEFAULT, SHIFT_ALPHA_VOLATILE,
    SHIFT_ALPHA_CV_STABLE, SHIFT_ALPHA_CV_NOISY,
    WEEKEND_EVENING_WEIGHT_BASE, WEEKEND_EVENING_WEIGHT_HIGH,
)
from forecast_system.utils.logger import get_logger
from forecast_system.utils.date_utils import get_lunar_info, get_holiday_info, get_special_event_info
from forecast_system.agents.ml_forecast_agent import MLForecastAgent

logger = get_logger('ensemble_agent')

# Safe imports
try:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    from prophet import Prophet
    import logging
    logging.getLogger('prophet').setLevel(logging.WARNING)
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    from forecast_system.agents.neuralprophet_agent import NeuralProphetAgent, HAS_NEURALPROPHET
except ImportError:
    HAS_NEURALPROPHET = False


class EnsembleMLAgent:
    """
    Kết hợp nhiều ML models bằng Stacking Ensemble.
    
    Level 0 (Base models):
        - XGBoost
        - CatBoost 
        - LightGBM
        - RandomForest
    
    Level 1 (Meta-learner):
        - Ridge Regression (kết hợp predictions từ base models)
    
    Output:
        - Stacked prediction (hourly)
        - Individual model predictions (để debug/compare)
        - Training metrics (MAE per model)
    """
    
    @staticmethod
    def get_available_models() -> Dict:
        """
        Trả về dict các models có sẵn với cấu hình tối ưu.
        Mỗi model được tune cho bài toán guest count forecasting.
        """
        models: Dict = {}
        
        if HAS_XGB:
            models['xgboost'] = {
                'class': XGBRegressor,  # type: ignore[reportPossiblyUnboundVariable]
                'params': {
                    'n_estimators': 150,
                    'learning_rate': 0.08,
                    'max_depth': 6,
                    'min_child_weight': 3,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': 42,
                    'objective': 'reg:squarederror',
                    'verbosity': 0,
                },
                'weight': 0.35,  # Default weight nếu không dùng stacking
            }
        
        if HAS_CAT:
            models['catboost'] = {
                'class': CatBoostRegressor,  # type: ignore[reportPossiblyUnboundVariable]
                'params': {
                    'iterations': 150,
                    'learning_rate': 0.08,
                    'depth': 6,
                    'l2_leaf_reg': 3.0,
                    'random_seed': 42,
                    'verbose': 0,
                },
                'weight': 0.30,
            }
        
        if HAS_LGBM:
            models['lightgbm'] = {
                'class': LGBMRegressor,  # type: ignore[reportPossiblyUnboundVariable]
                'params': {
                    'n_estimators': 150,
                    'learning_rate': 0.08,
                    'max_depth': 6,
                    'num_leaves': 31,
                    'min_child_samples': 10,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': 42,
                    'verbose': -1,
                },
                'weight': 0.25,
            }
        
        if HAS_SKLEARN:
            from sklearn.ensemble import RandomForestRegressor
            models['random_forest'] = {
                'class': RandomForestRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 3,
                    'random_state': 42,
                    'n_jobs': -1,
                },
                'weight': 0.10,
            }
        
        return models
    
    @staticmethod
    def train_stacking_ensemble(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_columns: List[str],
        sample_weight: np.ndarray = None,  # ⭐ V8: Weekend×EVENING weighting  # type: ignore[reportArgumentType]
    ) -> Dict:
        """
        Train Stacking Ensemble với multiple base models.
        
        Architecture:
            1. Split data: 70% train base, 30% train meta
            2. Train tất cả base models trên 70% data
            3. Predict trên 30% data → tạo meta-features
            4. Train Ridge meta-learner trên meta-features
            5. Re-train base models trên 100% data
        
        Returns:
            dict: {
                'base_models': {name: fitted_model},
                'meta_learner': fitted Ridge model,
                'model_names': list of model names,
                'metrics': {name: MAE},
                'feature_columns': feature columns used,
            }
        """
        available = EnsembleMLAgent.get_available_models()
        
        if not available:
            logger.error("No ML models available for stacking")
            return {}
        
        if len(X_train) < 20:
            logger.warning("Not enough data for stacking, using single model")
            return EnsembleMLAgent._train_single_best(X_train, y_train, available)
        
        X = X_train[feature_columns].copy()
        # Ensure all feature columns are numeric (fix object dtype from guest_mapping lookups)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(X.median())
        y = y_train.copy()
        
        # --- Step 1: Split for meta-learning ---
        split_idx = int(len(X) * 0.7)
        X_base, X_meta = X.iloc[:split_idx], X.iloc[split_idx:]
        y_base, y_meta = y.iloc[:split_idx], y.iloc[split_idx:]
        # ⭐ V8: Split sample weights in sync
        sw_base = sample_weight[:split_idx] if sample_weight is not None else None
        sw_all = sample_weight if sample_weight is not None else None
        
        # --- Step 2: Train base models on 70% ---
        base_models = {}
        base_predictions_meta = {}
        metrics = {}
        
        # Check for constant targets (causes CatBoost to crash)
        y_base_unique = y_base.nunique()
        y_all_unique = y.nunique()
        constant_target_base = y_base_unique <= 1
        constant_target_all = y_all_unique <= 1
        if constant_target_base:
            logger.debug(f"Stacking: base target has {y_base_unique} unique value(s), "
                        f"will skip models that require variance (e.g. catboost)")
        
        for name, config in available.items():
            try:
                # CatBoost hard-fails on constant targets
                if name == 'catboost' and constant_target_base:
                    logger.debug(f"Stacking: skipping {name} (constant target in base split)")
                    continue
                
                model = config['class'](**config['params'])
                # ⭐ V8: Pass sample_weight to models that support it
                if sw_base is not None and name in ('xgboost', 'lightgbm', 'catboost'):
                    model.fit(X_base, y_base, sample_weight=sw_base)
                else:
                    model.fit(X_base, y_base)
                
                # Predict on meta set
                preds_meta = model.predict(X_meta)
                preds_meta = np.maximum(preds_meta, 0)  # Non-negative
                
                base_predictions_meta[name] = preds_meta
                base_models[name] = model
                
                # Metrics
                mae = mean_absolute_error(y_meta, preds_meta)  # type: ignore[reportPossiblyUnboundVariable]
                metrics[name] = round(mae, 2)
                
            except Exception as e:
                logger.warning(f"Stacking: {name} training failed: {e}")
                continue
        
        if not base_models:
            logger.error("All base models failed in stacking")
            return {}
        
        # --- Step 3: Train meta-learner ---
        meta_features = pd.DataFrame(base_predictions_meta)
        meta_learner = None
        meta_type = 'none'
        
        if len(base_models) >= 2:
            # ⭐ v4: LightGBM meta-learner with contextual features
            if META_LEARNER_ENABLED and HAS_LGBM:
                try:
                    # Add context features to meta-learner
                    meta_ctx = meta_features.copy()
                    if 'weekday' in X_meta.columns:
                        meta_ctx['weekday'] = X_meta['weekday'].values
                    if 'is_weekend' in X_meta.columns:
                        meta_ctx['is_weekend'] = X_meta['is_weekend'].values
                    if 'month' in X_meta.columns:
                        meta_ctx['month'] = X_meta['month'].values
                    if 'is_friday' in X_meta.columns:
                        meta_ctx['is_friday'] = X_meta['is_friday'].values
                    if 'is_saturday' in X_meta.columns:
                        meta_ctx['is_saturday'] = X_meta['is_saturday'].values
                    if 'is_sunday' in X_meta.columns:
                        meta_ctx['is_sunday'] = X_meta['is_sunday'].values
                    
                    meta_learner = LGBMRegressor(  # type: ignore[reportPossiblyUnboundVariable]
                        n_estimators=100,
                        max_depth=4,
                        learning_rate=0.08,
                        num_leaves=15,
                        verbose=-1,
                        random_state=42,
                    )
                    meta_learner.fit(meta_ctx, y_meta)
                    
                    meta_preds = meta_learner.predict(meta_ctx)
                    meta_mae = mean_absolute_error(y_meta, meta_preds)  # type: ignore[reportPossiblyUnboundVariable]
                    metrics['meta_lgbm'] = round(meta_mae, 2)
                    meta_type = 'lgbm'
                    
                    logger.debug(f"⭐ LightGBM meta-learner trained: MAE={meta_mae:.2f}")
                except Exception as e:
                    logger.warning(f"LightGBM meta-learner failed: {e}, falling back to Ridge")
                    meta_learner = None
            
            # Fallback: Ridge meta-learner
            if meta_learner is None and HAS_SKLEARN:
                try:
                    meta_learner = Ridge(alpha=1.0)  # type: ignore[reportPossiblyUnboundVariable]
                    meta_learner.fit(meta_features, y_meta)  # type: ignore[reportArgumentType]
                    
                    meta_preds = meta_learner.predict(meta_features)
                    meta_mae = mean_absolute_error(y_meta, meta_preds)  # type: ignore[reportPossiblyUnboundVariable]
                    metrics['stacking_ensemble'] = round(meta_mae, 2)
                    meta_type = 'ridge'
                    
                    logger.debug(f"Ridge meta-learner trained: MAE={meta_mae:.2f}")
                except Exception as e:
                    logger.warning(f"Meta-learner failed: {e}, using weighted avg")
                    meta_learner = None
        
        # --- Step 4: Re-train base models on 100% data ---
        final_models = {}
        for name, config in available.items():
            if name in base_models:
                try:
                    # CatBoost hard-fails on constant targets
                    if name == 'catboost' and constant_target_all:
                        final_models[name] = base_models[name]  # Use partially trained
                        continue
                    
                    model = config['class'](**config['params'])
                    # ⭐ V8: Pass sample_weight for final training too
                    if sw_all is not None and name in ('xgboost', 'lightgbm', 'catboost'):
                        model.fit(X, y, sample_weight=sw_all)
                    else:
                        model.fit(X, y)
                    final_models[name] = model
                except Exception:
                    final_models[name] = base_models[name]  # Use partially trained
        
        model_names = list(final_models.keys())
        
        result = {
            'base_models': final_models,
            'meta_learner': meta_learner,
            'meta_type': meta_type,  # ⭐ v4: 'lgbm', 'ridge', or 'none'
            'model_names': model_names,
            'metrics': metrics,
            'feature_columns': feature_columns,
            'model_weights': {
                name: available[name]['weight'] 
                for name in model_names if name in available
            },
        }
        
        # Log training summary
        logger.debug(f"Stacking ensemble trained: "
                    f"{len(final_models)} models, "
                    f"metrics={metrics}")
        
        return result
    
    @staticmethod
    def _train_single_best(X_train, y_train, available):
        """Fallback: train chỉ model tốt nhất khi data quá ít"""
        for name, config in available.items():
            try:
                model = config['class'](**config['params'])
                model.fit(X_train, y_train)
                return {
                    'base_models': {name: model},
                    'meta_learner': None,
                    'model_names': [name],
                    'metrics': {},
                    'feature_columns': list(X_train.columns),
                    'model_weights': {name: 1.0},
                }
            except Exception:
                continue
        return {}
    
    @staticmethod
    def predict_stacking(
        ensemble: Dict,
        X_test: pd.DataFrame
    ) -> np.ndarray:
        """
        Dự đoán bằng Stacking Ensemble.
        
        Logic:
        1. Mỗi base model predict
        2. Nếu có meta-learner → dùng meta-learner kết hợp
        3. Nếu không → dùng weighted average
        
        Returns:
            np.ndarray predictions (non-negative)
        """
        if not ensemble or 'base_models' not in ensemble:
            return np.zeros(len(X_test))
        
        base_models = ensemble['base_models']
        meta_learner = ensemble.get('meta_learner')
        meta_type = ensemble.get('meta_type', 'ridge')  # ⭐ v4
        model_weights = ensemble.get('model_weights', {})
        feature_columns = ensemble.get('feature_columns', [])
        
        # Use only available features
        available_cols = [c for c in feature_columns if c in X_test.columns]
        X = X_test[available_cols] if available_cols else X_test
        
        # Get predictions from all base models
        all_preds = {}
        for name, model in base_models.items():
            try:
                preds = model.predict(X)
                all_preds[name] = np.maximum(preds, 0)  # Non-negative
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
                continue
        
        if not all_preds:
            return np.zeros(len(X_test))
        
        # Combine predictions
        if meta_learner is not None and len(all_preds) >= 2:
            try:
                meta_features = pd.DataFrame(all_preds)
                
                # ⭐ v4: LightGBM meta-learner uses context features
                if meta_type == 'lgbm':
                    ctx_cols = ['weekday', 'is_weekend', 'month',
                                'is_friday', 'is_saturday', 'is_sunday']
                    for col in ctx_cols:
                        if col in X.columns:
                            meta_features[col] = X[col].values  # type: ignore[reportAttributeAccessIssue]
                
                combined = meta_learner.predict(meta_features)
                return np.maximum(combined, 0)
            except Exception:
                pass  # Fallback to weighted average
        
        # Weighted average fallback
        total_weight = 0
        weighted_sum = np.zeros(len(X_test))
        
        for name, preds in all_preds.items():
            w = model_weights.get(name, 1.0 / len(all_preds))
            weighted_sum += preds * w
            total_weight += w
        
        if total_weight > 0:
            return weighted_sum / total_weight
        
        # Simple average as last resort
        return np.mean(list(all_preds.values()), axis=0)


class ProphetDailyAgent:
    """
    Prophet model cho daily trend forecasting.
    Prophet tốt ở:
    - Seasonality detection (weekly, yearly)
    - Trend decomposition
    - Holiday effects
    
    Dùng để bổ sung cho hourly ML models.
    """
    
    @staticmethod
    def train_and_predict(
        df_res: pd.DataFrame,
        next_days_info: List[Dict],
        vn_holidays
    ) -> Dict[str, float]:  # type: ignore[reportReturnType]
        """
        Dùng Prophet dự đoán daily total guests.
        
        Args:
            df_res: Transaction data cho 1 nhà hàng
            next_days_info: List forecast target days
            vn_holidays: holidays.VN object
        
        Returns:
            Dict[str, float]: {date_str: predicted_daily_total}
        """
        if not HAS_PROPHET or df_res.empty:
            return {}
        
        try:
            # Prepare Prophet format: ds (date), y (value)
            daily = df_res.groupby('date')['guest_count'].sum().reset_index()
            daily.columns = ['ds', 'y']
            daily['ds'] = pd.to_datetime(daily['ds'])
            
            # Remove any NaN or zero-only rows
            daily = daily[daily['y'] > 0].copy()
            
            # Ensure no duplicate dates and proper index
            daily = daily.drop_duplicates(subset='ds').sort_values('ds').reset_index(drop=True)  # type: ignore[reportCallIssue]
            
            if len(daily) < 14:
                return {}  # Need at least 2 weeks
            
            # Configure Prophet (yearly_seasonality=True to capture annual patterns
            # like Valentine, Christmas, etc.)
            model = Prophet(  # type: ignore[reportPossiblyUnboundVariable]
                yearly_seasonality=True,  # type: ignore[reportArgumentType]
                weekly_seasonality=True,  # type: ignore[reportArgumentType]
                daily_seasonality=False,  # type: ignore[reportArgumentType]
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
            )
            
            # WORKAROUND: Prophet 1.x + pandas 3.x incompatibility
            # Prophet's internal pd.crosstab fails with "duplicate labels" error
            # Monkey-patch crosstab during fit/predict
            _original_crosstab = pd.crosstab
            
            def _patched_crosstab(*args, **kwargs):
                """Wrap crosstab to handle duplicate index issue"""
                try:
                    return _original_crosstab(*args, **kwargs)
                except ValueError:
                    # Fallback: re-run with reset index on inputs
                    if len(args) >= 2:
                        idx = args[0]
                        cols = args[1]
                        if hasattr(idx, 'reset_index'):
                            idx = idx.reset_index(drop=True)
                        if hasattr(cols, 'reset_index'):
                            cols = cols.reset_index(drop=True)
                        return _original_crosstab(idx, cols, **kwargs)
                    raise
            
            pd.crosstab = _patched_crosstab
            
            try:
                model.fit(daily)
                
                # Predict
                future_dates = pd.DataFrame({
                    'ds': pd.to_datetime([d['date'] for d in next_days_info])
                })
                
                forecast = model.predict(future_dates)
            finally:
                pd.crosstab = _original_crosstab  # Always restore
            
            # Build result map
            result = {}
            for _, row in forecast.iterrows():
                date_str = row['ds'].strftime('%Y-%m-%d')  # type: ignore[reportAttributeAccessIssue]
                prediction = max(0.0, float(round(row['yhat'], 1)))  # type: ignore[reportArgumentType, reportCallIssue]
                result[date_str] = prediction
            
            return result
            
        except Exception as e:
            logger.debug(f"Prophet prediction failed: {e}")
            return {}


class EnsembleForecastAgent:
    """
    MAIN ENSEMBLE AGENT (Phase 8: Shift-Based)
    
    Kết hợp:
    1. ML Stacking Ensemble (multiple models) → shift predictions
    2. Prophet daily forecast
    3. AI (LLM) daily forecast
    
    Theo strategy weights từ AnalysisAgent restaurant classification.
    
    Flow (Phase 8 - Shift-Based):
        ML Stacking → shift predictions (ML_shift: MORNING + EVENING)
        Prophet → daily predictions (Prophet_daily)
        AI (LLM) → daily predictions (AI_daily)
        
        ML_daily_total = ML_shift[MORNING] + ML_shift[EVENING]
        
        # Weighted combination (daily level)
        combined_daily = w_ml * ML_daily_total + w_prophet * Prophet_daily + w_ai * AI_daily
        
        # Distribute back to 2 shifts using ML shift ratios or historical shift ratios
        Final_shift = combined_daily * (ML_shift[s] / ML_daily_total)
    """
    
    # Shift ID mapping (must match MLForecastAgent)
    SHIFT_ID_MAP = {'MORNING': 0, 'EVENING': 1}
    
    # ==========================================
    # ⭐ V6: BASELINE FORECAST (Low Volume)
    # ==========================================
    
    @staticmethod
    def run_baseline_forecast(
        res_code: str,
        df_res_cleaned: pd.DataFrame,
        next_days_info: List[Dict],
        analysis_report: Dict,
    ) -> Tuple[List[Dict], Dict]:
        """
        ⭐ v6: Baseline forecast for LOW_VOLUME restaurants.
        
        Instead of complex ML ensemble, use simple median of same-weekday
        historical data for each shift. Much more stable for restaurants
        with avg daily guests < 20.
        
        Logic:
        1. Group historical data by (weekday, shift)
        2. forecast = median(same_weekday_same_shift)
        3. If forecast < 5: round to nearest integer
        4. Apply holiday impact factor
        
        Args:
            res_code: Restaurant code
            df_res_cleaned: Cleaned historical data
            next_days_info: Future days to forecast
            analysis_report: Analysis report from AnalysisAgent
            
        Returns:
            (predictions: List[dict], ensemble_info: dict)
        """
        predictions = []
        ensemble_info = {
            'strategy': 'BASELINE_ONLY',
            'weights': {'ml': 0, 'ai': 0},
            'models_used': ['baseline_median'],
            'metrics': {'model_type': 'v6_baseline'},
        }
        
        if df_res_cleaned.empty:
            return predictions, ensemble_info
        
        # Prepare historical shift-level medians
        df = df_res_cleaned.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Detect if data has shift_id column
        has_shift = 'shift_id' in df.columns
        
        if has_shift:
            # Aggregate by (weekday, shift_id)
            weekday_shift_medians = df.groupby(
                ['weekday', 'shift_id']
            )['guest_count'].median().to_dict()
        else:
            # No shift data: estimate shift split from daily totals
            daily = df.groupby(['date', 'weekday'])['guest_count'].sum().reset_index()
            weekday_medians = daily.groupby('weekday')['guest_count'].median().to_dict()
            weekday_shift_medians = {}
            for wd, total in weekday_medians.items():
                weekday_shift_medians[(wd, 0)] = total * 0.45  # MORNING
                weekday_shift_medians[(wd, 1)] = total * 0.55  # EVENING
        
        # Global fallback
        overall_median = df['guest_count'].median() if not df.empty else 0
        
        DEFAULT_SHIFT_RATIOS = {'MORNING': 0.45, 'EVENING': 0.55}
        
        for d_info in next_days_info:
            d_date = d_info['date']
            d_str = str(d_date)
            weekday = d_info['weekday']
            holiday_impact = d_info.get('holiday_impact', 1.0)
            is_holiday = d_info.get('is_holiday', False)
            is_veg = d_info.get('is_veg', False)
            
            for shift_key in ['MORNING', 'EVENING']:
                shift_id = EnsembleForecastAgent.SHIFT_ID_MAP.get(shift_key, 0 if shift_key == 'MORNING' else 1)
                
                # Get same-weekday, same-shift median
                forecast = weekday_shift_medians.get(
                    (weekday, shift_id),
                    overall_median * DEFAULT_SHIFT_RATIOS[shift_key]
                )
                
                # Apply holiday impact
                if holiday_impact != 1.0 and not is_holiday:
                    forecast *= holiday_impact
                
                # ⭐ v6: Round low forecasts
                if forecast < LOW_VOLUME_ROUND_THRESHOLD:  # type: ignore[reportOptionalOperand]
                    forecast = round(forecast)  # type: ignore[reportArgumentType]
                
                forecast = max(0, int(round(forecast)))  # type: ignore[reportArgumentType]
                
                predictions.append({
                    'date': d_date,
                    'weekday': weekday,
                    'hour': None,
                    'shift': shift_key,
                    'forecast': forecast,
                    'combined_daily': None,
                    'is_holiday': is_holiday,
                    'is_veg': is_veg,
                    'forecast_mode': 'baseline',
                    'volume_segment': 'LOW_VOLUME',
                })
        
        logger.info(
            f"📊 v6 BASELINE forecast: {res_code} "
            f"({len(predictions)} shifts, avg_daily="
            f"{analysis_report.get('profile', {}).get('avg_daily', 0):.1f})"
        )
        
        return predictions, ensemble_info
    
    @staticmethod
    def run_ensemble_forecast(
        res_code: str,
        df_res_cleaned: pd.DataFrame,
        df_processed: pd.DataFrame,
        next_days_info: List[Dict],
        vn_holidays,
        analysis_report: Dict,
        ai_daily_map: Dict[str, float],
        neuralprophet_model=None,
        booking_lookup: Dict = None,  # ⭐ V8 Task 1: booking data pre-loaded  # type: ignore[reportArgumentType]
    ) -> Tuple[List[Dict], Dict]:
        """
        Phase 8: Run complete SHIFT-BASED ensemble forecast cho 1 nhà hàng.
        
        Output: 2 rows/ngày (MORNING + EVENING) thay vì 15 rows (hourly).
        
        Args:
            res_code: Restaurant code
            df_res_cleaned: Cleaned transaction data (outliers removed)
            df_processed: Feature-engineered data từ MLForecastAgent.prepare_data()
                          → ĐÃ là shift-level data (shift_id, not hour)
            next_days_info: Forecast target days
            vn_holidays: Vietnam holidays
            analysis_report: Analysis report từ AnalysisAgent
            ai_daily_map: AI predictions {date_str: daily_total}
        
        Returns:
            (predictions: List[dict], ensemble_info: dict)
        """
        strategy = analysis_report.get('strategy', 'ENSEMBLE_EQUAL')
        weights = STRATEGY_WEIGHTS.get(strategy, {'ml': 0.5, 'ai': 0.5})

        # Memento: Override with per-restaurant dynamic weights if learned
        try:
            from forecast_system.agents.forecast_brain import ForecastBrain as _FB
            _dyn_w = _FB.get_dynamic_weights(res_code)
            if _dyn_w:
                weights = _dyn_w
                ml_f = _dyn_w['ml']
                ai_f = _dyn_w['ai']
                strategy = f'DYNAMIC({ml_f:.2f}ML/{ai_f:.2f}AI)'
                logger.debug(
                    f'Memento: {res_code} dynamic weights ml={ml_f:.2f} ai={ai_f:.2f}'
                )
        except Exception:
            pass

        
        # Phase 8: Use FULL feature list from MLForecastAgent (shift-based)
        feature_columns = MLForecastAgent.get_feature_columns()
        available_features = [f for f in feature_columns if f in df_processed.columns]
        
        # ==========================================
        # 1. ML STACKING ENSEMBLE (Shift-Based)
        # ==========================================
        ml_shift = {}   # {date_str: {shift_key: prediction}}
        ml_daily = {}   # {date_str: total}
        ensemble_info = {
            'strategy': strategy,
            'weights': weights,
            'models_used': [],
            'metrics': {},
        }
        
        # ⭐ V8: Init shift P20/P95 dicts (populated below if ML is used)
        weekday_shift_p95 = {}
        weekday_shift_p20 = {}
        
        if weights['ml'] > 0 and not df_processed.empty:
            # Train stacking ensemble on shift-level data
            X_train = df_processed[available_features].copy()
            for col in X_train.columns:
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            X_train = X_train.fillna(X_train.median())
            y_train = df_processed['guest_count']
            
            # ⭐ V8 Task 4: Build sample weights (3x-5x for Weekend×EVENING)
            sample_weight = None
            sw = np.ones(len(X_train))
            has_weight_signal = False

            if 'outlier_weight' in df_processed.columns:
                outlier_w = pd.to_numeric(
                    df_processed['outlier_weight'].iloc[:len(sw)],
                    errors='coerce'
                ).fillna(1.0).clip(0.20, 1.0).values
                sw *= outlier_w
                has_weight_signal = bool(np.any(outlier_w < 0.999))

            if 'is_weekend' in df_processed.columns and 'shift_id' in df_processed.columns:
                we_mask = (
                    (df_processed['is_weekend'].iloc[:len(sw)] == 1) &
                    (df_processed['shift_id'].iloc[:len(sw)] == EnsembleForecastAgent.SHIFT_ID_MAP.get('EVENING', 1))
                ).values.astype(bool)
                sw[we_mask] *= WEEKEND_EVENING_WEIGHT_BASE  # 3.0
                # High-volume weekend evening samples get even higher weight
                if 'guest_count' in df_processed.columns:
                    high_vol = np.array(df_processed['guest_count'].iloc[:len(sw)].values, dtype=float) > 15
                    high_we_mask = np.array(we_mask) & np.array(high_vol)
                    sw[high_we_mask] = sw[high_we_mask] / WEEKEND_EVENING_WEIGHT_BASE * WEEKEND_EVENING_WEIGHT_HIGH
                n_weighted = int(np.sum(we_mask))
                if n_weighted > 0:
                    logger.debug(
                        f"⭐ V8 sample weights: {n_weighted} Weekend×EVENING samples "
                        f"weighted {WEEKEND_EVENING_WEIGHT_BASE}x-{WEEKEND_EVENING_WEIGHT_HIGH}x"
                    )
            
            if has_weight_signal or not np.allclose(sw, 1.0):
                sample_weight = sw

            ensemble = EnsembleMLAgent.train_stacking_ensemble(
                X_train, y_train, available_features,  # type: ignore[reportArgumentType]
                sample_weight=sample_weight,  # type: ignore[reportArgumentType]  # ⭐ V8
            )
            
            if ensemble:
                ensemble_info['models_used'] = ensemble.get('model_names', [])
                ensemble_info['metrics'] = ensemble.get('metrics', {})
                
                # Phase 8: Shift-based guest mapping
                guest_mapping = df_processed.set_index(
                    df_processed['date'].dt.date.astype(str) + "_" +  # type: ignore[reportAttributeAccessIssue]
                    df_processed['shift_id'].astype(str)
                )['guest_count'].to_dict()
                default_val = df_processed['guest_count'].median()
                
                # Smart fallback: weekday+shift medians
                weekday_shift_medians = df_processed.groupby(
                    ['weekday', 'shift_id']
                )['guest_count'].median().to_dict()
                weekday_medians = df_processed.groupby(
                    'weekday'
                )['guest_count'].median().to_dict()
                
                def smart_lag_fallback(weekday, shift_id):
                    """Use weekday+shift median as fallback."""
                    val = weekday_shift_medians.get((weekday, shift_id))
                    if val is not None:
                        return val
                    val = weekday_medians.get(weekday)
                    if val is not None:
                        return val / 2  # Split daily into 2 shifts
                    return default_val
                
                daily_map_hist = df_processed.groupby(
                    df_processed['date'].dt.date  # type: ignore[reportAttributeAccessIssue]
                )['guest_count'].sum().to_dict()
                recent_dates = sorted(daily_map_hist.keys())[-14:]
                rolling_default = (
                    np.median([daily_map_hist[d] for d in recent_dates])
                    if recent_dates else default_val
                )
                
                # Historical P95 caps per weekday+shift (replaces blunt 2x-median)
                # This avoids clipping Saturday peaks based on weekday-dominated medians
                weekday_shift_p95 = {}  # (weekday, shift_id) → P95 value
                weekday_shift_p20 = {}  # ⭐ V8 Task 5: (weekday, shift_id) → P20 value
                weekday_daily_p95 = {}  # weekday → P95 of daily total
                for wd in range(7):
                    wd_mask = df_processed['weekday'] == wd
                    for sid in EnsembleForecastAgent.SHIFT_ID_MAP.values():
                        vals = df_processed.loc[
                            wd_mask & (df_processed['shift_id'] == sid),
                            'guest_count'
                        ].values
                        if len(vals) >= 3:
                            vals_f = np.array(vals, dtype=float)
                            weekday_shift_p95[(wd, sid)] = float(np.percentile(vals_f, 95))
                            weekday_shift_p20[(wd, sid)] = float(np.percentile(vals_f, 20))
                    # Daily total P95 per weekday
                    wd_daily_vals = [
                        v for d, v in daily_map_hist.items()
                        if hasattr(d, 'weekday') and d.weekday() == wd
                    ]
                    if len(wd_daily_vals) >= 3:
                        weekday_daily_p95[wd] = float(np.percentile(np.array(wd_daily_vals, dtype=float), 95))
                
                # Fallback: overall daily median (only when no weekday P95)
                hist_daily_median = np.median(list(daily_map_hist.values()))
                
                # YoY growth rate
                daily_map_sorted = sorted(daily_map_hist.items())
                if len(daily_map_sorted) > 60:
                    recent_vals = [v for _, v in daily_map_sorted[-28:]]
                    old_vals = [v for _, v in daily_map_sorted[:28]]
                    yoy_rate = np.mean(recent_vals) / max(np.mean(old_vals), 1)
                    yoy_rate = min(max(yoy_rate, 0.5), 2.0)
                else:
                    yoy_rate = 1.0
                
                EVENT_TYPE_MAP = MLForecastAgent.EVENT_TYPE_MAP
                
                # Predict for each day × each shift
                for d_info in next_days_info:
                    target_date = d_info['date']
                    target_date_dt = pd.to_datetime(target_date)
                    d_str = str(target_date)
                    lunar = get_lunar_info(target_date)
                    target_weekday = target_date_dt.dayofweek
                    
                    shift_features = []
                    shift_keys_ordered = []
                    for shift_key, shift_id in EnsembleForecastAgent.SHIFT_ID_MAP.items():
                        fallback = smart_lag_fallback(target_weekday, shift_id)
                        
                        feat = {
                            'shift_id': shift_id,
                            'weekday': target_weekday,
                            'month': target_date_dt.month,
                            'day_of_month': target_date_dt.day,
                            'is_weekend': 1 if target_weekday >= 5 else 0,
                            'is_holiday': 1 if d_info['is_holiday'] else 0,
                            'is_tet': 1 if d_info.get('holiday_type') == 'TET_NGUYEN_DAN' else 0,
                            'is_pre_holiday': 1 if d_info.get('is_pre_holiday') else 0,
                            'is_post_holiday': 1 if d_info.get('is_post_holiday') else 0,
                            'holiday_impact': d_info.get('holiday_impact', 1.0),
                            'days_to_holiday': d_info.get('days_to_holiday', 0),
                            'is_holiday_window': 1 if d_info.get('is_holiday_window') else 0,
                            'is_T_minus_1': 1 if d_info.get('days_to_holiday', 0) == -1 else 0,
                            'is_T_minus_2': 1 if d_info.get('days_to_holiday', 0) == -2 else 0,
                            'is_T_plus_1': 1 if d_info.get('days_to_holiday', 0) == 1 else 0,
                            'lunar_day': lunar['lunar_day'],
                            'lunar_month': lunar['lunar_month'],
                            'is_veg': 1 if lunar['is_veg'] else 0,
                            'rolling_7d_mean': rolling_default,
                            'is_special_event': 1 if d_info.get('is_special_event') else 0,
                            'event_type_encoded': EVENT_TYPE_MAP.get(
                                d_info.get('event_type'), 0
                            ),
                            'yoy_growth_rate': yoy_rate,
                            # ⭐ v4 features
                            'is_friday': 1 if target_weekday == 4 else 0,
                            'is_saturday': 1 if target_weekday == 5 else 0,
                            'is_sunday': 1 if target_weekday == 6 else 0,
                            'trend_7d': 1.0,
                            'momentum': 1.0,
                            # ⭐ V8 Task 1: Real booking data from pre-loaded lookup
                            'booking_count': (
                                booking_lookup.get((str(res_code), d_str), 0)
                                if booking_lookup else 0
                            ),
                            'booking_ratio': (
                                booking_lookup.get((str(res_code), d_str), 0)
                                / max(analysis_report.get('profile', {}).get('avg_daily', 50), 1)
                                if booking_lookup else 0.0
                            ),
                            'booking_flag': (
                                1 if booking_lookup and
                                booking_lookup.get((str(res_code), d_str), 0)
                                / max(analysis_report.get('profile', {}).get('avg_daily', 50), 1)
                                > BOOKING_THRESHOLD_RATIO else 0
                            ),
                            # ⭐ v5 features
                            'trend_short': 1.0,
                            'weighted_lag': fallback,
                            'delta_7d': 0.0,
                            'trend_signal': 0,
                            'is_outlier_day': 0,
                        }
                        # Shift-based lag features
                        for lag_days in [7, 14, 28]:
                            lag_date_str = (
                                target_date - datetime.timedelta(days=lag_days)
                            ).strftime('%Y-%m-%d')
                            lag_key = f"{lag_date_str}_{shift_id}"
                            feat[f'lag_{lag_days}d'] = guest_mapping.get(
                                lag_key, fallback
                            )
                        
                        # lag_365d
                        lag_365_date_str = (
                            target_date - datetime.timedelta(days=365)
                        ).strftime('%Y-%m-%d')
                        feat['lag_365d'] = guest_mapping.get(
                            f"{lag_365_date_str}_{shift_id}", fallback
                        )
                        
                        # same_weekday_last_year (364 days)
                        same_wd_date_str = (
                            target_date - datetime.timedelta(days=364)
                        ).strftime('%Y-%m-%d')
                        feat['same_weekday_last_year'] = guest_mapping.get(
                            f"{same_wd_date_str}_{shift_id}", fallback
                        )
                        
                        # ⭐ v5: Recalculate weighted_lag after lags populated
                        feat['weighted_lag'] = (
                            0.6 * feat['lag_7d'] +
                            0.3 * feat['lag_14d'] +
                            0.1 * feat['lag_28d']
                        )
                        
                        shift_features.append(feat)
                        shift_keys_ordered.append(shift_key)
                    
                    X_test = pd.DataFrame(shift_features)
                    for missing_feature in available_features:
                        if missing_feature not in X_test.columns:
                            X_test[missing_feature] = 0
                    X_test = X_test[available_features]
                    
                    try:
                        preds = EnsembleMLAgent.predict_stacking(ensemble, X_test)  # type: ignore[reportArgumentType]
                        
                        shift_preds = {}
                        for sk, p in zip(shift_keys_ordered, preds):
                            shift_preds[sk] = max(0, float(p))
                        
                        ml_shift[d_str] = shift_preds
                        raw_daily = sum(shift_preds.values())
                        
                        # Soft sanity check: cap at weekday-specific P95
                        wd_p95 = weekday_daily_p95.get(target_weekday)
                        ml_extreme_cap = wd_p95 if wd_p95 else hist_daily_median * 2.0
                        if raw_daily > ml_extreme_cap and ml_extreme_cap > 0:
                            scale = ml_extreme_cap / raw_daily
                            shift_preds = {s: v * scale for s, v in shift_preds.items()}
                            ml_shift[d_str] = shift_preds  # type: ignore[reportArgumentType]
                            raw_daily = ml_extreme_cap
                        
                        ml_daily[d_str] = raw_daily
                        
                    except Exception as e:
                        logger.warning(f"Stacking predict failed for {res_code} "
                                     f"on {d_str}: {e}")
        
        # ==========================================
        # 2. PROPHET DAILY FORECAST
        # ==========================================
        prophet_daily = {}
        if HAS_PROPHET and not df_res_cleaned.empty:
            prophet_daily = ProphetDailyAgent.train_and_predict(
                df_res_cleaned, next_days_info, vn_holidays
            )
            if prophet_daily:
                ensemble_info['models_used'].append('prophet')
        
        # ==========================================
        # 2.5 NEURALPROPHET DAILY FORECAST
        # ==========================================
        neuralprophet_daily = {}
        if HAS_NEURALPROPHET and not df_res_cleaned.empty:
            if neuralprophet_model is not None:
                neuralprophet_daily = NeuralProphetAgent.predict_from_global_model(  # type: ignore[reportPossiblyUnboundVariable]
                    neuralprophet_model, df_res_cleaned, next_days_info, vn_holidays
                )
            if neuralprophet_daily:
                ensemble_info['models_used'].append('neuralprophet')
                logger.debug(f"NeuralProphet contributed {len(neuralprophet_daily)} daily predictions")
        
        # ==========================================
        # 2.7 WEEKDAY-SPECIFIC SHIFT RATIOS
        # ==========================================
        from forecast_system.agents.data_agent import DataAgent
        weekday_shift_ratios = DataAgent.get_weekday_shift_ratios(df_res_cleaned)
        
        # ==========================================
        # 3. SHIFT-BASED ENSEMBLE COMBINATION
        # ==========================================
        # Smart historical max cap: ngày thường vs ngày lễ/special event
        profile = analysis_report.get('profile', {})
        hist_max_normal = profile.get('max_daily_normal') or profile.get('max_daily')
        hist_max_normal_weekday = profile.get('max_daily_normal_weekday') or hist_max_normal
        hist_max_normal_weekend = profile.get('max_daily_normal_weekend') or hist_max_normal
        hist_max_by_holiday = profile.get('max_daily_by_holiday', {})
        
        # Smart historical min floor: ngày thường (3 tháng gần nhất)
        hist_min_normal = profile.get('min_daily_normal', 0)
        hist_min_normal_weekday = profile.get('min_daily_normal_weekday') or hist_min_normal
        hist_min_normal_weekend = profile.get('min_daily_normal_weekend') or hist_min_normal
        
        predictions = EnsembleForecastAgent._combine_predictions(
            next_days_info=next_days_info,
            ml_shift=ml_shift,
            ml_daily=ml_daily,
            prophet_daily=prophet_daily,
            neuralprophet_daily=neuralprophet_daily,
            ai_daily_map=ai_daily_map,
            weights=weights,
            analysis_report=analysis_report,
            weekday_shift_ratios=weekday_shift_ratios,
            historical_max_normal=hist_max_normal,
            historical_max_normal_weekday=hist_max_normal_weekday,
            historical_max_normal_weekend=hist_max_normal_weekend,
            historical_max_by_holiday=hist_max_by_holiday,
            historical_min_normal=hist_min_normal,
            historical_min_normal_weekday=hist_min_normal_weekday,
            historical_min_normal_weekend=hist_min_normal_weekend,
            # ⭐ V8: Pass shift-level P20/P95 for per-shift constraints
            weekday_shift_p95=weekday_shift_p95,
            weekday_shift_p20=weekday_shift_p20,
        )
        
        ensemble_info['total_predictions'] = len(predictions)
        
        return predictions, ensemble_info
    
    @staticmethod
    def _combine_predictions(
        next_days_info: List[Dict],
        ml_shift: Dict,
        ml_daily: Dict,
        prophet_daily: Dict,
        neuralprophet_daily: Dict,
        ai_daily_map: Dict,
        weights: Dict,
        analysis_report: Dict,
        weekday_shift_ratios: Dict = None,  # type: ignore[reportArgumentType]
        historical_max_normal: int = None,  # type: ignore[reportArgumentType]
        historical_max_normal_weekday: int = None,  # type: ignore[reportArgumentType]
        historical_max_normal_weekend: int = None,  # type: ignore[reportArgumentType]
        historical_max_by_holiday: Dict = None,  # type: ignore[reportArgumentType]
        historical_min_normal: int = None,  # type: ignore[reportArgumentType]
        historical_min_normal_weekday: int = None,  # type: ignore[reportArgumentType]
        historical_min_normal_weekend: int = None,  # type: ignore[reportArgumentType]
        weekday_shift_p95: Dict = None,  # ⭐ V8 Task 5  # type: ignore[reportArgumentType]
        weekday_shift_p20: Dict = None,  # ⭐ V8 Task 5  # type: ignore[reportArgumentType]
    ) -> List[Dict]:
        """
        Phase 8: Combine ML + Prophet + NeuralProphet + AI predictions.
        
        Output: 2 rows per day (MORNING + EVENING shifts).
        
        Chiến lược:
        1. Tính combined daily total = weighted average of sources
        2. Apply holiday_impact factor
        3. Smart cap:
           - Ngày thường: cap tại max_daily_normal (3 tháng gần nhất)
           - Ngày lễ/event: cap tại max của đúng loại lễ đó trong năm trước
        4. Smart floor:
           - Ngày thường: floor tại min_daily_normal (3 tháng gần nhất)
           - Tách weekday/weekend để tránh floor sai ngữ cảnh
        5. Distribute to 2 shifts using weekday-specific shift ratios (preferred)
           or ML shift ratios (fallback)
        """
        predictions = []
        trend_score = analysis_report.get('trend_score', 0)
        weekly_growth = trend_score / 2000  # Conservative: 10 score → 0.5% per week
        
        if weekday_shift_ratios is None:
            weekday_shift_ratios = {}
        if historical_max_by_holiday is None:
            historical_max_by_holiday = {}
        
        # Default shift ratios (if no historical data)
        DEFAULT_SHIFT_RATIOS = {'MORNING': 0.45, 'EVENING': 0.55}
        
        n_capped = 0  # Track historical max cap hits
        
        for d_info in next_days_info:
            d_str = str(d_info['date'])
            weekday_name = d_info.get('weekday', '')
            
            # --- Collect available daily totals ---
            ml_total = ml_daily.get(d_str, None)
            prophet_total = prophet_daily.get(d_str, None)
            ai_total = ai_daily_map.get(d_str, None)
            
            if ai_total is not None:
                try:
                    ai_total = float(ai_total)
                except (ValueError, TypeError):
                    ai_total = None
            
            # --- Weighted combination ---
            combined_daily = EnsembleForecastAgent._weighted_combine(
                ml_total=ml_total,
                prophet_total=prophet_total,
                neuralprophet_total=neuralprophet_daily.get(d_str, None),
                ai_total=ai_total,
                ml_weight=weights.get('ml', 0.5),
                ai_weight=weights.get('ai', 0.5),
            )
            
            if combined_daily is None or combined_daily <= 0:
                continue
            
            # --- Growth adjustment (±15% clamp) ---
            days_from_today = (d_info['date'] - CURRENT_DATE).days
            if days_from_today > 0:
                weeks_ahead = days_from_today / 7
                growth_factor = 1 + (weekly_growth * weeks_ahead)
                growth_factor = max(0.85, min(1.15, growth_factor))
                combined_daily *= growth_factor
            
            # --- ⭐ v5: Trend Adjustment Layer ---
            # Apply post-prediction trend adjustment based on recent momentum.
            # This catches rapid trend changes that the models haven't fully learned yet.
            # trend_short = mean_3d / mean_7d (from analysis_report)
            profile = analysis_report.get('profile', {})
            recent_trend = profile.get('trend_short')
            if recent_trend is None:
                # Calculate from ml_shift data if available
                recent_trend = 1.0
            
            if recent_trend > 1.1:
                # Spike detected: scale up proportionally (1.05 ~ 1.15)
                trend_adjust = min(
                    TREND_ADJUST_SPIKE_MAX,
                    max(TREND_ADJUST_SPIKE_MIN, recent_trend)
                )
                combined_daily *= trend_adjust
                logger.debug(
                    f"📈 v5 Trend spike on {d_str}: "
                    f"trend_short={recent_trend:.3f}, adjust=×{trend_adjust:.3f}"
                )
            elif recent_trend < 0.9:
                # Drop detected: scale down proportionally (0.85 ~ 0.95)
                trend_adjust = max(
                    TREND_ADJUST_DROP_MIN,
                    min(TREND_ADJUST_DROP_MAX, recent_trend)
                )
                combined_daily *= trend_adjust
                logger.debug(
                    f"📉 v5 Trend drop on {d_str}: "
                    f"trend_short={recent_trend:.3f}, adjust=×{trend_adjust:.3f}"
                )
            # NOTE: trend_signal feature (v5) effectively creates separate
            # prediction paths (trend vs non-trend) within the same model,
            # as the tree-based models will split on trend_signal=1/-1/0
            
            # --- ⭐ v7: Holiday Curve Adjustment (per-segment) ---
            holiday_impact = d_info.get('holiday_impact', 1.0)
            is_special_event = d_info.get('is_special_event', False)
            event_type = d_info.get('event_type')
            is_holiday = d_info.get('is_holiday', False)
            holiday_type = d_info.get('holiday_type')
            days_to_holiday = d_info.get('days_to_holiday', 0)
            is_holiday_window = d_info.get('is_holiday_window', False)
            # [FIX] HIGH-TRAFFIC holidays: 30/4, 1/5, 2/9 are NOT closed → apply boost
            # Closed-likely holidays (TET) still skip factor (handled by closure logic)
            closed_likely = d_info.get('closed_likely', False)
            HIGH_TRAFFIC_HOLIDAY_TYPES = {
                'LIBERATION_DAY', 'LABOR_DAY', 'NATIONAL_DAY', 'HUNG_KINGS',
                'TET_DUONG_LICH'
            }
            is_high_traffic_holiday = (
                is_holiday and holiday_type in HIGH_TRAFFIC_HOLIDAY_TYPES and not closed_likely
            )
            
            # Select per-segment holiday curve based on volume
            avg_daily = analysis_report.get('profile', {}).get('avg_daily', 50)
            if avg_daily >= MEDIUM_VOLUME_DAILY_THRESHOLD:
                holiday_curve = HOLIDAY_CURVE_HIGH_VOLUME
            elif avg_daily >= LOW_VOLUME_DAILY_THRESHOLD:
                holiday_curve = HOLIDAY_CURVE_MEDIUM_VOLUME
            else:
                holiday_curve = HOLIDAY_CURVE_LOW_VOLUME
            
            # ⭐ v7 Feature 4: Booking override during holiday window
            # If booking data shows higher-than-expected demand, trust bookings
            booking_count = d_info.get('booking_count', 0)
            booking_ratio = d_info.get('booking_ratio', 0.0)
            booking_override = False
            
            if is_high_traffic_holiday and holiday_impact != 1.0:
                # 30/4, 1/5, 2/9, etc. are demand spikes, not closure-like holidays.
                # Handle them before the generic holiday-window curve, whose day-0
                # factor is intentionally low for closure-heavy holidays such as Tet.
                holiday_factor = max(1.0, holiday_impact)
                logger.debug(
                    f"High-traffic holiday on {d_str} [{holiday_type}]: "
                    f"applying impact={holiday_factor:.2f}"
                )
            elif is_holiday_window and booking_ratio > HOLIDAY_BOOKING_OVERRIDE_THRESHOLD:
                # Booking > normal → override holiday factor (don't suppress forecast)
                holiday_factor = max(1.0, booking_ratio * 0.8)  # Cap at booking level
                booking_override = True
                logger.debug(
                    f"🎫 v7 Booking override on {d_str}: "
                    f"booking_ratio={booking_ratio:.2f} > {HOLIDAY_BOOKING_OVERRIDE_THRESHOLD}, "
                    f"factor={holiday_factor:.2f} (was {holiday_impact:.2f})"
                )
            elif is_holiday_window and days_to_holiday in holiday_curve:
                # ⭐ v7 Feature 2+3: Per-segment holiday curve
                holiday_factor = holiday_curve[days_to_holiday]
                logger.debug(
                    f"🎆 v7 Holiday curve on {d_str}: "
                    f"days={days_to_holiday}, curve_factor={holiday_factor:.2f} "
                    f"(old holiday_impact={holiday_impact:.2f})"
                )
            elif is_high_traffic_holiday and holiday_impact != 1.0:
                # [FIX] Apply boost for 30/4, 1/5, 2/9 etc (high-traffic holidays)
                # These days have MORE guests than normal, so we MUST apply the factor
                holiday_factor = holiday_impact
                logger.debug(
                    f"🎉 High-traffic holiday on {d_str} [{holiday_type}]: "
                    f"applying impact={holiday_factor:.2f}"
                )
            elif holiday_impact != 1.0 and not is_holiday:
                # Fallback: use existing holiday_impact from date_utils for pre/post days
                holiday_factor = holiday_impact
            else:
                holiday_factor = 1.0
            
            # [FIX] Apply holiday factor:
            # - High-traffic holidays (30/4, 1/5): apply boost
            # - TET/closed-likely: skip (handled by closure logic)
            # - Pre/post holidays: apply as before
            if holiday_factor != 1.0 and (is_high_traffic_holiday or not is_holiday):
                combined_daily *= holiday_factor
                if is_high_traffic_holiday:
                    logger.debug(
                        f"🎉 Applied holiday boost: {d_str} [{holiday_type}] "
                        f"× {holiday_factor:.2f} → {combined_daily:.0f}"
                    )
            
            # Log special events
            if is_special_event:
                logger.debug(
                    f"🎯 Special event {event_type} on {d_str}: "
                    f"impact={holiday_factor:.2f} → {combined_daily:.0f}"
                    f"{' (booking override)' if booking_override else ''}"
                )
            
            # --- Smart Historical Max Cap ---
            # Ngày lễ/event: dùng max lịch sử của ĐÚNG loại lễ đó
            # Ngày thường: dùng max 3 tháng gần nhất (loại trừ ngày lễ)
            effective_cap = None
            cap_source = None
            
            if is_holiday and holiday_type and holiday_type in historical_max_by_holiday:
                effective_cap = historical_max_by_holiday[holiday_type]
                cap_source = f"holiday:{holiday_type}"
            elif is_special_event and event_type and event_type in historical_max_by_holiday:
                effective_cap = historical_max_by_holiday[event_type]
                cap_source = f"event:{event_type}"
            elif historical_max_normal:
                # Use weekday/weekend-specific cap to avoid clipping weekend peaks
                is_weekend_day = d_info['date'].weekday() >= 5
                if is_weekend_day and historical_max_normal_weekend:
                    effective_cap = historical_max_normal_weekend
                    cap_source = "normal_weekend(3mo)"
                elif not is_weekend_day and historical_max_normal_weekday:
                    effective_cap = historical_max_normal_weekday
                    cap_source = "normal_weekday(3mo)"
                else:
                    effective_cap = historical_max_normal
                    cap_source = "normal(3mo)"
            
            # Guard: effective_cap=0 means bad historical data (all zeros)
            # → skip cap to avoid killing valid AI/ensemble predictions
            if effective_cap is not None and effective_cap <= 0:
                logger.debug(
                    f"📊 Smart max cap SKIPPED for {d_str}: effective_cap={effective_cap} "
                    f"({cap_source}) — likely zero-guest historical data"
                )
                effective_cap = None

            if (
                is_high_traffic_holiday
                and effective_cap
                and cap_source
                and cap_source.startswith("normal")
            ):
                # Normal-day caps are too tight for traffic-positive holidays.
                # If no holiday-specific cap exists, relax the cap by the same
                # calibrated factor used for the forecast boost.
                effective_cap = float(effective_cap) * max(1.0, holiday_factor)
                cap_source = f"{cap_source}*holiday_factor"
            
            if effective_cap and combined_daily > effective_cap:
                logger.debug(
                    f"📊 Smart max cap: {d_str} forecast {combined_daily:.0f} "
                    f"→ capped at {effective_cap} ({cap_source})"
                )
                combined_daily = float(effective_cap)
                n_capped += 1
            
            # --- Smart Historical Min Floor ---
            # Ngày thường: forecast không được thấp hơn min lịch sử 3 tháng
            # KHÔNG áp dụng cho ngày lễ (vì lễ có thể đóng cửa/giảm khách)
            if not is_holiday and not is_special_event:
                effective_floor = None
                floor_source = None
                
                is_weekend_day = d_info['date'].weekday() >= 5
                if is_weekend_day and historical_min_normal_weekend:
                    effective_floor = historical_min_normal_weekend
                    floor_source = "min_weekend(3mo)"
                elif not is_weekend_day and historical_min_normal_weekday:
                    effective_floor = historical_min_normal_weekday
                    floor_source = "min_weekday(3mo)"
                elif historical_min_normal:
                    effective_floor = historical_min_normal
                    floor_source = "min_normal(3mo)"
                
                if effective_floor and effective_floor > 0 and combined_daily < effective_floor:
                    logger.debug(
                        f"📊 Smart min floor: {d_str} forecast {combined_daily:.0f} "
                        f"→ raised to {effective_floor} ({floor_source})"
                    )
                    combined_daily = float(effective_floor)
            
            # --- ⭐ V8 Task 3: Adaptive alpha-blend shift distribution ---
            ml_shift_preds = ml_shift.get(d_str, {})
            wd_ratios = weekday_shift_ratios.get(weekday_name, {})
            
            if ml_shift_preds and wd_ratios:
                # Both available → compute adaptive alpha based on ML stability
                ml_shift_total = sum(ml_shift_preds.values())
                if ml_shift_total > 0:
                    ml_ratios = {s: v / ml_shift_total for s, v in ml_shift_preds.items()}
                else:
                    ml_ratios = wd_ratios
                
                # Compute CV of ML shift ratios across shifts as stability signal
                ml_ratio_vals = list(ml_ratios.values())
                if len(ml_ratio_vals) >= 2 and np.mean(ml_ratio_vals) > 0:
                    ml_cv = float(np.std(ml_ratio_vals) / np.mean(ml_ratio_vals))
                else:
                    ml_cv = 0.0
                
                # Adaptive alpha: stable ML → trust ML, noisy ML → trust historical
                if ml_cv < SHIFT_ALPHA_CV_STABLE:
                    alpha = SHIFT_ALPHA_DEFAULT  # 0.7
                elif ml_cv > SHIFT_ALPHA_CV_NOISY:
                    alpha = SHIFT_ALPHA_VOLATILE  # 0.3
                else:
                    # Linear interpolation between stable and noisy
                    t = (ml_cv - SHIFT_ALPHA_CV_STABLE) / max(
                        SHIFT_ALPHA_CV_NOISY - SHIFT_ALPHA_CV_STABLE, 0.01
                    )
                    alpha = SHIFT_ALPHA_DEFAULT * (1 - t) + SHIFT_ALPHA_VOLATILE * t
                
                # Blend: alpha * ML + (1-alpha) * historical
                shift_ratios = {}
                all_shifts = set(list(ml_ratios.keys()) + list(wd_ratios.keys()))
                for s in all_shifts:
                    ml_r = ml_ratios.get(s, 0.5)
                    hist_r = wd_ratios.get(s, 0.5)
                    shift_ratios[s] = alpha * ml_r + (1 - alpha) * hist_r
                
                # Normalize
                total_r = sum(shift_ratios.values())
                if total_r > 0:
                    shift_ratios = {s: v / total_r for s, v in shift_ratios.items()}
            elif ml_shift_preds:
                # ML only
                ml_shift_total = sum(ml_shift_preds.values())
                if ml_shift_total > 0:
                    shift_ratios = {s: v / ml_shift_total for s, v in ml_shift_preds.items()}
                else:
                    shift_ratios = DEFAULT_SHIFT_RATIOS
            elif wd_ratios:
                shift_ratios = wd_ratios
            else:
                shift_ratios = DEFAULT_SHIFT_RATIOS
            
            # --- Generate 2 rows (MORNING + EVENING) ---
            target_weekday = d_info['date'].weekday()
            for shift_key in ['MORNING', 'EVENING']:
                ratio = shift_ratios.get(shift_key, 0.5)
                final_val = max(0, int(round(combined_daily * ratio)))
                ml_val = ml_shift_preds.get(shift_key, 0)
                
                # ⭐ V8 Task 5: Shift-level P20/P95 constraints
                shift_id = EnsembleForecastAgent.SHIFT_ID_MAP.get(shift_key, 0)
                if weekday_shift_p95:
                    p95_val = weekday_shift_p95.get((target_weekday, shift_id))
                    if p95_val and is_high_traffic_holiday:
                        p95_val = float(p95_val) * max(1.0, holiday_factor)
                    if p95_val and final_val > p95_val:
                        logger.debug(
                            f"📊 V8 shift cap: {d_str} {shift_key} {final_val} → {int(p95_val)} "
                            f"(P95 wd={target_weekday})"
                        )
                        final_val = int(p95_val)
                
                if weekday_shift_p20 and not is_holiday and not is_special_event:
                    p20_val = weekday_shift_p20.get((target_weekday, shift_id))
                    if p20_val and p20_val > 0 and final_val < p20_val:
                        logger.debug(
                            f"📊 V8 shift floor: {d_str} {shift_key} {final_val} → {int(p20_val)} "
                            f"(P20 wd={target_weekday})"
                        )
                        final_val = int(p20_val)
                
                predictions.append({
                    'date': d_info['date'],
                    'shift': shift_key,
                    'shift_id': shift_id,
                    'hour': None,  # Phase 8: shift-based, no hour
                    'forecast': final_val,
                    'weekday': d_info['weekday'],
                    'is_holiday': d_info['is_holiday'],
                    'holiday_type': d_info.get('holiday_type'),
                    'is_veg': d_info.get('is_veg', False),
                    'is_special_event': d_info.get('is_special_event', False),
                    'event_type': d_info.get('event_type'),
                    'holiday_impact': holiday_factor,
                    'ml_shift': int(round(ml_val)),
                    'combined_daily': int(round(combined_daily)),
                    'sources': {
                        'ml': ml_total,
                        'prophet': prophet_total,
                        'ai': ai_total,
                    },
                })
        
        if n_capped > 0:
            logger.info(
                f"📊 Smart max cap applied to {n_capped}/{len(next_days_info)} days "
                f"(normal_max={historical_max_normal}, holiday_caps={len(historical_max_by_holiday)})"
            )
        
        return predictions
    
    @staticmethod
    def run_ensemble_forecast_daily_only(
        res_code: str,
        df_res_cleaned: pd.DataFrame,
        df_processed: pd.DataFrame,
        next_days_info: List[Dict],
        vn_holidays,
        analysis_report: Dict,
        ai_daily_map: Dict[str, float],
        neuralprophet_model=None,
    ) -> Tuple[List[Dict], Dict]:
        """
        Phase 8: Run ensemble forecast ở chế độ DAILY-ONLY (shift-based).
        Dùng cho long-term forecast (>30 ngày) để tiết kiệm thời gian.
        
        Output: 1 row/ngày/nhà hàng với Hour=None, forecast = daily total.
        
        Sử dụng shift-based features (MORNING/EVENING) thay vì hourly,
        consistent với short-term forecast.
        """
        strategy = analysis_report.get('strategy', 'ENSEMBLE_EQUAL')
        weights = STRATEGY_WEIGHTS.get(strategy, {'ml': 0.5, 'ai': 0.5})
        
        # Phase 8: Use FULL shift-based feature list from MLForecastAgent
        feature_columns = MLForecastAgent.get_feature_columns()
        available_features = [f for f in feature_columns if f in df_processed.columns]
        
        # ==========================================
        # 1. ML DAILY PREDICTION (shift-based: predict 2 shifts → sum)
        # ==========================================
        ml_daily = {}
        ensemble_info = {
            'strategy': strategy,
            'weights': weights,
            'models_used': [],
            'metrics': {},
            'mode': 'daily_only',
        }
        
        if weights['ml'] > 0 and not df_processed.empty:
            X_train = df_processed[available_features].copy()
            for col in X_train.columns:
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            X_train = X_train.fillna(X_train.median())
            y_train = df_processed['guest_count']
            
            ensemble = EnsembleMLAgent.train_stacking_ensemble(
                X_train, y_train, available_features  # type: ignore[reportArgumentType]
            )
            
            if ensemble:
                ensemble_info['models_used'] = ensemble.get('model_names', [])
                ensemble_info['metrics'] = ensemble.get('metrics', {})
                
                # Phase 8: Shift-based guest mapping (same as short-term)
                guest_mapping = df_processed.set_index(
                    df_processed['date'].dt.date.astype(str) + "_" +  # type: ignore[reportAttributeAccessIssue]
                    df_processed['shift_id'].astype(str)
                )['guest_count'].to_dict()
                default_val = df_processed['guest_count'].median()
                
                # Smart fallback: weekday+shift medians
                weekday_shift_medians = df_processed.groupby(
                    ['weekday', 'shift_id']
                )['guest_count'].median().to_dict()
                weekday_medians = df_processed.groupby(
                    'weekday'
                )['guest_count'].median().to_dict()
                
                def smart_lag_fallback(weekday, shift_id):
                    val = weekday_shift_medians.get((weekday, shift_id))
                    if val is not None:
                        return val
                    val = weekday_medians.get(weekday)
                    if val is not None:
                        return val / 2  # Split daily into 2 shifts
                    return default_val
                
                daily_map_hist = df_processed.groupby(
                    df_processed['date'].dt.date  # type: ignore[reportAttributeAccessIssue]
                )['guest_count'].sum().to_dict()
                recent_dates = sorted(daily_map_hist.keys())[-14:]
                rolling_default = (
                    np.median([daily_map_hist[d] for d in recent_dates])
                    if recent_dates else default_val
                )
                
                # Historical P95 caps per weekday (replaces blunt 2x-median)
                weekday_daily_p95 = {}  # weekday → P95 of daily total
                for wd in range(7):
                    wd_daily_vals = [
                        v for d, v in daily_map_hist.items()
                        if hasattr(d, 'weekday') and d.weekday() == wd
                    ]
                    if len(wd_daily_vals) >= 3:
                        weekday_daily_p95[wd] = float(np.percentile(list(wd_daily_vals), 95))
                
                # Fallback: overall daily median (only when no weekday P95)
                hist_daily_median = np.median(list(daily_map_hist.values()))
                
                # YoY growth rate
                daily_map_sorted = sorted(daily_map_hist.items())
                if len(daily_map_sorted) > 60:
                    recent_vals = [v for _, v in daily_map_sorted[-28:]]
                    old_vals = [v for _, v in daily_map_sorted[:28]]
                    yoy_rate = np.mean(recent_vals) / max(np.mean(old_vals), 1)
                    yoy_rate = min(max(yoy_rate, 0.5), 2.0)
                else:
                    yoy_rate = 1.0
                
                EVENT_TYPE_MAP = MLForecastAgent.EVENT_TYPE_MAP
                
                for d_info in next_days_info:
                    target_date = d_info['date']
                    target_date_dt = pd.to_datetime(target_date)
                    d_str = str(target_date)
                    lunar = get_lunar_info(target_date)
                    target_weekday = target_date_dt.dayofweek
                    
                    # Predict 2 shifts → sum for daily total
                    shift_features = []
                    for shift_key, shift_id in EnsembleForecastAgent.SHIFT_ID_MAP.items():
                        fallback = smart_lag_fallback(target_weekday, shift_id)
                        
                        feat = {
                            'shift_id': shift_id,
                            'weekday': target_weekday,
                            'month': target_date_dt.month,
                            'day_of_month': target_date_dt.day,
                            'is_weekend': 1 if target_weekday >= 5 else 0,
                            'is_holiday': 1 if d_info['is_holiday'] else 0,
                            'is_tet': 1 if d_info.get('holiday_type') == 'TET_NGUYEN_DAN' else 0,
                            'is_pre_holiday': 1 if d_info.get('is_pre_holiday') else 0,
                            'is_post_holiday': 1 if d_info.get('is_post_holiday') else 0,
                            'holiday_impact': d_info.get('holiday_impact', 1.0),
                            # ⭐ v7: Distance-based holiday features
                            'days_to_holiday': d_info.get('days_to_holiday', 0),
                            'is_holiday_window': 1 if d_info.get('is_holiday_window', False) else 0,
                            'is_T_minus_1': 1 if d_info.get('days_to_holiday', 0) == -1 else 0,
                            'is_T_minus_2': 1 if d_info.get('days_to_holiday', 0) == -2 else 0,
                            'is_T_plus_1': 1 if d_info.get('days_to_holiday', 0) == 1 else 0,
                            'lunar_day': lunar['lunar_day'],
                            'lunar_month': lunar['lunar_month'],
                            'is_veg': 1 if lunar['is_veg'] else 0,
                            'rolling_7d_mean': rolling_default,
                            'is_special_event': 1 if d_info.get('is_special_event') else 0,
                            'event_type_encoded': EVENT_TYPE_MAP.get(
                                d_info.get('event_type'), 0
                            ),
                            'yoy_growth_rate': yoy_rate,
                            # ⭐ v4 features
                            'is_friday': 1 if target_weekday == 4 else 0,
                            'is_saturday': 1 if target_weekday == 5 else 0,
                            'is_sunday': 1 if target_weekday == 6 else 0,
                            'trend_7d': 1.0,
                            'momentum': 1.0,
                            'booking_count': 0,
                            'booking_ratio': 0.0,
                            'booking_flag': 0,
                            # ⭐ v5 features
                            'trend_short': 1.0,
                            'weighted_lag': fallback,
                            'delta_7d': 0.0,
                            'trend_signal': 0,
                            'is_outlier_day': 0,
                        }
                        # Shift-based lag features
                        for lag_days in [7, 14, 28]:
                            lag_date_str = (
                                target_date - datetime.timedelta(days=lag_days)
                            ).strftime('%Y-%m-%d')
                            lag_key = f"{lag_date_str}_{shift_id}"
                            feat[f'lag_{lag_days}d'] = guest_mapping.get(
                                lag_key, fallback
                            )
                        
                        # lag_365d
                        lag_365_date_str = (
                            target_date - datetime.timedelta(days=365)
                        ).strftime('%Y-%m-%d')
                        feat['lag_365d'] = guest_mapping.get(
                            f"{lag_365_date_str}_{shift_id}", fallback
                        )
                        
                        # same_weekday_last_year (364 days)
                        same_wd_date_str = (
                            target_date - datetime.timedelta(days=364)
                        ).strftime('%Y-%m-%d')
                        feat['same_weekday_last_year'] = guest_mapping.get(
                            f"{same_wd_date_str}_{shift_id}", fallback
                        )
                        
                        # ⭐ v5: Recalculate weighted_lag after lags populated
                        feat['weighted_lag'] = (
                            0.6 * feat['lag_7d'] +
                            0.3 * feat['lag_14d'] +
                            0.1 * feat['lag_28d']
                        )
                        
                        shift_features.append(feat)
                    
                    X_test = pd.DataFrame(shift_features)
                    for missing_feature in available_features:
                        if missing_feature not in X_test.columns:
                            X_test[missing_feature] = 0
                    X_test = X_test[available_features]
                    
                    try:
                        preds = EnsembleMLAgent.predict_stacking(ensemble, X_test)  # type: ignore[reportArgumentType]
                        raw_daily = sum(max(0, float(p)) for p in preds)
                        # Soft cap at weekday-specific P95
                        wd_p95 = weekday_daily_p95.get(target_weekday)
                        ml_extreme_cap = wd_p95 if wd_p95 else hist_daily_median * 2.0
                        if raw_daily > ml_extreme_cap and ml_extreme_cap > 0:
                            raw_daily = ml_extreme_cap
                        ml_daily[d_str] = raw_daily
                    except Exception as e:
                        logger.warning(f"Stacking predict (daily-only) failed for "
                                     f"{res_code} on {d_str}: {e}")
        
        # ==========================================
        # 2. PROPHET DAILY FORECAST
        # ==========================================
        prophet_daily = {}
        if HAS_PROPHET and not df_res_cleaned.empty:
            prophet_daily = ProphetDailyAgent.train_and_predict(
                df_res_cleaned, next_days_info, vn_holidays
            )
            if prophet_daily:
                ensemble_info['models_used'].append('prophet')
        
        # ==========================================
        # 2.5 NEURALPROPHET DAILY FORECAST (daily-only mode)
        # ==========================================
        neuralprophet_daily = {}
        if HAS_NEURALPROPHET and not df_res_cleaned.empty:
            if neuralprophet_model is not None:
                neuralprophet_daily = NeuralProphetAgent.predict_from_global_model(  # type: ignore[reportPossiblyUnboundVariable]
                    neuralprophet_model, df_res_cleaned, next_days_info, vn_holidays
                )
            if neuralprophet_daily:
                ensemble_info['models_used'].append('neuralprophet')
        
        # ==========================================
        # 3. DAILY-ONLY ENSEMBLE COMBINATION
        # ==========================================
        # Smart historical max cap: ngày thường vs ngày lễ/special event
        profile = analysis_report.get('profile', {})
        hist_max_normal = profile.get('max_daily_normal') or profile.get('max_daily')
        hist_max_normal_weekday = profile.get('max_daily_normal_weekday') or hist_max_normal
        hist_max_normal_weekend = profile.get('max_daily_normal_weekend') or hist_max_normal
        hist_max_by_holiday = profile.get('max_daily_by_holiday', {})
        
        # Smart historical min floor: ngày thường (3 tháng gần nhất)
        hist_min_normal = profile.get('min_daily_normal', 0)
        hist_min_normal_weekday = profile.get('min_daily_normal_weekday') or hist_min_normal
        hist_min_normal_weekend = profile.get('min_daily_normal_weekend') or hist_min_normal
        
        predictions = EnsembleForecastAgent._combine_predictions_daily_only(
            next_days_info=next_days_info,
            ml_daily=ml_daily,
            prophet_daily=prophet_daily,
            neuralprophet_daily=neuralprophet_daily,
            ai_daily_map=ai_daily_map,
            weights=weights,
            analysis_report=analysis_report,
            historical_max_normal=hist_max_normal,
            historical_max_normal_weekday=hist_max_normal_weekday,
            historical_max_normal_weekend=hist_max_normal_weekend,
            historical_max_by_holiday=hist_max_by_holiday,
            historical_min_normal=hist_min_normal,
            historical_min_normal_weekday=hist_min_normal_weekday,
            historical_min_normal_weekend=hist_min_normal_weekend,
        )
        
        ensemble_info['total_predictions'] = len(predictions)
        
        return predictions, ensemble_info
    
    @staticmethod
    def _combine_predictions_daily_only(
        next_days_info: List[Dict],
        ml_daily: Dict,
        prophet_daily: Dict,
        neuralprophet_daily: Dict,
        ai_daily_map: Dict,
        weights: Dict,
        analysis_report: Dict,
        historical_max_normal: int = None,  # type: ignore[reportArgumentType]
        historical_max_normal_weekday: int = None,  # type: ignore[reportArgumentType]
        historical_max_normal_weekend: int = None,  # type: ignore[reportArgumentType]
        historical_max_by_holiday: Dict = None,  # type: ignore[reportArgumentType]
        historical_min_normal: int = None,  # type: ignore[reportArgumentType]
        historical_min_normal_weekday: int = None,  # type: ignore[reportArgumentType]
        historical_min_normal_weekend: int = None,  # type: ignore[reportArgumentType]
    ) -> List[Dict]:
        """
        Combine ML + Prophet + AI predictions ở level DAILY-ONLY.
        
        Khác với _combine_predictions:
        - KHÔNG phân bổ theo giờ
        - Mỗi ngày chỉ tạo 1 row với Hour=None
        - Vẫn áp dụng growth adjustment, holiday impact
        - Smart cap: ngày thường dùng max 3 tháng, ngày lễ dùng max cùng loại lễ
        - Smart floor: ngày thường dùng min 3 tháng (tách weekday/weekend)
        
        Output format:
            {'date': ..., 'hour': None, 'forecast': daily_total, 'weekday': ..., ...}
        """
        predictions = []
        trend_score = analysis_report.get('trend_score', 0)
        weekly_growth = trend_score / 2000  # Conservative: same as hourly
        n_capped = 0
        
        if historical_max_by_holiday is None:
            historical_max_by_holiday = {}
        
        for d_info in next_days_info:
            d_str = str(d_info['date'])
            
            # --- Collect available daily totals ---
            ml_total = ml_daily.get(d_str, None)
            prophet_total = prophet_daily.get(d_str, None)
            ai_total = ai_daily_map.get(d_str, None)
            
            if ai_total is not None:
                try:
                    ai_total = float(ai_total)
                except (ValueError, TypeError):
                    ai_total = None
            
            # --- Weighted combination ---
            combined_daily = EnsembleForecastAgent._weighted_combine(
                ml_total=ml_total,
                prophet_total=prophet_total,
                neuralprophet_total=neuralprophet_daily.get(d_str, None),
                ai_total=ai_total,
                ml_weight=weights.get('ml', 0.5),
                ai_weight=weights.get('ai', 0.5),
            )
            
            if combined_daily is None or combined_daily <= 0:
                continue
            
            # --- Growth adjustment (±15% clamp) ---
            days_from_today = (d_info['date'] - CURRENT_DATE).days
            if days_from_today > 0:
                weeks_ahead = days_from_today / 7
                growth_factor = 1 + (weekly_growth * weeks_ahead)
                growth_factor = max(0.85, min(1.15, growth_factor))
                combined_daily *= growth_factor
            
            # --- ⭐ v7: Holiday Curve Adjustment (per-segment) ---
            holiday_impact = d_info.get('holiday_impact', 1.0)
            is_special_event = d_info.get('is_special_event', False)
            event_type = d_info.get('event_type')
            is_holiday = d_info.get('is_holiday', False)
            holiday_type = d_info.get('holiday_type')
            days_to_holiday = d_info.get('days_to_holiday', 0)
            is_holiday_window = d_info.get('is_holiday_window', False)
            # [FIX] HIGH-TRAFFIC holidays: 30/4, 1/5, 2/9 are NOT closed → apply boost
            closed_likely = d_info.get('closed_likely', False)
            HIGH_TRAFFIC_HOLIDAY_TYPES = {
                'LIBERATION_DAY', 'LABOR_DAY', 'NATIONAL_DAY', 'HUNG_KINGS',
                'TET_DUONG_LICH'
            }
            is_high_traffic_holiday = (
                is_holiday and holiday_type in HIGH_TRAFFIC_HOLIDAY_TYPES and not closed_likely
            )
            
            # Select per-segment holiday curve
            avg_daily = analysis_report.get('profile', {}).get('avg_daily', 50)
            if avg_daily >= MEDIUM_VOLUME_DAILY_THRESHOLD:
                holiday_curve = HOLIDAY_CURVE_HIGH_VOLUME
            elif avg_daily >= LOW_VOLUME_DAILY_THRESHOLD:
                holiday_curve = HOLIDAY_CURVE_MEDIUM_VOLUME
            else:
                holiday_curve = HOLIDAY_CURVE_LOW_VOLUME
            
            # v7: Booking override
            booking_ratio = d_info.get('booking_ratio', 0.0)
            if is_high_traffic_holiday and holiday_impact != 1.0:
                holiday_factor = max(1.0, holiday_impact)
            elif is_holiday_window and booking_ratio > HOLIDAY_BOOKING_OVERRIDE_THRESHOLD:
                holiday_factor = max(1.0, booking_ratio * 0.8)
            elif is_holiday_window and days_to_holiday in holiday_curve:
                holiday_factor = holiday_curve[days_to_holiday]
            elif holiday_impact != 1.0 and not is_holiday:
                holiday_factor = holiday_impact
            else:
                holiday_factor = 1.0
            
            # [FIX] Apply holiday factor for high-traffic holidays AND pre/post days
            if holiday_factor != 1.0 and (is_high_traffic_holiday or not is_holiday):
                combined_daily *= holiday_factor
                if is_high_traffic_holiday:
                    logger.debug(
                        f"🎉 [daily-only] Holiday boost: {d_str} [{holiday_type}] "
                        f"× {holiday_factor:.2f} → {combined_daily:.0f}"
                    )
            
            # --- Smart Historical Max Cap ---
            # Ngày lễ/event: dùng max lịch sử của ĐÚNG loại lễ đó
            # Ngày thường: dùng max 3 tháng gần nhất (loại trừ ngày lễ)
            effective_cap = None
            cap_source = None
            
            if is_holiday and holiday_type and holiday_type in historical_max_by_holiday:
                effective_cap = historical_max_by_holiday[holiday_type]
                cap_source = f"holiday:{holiday_type}"
            elif is_special_event and event_type and event_type in historical_max_by_holiday:
                effective_cap = historical_max_by_holiday[event_type]
                cap_source = f"event:{event_type}"
            elif historical_max_normal:
                # Use weekday/weekend-specific cap to avoid clipping weekend peaks
                is_weekend_day = d_info['date'].weekday() >= 5
                if is_weekend_day and historical_max_normal_weekend:
                    effective_cap = historical_max_normal_weekend
                    cap_source = "normal_weekend(3mo)"
                elif not is_weekend_day and historical_max_normal_weekday:
                    effective_cap = historical_max_normal_weekday
                    cap_source = "normal_weekday(3mo)"
                else:
                    effective_cap = historical_max_normal
                    cap_source = "normal(3mo)"
            
            if (
                is_high_traffic_holiday
                and effective_cap
                and cap_source
                and cap_source.startswith("normal")
            ):
                effective_cap = float(effective_cap) * max(1.0, holiday_factor)
                cap_source = f"{cap_source}*holiday_factor"
            
            if effective_cap and combined_daily > effective_cap:
                logger.debug(
                    f"📊 Smart max cap (daily-only): {d_str} forecast "
                    f"{combined_daily:.0f} → capped at {effective_cap} ({cap_source})"
                )
                combined_daily = float(effective_cap)
                n_capped += 1
            
            # --- Smart Historical Min Floor ---
            # Ngày thường: forecast không được thấp hơn min lịch sử 3 tháng
            # KHÔNG áp dụng cho ngày lễ (vì lễ có thể đóng cửa/giảm khách)
            if not is_holiday and not is_special_event:
                effective_floor = None
                floor_source = None
                
                is_weekend_day = d_info['date'].weekday() >= 5
                if is_weekend_day and historical_min_normal_weekend:
                    effective_floor = historical_min_normal_weekend
                    floor_source = "min_weekend(3mo)"
                elif not is_weekend_day and historical_min_normal_weekday:
                    effective_floor = historical_min_normal_weekday
                    floor_source = "min_weekday(3mo)"
                elif historical_min_normal:
                    effective_floor = historical_min_normal
                    floor_source = "min_normal(3mo)"
                
                if effective_floor and effective_floor > 0 and combined_daily < effective_floor:
                    logger.debug(
                        f"📊 Smart min floor (daily-only): {d_str} forecast "
                        f"{combined_daily:.0f} → raised to {effective_floor} ({floor_source})"
                    )
                    combined_daily = float(effective_floor)
            
            final_val = max(0, int(round(combined_daily)))
            
            predictions.append({
                'date': d_info['date'],
                'hour': None,  # DAILY-ONLY: không có thông tin giờ
                'forecast': final_val,
                'weekday': d_info['weekday'],
                'is_holiday': d_info['is_holiday'],
                'holiday_type': d_info.get('holiday_type'),
                'is_veg': d_info.get('is_veg', False),
                'is_special_event': d_info.get('is_special_event', False),
                'event_type': d_info.get('event_type'),
                'holiday_impact': holiday_factor,
                'ml_hourly': None,
                'combined_daily': final_val,
                'forecast_mode': 'daily_only',
                'sources': {
                    'ml': ml_total,
                    'prophet': prophet_total,
                    'ai': ai_total,
                },
            })
        
        if n_capped > 0:
            logger.info(
                f"📊 Smart max cap (daily-only) applied to {n_capped}/{len(next_days_info)} days "
                f"(normal_max={historical_max_normal}, holiday_caps={len(historical_max_by_holiday)})"
            )
        
        return predictions
    

    @staticmethod
    def _weighted_combine(
        ml_total: Optional[float],
        prophet_total: Optional[float],
        neuralprophet_total: Optional[float] = None,
        ai_total: Optional[float] = None,
        ml_weight: float = 0.5,
        ai_weight: float = 0.5,
    ) -> Optional[float]:
        """
        Weighted combination of multiple forecast sources.
        
        ⭐ v5 Weight distribution (ML-heavy):
        - ml_weight is split: 60% ML stacking, 20% NeuralProphet, 20% Prophet
        - ai_weight goes to AI (LLM)
        - If any source is missing, redistribute its weight proportionally
        
        Returns:
            float combined prediction hoặc None
        """
        sources = {}
        
        # ⭐ v5: ML-heavy weight split (configurable from settings)
        if ml_total is not None and ml_total >= 0:
            sources['ml'] = (ml_total, ml_weight * ML_STACKING_SHARE)
        
        if neuralprophet_total is not None and neuralprophet_total >= 0:
            sources['neuralprophet'] = (neuralprophet_total, ml_weight * NP_SHARE)
        
        if prophet_total is not None and prophet_total >= 0:
            sources['prophet'] = (prophet_total, ml_weight * PROPHET_SHARE)
        
        if ai_total is not None and ai_total >= 0:
            sources['ai'] = (ai_total, ai_weight)
        
        if not sources:
            return None
        
        # Normalize weights
        total_weight = sum(w for _, w in sources.values())
        
        if total_weight <= 0:
            return None
        
        combined = sum(
            val * (w / total_weight)
            for val, w in sources.values()
        )
        
        return max(0, combined)
    
    @staticmethod
    def calculate_confidence(
        ensemble_info: Dict,
        analysis_report: Dict
    ) -> float:
        """
        Tính confidence score (0.0 - 1.0) cho forecast.
        
        Factors:
        - Số models thành công
        - Restaurant category & confidence
        - ML training metrics
        - Data sufficiency
        
        Returns:
            float (0.0 = không tin cậy, 1.0 = rất tin cậy)
        """
        score = 0.5  # Base
        
        # Model diversity bonus
        n_models = len(ensemble_info.get('models_used', []))
        if n_models >= 4:
            score += 0.15
        elif n_models >= 2:
            score += 0.10
        elif n_models >= 1:
            score += 0.05
        
        # Restaurant confidence
        confidence = analysis_report.get('confidence', 'MEDIUM')
        if confidence == 'HIGH':
            score += 0.15
        elif confidence == 'MEDIUM':
            score += 0.05
        elif confidence == 'LOW':
            score -= 0.10
        
        # Data completeness
        profile = analysis_report.get('profile', {})
        active_days = profile.get('active_days', 0)
        if active_days > 60:
            score += 0.10
        elif active_days > 30:
            score += 0.05
        elif active_days < 14:
            score -= 0.15
        
        # Volatility penalty
        cv = profile.get('cv', 0)
        if cv > 0.8:
            score -= 0.10
        elif cv > 0.5:
            score -= 0.05
        
        return max(0.0, min(1.0, round(score, 2)))
