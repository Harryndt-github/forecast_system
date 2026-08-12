"""
==============================================
ML FORECAST AGENT
==============================================
Trách nhiệm:
- Feature engineering từ transaction data
- Train ML models (XGBoost, CatBoost, LightGBM, Prophet)
- Dự báo hourly guest count
- Model caching

Refactored từ forecast_fb.py MLForecastAgent class.
Fixes: Thêm nhiều lag features, model caching, better error handling.
"""

import pandas as pd
import numpy as np
import datetime
import pickle
import traceback
from pathlib import Path

from forecast_system.config.settings import (
    CURRENT_DATE, MODEL_CACHE_DIR,
    SHIFT_DEFINITIONS, ALL_OPERATING_HOURS, FEATURE_IMPORTANCE_THRESHOLD,
    BOOKING_THRESHOLD_RATIO,
    TREND_SPIKE_THRESHOLD, TREND_DROP_THRESHOLD,
)
from forecast_system.utils.date_utils import get_lunar_info
from forecast_system.agents.data_agent import DataAgent
from forecast_system.utils.logger import get_logger

logger = get_logger('ml_forecast_agent')

# ==========================================
# Safe Imports cho ML Libraries
# ==========================================
print("--- KHỞI TẠO CÁC MODEL AGENT ---")

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import StackingRegressor, RandomForestRegressor
    HAS_SKLEARN = True
    print("✅ Scikit-learn: Ready")
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ Scikit-learn: Missing")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
    print("✅ XGBoost: Ready")
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost: Missing")

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
    print("✅ LightGBM: Ready")
except ImportError:
    HAS_LGBM = False
    print("⚠️ LightGBM: Missing")

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
    print("✅ CatBoost: Ready")
except ImportError:
    HAS_CAT = False
    print("⚠️ CatBoost: Missing")

try:
    from prophet import Prophet
    HAS_PROPHET = True
    print("✅ Prophet: Ready")
except ImportError:
    HAS_PROPHET = False
    print("⚠️ Prophet: Missing")


class MLForecastAgent:
    """
    Agent xử lý ML-based forecasting.
    Hỗ trợ: XGBoost, CatBoost, LightGBM, Random Forest, Prophet.
    """
    
    # ==========================================
    # FEATURE ENGINEERING
    # ==========================================
    
    # Event type encoding map (shared across methods)
    EVENT_TYPE_MAP = {
        None: 0,
        'VALENTINE': 1, 'PRE_VALENTINE': 1,
        'WOMENS_DAY': 2, 'PRE_WOMENS_DAY': 2,
        'VN_WOMENS_DAY': 3, 'PRE_VN_WOMENS_DAY': 3,
        'TEACHERS_DAY': 4, 'PRE_TEACHERS_DAY': 4,
        'CHRISTMAS': 5,
        'HALLOWEEN': 6,
        'CHILDRENS_DAY': 7,
        'NEW_YEARS_EVE': 8,
        'WHITE_DAY': 9,
        'FATHERS_DAY': 10,
        'MOTHERS_DAY': 11,
        'MID_AUTUMN': 12,
    }
    
    # Shift ID mapping
    SHIFT_ID_MAP = {'MORNING': 0, 'EVENING': 1}
    
    @staticmethod
    def prepare_data(df_res, vn_holidays):
        """
        Phase 8: Chuyển đổi transaction data thành SHIFT-BASED time series.
        
        Thay vì 15 rows/ngày (hourly), chỉ tạo 2 rows/ngày:
        - MORNING (8h-15h): Tổng guest ca sáng
        - EVENING (16h-23h): Tổng guest ca tối
        
        Features:
        - Shift: shift_id (0=MORNING, 1=EVENING)
        - Time: weekday, month, day_of_month, is_weekend
        - Calendar: is_holiday, lunar_day, lunar_month, is_veg
        - Lag: lag_7d, lag_14d, lag_28d (cùng shift tuần trước)
        - YoY: lag_365d, same_weekday_last_year, yoy_growth_rate
        - Special Events: is_special_event, event_type_encoded
        - Rolling: rolling_7d_mean
        
        Returns:
            pd.DataFrame với shift-level features
        """
        if df_res.empty:
            return pd.DataFrame()
        
        # 1. Map each transaction to its shift
        df = df_res.copy()
        df['shift'] = df['hour'].apply(DataAgent.map_to_shift)
        df = df[df['shift'] != 'OTHER']  # Exclude non-operating hours
        
        # 2. Aggregate to SHIFT level (date + shift). Outlier rows are kept,
        # but their flags/weights are propagated so model training can downweight
        # abnormal closures or one-off demand spikes instead of deleting them.
        if 'is_outlier_day' not in df.columns:
            df['is_outlier_day'] = 0
        if 'outlier_weight' not in df.columns:
            df['outlier_weight'] = 1.0

        df_shift = df.groupby(['date', 'shift']).agg(
            guest_count=('guest_count', 'sum'),
            is_outlier_day=('is_outlier_day', 'max'),
            outlier_weight=('outlier_weight', 'min'),
        ).reset_index()
        df_shift['date'] = pd.to_datetime(df_shift['date'])
        df_shift['shift_id'] = df_shift['shift'].map(MLForecastAgent.SHIFT_ID_MAP)
        
        # 3. Base Time Features
        df_shift['weekday'] = df_shift['date'].dt.dayofweek
        df_shift['month'] = df_shift['date'].dt.month
        df_shift['day_of_month'] = df_shift['date'].dt.day
        df_shift['is_weekend'] = df_shift['weekday'].isin([5, 6]).astype(int)
        df_shift['is_holiday'] = df_shift['date'].apply(
            lambda x: 1 if x in vn_holidays else 0
        )
        
        # ⭐ v4: Explicit day-of-week flags (replace hardcoded weekend multipliers)
        df_shift['is_friday'] = (df_shift['weekday'] == 4).astype(int)
        df_shift['is_saturday'] = (df_shift['weekday'] == 5).astype(int)
        df_shift['is_sunday'] = (df_shift['weekday'] == 6).astype(int)
        
        # 4. Holiday + Special Event Features
        from forecast_system.utils.date_utils import get_holiday_info
        
        holiday_data = []
        for d in df_shift['date']:
            d_date = d.date() if hasattr(d, 'date') else d
            h_info = get_holiday_info(d_date, vn_holidays)
            holiday_data.append({
                'is_tet': 1 if h_info.get('holiday_type') == 'TET_NGUYEN_DAN' else 0,
                'is_pre_holiday': 1 if h_info.get('is_pre_holiday') else 0,
                'is_post_holiday': 1 if h_info.get('is_post_holiday') else 0,
                'holiday_impact': h_info.get('holiday_impact', 1.0),
                # ⭐ v7: Distance-based holiday features
                'days_to_holiday': h_info.get('days_to_holiday', 0),
                'is_holiday_window': 1 if h_info.get('is_holiday_window', False) else 0,
                'is_T_minus_1': 1 if h_info.get('days_to_holiday', 0) == -1 else 0,
                'is_T_minus_2': 1 if h_info.get('days_to_holiday', 0) == -2 else 0,
                'is_T_plus_1': 1 if h_info.get('days_to_holiday', 0) == 1 else 0,
                'is_special_event': 1 if h_info.get('is_special_event', False) else 0,
                'event_type_encoded': MLForecastAgent.EVENT_TYPE_MAP.get(
                    h_info.get('event_type'), 0
                ),
            })
        hol_df = pd.DataFrame(holiday_data)
        df_shift = pd.concat([df_shift.reset_index(drop=True), hol_df], axis=1)
        
        # 5. Lunar Calendar Features
        lunar_data = []
        for d in df_shift['date']:
            info = get_lunar_info(d)
            lunar_data.append(info)
        ldf = pd.DataFrame(lunar_data)
        ldf['is_veg'] = ldf['is_veg'].astype(int)
        df_shift = pd.concat([df_shift.reset_index(drop=True), ldf], axis=1)
        
        # 6. LAG Features (shift-based: same shift, N days ago)
        df_shift = df_shift.sort_values(['date', 'shift_id'])
        
        # Create shift-level mapping key: "date_shiftid"
        base_key = df_shift['date'].dt.date.astype(str) + "_" + df_shift['shift_id'].astype(str)
        guest_mapping = df_shift.set_index(base_key)['guest_count'].to_dict()
        default_val = df_shift['guest_count'].median()  # Median more robust than mean
        
        for lag_days in [7, 14, 28]:
            lag_key = (
                (df_shift['date'] - datetime.timedelta(days=lag_days))
                .dt.date.astype(str) + "_" + df_shift['shift_id'].astype(str)
            )
            df_shift[f'lag_{lag_days}d'] = lag_key.map(guest_mapping).fillna(default_val)
        
        # 6b. Lag 365d (same shift, same date last year)
        lag_365_key = (
            (df_shift['date'] - datetime.timedelta(days=365))
            .dt.date.astype(str) + "_" + df_shift['shift_id'].astype(str)
        )
        lag_365_raw = lag_365_key.map(guest_mapping)
        df_shift['lag_365d'] = lag_365_raw.fillna(default_val)
        
        # 6c. Same weekday last year (364 days = same weekday)
        same_wd_key = (
            (df_shift['date'] - datetime.timedelta(days=364))
            .dt.date.astype(str) + "_" + df_shift['shift_id'].astype(str)
        )
        same_wd_raw = same_wd_key.map(guest_mapping)
        df_shift['same_weekday_last_year'] = same_wd_raw.fillna(default_val)
        
        # 6d. YoY Growth Rate (with chain/brand fallback)
        # Count how many YoY data points actually exist (not NaN from mapping)
        yoy_data_days = lag_365_raw.notna().sum()
        
        daily_total_all = df_shift.groupby('date')['guest_count'].sum()
        if len(daily_total_all) > 0:
            recent_avg = daily_total_all.tail(28).mean()
            all_dates = daily_total_all.index.sort_values()
            if len(all_dates) > 60 and yoy_data_days >= 20:
                # Sufficient YoY data → calculate restaurant-specific growth
                old_avg = daily_total_all.head(28).mean()
                yoy_rate = recent_avg / max(old_avg, 1) if old_avg > 0 else 1.0
            else:
                # Fallback: insufficient YoY data (<20 days with actual last year)
                # Use growth from recent trend (30d vs 60d) as proxy
                if len(daily_total_all) >= 60:
                    recent_30 = daily_total_all.tail(30).mean()
                    prev_30 = daily_total_all.iloc[-60:-30].mean() if len(daily_total_all) >= 60 else recent_30
                    yoy_rate = recent_30 / max(prev_30, 1) if prev_30 > 0 else 1.0
                    logger.debug(
                        f"YoY fallback (trend-based): yoy_data_days={yoy_data_days}, "
                        f"rate={yoy_rate:.3f}"
                    )
                else:
                    yoy_rate = 1.0
        else:
            yoy_rate = 1.0
        df_shift['yoy_growth_rate'] = min(max(yoy_rate, 0.5), 2.0)
        
        # 7. Rolling Mean (7-day average of daily totals)
        daily_total = df_shift.groupby('date')['guest_count'].sum().reset_index()
        daily_total = daily_total.sort_values('date')
        daily_total['rolling_7d_mean'] = daily_total['guest_count'].rolling(
            window=7, min_periods=1
        ).mean()
        rolling_map = daily_total.set_index('date')['rolling_7d_mean'].to_dict()
        df_shift['rolling_7d_mean'] = df_shift['date'].map(rolling_map).fillna(default_val)  # type: ignore[reportArgumentType]
        
        # ⭐ v4: Trend & Momentum features
        daily_total['rolling_3d_mean'] = daily_total['guest_count'].rolling(
            window=3, min_periods=1
        ).mean()
        daily_total['rolling_14d_mean'] = daily_total['guest_count'].rolling(
            window=14, min_periods=1
        ).mean()
        
        # trend_7d = mean_7d / mean_14d → >1.0 = uptrend
        daily_total['trend_7d'] = (
            daily_total['rolling_7d_mean'] /
            daily_total['rolling_14d_mean'].replace(0, np.nan)
        ).fillna(1.0)
        # momentum = last_3d / last_7d → >1.0 = accelerating
        daily_total['momentum'] = (
            daily_total['rolling_3d_mean'] /
            daily_total['rolling_7d_mean'].replace(0, np.nan)
        ).fillna(1.0)
        
        trend_map = daily_total.set_index('date')['trend_7d'].to_dict()
        momentum_map = daily_total.set_index('date')['momentum'].to_dict()
        df_shift['trend_7d'] = df_shift['date'].map(trend_map).fillna(1.0)  # type: ignore[reportArgumentType]
        df_shift['momentum'] = df_shift['date'].map(momentum_map).fillna(1.0)  # type: ignore[reportArgumentType]
        
        # ⭐ v5: trend_short = mean_3d / mean_7d (short-term acceleration)
        daily_total['trend_short'] = (
            daily_total['rolling_3d_mean'] /
            daily_total['rolling_7d_mean'].replace(0, np.nan)
        ).fillna(1.0)
        trend_short_map = daily_total.set_index('date')['trend_short'].to_dict()
        df_shift['trend_short'] = df_shift['date'].map(trend_short_map).fillna(1.0)  # type: ignore[reportArgumentType]
        
        # ⭐ v5: Weighted lag = 0.6*lag_7d + 0.3*lag_14d + 0.1*lag_28d
        df_shift['weighted_lag'] = (
            0.6 * df_shift['lag_7d'] +
            0.3 * df_shift['lag_14d'] +
            0.1 * df_shift['lag_28d']
        )
        
        # ⭐ v5: Delta feature = guest_count - lag_7d (for delta prediction)
        df_shift['delta_7d'] = df_shift['guest_count'] - df_shift['lag_7d']
        
        # ⭐ v5: Early trend detection (spike/drop signal)
        # spike=1 if last_3d > last_7d * 1.2, drop=-1 if < 0.8, neutral=0
        daily_total['trend_signal'] = 0
        daily_total.loc[
            daily_total['rolling_3d_mean'] > daily_total['rolling_7d_mean'] * TREND_SPIKE_THRESHOLD,
            'trend_signal'
        ] = 1
        daily_total.loc[
            daily_total['rolling_3d_mean'] < daily_total['rolling_7d_mean'] * TREND_DROP_THRESHOLD,
            'trend_signal'
        ] = -1
        signal_map = daily_total.set_index('date')['trend_signal'].to_dict()
        df_shift['trend_signal'] = df_shift['date'].map(signal_map).fillna(0).astype(int)  # type: ignore[reportArgumentType]
        
        # ⭐ v4: Booking features (placeholder - will be populated when booking data available)
        df_shift['booking_count'] = 0
        df_shift['booking_ratio'] = 0.0
        df_shift['booking_flag'] = 0
        
        return df_shift
    
    # ==========================================
    # MODEL TRAINING & PREDICTION
    # ==========================================
    
    @staticmethod
    def get_feature_columns():
        """Phase 8 + v4/v5: Features for shift-based ML model"""
        return [
            'shift_id',                          # 0=MORNING, 1=EVENING
            'weekday', 'month', 'day_of_month',
            'is_weekend', 'is_holiday',
            'is_friday', 'is_saturday', 'is_sunday',  # ⭐ v4: explicit weekend flags
            'is_tet', 'is_pre_holiday', 'is_post_holiday', 'holiday_impact',
            # ⭐ v7: Distance-based holiday features
            'days_to_holiday',                   # continuous: -3,-2,-1,0,+1,+2,+3
            'is_holiday_window',                 # 1 if within ±3 days
            'is_T_minus_1',                      # exactly 1 day before holiday
            'is_T_minus_2',                      # exactly 2 days before holiday
            'is_T_plus_1',                       # exactly 1 day after holiday
            'lunar_day', 'lunar_month', 'is_veg',
            # ⭐ v5: Replaced individual lag_7d/14d/28d with single weighted_lag
            # Individual lags still computed internally for delta_7d and weighted_lag calculation
            'lag_365d',                          # Same shift, same date last year
            'same_weekday_last_year',             # Same shift, same weekday last year
            'yoy_growth_rate',                    # Year-over-year growth
            'is_special_event',                   # Valentine, Noel, 8/3, etc.
            'event_type_encoded',                 # Numeric encoding of event type
            'rolling_7d_mean',
            'trend_7d',                          # ⭐ v4: mean_7d / mean_14d
            'momentum',                          # ⭐ v4: mean_3d / mean_7d
            'trend_short',                       # ⭐ v5: mean_3d / mean_7d (alias)
            'weighted_lag',                      # ⭐ v5: 0.6*lag7 + 0.3*lag14 + 0.1*lag28
            'delta_7d',                          # ⭐ v5: guest_count - lag_7d
            'trend_signal',                      # ⭐ v5: spike=1, drop=-1, neutral=0
            'booking_count',                     # ⭐ v4: booking guests for this date
            'booking_ratio',                     # ⭐ v4: booking / avg_daily
            'booking_flag',                      # ⭐ v4: 1 if booking_ratio > threshold
            'is_outlier_day',                    # Tagged abnormal days, not removed
        ]
    
    @staticmethod
    def train_and_predict(res_code, df_train_processed, next_days_info, vn_holidays):
        """
        Phase 8: Train ML model và dự báo theo SHIFT (2 ca/ngày).
        
        Thay vì predict 15 giờ/ngày, chỉ predict 2 shifts:
        - MORNING (shift_id=0): Tổng guest ca sáng
        - EVENING (shift_id=1): Tổng guest ca tối (16h-23h)
        
        Returns:
            List[dict] predictions với shift key thay vì hour
        """
        if df_train_processed.empty:
            return []
        
        features = MLForecastAgent.get_feature_columns()
        available_features = [f for f in features if f in df_train_processed.columns]
        
        if not available_features:
            logger.warning(f"No features available for {res_code}")
            return []
        
        X = df_train_processed[available_features].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(X.median())
        y = df_train_processed['guest_count']
        
        # Select Model
        model = None
        model_name = "Unknown"
        cached = MLForecastAgent.load_model_cache(res_code, max_age_hours=48)
        cached_model = cached.get('model') if cached else None
        
        if HAS_XGB:
            model = XGBRegressor(  # type: ignore[reportPossiblyUnboundVariable]
                n_estimators=150, learning_rate=0.08, max_depth=6,
                subsample=0.8, random_state=42,
                objective='reg:squarederror', verbosity=0
            )
            model_name = "XGBoost"
        elif HAS_CAT:
            model = CatBoostRegressor(  # type: ignore[reportPossiblyUnboundVariable]
                iterations=150, learning_rate=0.08, depth=6,
                l2_leaf_reg=3.0, random_seed=42, verbose=0
            )
            model_name = "CatBoost"
        elif HAS_LGBM:
            model = LGBMRegressor(  # type: ignore[reportPossiblyUnboundVariable]
                n_estimators=150, learning_rate=0.08, max_depth=6,
                num_leaves=31, random_state=42, verbose=-1
            )
            model_name = "LightGBM"
        elif HAS_SKLEARN:
            model = RandomForestRegressor(  # type: ignore[reportPossiblyUnboundVariable]
                n_estimators=100, max_depth=10, random_state=42,
                warm_start=True
            )
            model_name = "RandomForest"
        else:
            logger.error(f"No ML library available for {res_code}")
            return []
        
        # Train
        try:
            if cached_model and type(cached_model).__name__ == type(model).__name__:
                if model_name == "XGBoost":
                    model.fit(X, y, xgb_model=cached_model.get_booster())  # type: ignore[reportCallIssue]
                elif model_name == "LightGBM":
                    model.fit(X, y, init_model=cached_model)  # type: ignore[reportCallIssue]
                elif model_name == "RandomForest":
                    model.n_estimators = cached_model.n_estimators + 20  # type: ignore[reportAttributeAccessIssue]
                    model.fit(X, y)
                else:
                    model.fit(X, y)
            else:
                model.fit(X, y)
            MLForecastAgent.save_model_cache(res_code, model)
        except Exception as e:
            logger.debug(f"Warm-start failed for {res_code}, training fresh: {e}")
            try:
                model.fit(X, y)
                MLForecastAgent.save_model_cache(res_code, model)
            except Exception as e2:
                logger.error(f"Model Fit Error for {res_code} ({model_name}): {e2}")
                return []
        
        # ======= Feature Importance Validation + Auto-Pruning =======
        feature_importances = {}
        pruned_features = []
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                total_imp = sum(importances)
                if total_imp > 0:
                    for fname, imp in zip(available_features, importances):
                        feature_importances[fname] = imp / total_imp
                    
                    # Identify features below importance threshold
                    low_features = [
                        f for f, imp in feature_importances.items()
                        if imp < FEATURE_IMPORTANCE_THRESHOLD
                    ]
                    
                    # Protected features: never prune these core features
                    protected = {'shift_id', 'weekday', 'is_weekend', 'lag_7d', 
                                 'rolling_7d_mean', 'is_holiday'}
                    prunable = [f for f in low_features if f not in protected]
                    
                    if prunable and len(available_features) - len(prunable) >= 6:
                        # Retrain without noise features for cleaner predictions
                        pruned_features = prunable
                        clean_features = [f for f in available_features if f not in prunable]
                        X_clean = df_train_processed[clean_features].copy()
                        for col in X_clean.columns:
                            X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
                        X_clean = X_clean.fillna(X_clean.median())
                        
                        try:
                            model.fit(X_clean, y)
                            available_features = clean_features  # Update for prediction
                            logger.debug(
                                f"{res_code}: Pruned {len(prunable)} low-importance features: "
                                f"{prunable} → retrained with {len(clean_features)} features"
                            )
                        except Exception:
                            # If retrain fails, keep original model
                            pass
                    elif low_features:
                        logger.debug(
                            f"{res_code}: Low importance features "
                            f"(<{FEATURE_IMPORTANCE_THRESHOLD}): {low_features}"
                        )
        except Exception:
            pass
        
        # ======= Predict per shift =======
        predictions = []
        
        # Shift-based guest mapping
        guest_mapping = df_train_processed.set_index(
            df_train_processed['date'].dt.date.astype(str) + "_" +
            df_train_processed['shift_id'].astype(str)
        )['guest_count'].to_dict()
        default_val = df_train_processed['guest_count'].median()
        
        # Smart fallback: weekday+shift medians
        weekday_shift_medians = df_train_processed.groupby(
            ['weekday', 'shift_id']
        )['guest_count'].median().to_dict()
        weekday_medians = df_train_processed.groupby(
            'weekday'
        )['guest_count'].median().to_dict()
        
        def smart_lag_fallback(weekday, shift_id):
            val = weekday_shift_medians.get((weekday, shift_id))
            if val is not None:
                return val
            val = weekday_medians.get(weekday)
            if val is not None:
                return val / 2  # Split daily median into 2 shifts
            return default_val
        
        # Rolling mean
        daily_map = df_train_processed.groupby(
            df_train_processed['date'].dt.date
        )['guest_count'].sum().to_dict()
        recent_dates = sorted(daily_map.keys())[-7:]
        rolling_default = (
            np.median([daily_map[d] for d in recent_dates])
            if recent_dates else default_val
        )
        
        # YoY growth rate
        daily_map_sorted = sorted(daily_map.items())
        if len(daily_map_sorted) > 60:
            recent_vals = [v for _, v in daily_map_sorted[-28:]]
            old_vals = [v for _, v in daily_map_sorted[:28]]
            yoy_rate = np.mean(recent_vals) / max(np.mean(old_vals), 1)
            yoy_rate = min(max(yoy_rate, 0.5), 2.0)
        else:
            yoy_rate = 1.0
        
        for d_info in next_days_info:
            target_date = d_info['date']
            target_date_dt = pd.to_datetime(target_date)
            lunar = get_lunar_info(target_date)
            target_weekday = target_date_dt.dayofweek
            
            # Predict for each SHIFT (MORNING=0, EVENING=1)
            for shift_key, shift_id in MLForecastAgent.SHIFT_ID_MAP.items():
                fallback = smart_lag_fallback(target_weekday, shift_id)
                
                feat = {
                    'shift_id': shift_id,
                    'weekday': target_weekday,
                    'month': target_date_dt.month,
                    'day_of_month': target_date_dt.day,
                    'is_weekend': 1 if target_weekday >= 5 else 0,
                    'is_holiday': 1 if d_info['is_holiday'] else 0,
                    'is_friday': 1 if target_weekday == 4 else 0,     # ⭐ v4
                    'is_saturday': 1 if target_weekday == 5 else 0,   # ⭐ v4
                    'is_sunday': 1 if target_weekday == 6 else 0,     # ⭐ v4
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
                    'trend_7d': 1.0,                                  # ⭐ v4
                    'momentum': 1.0,                                  # ⭐ v4
                    'trend_short': 1.0,                               # ⭐ v5
                    'weighted_lag': fallback,                          # ⭐ v5: will be recalculated below
                    'delta_7d': 0.0,                                  # ⭐ v5
                    'trend_signal': 0,                                # ⭐ v5
                    'booking_count': 0,                               # ⭐ v4
                    'booking_ratio': 0.0,                             # ⭐ v4
                    'booking_flag': 0,                                # ⭐ v4
                    'is_special_event': 1 if d_info.get('is_special_event') else 0,
                    'event_type_encoded': MLForecastAgent.EVENT_TYPE_MAP.get(
                        d_info.get('event_type'), 0
                    ),
                    'yoy_growth_rate': yoy_rate,
                }
                
                # Shift-based lag features
                for lag_days in [7, 14, 28]:
                    lag_date_str = (
                        target_date - datetime.timedelta(days=lag_days)
                    ).strftime('%Y-%m-%d')
                    lag_key = f"{lag_date_str}_{shift_id}"
                    feat[f'lag_{lag_days}d'] = guest_mapping.get(lag_key, fallback)
                
                # Lag 365d
                lag_365_date_str = (
                    target_date - datetime.timedelta(days=365)
                ).strftime('%Y-%m-%d')
                feat['lag_365d'] = guest_mapping.get(
                    f"{lag_365_date_str}_{shift_id}", fallback
                )
                
                # Same weekday last year
                same_wd_date_str = (
                    target_date - datetime.timedelta(days=364)
                ).strftime('%Y-%m-%d')
                feat['same_weekday_last_year'] = guest_mapping.get(
                    f"{same_wd_date_str}_{shift_id}", fallback
                )
                
                # ⭐ v5: Recalculate weighted_lag now that lag values are populated
                feat['weighted_lag'] = (
                    0.6 * feat['lag_7d'] +
                    0.3 * feat['lag_14d'] +
                    0.1 * feat['lag_28d']
                )
                # ⭐ v5: delta_7d at prediction time (expected delta = 0, model learns)
                feat['delta_7d'] = 0.0
                
                X_test = pd.DataFrame([feat])[available_features]
                
                try:
                    pred = model.predict(X_test)[0]  # type: ignore[reportIndexIssue]
                    val = max(0, int(round(pred)))  # type: ignore[reportArgumentType]
                except Exception as e:
                    logger.warning(
                        f"Predict error {res_code} {target_date} {shift_key}: {e}"
                    )
                    val = int(round(fallback))
                
                predictions.append({
                    'date': target_date,
                    'shift': shift_key,
                    'shift_id': shift_id,
                    'hour': None,  # Phase 8: no hour, use shift
                    'forecast': val,
                    'weekday': d_info['weekday'],
                    'is_holiday': d_info['is_holiday'],
                    'is_veg': d_info.get('is_veg', False),
                    'is_special_event': d_info.get('is_special_event', False),
                    'event_type': d_info.get('event_type'),
                    'model': model_name,
                    'feature_importances': feature_importances,
                })
        
        return predictions
    
    # ==========================================
    # MODEL CACHING
    # ==========================================
    
    @staticmethod
    def save_model_cache(res_code, model, metrics=None):
        """Cache trained model cho restaurant"""
        cache_dir = Path(MODEL_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        path = cache_dir / f"{res_code}.pkl"
        data = {
            'model': model,
            'metrics': metrics or {},
            'created_at': datetime.datetime.now(),
        }
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Model cache save failed for {res_code}: {e}")
    
    @staticmethod
    def load_model_cache(res_code, max_age_hours=24):
        """Load cached model nếu còn hạn"""
        path = Path(MODEL_CACHE_DIR) / f"{res_code}.pkl"
        
        if not path.exists():
            return None
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            age = (datetime.datetime.now() - data['created_at']).total_seconds() / 3600
            if age > max_age_hours:
                path.unlink()
                return None
            
            return data
        except Exception:
            return None
