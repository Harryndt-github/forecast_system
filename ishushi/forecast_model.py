"""
==============================================
ISHUSHI FORECAST MODEL
==============================================
Dual-target forecasting:
1. Guest Count (số khách) per restaurant per day
2. Item Quantity (số lượng món) per restaurant per day per sap_code/group

Models used:
- XGBoost + LightGBM + CatBoost ensemble
- Prophet for trend/seasonality capture

Output: Forecast cho 30 ngày tới
"""

import pandas as pd
import numpy as np
import datetime
import os
import warnings
import traceback

warnings.filterwarnings('ignore')

from forecast_system.utils.logger import get_logger
from forecast_system.ishushi.config import (
    ISHUSHI_CONFIG, ISHUSHI_SAP_CATALOG, ISHUSHI_SAP_GROUPS,
    ISHUSHI_SHIFT_HOURS, ISHUSHI_HOUR_TO_SHIFT
)
from forecast_system.ishushi.data_agent import IshushiDataAgent

logger = get_logger('ishushi_forecast_model')


class IshushiForecastModel:
    """
    Model dự đoán cho chuỗi nhà hàng Ishushi.
    
    2 mục tiêu song song:
    1. Dự đoán số khách (total_guests) theo ngày cho mỗi nhà hàng
    2. Dự đoán số lượng món ăn (total_quantity) theo ngày, theo sap_code
    """
    
    # ==========================================
    # GUEST FORECAST (SỐ KHÁCH)
    # ==========================================
    
    @staticmethod
    def prepare_guest_features(df_daily_guests, df_bookings=None):
        """
        Chuẩn bị features cho việc dự đoán số khách.
        ⭐ v4: Thêm trend, momentum, booking features
        
        Args:
            df_daily_guests: Từ IshushiDataAgent.build_daily_guest_summary()
            df_bookings: Booking data (optional)
            
        Returns:
            pd.DataFrame with features
        """
        if df_daily_guests.empty:
            return pd.DataFrame()
        
        df = df_daily_guests.copy()
        
        # Time features (⭐ includes is_friday, is_saturday, is_sunday)
        df = IshushiDataAgent.add_time_features(df)
        
        # Lag features for total_guests
        df = IshushiDataAgent.add_lag_features(
            df, 'total_guests', 
            lag_days=[1, 2, 3, 7, 14, 21, 28],
            group_cols=['restaurant_code']
        )
        
        # Rolling features
        df = IshushiDataAgent.add_rolling_features(
            df, 'total_guests',
            windows=[3, 7, 14, 28, 56],
            group_cols=['restaurant_code']
        )
        
        # ⭐ v4: Trend features
        df = IshushiDataAgent.add_trend_features(
            df, 'total_guests',
            group_cols=['restaurant_code']
        )
        
        # YoY features
        df = IshushiDataAgent.add_yoy_features(
            df, 'total_guests',
            group_cols=['restaurant_code']
        )
        
        # Lag features for num_transactions (number of orders)
        df = IshushiDataAgent.add_lag_features(
            df, 'num_transactions',
            lag_days=[1, 7, 14, 28],
            group_cols=['restaurant_code']
        )
        df = IshushiDataAgent.add_rolling_features(
            df, 'num_transactions',
            windows=[7, 14, 28],
            group_cols=['restaurant_code']
        )
        
        # Guests per transaction (avg group size)
        df['guests_per_txn'] = (df['total_guests'] / df['num_transactions'].replace(0, 1))
        df = IshushiDataAgent.add_rolling_features(
            df, 'guests_per_txn',
            windows=[7, 14],
            group_cols=['restaurant_code']
        )
        
        # ⭐ v4: Booking features
        df = IshushiDataAgent.add_booking_features(
            df, df_bookings=df_bookings,
            group_cols=['restaurant_code']
        )
        
        return df
    
    @staticmethod
    def prepare_item_features(df_daily_items):
        """
        Chuẩn bị features cho việc dự đoán số lượng món ăn.
        Từng (restaurant_code, sap_code) là một series riêng.
        
        Args:
            df_daily_items: Từ IshushiDataAgent.build_daily_item_summary()
            
        Returns:
            pd.DataFrame with features
        """
        if df_daily_items.empty:
            return pd.DataFrame()
        
        df = df_daily_items.copy()
        
        # Time features
        df = IshushiDataAgent.add_time_features(df)
        
        # Lag features for total_quantity
        df = IshushiDataAgent.add_lag_features(
            df, 'total_quantity',
            lag_days=[1, 7, 14, 28],
            group_cols=['restaurant_code', 'sap_code']
        )
        
        # Rolling features
        df = IshushiDataAgent.add_rolling_features(
            df, 'total_quantity',
            windows=[7, 14, 28],
            group_cols=['restaurant_code', 'sap_code']
        )
        
        # YoY features
        df = IshushiDataAgent.add_yoy_features(
            df, 'total_quantity',
            group_cols=['restaurant_code', 'sap_code']
        )
        
        # Lag features for num_orders
        df = IshushiDataAgent.add_lag_features(
            df, 'num_orders',
            lag_days=[1, 7, 14],
            group_cols=['restaurant_code', 'sap_code']
        )
        
        # Quantity per order
        df['qty_per_order'] = (
            df['total_quantity'] / df['num_orders'].replace(0, 1)
        )
        
        return df
    
    @staticmethod
    def prepare_group_features(df_daily_groups):
        """
        Chuẩn bị features cho dự đoán theo nhóm món ăn.
        """
        if df_daily_groups.empty:
            return pd.DataFrame()
        
        df = df_daily_groups.copy()
        df = IshushiDataAgent.add_time_features(df)
        
        df = IshushiDataAgent.add_lag_features(
            df, 'total_quantity',
            lag_days=[1, 7, 14, 28],
            group_cols=['restaurant_code', 'item_group']
        )
        df = IshushiDataAgent.add_rolling_features(
            df, 'total_quantity',
            windows=[7, 14, 28],
            group_cols=['restaurant_code', 'item_group']
        )
        
        return df
    
    # ==========================================
    # ML ENSEMBLE MODEL
    # ==========================================
    
    @staticmethod
    def _get_feature_columns(df, target_col):
        """Xác định feature columns (loại bỏ target, identifiers, dates)"""
        exclude_cols = {
            target_col, 'date', 'date_dt', 'weekday', 'restaurant_code',
            'transaction_id', 'sap_code', 'item_name', 'item_group',
            'num_transactions', 'num_orders', 'total_guests', 'total_quantity',
            'guests_per_txn', 'qty_per_order',
        }
        # Keep lag/rolling versions of excluded cols
        feature_cols = [
            c for c in df.columns 
            if c not in exclude_cols 
            and not df[c].dtype == 'object'
            and df[c].notna().sum() > 0
        ]
        return feature_cols
    
    @staticmethod
    def train_and_forecast_xgboost(df_train, df_future, target_col, feature_cols):
        """
        Train XGBoost model và forecast.
        
        Args:
            df_train: Training data
            df_future: Future data (features only, no target)
            target_col: Target column name
            feature_cols: List of feature column names
            
        Returns:
            np.array: Predictions
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("XGBoost not installed, skipping")
            return None
        
        X_train = df_train[feature_cols].fillna(0)
        y_train = df_train[target_col].fillna(0)
        X_future = df_future[feature_cols].fillna(0)
        
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_future)
        
        return np.maximum(preds, 0)  # Non-negative
    
    @staticmethod
    def train_and_forecast_lightgbm(df_train, df_future, target_col, feature_cols):
        """Train LightGBM model và forecast."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("LightGBM not installed, skipping")
            return None
        
        X_train = df_train[feature_cols].fillna(0)
        y_train = df_train[target_col].fillna(0)
        X_future = df_future[feature_cols].fillna(0)
        
        model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_future)
        
        return np.maximum(preds, 0)  # type: ignore[reportArgumentType, reportCallIssue]
    
    @staticmethod
    def train_and_forecast_catboost(df_train, df_future, target_col, feature_cols):
        """Train CatBoost model và forecast."""
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            logger.warning("CatBoost not installed, skipping")
            return None
        
        X_train = df_train[feature_cols].fillna(0)
        y_train = df_train[target_col].fillna(0)
        X_future = df_future[feature_cols].fillna(0)
        
        model = CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            random_seed=42,
            verbose=0,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_future)
        
        return np.maximum(preds, 0)
    
    @staticmethod
    def train_and_forecast_prophet(df_series, forecast_days):
        """
        Train Prophet model cho một time series.
        
        Args:
            df_series: DataFrame with 'date_dt' and target column
            forecast_days: Số ngày dự đoán
            
        Returns:
            pd.DataFrame with 'ds' and 'yhat'
        """
        try:
            from prophet import Prophet
        except ImportError:
            logger.warning("Prophet not installed, skipping")
            return None
        
        if len(df_series) < 14:
            return None
        
        prophet_df = df_series[['date_dt', 'target']].rename(
            columns={'date_dt': 'ds', 'target': 'y'}
        )
        
        model = Prophet(
            yearly_seasonality=True,  # type: ignore[reportArgumentType]
            weekly_seasonality=True,  # type: ignore[reportArgumentType]
            daily_seasonality=False,  # type: ignore[reportArgumentType]
            changepoint_prior_scale=0.1,
        )
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        
        # Chỉ lấy phần forecast mới
        forecast = forecast.tail(forecast_days)[['ds', 'yhat']].reset_index(drop=True)
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        
        return forecast
    
    @staticmethod
    def ensemble_predict(predictions_dict, weights=None):
        """
        Combine predictions từ nhiều models.
        ⭐ v4: Simple weighted average (used as fallback for meta-learner).
        """
        valid_preds = {k: v for k, v in predictions_dict.items() if v is not None}
        
        if not valid_preds:
            return None
        
        if weights is None:
            weights = {k: 1.0 / len(valid_preds) for k in valid_preds}
        
        # Normalize weights
        total_w = sum(weights.get(k, 0) for k in valid_preds)
        if total_w == 0:
            total_w = 1.0
        
        result = np.zeros(len(next(iter(valid_preds.values()))))
        for model_name, preds in valid_preds.items():
            w = weights.get(model_name, 1.0 / len(valid_preds)) / total_w
            result += preds * w
        
        return np.maximum(result, 0)
    
    @staticmethod
    def meta_learner_ensemble(df_train, df_future, preds_train, preds_future,
                              target_col='total_guests'):
        """
        ⭐ v4: Meta-Learning Ensemble using LightGBM.
        
        Instead of weighted average, train a LightGBM that learns
        which model to trust based on context (weekday, month, etc.)
        
        Args:
            df_train: Training DataFrame with time features
            df_future: Future DataFrame with time features
            preds_train: Dict of {model_name: train_predictions}
            preds_future: Dict of {model_name: future_predictions}
            target_col: Target column name in df_train
            
        Returns:
            np.array: Final predictions, or None if fails
        """
        try:
            import lightgbm as lgb
        except ImportError:
            return None
        
        if not preds_train or not preds_future:
            return None
        
        # Build meta-features for training
        meta_train = pd.DataFrame()
        meta_future = pd.DataFrame()
        
        # ⭐ Only add model predictions that exist in BOTH train and future
        common_models = set(preds_train.keys()) & set(preds_future.keys())
        
        for name in sorted(common_models):
            p_train = preds_train[name]
            p_fut = preds_future[name]
            if (p_train is not None and len(p_train) == len(df_train) and
                p_fut is not None and len(p_fut) == len(df_future)):
                meta_train[f'pred_{name}'] = p_train
                meta_future[f'pred_{name}'] = p_fut
        
        if meta_train.empty or meta_future.empty:
            return None
        
        # Add context features (only if present in BOTH DataFrames)
        context_cols = ['day_of_week', 'is_weekend', 'month', 'is_friday',
                       'is_saturday', 'is_sunday']
        for col in context_cols:
            if col in df_train.columns and col in df_future.columns:
                meta_train[col] = df_train[col].values
                meta_future[col] = df_future[col].values
        
        # ⭐ Safety: ensure exact same columns in both DataFrames
        common_cols = sorted(set(meta_train.columns) & set(meta_future.columns))
        meta_train = meta_train[common_cols]
        meta_future = meta_future[common_cols]
        
        # Target
        y_train = df_train[target_col].values
        
        if len(meta_train) < 14 or len(y_train) < 14:
            return None
        
        try:
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=5,
                random_state=42,
                verbose=-1,
            )
            model.fit(meta_train, y_train)
            preds = model.predict(meta_future)
            return np.maximum(preds, 0)
        except Exception:
            return None
    
    # ==========================================
    # FORECAST PIPELINE
    # ==========================================
    
    @staticmethod
    def _build_future_dates(last_date, forecast_days):
        """Build future dates DataFrame for forecasting.
        
        ⭐ Đảm bảo forecast luôn bắt đầu từ ngày hôm nay (run_date) trở đi,
        KHÔNG bỏ qua ngày chạy model.
        """
        today = pd.Timestamp(datetime.date.today())
        
        # Nếu last_date < today → bắt đầu từ today (bao gồm ngày chạy model)
        # Nếu last_date == today → cũng bắt đầu từ today (dữ liệu hôm nay chưa đầy đủ)
        # Nếu last_date > today (hiếm) → bắt đầu từ last_date + 1
        if isinstance(last_date, datetime.date) and not isinstance(last_date, pd.Timestamp):
            last_date_ts = pd.Timestamp(last_date)
        else:
            last_date_ts = last_date
        
        if last_date_ts >= today:  # type: ignore[reportOperatorIssue]
            # last_date là hôm nay hoặc tương lai → bắt đầu từ chính ngày đó
            start_date = last_date_ts
        else:
            # last_date ở quá khứ → bắt đầu từ hôm nay
            start_date = today
        
        future_dates = pd.date_range(
            start=start_date,
            periods=forecast_days,
            freq='D'
        )
        
        df_future = pd.DataFrame({'date_dt': future_dates})
        df_future['date'] = df_future['date_dt'].dt.date
        df_future['weekday'] = df_future['date_dt'].dt.day_name()
        
        # Add time features (MUST match add_time_features exactly)
        df_future['day_of_week'] = df_future['date_dt'].dt.dayofweek
        df_future['day_of_month'] = df_future['date_dt'].dt.day
        df_future['month'] = df_future['date_dt'].dt.month
        df_future['year'] = df_future['date_dt'].dt.year
        df_future['week_of_year'] = df_future['date_dt'].dt.isocalendar().week.astype(int)
        df_future['is_weekend'] = df_future['day_of_week'].isin([5, 6]).astype(int)
        df_future['quarter'] = df_future['date_dt'].dt.quarter
        
        # ⭐ v4: Explicit day-of-week flags (must match add_time_features)
        df_future['is_friday'] = (df_future['day_of_week'] == 4).astype(int)
        df_future['is_saturday'] = (df_future['day_of_week'] == 5).astype(int)
        df_future['is_sunday'] = (df_future['day_of_week'] == 6).astype(int)
        
        df_future['dow_sin'] = np.sin(2 * np.pi * df_future['day_of_week'] / 7)
        df_future['dow_cos'] = np.cos(2 * np.pi * df_future['day_of_week'] / 7)
        df_future['month_sin'] = np.sin(2 * np.pi * df_future['month'] / 12)
        df_future['month_cos'] = np.cos(2 * np.pi * df_future['month'] / 12)
        
        df_future['is_month_start'] = (df_future['day_of_month'] <= 5).astype(int)
        df_future['is_month_end'] = (df_future['day_of_month'] >= 25).astype(int)
        
        return df_future
    
    @staticmethod
    def _propagate_lag_features(df_train, df_future, target_col, feature_cols, group_cols=None):
        """
        Điền lag/rolling features cho future dates dựa trên dữ liệu train cuối cùng.
        """
        if group_cols is None:
            group_cols = ['restaurant_code']
        
        # Lấy dữ liệu cuối cùng của training
        for col in feature_cols:
            if col not in df_future.columns:
                # Tìm giá trị cuối cùng từ training data
                last_val = df_train[col].iloc[-1] if col in df_train.columns else 0
                df_future[col] = last_val
        
        return df_future
    
    @staticmethod
    def forecast_guests(df_daily_guests, forecast_days=None, df_bookings=None):
        """
        Dự đoán số khách cho mỗi nhà hàng Ishushi.
        ⭐ v4: Meta-learner ensemble + booking features
        
        Args:
            df_daily_guests: Daily guest summary
            forecast_days: Số ngày dự đoán (default: 30)
            df_bookings: Booking data for feature injection
            
        Returns:
            pd.DataFrame: [restaurant_code, date, predicted_guests, 
                          predicted_transactions, model_used]
        """
        if forecast_days is None:
            forecast_days = ISHUSHI_CONFIG['forecast_days']
        
        if df_daily_guests.empty:
            logger.warning("No guest data for forecasting")
            return pd.DataFrame()
        
        logger.info(f"\n{'='*50}")
        logger.info(f"🔮 GUEST FORECAST v4 ({forecast_days} days)")
        logger.info(f"{'='*50}")
        
        use_meta = ISHUSHI_CONFIG.get('meta_learner_enabled', True)
        
        # Prepare features (⭐ includes trend + booking)
        df_featured = IshushiForecastModel.prepare_guest_features(
            df_daily_guests, df_bookings=df_bookings
        )
        
        all_forecasts = []
        restaurants = df_featured['restaurant_code'].unique()
        
        for res_code in restaurants:
            logger.info(f"\n🏪 Restaurant: {res_code}")
            
            df_res = df_featured[df_featured['restaurant_code'] == res_code].copy()
            df_res = df_res.sort_values('date_dt')  # type: ignore[reportCallIssue]
            
            if len(df_res) < ISHUSHI_CONFIG['min_data_days']:
                logger.warning(f"   Not enough data ({len(df_res)} days)")
                continue
            
            target_col = 'total_guests'
            feature_cols = IshushiForecastModel._get_feature_columns(df_res, target_col)
            
            # Remove NaN rows from training
            df_train = df_res.dropna(subset=feature_cols + [target_col])
            
            if len(df_train) < 14:
                logger.warning(f"   Not enough clean data ({len(df_train)} rows)")
                continue
            
            # Build future dates
            last_date = df_train['date_dt'].max()
            df_future = IshushiForecastModel._build_future_dates(last_date, forecast_days)
            df_future['restaurant_code'] = res_code
            
            # Propagate lag features
            df_future = IshushiForecastModel._propagate_lag_features(
                df_train, df_future, target_col, feature_cols
            )
            
            # Calculate days_since_start for future
            min_date = df_train['date_dt'].min()
            df_future['days_since_start'] = (df_future['date_dt'] - min_date).dt.days
            
            # --- ML Models ---
            preds_future = {}
            preds_train_insample = {}  # ⭐ v4: for meta-learner
            
            # XGBoost
            try:
                p = IshushiForecastModel.train_and_forecast_xgboost(
                    df_train, df_future, target_col, feature_cols
                )
                if p is not None:
                    preds_future['xgboost'] = p
                    # In-sample predictions for meta-learner
                    p_train = IshushiForecastModel.train_and_forecast_xgboost(
                        df_train, df_train, target_col, feature_cols
                    )
                    if p_train is not None:
                        preds_train_insample['xgboost'] = p_train
                    logger.info(f"   ✅ XGBoost: avg={p.mean():.1f}")
            except Exception as e:
                logger.warning(f"   ❌ XGBoost failed: {e}")
            
            # LightGBM
            try:
                p = IshushiForecastModel.train_and_forecast_lightgbm(
                    df_train, df_future, target_col, feature_cols
                )
                if p is not None:
                    preds_future['lightgbm'] = p
                    p_train = IshushiForecastModel.train_and_forecast_lightgbm(
                        df_train, df_train, target_col, feature_cols
                    )
                    if p_train is not None:
                        preds_train_insample['lightgbm'] = p_train
                    logger.info(f"   ✅ LightGBM: avg={p.mean():.1f}")
            except Exception as e:
                logger.warning(f"   ❌ LightGBM failed: {e}")
            
            # CatBoost
            try:
                p = IshushiForecastModel.train_and_forecast_catboost(
                    df_train, df_future, target_col, feature_cols
                )
                if p is not None:
                    preds_future['catboost'] = p
                    p_train = IshushiForecastModel.train_and_forecast_catboost(
                        df_train, df_train, target_col, feature_cols
                    )
                    if p_train is not None:
                        preds_train_insample['catboost'] = p_train
                    logger.info(f"   ✅ CatBoost: avg={p.mean():.1f}")
            except Exception as e:
                logger.warning(f"   ❌ CatBoost failed: {e}")
            
            # --- Prophet ---
            try:
                prophet_df = df_res[['date_dt', target_col]].copy()
                prophet_df = prophet_df.rename(columns={target_col: 'target'})  # type: ignore[reportCallIssue]
                prophet_result = IshushiForecastModel.train_and_forecast_prophet(
                    prophet_df, forecast_days
                )
                if prophet_result is not None:
                    preds_future['prophet'] = prophet_result['yhat'].values  # type: ignore[reportAttributeAccessIssue]
                    # In-sample: use fitted values on training period
                    from prophet import Prophet
                    p_model = Prophet(
                        yearly_seasonality=True, weekly_seasonality=True,  # type: ignore[reportArgumentType]
                        daily_seasonality=False, changepoint_prior_scale=0.1,  # type: ignore[reportArgumentType]
                    )
                    p_fit_df = prophet_df.rename(columns={'date_dt': 'ds', 'target': 'y'})
                    p_model.fit(p_fit_df)
                    p_in_sample = p_model.predict(p_fit_df[['ds']])  # type: ignore[reportArgumentType]
                    preds_train_insample['prophet'] = p_in_sample['yhat'].clip(lower=0).values
                    logger.info(f"   ✅ Prophet: avg={prophet_result['yhat'].mean():.1f}")
            except Exception as e:
                logger.warning(f"   ❌ Prophet failed: {e}")
            
            # --- ⭐ v4: Meta-Learner Ensemble ---
            if not preds_future:
                # Fallback: Dùng trung bình 28 ngày cuối
                avg_28 = df_res.tail(28)[target_col].mean()
                ensemble = np.full(forecast_days, avg_28)
                model_used = 'historical_avg'
                logger.info(f"   ⚠️ All models failed, using 28-day avg: {avg_28:.1f}")
            else:
                meta_result = None
                if use_meta and len(preds_train_insample) >= 2:
                    meta_result = IshushiForecastModel.meta_learner_ensemble(
                        df_train, df_future, preds_train_insample, preds_future,
                        target_col=target_col
                    )
                
                if meta_result is not None:
                    ensemble = meta_result
                    model_used = f"meta_lgbm({'+'.join(preds_future.keys())})"
                    logger.info(f"   🧠 Meta-Learner avg: {ensemble.mean():.1f}")
                else:
                    # Fallback: weighted average
                    ml_models = {k: v for k, v in preds_future.items() if k != 'prophet'}
                    if ml_models and 'prophet' in preds_future:
                        ml_ensemble = IshushiForecastModel.ensemble_predict(ml_models)
                        ensemble = 0.7 * ml_ensemble + 0.3 * preds_future['prophet']  # type: ignore[reportOperatorIssue]
                    else:
                        ensemble = IshushiForecastModel.ensemble_predict(preds_future)
                    model_used = f"weighted_avg({'+'.join(preds_future.keys())})"
                    logger.info(f"   🎯 Weighted avg: {ensemble.mean():.1f}")  # type: ignore[reportOptionalMemberAccess]
            
            # Build forecast DataFrame
            df_forecast = df_future[['restaurant_code', 'date', 'weekday']].copy()
            df_forecast['predicted_guests'] = np.round(ensemble).astype(int)  # type: ignore[reportArgumentType, reportCallIssue]
            df_forecast['model_used'] = model_used
            
            # Estimate transactions (based on avg guests per transaction)
            avg_gpt = df_res['guests_per_txn'].dropna().tail(28).mean()
            if pd.isna(avg_gpt) or avg_gpt <= 0:  # type: ignore[reportGeneralTypeIssues]
                avg_gpt = 2.0  # Default: 2 guests per transaction
            df_forecast['predicted_transactions'] = np.round(
                df_forecast['predicted_guests'] / avg_gpt
            ).astype(int)
            
            all_forecasts.append(df_forecast)
        
        if all_forecasts:
            result = pd.concat(all_forecasts, ignore_index=True)
            logger.info(f"\n✅ Guest forecast v4 complete: {len(result)} records")
            return result
        
        return pd.DataFrame()
    
    @staticmethod
    def _compute_historical_shift_ratios(df_daily_guests_shift, res_code):
        """
        Tính tỷ lệ shift ratio lịch sử cho từng weekday.
        ⭐ v4: Kept as fallback when shift ML data is insufficient.
        
        Returns:
            dict: {weekday_name: {'MORNING': ratio, 'EVENING': ratio}}
        """
        df_res = df_daily_guests_shift[
            df_daily_guests_shift['restaurant_code'] == res_code
        ].copy()
        
        if df_res.empty:
            return {}
        
        ratios = {}
        
        if 'weekday' not in df_res.columns:
            return {}
        
        for weekday in df_res['weekday'].unique():
            df_wd = df_res[df_res['weekday'] == weekday]
            
            morning_total = df_wd[
                df_wd['shift'] == 'MORNING'
            ]['total_guests'].sum()
            evening_total = df_wd[
                df_wd['shift'] == 'EVENING'
            ]['total_guests'].sum()
            
            daily_total = morning_total + evening_total
            if daily_total > 0:
                ratios[weekday] = {
                    'MORNING': morning_total / daily_total,
                    'EVENING': evening_total / daily_total,
                }
            else:
                ratios[weekday] = {'MORNING': 0.45, 'EVENING': 0.55}
        
        return ratios
    
    @staticmethod
    def _prepare_shift_features(df_shift_data, shift_name, df_bookings=None):
        """
        ⭐ v4: Prepare features specifically for a shift model.
        Target = total_guests for that specific shift.
        
        Args:
            df_shift_data: DataFrame filtered for one shift
            shift_name: 'MORNING' or 'EVENING'
            df_bookings: Booking data (optional)
            
        Returns:
            pd.DataFrame with features
        """
        if df_shift_data.empty:
            return pd.DataFrame()
        
        df = df_shift_data.copy()
        
        # Time features (includes is_friday, is_saturday, is_sunday)
        df = IshushiDataAgent.add_time_features(df)
        
        # Lag features for shift-specific guests
        df = IshushiDataAgent.add_lag_features(
            df, 'total_guests', 
            lag_days=[1, 2, 3, 7, 14, 21, 28],
            group_cols=['restaurant_code']
        )
        
        # Rolling features
        df = IshushiDataAgent.add_rolling_features(
            df, 'total_guests',
            windows=[3, 7, 14, 28],
            group_cols=['restaurant_code']
        )
        
        # Trend features
        df = IshushiDataAgent.add_trend_features(
            df, 'total_guests',
            group_cols=['restaurant_code']
        )
        
        # YoY
        df = IshushiDataAgent.add_yoy_features(
            df, 'total_guests',
            group_cols=['restaurant_code']
        )
        
        # Booking features
        df = IshushiDataAgent.add_booking_features(
            df, df_bookings=df_bookings,
            group_cols=['restaurant_code']
        )
        
        return df
    
    @staticmethod
    def _forecast_single_shift(df_shift_featured, res_code, shift_name,
                                forecast_days):
        """
        ⭐ v4: Train ML models directly for one shift.
        Returns (predictions, model_used) or (None, None).
        """
        df_res = df_shift_featured[
            df_shift_featured['restaurant_code'] == res_code
        ].copy()
        df_res = df_res.sort_values('date_dt')
        
        if len(df_res) < 14:
            return None, None
        
        target_col = 'total_guests'
        feature_cols = IshushiForecastModel._get_feature_columns(df_res, target_col)
        
        df_train = df_res.dropna(subset=feature_cols + [target_col])
        
        if len(df_train) < 14:
            return None, None
        
        # Build future dates
        last_date = df_train['date_dt'].max()
        df_future = IshushiForecastModel._build_future_dates(last_date, forecast_days)
        df_future['restaurant_code'] = res_code
        
        # Propagate lag features
        df_future = IshushiForecastModel._propagate_lag_features(
            df_train, df_future, target_col, feature_cols
        )
        
        min_date = df_train['date_dt'].min()
        df_future['days_since_start'] = (df_future['date_dt'] - min_date).dt.days
        
        use_meta = ISHUSHI_CONFIG.get('meta_learner_enabled', True)
        
        # Train models
        preds_future = {}
        preds_train = {}
        
        for name, train_fn in [
            ('xgboost', IshushiForecastModel.train_and_forecast_xgboost),
            ('lightgbm', IshushiForecastModel.train_and_forecast_lightgbm),
            ('catboost', IshushiForecastModel.train_and_forecast_catboost),
        ]:
            try:
                p = train_fn(df_train, df_future, target_col, feature_cols)
                if p is not None:
                    preds_future[name] = p
                    p_t = train_fn(df_train, df_train, target_col, feature_cols)
                    if p_t is not None:
                        preds_train[name] = p_t
            except Exception:
                pass
        
        # Prophet
        try:
            prophet_df = df_res[['date_dt', target_col]].copy()
            prophet_df = prophet_df.rename(columns={target_col: 'target'})
            prophet_result = IshushiForecastModel.train_and_forecast_prophet(
                prophet_df, forecast_days
            )
            if prophet_result is not None:
                preds_future['prophet'] = prophet_result['yhat'].values  # type: ignore[reportAttributeAccessIssue]
                from prophet import Prophet
                p_model = Prophet(
                    yearly_seasonality=True, weekly_seasonality=True,  # type: ignore[reportArgumentType]
                    daily_seasonality=False, changepoint_prior_scale=0.1,  # type: ignore[reportArgumentType]
                )
                p_fit_df = prophet_df.rename(columns={'date_dt': 'ds', 'target': 'y'})
                p_model.fit(p_fit_df)
                p_in = p_model.predict(p_fit_df[['ds']])
                preds_train['prophet'] = p_in['yhat'].clip(lower=0).values
        except Exception:
            pass
        
        if not preds_future:
            return None, None
        
        # Meta-learner ensemble
        meta_result = None
        if use_meta and len(preds_train) >= 2:
            meta_result = IshushiForecastModel.meta_learner_ensemble(
                df_train, df_future, preds_train, preds_future,
                target_col=target_col
            )
        
        if meta_result is not None:
            preds_final = meta_result
            model_used = f"shift_meta_{shift_name.lower()}({'+'.join(preds_future.keys())})"
        else:
            ml_models = {k: v for k, v in preds_future.items() if k != 'prophet'}
            if ml_models and 'prophet' in preds_future:
                ml_ens = IshushiForecastModel.ensemble_predict(ml_models)
                preds_final = 0.7 * ml_ens + 0.3 * preds_future['prophet']  # type: ignore[reportOperatorIssue]
            else:
                preds_final = IshushiForecastModel.ensemble_predict(preds_future)
            model_used = f"shift_wt_{shift_name.lower()}({'+'.join(preds_future.keys())})"
        
        return (df_future, preds_final, model_used)
    
    @staticmethod
    def forecast_guests_by_shift(df_daily_guests, df_daily_guests_shift,
                                 forecast_days=None, df_bookings=None):
        """
        ⭐ v4: Dự đoán số khách trực tiếp theo CA (MORNING/EVENING).
        
        THAY ĐỔI CHÍNH: Không còn forecast daily total rồi chia ratio.
        Thay vào đó, train ML model riêng cho MỖI CA:
        - Model MORNING: target = morning_guests, features incl. weekday flags + booking + trend
        - Model EVENING: target = evening_guests, features incl. weekday flags + booking + trend
        
        Fallback to ratio-based split if shift data < 14 days.
        
        Args:
            df_daily_guests: Daily guest summary (total)
            df_daily_guests_shift: Daily guest summary BY SHIFT
            forecast_days: Số ngày dự đoán
            df_bookings: Booking data (optional)
            
        Returns:
            pd.DataFrame: [restaurant_code, date, shift, weekday,
                          predicted_guests, predicted_transactions, model_used]
        """
        if forecast_days is None:
            forecast_days = ISHUSHI_CONFIG['forecast_days']
        
        logger.info(f"\n{'='*50}")
        logger.info(f"🔮 SHIFT FORECAST v4 - DIRECT SHIFT MODELS")
        logger.info(f"{'='*50}")
        
        all_shift_rows = []
        restaurants = df_daily_guests['restaurant_code'].unique()
        
        for res_code in restaurants:
            logger.info(f"\n🏪 Restaurant: {res_code}")
            
            # Check shift data availability
            df_res_shift = df_daily_guests_shift[
                df_daily_guests_shift['restaurant_code'] == res_code
            ] if not df_daily_guests_shift.empty else pd.DataFrame()
            
            has_enough_shift_data = (
                not df_res_shift.empty and 
                len(df_res_shift[df_res_shift['shift'] == 'MORNING']) >= 14 and
                len(df_res_shift[df_res_shift['shift'] == 'EVENING']) >= 14
            )
            
            if has_enough_shift_data:
                # ⭐ v4: DIRECT SHIFT ML MODELS
                logger.info(f"   🧠 Using DIRECT shift ML models")
                
                for shift_name in ['MORNING', 'EVENING']:
                    # Filter shift data
                    df_shift = df_daily_guests_shift[
                        (df_daily_guests_shift['restaurant_code'] == res_code) &
                        (df_daily_guests_shift['shift'] == shift_name)
                    ].copy()
                    
                    # Prepare shift-specific features
                    df_featured = IshushiForecastModel._prepare_shift_features(
                        df_shift, shift_name, df_bookings=df_bookings
                    )
                    
                    # Forecast
                    result = IshushiForecastModel._forecast_single_shift(
                        df_featured, res_code, shift_name, forecast_days
                    )
                    
                    if result is not None and result[0] is not None:
                        df_future, preds_final, model_used = result
                        
                        for i, (_, row) in enumerate(df_future.iterrows()):
                            shift_guests = max(0, round(float(preds_final[i])))
                            shift_txns = max(1, round(shift_guests / 2.0))
                            
                            all_shift_rows.append({
                                'restaurant_code': res_code,
                                'date': row['date'],
                                'shift': shift_name,
                                'weekday': row.get('weekday', ''),
                                'predicted_guests': int(shift_guests),
                                'predicted_transactions': int(shift_txns),
                                'model_used': model_used,
                            })
                        
                        avg = np.mean(preds_final)
                        logger.info(f"   ✅ {shift_name}: avg={avg:.1f} guests/day")
                    else:
                        # Fallback for this specific shift
                        logger.warning(f"   ⚠️ {shift_name} direct model failed, using ratio fallback")
                        # Use ratio fallback just for this shift
                        avg_shift = df_shift['total_guests'].tail(28).mean() if not df_shift.empty else 10
                        last_date = pd.to_datetime(df_shift['date']).max() if not df_shift.empty else pd.Timestamp.now()
                        # Bắt đầu từ hôm nay (không bỏ qua ngày chạy model)
                        today_ts = pd.Timestamp(datetime.date.today())
                        fallback_start = max(last_date, today_ts) if last_date >= today_ts else today_ts
                        future_dates = pd.date_range(
                            start=fallback_start,
                            periods=forecast_days, freq='D'
                        )
                        for d in future_dates:
                            all_shift_rows.append({
                                'restaurant_code': res_code,
                                'date': d.date(),
                                'shift': shift_name,
                                'weekday': d.day_name(),
                                'predicted_guests': int(round(avg_shift)),
                                'predicted_transactions': max(1, int(round(avg_shift / 2))),
                                'model_used': 'shift_avg_fallback',
                            })
            else:
                # LEGACY FALLBACK: ratio-based split (when not enough shift data)
                logger.info(f"   📊 Insufficient shift data → ratio fallback")
                
                df_daily_forecast = IshushiForecastModel.forecast_guests(
                    df_daily_guests[df_daily_guests['restaurant_code'] == res_code],
                    forecast_days=forecast_days, df_bookings=df_bookings
                )
                
                if df_daily_forecast.empty:
                    continue
                
                hist_ratios = IshushiForecastModel._compute_historical_shift_ratios(
                    df_daily_guests_shift, res_code
                )
                
                for _, row in df_daily_forecast.iterrows():
                    daily_total = float(row['predicted_guests'])
                    weekday = str(row.get('weekday', ''))
                    wd_ratio = hist_ratios.get(weekday, {'MORNING': 0.45, 'EVENING': 0.55})
                    
                    for shift in ['MORNING', 'EVENING']:
                        ratio = wd_ratio.get(shift, 0.5)
                        shift_guests = max(0, round(daily_total * ratio))
                        shift_txns = max(1, round(shift_guests / 2.0))
                        
                        all_shift_rows.append({
                            'restaurant_code': res_code,
                            'date': row['date'],
                            'shift': shift,
                            'weekday': weekday,
                            'predicted_guests': int(shift_guests),
                            'predicted_transactions': int(shift_txns),
                            'model_used': f"ratio_fallback({row.get('model_used', 'ensemble')})",
                        })
        
        if all_shift_rows:
            result = pd.DataFrame(all_shift_rows)
            
            for shift in ['MORNING', 'EVENING']:
                df_shift = result[result['shift'] == shift]
                if not df_shift.empty:
                    logger.info(
                        f"   {shift}: avg {df_shift['predicted_guests'].mean():.0f} guests/day"
                    )
            
            logger.info(f"\n✅ Shift forecast v4 complete: {len(result)} records "
                       f"({len(result)//2} days × 2 shifts)")
            return result
        
        return pd.DataFrame()
    
    @staticmethod
    def forecast_items(df_daily_items, df_daily_guests, forecast_days=None):
        """
        Dự đoán số lượng món ăn theo sap_code cho mỗi nhà hàng.
        
        Approach: 
        - Tính tỉ lệ (ratio) mỗi sap_code so với tổng khách
        - Dùng ML để dự đoán ratio + absolute quantity
        - Combine: predicted_quantity = avg(ML_qty, predicted_guests * ratio)
        
        Args:
            df_daily_items: Daily item summary
            df_daily_guests: Daily guest summary (for ratio calculation)
            forecast_days: Số ngày dự đoán
            
        Returns:
            pd.DataFrame: [restaurant_code, date, sap_code, item_name,
                          item_group, predicted_quantity, model_used]
        """
        if forecast_days is None:
            forecast_days = ISHUSHI_CONFIG['forecast_days']
        
        if df_daily_items.empty:
            logger.warning("No item data for forecasting")
            return pd.DataFrame()
        
        logger.info(f"\n{'='*50}")
        logger.info(f"🍱 ITEM FORECAST ({forecast_days} days)")
        logger.info(f"{'='*50}")
        
        # Prepare features
        df_featured = IshushiForecastModel.prepare_item_features(df_daily_items)
        
        all_forecasts = []
        
        # Get unique (restaurant, sap_code) pairs
        pairs = df_featured.groupby(['restaurant_code', 'sap_code']).size().reset_index(name='count')  # type: ignore[reportCallIssue]
        
        logger.info(f"   Total series to forecast: {len(pairs)}")
        
        for _, pair in pairs.iterrows():
            res_code = pair['restaurant_code']
            sap_code = pair['sap_code']
            item_name = ISHUSHI_SAP_CATALOG.get(sap_code, f"Unknown ({sap_code})")  # type: ignore[reportArgumentType, reportCallIssue]
            
            # Get group
            sap_to_group = {}
            for group, codes in ISHUSHI_SAP_GROUPS.items():
                for code in codes:
                    sap_to_group[code] = group
            item_group = sap_to_group.get(sap_code, "Other")
            
            df_series = df_featured[  # type: ignore[reportCallIssue]
                (df_featured['restaurant_code'] == res_code) &
                (df_featured['sap_code'] == sap_code)
            ].copy().sort_values('date_dt')
            
            if len(df_series) < 7:
                # Quá ít data → dùng simple average
                avg_qty = df_series['total_quantity'].mean()
                
                future_dates = IshushiForecastModel._build_future_dates(
                    df_series['date_dt'].max(), forecast_days
                )
                df_fc = future_dates[['date', 'weekday']].copy()
                df_fc['restaurant_code'] = res_code
                df_fc['sap_code'] = sap_code
                df_fc['item_name'] = item_name
                df_fc['item_group'] = item_group
                df_fc['predicted_quantity'] = round(avg_qty)
                df_fc['model_used'] = 'simple_avg'
                all_forecasts.append(df_fc)
                continue
            
            target_col = 'total_quantity'
            feature_cols = IshushiForecastModel._get_feature_columns(df_series, target_col)
            
            df_train = df_series.dropna(subset=[c for c in feature_cols if c in df_series.columns])
            
            if len(df_train) < 7:
                continue
            
            # Build future
            last_date = df_train['date_dt'].max()
            df_future = IshushiForecastModel._build_future_dates(last_date, forecast_days)
            df_future['restaurant_code'] = res_code
            df_future['sap_code'] = sap_code
            
            # Propagate features
            df_future = IshushiForecastModel._propagate_lag_features(
                df_train, df_future, target_col, feature_cols,
                group_cols=['restaurant_code', 'sap_code']
            )
            min_date = df_train['date_dt'].min()
            df_future['days_since_start'] = (df_future['date_dt'] - min_date).dt.days
            
            # ML predictions
            preds = {}
            
            try:
                p = IshushiForecastModel.train_and_forecast_xgboost(
                    df_train, df_future, target_col, feature_cols
                )
                if p is not None:
                    preds['xgboost'] = p
            except Exception:
                pass
            
            try:
                p = IshushiForecastModel.train_and_forecast_lightgbm(
                    df_train, df_future, target_col, feature_cols
                )
                if p is not None:
                    preds['lightgbm'] = p
            except Exception:
                pass
            
            # Ensemble
            if preds:
                ensemble = IshushiForecastModel.ensemble_predict(preds)
                model_used = f"ml_ensemble({'+'.join(preds.keys())})"
            else:
                # Fallback: weekday average from last 28 days
                recent = df_series.tail(28)
                weekday_avg = recent.groupby('day_of_week')['total_quantity'].mean()
                
                ensemble = np.array([
                    weekday_avg.get(d, recent['total_quantity'].mean())
                    for d in df_future['day_of_week']
                ])
                model_used = 'weekday_avg'
            
            # Build forecast
            df_fc = df_future[['date', 'weekday']].copy()
            df_fc['restaurant_code'] = res_code
            df_fc['sap_code'] = sap_code
            df_fc['item_name'] = item_name
            df_fc['item_group'] = item_group
            df_fc['predicted_quantity'] = np.round(ensemble).astype(int)  # type: ignore[reportArgumentType, reportCallIssue]
            df_fc['model_used'] = model_used
            
            all_forecasts.append(df_fc)
        
        if all_forecasts:
            result = pd.concat(all_forecasts, ignore_index=True)
            logger.info(f"\n✅ Item forecast complete: {len(result)} records")
            return result
        
        return pd.DataFrame()
    
    @staticmethod
    def forecast_items_by_shift(df_daily_items, df_daily_items_shift,
                                df_daily_guests, forecast_days=None):
        """
        Dự đoán số lượng món ăn theo CA LÀM VIỆC.
        
        Approach:
        1. Forecast daily total per (restaurant, sap_code)
        2. Phân phối vào MORNING/EVENING dùng tỷ lệ lịch sử
        
        Args:
            df_daily_items: Daily item summary (total)
            df_daily_items_shift: Daily item summary BY SHIFT
            df_daily_guests: Daily guest summary (for ratio)
            forecast_days: Số ngày dự đoán
            
        Returns:
            pd.DataFrame: [restaurant_code, date, shift, sap_code, item_name,
                          item_group, predicted_quantity, model_used]
        """
        if forecast_days is None:
            forecast_days = ISHUSHI_CONFIG['forecast_days']
        
        # Get daily total item forecast
        df_item_forecast = IshushiForecastModel.forecast_items(
            df_daily_items, df_daily_guests, forecast_days=forecast_days
        )
        
        if df_item_forecast.empty:
            return pd.DataFrame()
        
        logger.info(f"\n{'='*50}")
        logger.info(f"🔄 DISTRIBUTING ITEM FORECAST TO SHIFTS")
        logger.info(f"{'='*50}")
        
        all_shift_rows = []
        
        # Compute shift ratios per (restaurant, sap_code) from historical data
        if not df_daily_items_shift.empty:
            for (res_code, sap_code), grp in df_daily_items_shift.groupby(
                ['restaurant_code', 'sap_code']
            ):
                morning_total = grp[
                    grp['shift'] == 'MORNING'
                ]['total_quantity'].sum()
                evening_total = grp[
                    grp['shift'] == 'EVENING'
                ]['total_quantity'].sum()
                total = morning_total + evening_total
                
                if total > 0:
                    ratio = {
                        'MORNING': morning_total / total,
                        'EVENING': evening_total / total,
                    }
                else:
                    ratio = {'MORNING': 0.45, 'EVENING': 0.55}
                
                # Get forecasts for this (restaurant, sap_code)
                mask = (
                    (df_item_forecast['restaurant_code'] == res_code) &
                    (df_item_forecast['sap_code'] == sap_code)
                )
                df_fc = df_item_forecast[mask]
                
                for _, row in df_fc.iterrows():
                    daily_qty = float(row['predicted_quantity'])
                    
                    for shift in ['MORNING', 'EVENING']:
                        shift_qty = max(0, round(daily_qty * ratio[shift]))
                        
                        all_shift_rows.append({
                            'restaurant_code': res_code,
                            'date': row['date'],
                            'shift': shift,
                            'weekday': row.get('weekday', ''),
                            'sap_code': row.get('sap_code', ''),
                            'item_name': row.get('item_name', ''),
                            'item_group': row.get('item_group', ''),
                            'predicted_quantity': int(shift_qty),
                            'model_used': row.get('model_used', 'ensemble'),
                        })
        else:
            # No shift history → default 45/55 split
            for _, row in df_item_forecast.iterrows():
                daily_qty = float(row['predicted_quantity'])
                for shift, ratio in [('MORNING', 0.45), ('EVENING', 0.55)]:
                    shift_qty = max(0, round(daily_qty * ratio))
                    all_shift_rows.append({
                        'restaurant_code': row['restaurant_code'],
                        'date': row['date'],
                        'shift': shift,
                        'weekday': row.get('weekday', ''),
                        'sap_code': row.get('sap_code', ''),
                        'item_name': row.get('item_name', ''),
                        'item_group': row.get('item_group', ''),
                        'predicted_quantity': int(shift_qty),
                        'model_used': row.get('model_used', 'ensemble'),
                    })
        
        # Handle items in forecast but not in shift history
        if not df_daily_items_shift.empty:
            shift_pairs = set(
                zip(df_daily_items_shift['restaurant_code'],
                    df_daily_items_shift['sap_code'].astype(str))
            )
            for _, row in df_item_forecast.iterrows():
                pair = (str(row['restaurant_code']), str(row.get('sap_code', '')))
                if pair not in shift_pairs:
                    daily_qty = float(row['predicted_quantity'])
                    for shift, ratio in [('MORNING', 0.45), ('EVENING', 0.55)]:
                        shift_qty = max(0, round(daily_qty * ratio))
                        all_shift_rows.append({
                            'restaurant_code': row['restaurant_code'],
                            'date': row['date'],
                            'shift': shift,
                            'weekday': row.get('weekday', ''),
                            'sap_code': row.get('sap_code', ''),
                            'item_name': row.get('item_name', ''),
                            'item_group': row.get('item_group', ''),
                            'predicted_quantity': int(shift_qty),
                            'model_used': row.get('model_used', 'ensemble'),
                        })
        
        if all_shift_rows:
            result = pd.DataFrame(all_shift_rows)
            # Deduplicate in case
            result = result.drop_duplicates(
                subset=['restaurant_code', 'date', 'shift', 'sap_code'],
                keep='first'
            )
            logger.info(f"✅ Item shift forecast complete: {len(result)} records")
            return result
        
        return pd.DataFrame()
    
    # ==========================================
    # BACKTEST / EVALUATION
    # ==========================================
    
    @staticmethod
    def backtest_guest_model(df_daily_guests, test_days=None):
        """
        Backtest guest forecast model.
        Dùng N ngày cuối làm test set.
        
        Returns:
            dict: {restaurant_code: {mae, mape, rmse, r2}}
        """
        if test_days is None:
            test_days = ISHUSHI_CONFIG['test_days']
        
        logger.info(f"\n📊 BACKTESTING (last {test_days} days as test)")
        
        df_featured = IshushiForecastModel.prepare_guest_features(df_daily_guests)
        results = {}
        
        for res_code in df_featured['restaurant_code'].unique():
            df_res = df_featured[df_featured['restaurant_code'] == res_code].copy()
            df_res = df_res.sort_values('date_dt')  # type: ignore[reportCallIssue]
            
            if len(df_res) < test_days + 30:
                continue
            
            # Split
            df_train = df_res.iloc[:-test_days]
            df_test = df_res.iloc[-test_days:]
            
            target_col = 'total_guests'
            feature_cols = IshushiForecastModel._get_feature_columns(df_res, target_col)
            
            # Clean NaN
            df_train = df_train.dropna(subset=feature_cols + [target_col])
            df_test_clean = df_test.dropna(subset=feature_cols + [target_col])
            
            if len(df_train) < 14 or len(df_test_clean) < 7:
                continue
            
            # Predictions
            preds = {}
            try:
                p = IshushiForecastModel.train_and_forecast_xgboost(
                    df_train, df_test_clean, target_col, feature_cols
                )
                if p is not None:
                    preds['xgboost'] = p
            except Exception:
                pass
            
            try:
                p = IshushiForecastModel.train_and_forecast_lightgbm(
                    df_train, df_test_clean, target_col, feature_cols
                )
                if p is not None:
                    preds['lightgbm'] = p
            except Exception:
                pass
            
            if not preds:
                continue
            
            ensemble = IshushiForecastModel.ensemble_predict(preds)
            actuals = df_test_clean[target_col].values
            
            # ⭐ v4: Metrics with SMAPE for low-volume restaurants
            mae = np.mean(np.abs(ensemble - actuals))
            rmse = np.sqrt(np.mean((ensemble - actuals) ** 2))
            
            ss_res = np.sum((actuals - ensemble) ** 2)
            ss_tot = np.sum((actuals - actuals.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            avg_daily = actuals.mean()
            low_volume_threshold = ISHUSHI_CONFIG.get('low_volume_threshold', 30)
            
            if avg_daily < low_volume_threshold:
                # ⭐ v4: SMAPE for low-volume (MAPE explodes with small numbers)
                denominator = np.abs(ensemble) + np.abs(actuals)  # type: ignore[reportArgumentType, reportCallIssue]
                smape = np.mean(
                    2 * np.abs(ensemble - actuals) / np.maximum(denominator, 1)
                ) * 100
                mape = smape  # Use SMAPE as primary metric
                metric_type = 'SMAPE'  # Symmetric MAPE
            else:
                mape = np.mean(np.abs((ensemble - actuals) / np.maximum(actuals, 1))) * 100
                metric_type = 'MAPE'
            
            # Weighted MAE (normalize by volume)
            weighted_mae = mae / max(avg_daily, 1)  # Relative MAE
            
            results[res_code] = {
                'mae': round(mae, 2),
                'mape': round(mape, 2),
                'rmse': round(rmse, 2),
                'r2': round(r2, 4),
                'test_days': len(df_test_clean),
                'metric_type': metric_type,
                'avg_daily': round(avg_daily, 1),
                'weighted_mae': round(weighted_mae, 4),
            }
            
            logger.info(
                f"   {res_code}: MAE={mae:.1f}, {metric_type}={mape:.1f}%, "
                f"RMSE={rmse:.1f}, R²={r2:.3f}, avg={avg_daily:.0f}"
            )
        
        return results
    
    # ==========================================
    # SAVE RESULTS
    # ==========================================
    
    @staticmethod
    def save_results(df_guest_forecast, df_item_forecast, backtest_results=None,
                     output_dir=None):
        """
        Lưu kết quả forecast ra Excel.
        
        Output files:
        - Ishushi_Guest_Forecast.xlsx: Dự đoán số khách
        - Ishushi_Item_Forecast.xlsx: Dự đoán số lượng món ăn
        - Ishushi_Backtest_Report.xlsx: Kết quả backtest
        """
        if output_dir is None:
            output_dir = ISHUSHI_CONFIG['output_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Guest Forecast
        if df_guest_forecast is not None and not df_guest_forecast.empty:
            guest_file = os.path.join(output_dir, f'Ishushi_Guest_Forecast.xlsx')
            
            with pd.ExcelWriter(guest_file, engine='openpyxl') as writer:
                # Summary per restaurant
                summary = df_guest_forecast.groupby('restaurant_code').agg(
                    avg_predicted_guests=('predicted_guests', 'mean'),
                    max_predicted_guests=('predicted_guests', 'max'),
                    min_predicted_guests=('predicted_guests', 'min'),
                    forecast_days=('date', 'count'),
                ).reset_index()
                summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Detailed forecast
                df_guest_forecast.to_excel(writer, sheet_name='Daily_Forecast', index=False)
                
                # Pivot: restaurants as rows, dates as columns
                pivot = df_guest_forecast.pivot_table(
                    index='restaurant_code',
                    columns='date',
                    values='predicted_guests',
                    aggfunc='sum'
                )
                pivot.to_excel(writer, sheet_name='Pivot_View')
            
            logger.info(f"💾 Guest forecast saved: {guest_file}")
        
        # 2. Item Forecast
        if df_item_forecast is not None and not df_item_forecast.empty:
            item_file = os.path.join(output_dir, f'Ishushi_Item_Forecast.xlsx')
            
            with pd.ExcelWriter(item_file, engine='openpyxl') as writer:
                # Summary per item group
                group_summary = df_item_forecast.groupby(['item_group', 'sap_code', 'item_name']).agg(
                    avg_daily_quantity=('predicted_quantity', 'mean'),
                    total_quantity=('predicted_quantity', 'sum'),
                    restaurants=('restaurant_code', 'nunique'),
                ).reset_index()
                group_summary.to_excel(writer, sheet_name='Item_Summary', index=False)
                
                # Detailed forecast
                df_item_forecast.to_excel(writer, sheet_name='Daily_Forecast', index=False)
                
                # Pivot by item group
                pivot_group = df_item_forecast.groupby(
                    ['restaurant_code', 'date', 'item_group']
                )['predicted_quantity'].sum().reset_index()
                
                pivot = pivot_group.pivot_table(
                    index=['restaurant_code', 'item_group'],
                    columns='date',
                    values='predicted_quantity',
                    aggfunc='sum'
                )
                pivot.to_excel(writer, sheet_name='Pivot_By_Group')
            
            logger.info(f"💾 Item forecast saved: {item_file}")
        
        # 3. Backtest Report
        if backtest_results:
            bt_file = os.path.join(output_dir, f'Ishushi_Backtest_Report.xlsx')
            
            df_bt = pd.DataFrame.from_dict(backtest_results, orient='index')
            df_bt.index.name = 'restaurant_code'
            df_bt = df_bt.reset_index()
            
            with pd.ExcelWriter(bt_file, engine='openpyxl') as writer:
                df_bt.to_excel(writer, sheet_name='Backtest_Results', index=False)
                
                # Overall summary
                overall = pd.DataFrame([{
                    'metric': 'Average MAE',
                    'value': df_bt['mae'].mean(),
                }, {
                    'metric': 'Average MAPE (%)',
                    'value': df_bt['mape'].mean(),
                }, {
                    'metric': 'Average RMSE',
                    'value': df_bt['rmse'].mean(),
                }, {
                    'metric': 'Average R²',
                    'value': df_bt['r2'].mean(),
                }, {
                    'metric': 'Restaurants Tested',
                    'value': len(df_bt),
                }])
                overall.to_excel(writer, sheet_name='Overall_Summary', index=False)
            
            logger.info(f"💾 Backtest report saved: {bt_file}")
        
        logger.info(f"\n📁 All outputs saved to: {output_dir}")
