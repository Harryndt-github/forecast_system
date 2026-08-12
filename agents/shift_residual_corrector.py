"""
============================================================
⭐ V8 SHIFT RESIDUAL CORRECTOR
============================================================
LightGBM/CatBoost residual correction model focused on 
Weekend × EVENING shift errors.

Target: Actual_Guest - Final_Predicted_Guests (residual)
Apply: ONLY to Weekend×EVENING predictions after ensemble + brain corrections.

Features:
    - restaurant_code_encoded (label)
    - weekday (0-6)
    - shift_id (0=MORNING, 1=EVENING) 
    - is_weekend (0/1)
    - is_weekend_evening (interaction)
    - booking_ratio
    - holiday_impact
    - lag_residual_7d, 14d, 28d
    - rolling_bias_4w
    - volume_segment (0=LOW, 1=MED, 2=HIGH)

Gate condition: Only apply if validation MAE improves ≥5% for Weekend×EVENING.
Correction clamp: ±40% of original forecast.

Usage:
    from forecast_system.agents.shift_residual_corrector import ShiftResidualCorrector
    
    # Training (in _run_learning_and_updates)
    corrector = ShiftResidualCorrector()
    corrector.train(df_master_history)
    
    # Inference (after ensemble + brain)  
    predictions = ShiftResidualCorrector.apply_corrections(predictions, res_code)
"""

import os
import json
import datetime
import pickle
import traceback
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from forecast_system.config.settings import (
    PROJECT_ROOT,
    SHIFT_RESIDUAL_MIN_SAMPLES,
    SHIFT_RESIDUAL_MIN_R2,
    SHIFT_RESIDUAL_MAX_CORRECTION_PCT,
    SHIFT_RESIDUAL_LOOKBACK_DAYS,
    LOW_VOLUME_THRESHOLD,
)
from forecast_system.utils.logger import get_logger

logger = get_logger('shift_residual_corrector')

# Model persistence paths
MODEL_DIR = PROJECT_ROOT / 'models'
MODEL_PATH = MODEL_DIR / 'shift_residual_corrector.pkl'
METADATA_PATH = MODEL_DIR / 'shift_residual_corrector_meta.json'

# Check available ML libraries
HAS_LGBM = False
HAS_CATBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    pass

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    pass


class ShiftResidualCorrector:
    """
    Shift-specific residual correction model.
    
    Learns to predict the error (Actual - Predicted) for each 
    (restaurant, weekday, shift) combination, with special emphasis
    on Weekend × EVENING patterns.
    """
    
    FEATURE_COLUMNS = [
        'restaurant_encoded',
        'weekday',
        'shift_id',
        'is_weekend',
        'is_weekend_evening',   # Key interaction feature
        'volume_segment',
        'lag_residual_7d',
        'lag_residual_14d',
        'lag_residual_28d',
        'rolling_bias_4w',
    ]
    
    VOLUME_MAP = {
        'LOW_VOLUME': 0,
        'MEDIUM_VOLUME': 1,
        'HIGH_VOLUME': 2,
    }
    
    def __init__(self):
        self.model = None
        self.restaurant_encoder = {}  # code → int
        self.metadata = {}
        self.is_trained = False
        self.weekend_evening_improvement = 0.0  # Validation improvement %
    
    def train(
        self,
        df_master: pd.DataFrame,
        lookback_days: int = None,  # type: ignore[reportArgumentType]
    ) -> bool:
        """
        Train the residual corrector from Master_Forecast_Tracking data.
        
        Args:
            df_master: Historical forecast tracking data with columns:
                Restaurant_Code, Date, Weekday, Shift, 
                Final_Predicted_Guests, Actual_Guest
            lookback_days: Days of history to use (default from config)
            
        Returns:
            True if model was trained and saved successfully
        """
        if lookback_days is None:
            lookback_days = SHIFT_RESIDUAL_LOOKBACK_DAYS
        
        if not HAS_LGBM and not HAS_CATBOOST:
            logger.warning("⚠️ ShiftResidualCorrector: No ML library available (need lightgbm or catboost)")
            return False
        
        logger.info("🔧 Training Shift Residual Corrector...")
        
        # --- 1. Prepare data ---
        df = self._prepare_training_data(df_master, lookback_days)
        
        if df is None or len(df) < SHIFT_RESIDUAL_MIN_SAMPLES:
            logger.warning(
                f"⚠️ Not enough samples for residual corrector "
                f"({len(df) if df is not None else 0} < {SHIFT_RESIDUAL_MIN_SAMPLES})"
            )
            return False
        
        # --- 2. Build features ---
        X, y = self._build_features(df)
        
        if X.empty or len(X) < SHIFT_RESIDUAL_MIN_SAMPLES:
            logger.warning("⚠️ Not enough valid features after processing")
            return False
        
        # --- 3. Time-based split (80/20 chronological) ---
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # --- 4. Train model ---
        model = self._train_model(X_train, y_train)
        
        if model is None:
            logger.warning("⚠️ Model training failed")
            return False
        
        # --- 5. Validate: Gate condition ---
        val_preds = model.predict(X_val)
        
        # Overall R²
        ss_res = np.sum((y_val.values - val_preds) ** 2)
        ss_tot = np.sum((y_val.values - np.mean(y_val.values)) ** 2)
        r2 = 1 - (ss_res / max(ss_tot, 1e-10))

        overall_mae_before = np.mean(np.abs(y_val.values))
        overall_mae_after = np.mean(np.abs(y_val.values - val_preds))
        overall_improvement_pct = (
            (overall_mae_before - overall_mae_after) / max(overall_mae_before, 1) * 100
        )
        
        # Weekend × EVENING specific validation
        we_mask = X_val['is_weekend_evening'] == 1
        if we_mask.sum() >= 5:
            mae_before = np.mean(np.abs(y_val[we_mask].values))  # Residual = baseline error
            mae_after = np.mean(np.abs(y_val[we_mask].values - val_preds[we_mask]))
            improvement_pct = (mae_before - mae_after) / max(mae_before, 1) * 100
        else:
            improvement_pct = 0.0
            mae_before = 0.0
            mae_after = 0.0
        
        logger.info(f"   📊 Overall R²: {r2:.3f}")
        logger.info(f"   📊 Weekend×EVENING: MAE {mae_before:.1f} → {mae_after:.1f} "
                    f"(improvement: {improvement_pct:.1f}%)")
        
        # Gate: Must pass minimum R² AND improve Weekend×EVENING
        if r2 < SHIFT_RESIDUAL_MIN_R2:
            logger.info(
                f"   ⛔ Gate FAILED: R² {r2:.3f} < {SHIFT_RESIDUAL_MIN_R2} — model not saved"
            )
            return False
        
        if overall_improvement_pct < 5.0 and (improvement_pct < 5.0 and we_mask.sum() >= 5):
            logger.info(
                f"   ⛔ Gate FAILED: Weekend×EVENING improvement {improvement_pct:.1f}% < 5% — model not saved"
            )
            return False
        
        # --- 6. Save model ---
        self.model = model
        self.is_trained = True
        self.weekend_evening_improvement = improvement_pct
        
        self.metadata = {
            'trained_at': datetime.datetime.now().isoformat(),
            'n_samples': len(X),
            'n_train': len(X_train),
            'n_val': len(X_val),
            'r2': round(r2, 4),
            'overall_improvement_pct': round(overall_improvement_pct, 1),
            'mae_overall_before': round(overall_mae_before, 2),
            'mae_overall_after': round(overall_mae_after, 2),
            'weekend_evening_improvement_pct': round(improvement_pct, 1),
            'mae_we_before': round(mae_before, 2),
            'mae_we_after': round(mae_after, 2),
            'n_restaurants': len(self.restaurant_encoder),
            'features': self.FEATURE_COLUMNS,
        }
        
        self._save_model()
        
        logger.info(
            f"   ✅ Shift Residual Corrector saved! "
            f"(R²={r2:.3f}, WE improvement={improvement_pct:.1f}%)"
        )
        
        return True
    
    def _prepare_training_data(
        self, df_master: pd.DataFrame, lookback_days: int
    ) -> Optional[pd.DataFrame]:
        """Prepare training data from Master tracking file."""
        
        required_cols = [
            'Restaurant_Code', 'Date', 'Weekday', 'Shift',
            'Final_Predicted_Guests', 'Actual_Guest'
        ]
        
        # Check required columns
        missing = [c for c in required_cols if c not in df_master.columns]
        if missing:
            logger.warning(f"Missing columns in Master data: {missing}")
            return None
        
        df = df_master[required_cols].copy()
        
        # Drop NaN in target columns
        df = df.dropna(subset=['Final_Predicted_Guests', 'Actual_Guest', 'Shift'])
        
        # Compute residual (target)
        df['residual'] = df['Actual_Guest'] - df['Final_Predicted_Guests']
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Filter by lookback period
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        df = df[df['Date'] >= cutoff]
        
        # Map shift to shift_id
        shift_map = {'MORNING': 0, 'EVENING': 1}
        df['shift_id'] = df['Shift'].map(shift_map)
        df = df.dropna(subset=['shift_id'])
        df['shift_id'] = df['shift_id'].astype(int)
        
        # Weekday numeric
        df['weekday'] = df['Date'].dt.dayofweek
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['is_weekend_evening'] = ((df['is_weekend'] == 1) & (df['shift_id'] == 1)).astype(int)
        
        # Encode restaurant codes
        unique_codes = df['Restaurant_Code'].unique()
        self.restaurant_encoder = {
            code: idx for idx, code in enumerate(unique_codes)
        }
        df['restaurant_encoded'] = df['Restaurant_Code'].map(self.restaurant_encoder)
        
        # Sort by date for time-series split
        df = df.sort_values(['Restaurant_Code', 'Date', 'shift_id']).reset_index(drop=True)
        
        logger.info(
            f"   Prepared {len(df):,} samples "
            f"({len(unique_codes)} restaurants, "
            f"WE samples: {(df['is_weekend_evening'] == 1).sum()})"
        )
        
        return df
    
    def _build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Build feature matrix and target from prepared data."""
        
        # Volume segment (based on restaurant average)
        res_avg = df.groupby('Restaurant_Code')['Actual_Guest'].mean()
        df['volume_segment'] = df['Restaurant_Code'].map(
            lambda x: (
                0 if res_avg.get(x, 0) < LOW_VOLUME_THRESHOLD
                else (2 if res_avg.get(x, 0) >= 50 else 1)
            )
        )
        
        # Lag residuals per (restaurant, shift)
        for lag_weeks in [1, 2, 4]:  # 7d, 14d, 28d
            col_name = f'lag_residual_{lag_weeks * 7}d'
            df[col_name] = df.groupby(
                ['Restaurant_Code', 'shift_id']
            )['residual'].shift(lag_weeks)
        
        # Rolling bias (4-week mean residual)
        df['rolling_bias_4w'] = df.groupby(
            ['Restaurant_Code', 'shift_id']
        )['residual'].transform(
            lambda x: x.rolling(4, min_periods=2).mean()
        )
        
        # Drop rows with NaN features
        feature_cols = self.FEATURE_COLUMNS
        df = df.dropna(subset=feature_cols)
        
        X = df[feature_cols].copy()
        y = df['residual'].copy()
        
        # Ensure numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0)
        
        return X, y
    
    def _train_model(self, X_train, y_train):
        """Train LightGBM (preferred) or CatBoost model."""
        
        if HAS_LGBM:
            try:
                model = LGBMRegressor(  # type: ignore[reportPossiblyUnboundVariable]
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=10,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    verbose=-1,
                    random_state=42,
                )
                model.fit(X_train, y_train)
                logger.info("   Trained LightGBM residual model")
                return model
            except Exception as e:
                logger.warning(f"LightGBM training failed: {e}")
        
        if HAS_CATBOOST:
            try:
                model = CatBoostRegressor(  # type: ignore[reportPossiblyUnboundVariable]
                    iterations=200,
                    depth=5,
                    learning_rate=0.05,
                    verbose=0,
                    random_state=42,
                )
                model.fit(X_train, y_train)
                logger.info("   Trained CatBoost residual model")
                return model
            except Exception as e:
                logger.warning(f"CatBoost training failed: {e}")
        
        return None
    
    def _save_model(self):
        """Save model and metadata to disk."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'restaurant_encoder': self.restaurant_encoder,
                }, f)
            
            with open(METADATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            
            logger.info(f"   💾 Model saved to {MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Failed to save model: {e}")
    
    @staticmethod
    def load_model() -> Optional['ShiftResidualCorrector']:
        """Load trained model from disk."""
        if not MODEL_PATH.exists():
            return None
        
        try:
            with open(MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
            
            corrector = ShiftResidualCorrector()
            corrector.model = data['model']
            corrector.restaurant_encoder = data['restaurant_encoder']
            corrector.is_trained = True
            
            if METADATA_PATH.exists():
                with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                    corrector.metadata = json.load(f)
            
            return corrector
        except Exception as e:
            logger.warning(f"Failed to load residual corrector model: {e}")
            return None
    
    @staticmethod
    def apply_corrections(
        predictions: List[Dict],
        res_code: str,
    ) -> List[Dict]:
        """
        Apply shift residual corrections to predictions.
        
        ONLY applies to Weekend × EVENING predictions.
        Corrections are clamped to ±SHIFT_RESIDUAL_MAX_CORRECTION_PCT of original forecast.
        
        Args:
            predictions: List of prediction dicts with 'forecast', 'shift', 'weekday' etc.
            res_code: Restaurant code
            
        Returns:
            Modified predictions list
        """
        # Load model (cached after first load)
        corrector = ShiftResidualCorrector.load_model()
        if corrector is None or not corrector.is_trained:
            return predictions
        
        model = corrector.model
        encoder = corrector.restaurant_encoder
        
        # Encode restaurant
        res_encoded = encoder.get(str(res_code))
        if res_encoded is None:
            # Unknown restaurant — skip correction
            return predictions
        
        n_corrected = 0
        
        for p in predictions:
            shift = p.get('shift', '')
            weekday = p.get('weekday', '')
            forecast = p.get('forecast', 0)
            
            # Only correct Weekend × EVENING
            is_weekend = weekday in ('Saturday', 'Sunday')
            is_evening = shift == 'EVENING'
            
            if forecast <= 0:
                continue
            
            try:
                # Build feature vector
                weekday_num = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
                             'Thursday': 3, 'Friday': 4, 'Saturday': 5,
                             'Sunday': 6}.get(weekday, 0)
                shift_id = 1 if is_evening else 0
                volume_segment = 0 if forecast < LOW_VOLUME_THRESHOLD else (2 if forecast >= 100 else 1)
                is_weekend_evening = int(is_weekend and is_evening)
                day_type = 'WEEKEND' if is_weekend else 'WEEKDAY'
                size_group = 'LARGE' if volume_segment == 2 else 'SMALL'
                segment_key = f"{shift}|{day_type}|{size_group}"

                eligible = (
                    segment_key == 'EVENING|WEEKDAY|LARGE' or
                    segment_key == 'MORNING|WEEKDAY|LARGE' or
                    is_weekend_evening
                )
                if not eligible:
                    continue
                
                feat = {
                    'restaurant_encoded': res_encoded,
                    'weekday': weekday_num,
                    'shift_id': shift_id,
                    'is_weekend': int(is_weekend),
                    'is_weekend_evening': is_weekend_evening,
                    'volume_segment': volume_segment,
                    'lag_residual_7d': 0.0,
                    'lag_residual_14d': 0.0,
                    'lag_residual_28d': 0.0,
                    'rolling_bias_4w': 0.0,
                }
                
                X_pred = pd.DataFrame([feat])
                correction = float(model.predict(X_pred)[0])
                
                if segment_key == 'EVENING|WEEKDAY|LARGE':
                    max_pct = 0.30
                elif segment_key == 'MORNING|WEEKDAY|LARGE':
                    max_pct = 0.12
                    if correction > 0:
                        correction *= 0.45
                else:
                    max_pct = min(SHIFT_RESIDUAL_MAX_CORRECTION_PCT, 0.25)
                max_correction = abs(forecast) * max_pct
                correction = max(-max_correction, min(max_correction, correction))
                if abs(correction) < 1.0:
                    continue
                
                corrected = max(0, int(round(forecast + correction)))
                
                if corrected != forecast:
                    p['forecast'] = corrected
                    p['residual_correction'] = round(correction, 1)
                    n_corrected += 1
                    
            except Exception:
                continue
        
        if n_corrected > 0:
            logger.debug(
                f"🔧 ShiftResidualCorrector: {res_code} — "
                f"{n_corrected} Weekend×EVENING predictions corrected"
            )
        
        return predictions
