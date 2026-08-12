"""
==============================================
NEURAL CORRECTOR (Replace Rule-Based Brain)
==============================================
Thay thế logic rule-based trong ForecastBrain bằng
lightweight MLP neural network.

Addresses Limitation #3: Brain correction là rule-based.

Architecture:
    Input (11 features) → Dense(32) → ReLU → Dense(16) → ReLU → Dense(1)
    
    Features:
    - raw_prediction (normalized)
    - weekday (0-6)
    - hour (8-23, or None for shift-based mode)
    - is_holiday, is_weekend, is_pre_holiday, is_post_holiday
    - holiday_impact
    - restaurant_category_encoded
    - overall_bias (from brain)
    - correction_factor (from brain)
    
    Target: actual - predicted (error to correct)
    
Training data: historical (predicted, actual) pairs from master file.
"""

import json
import pickle
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from forecast_system.config.settings import PROJECT_ROOT
from forecast_system.utils.logger import get_logger

logger = get_logger('neural_corrector')

# Check sklearn availability
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN_NN = True
except ImportError:
    HAS_SKLEARN_NN = False


class NeuralCorrector:
    """
    Lightweight neural network for prediction correction.
    Learns non-linear correction patterns from historical data.
    """

    MODEL_FILE = PROJECT_ROOT / 'neural_corrector.pkl'
    MIN_TRAINING_SAMPLES = 100
    
    CATEGORY_MAP = {
        'STANDARD': 0, 'HIGH_VOLUME': 1, 'VOLATILE': 2,
        'NEW': 3, 'YOUNG': 4, 'UNKNOWN': 5
    }
    
    WEEKDAY_MAP = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6,
    }

    @staticmethod
    def train(df_master: pd.DataFrame, brain_memory: Dict = None):  # type: ignore[reportArgumentType]
        """
        Train neural corrector from historical prediction vs actual data.
        
        Args:
            df_master: Master file with columns:
                       Final_Predicted_Guests, Actual_Guest, Date, Hour, Weekday,
                       Restaurant_Code
            brain_memory: brain_memory.json contents
        """
        if not HAS_SKLEARN_NN:
            logger.warning("sklearn not available, skipping Neural Corrector")
            return None
        
        logger.info("🧠 Training Neural Corrector...")
        
        # Prepare training data
        df = df_master.copy()
        
        # Need both predicted and actual
        required = ['Final_Predicted_Guests', 'Actual_Guest']
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}")
                return None
        
        df = df.dropna(subset=required)
        df = df[(df['Final_Predicted_Guests'] > 0) & (df['Actual_Guest'] >= 0)]
        
        # [FIX #4] Only use last 90 days to avoid stale patterns
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date  # type: ignore[reportAttributeAccessIssue]
            import datetime as _dt
            cutoff_90d = _dt.date.today() - _dt.timedelta(days=90)
            df_recent = df[df['Date'] >= cutoff_90d]
            if len(df_recent) >= NeuralCorrector.MIN_TRAINING_SAMPLES:
                df = df_recent
                logger.info(f"   Using last 90 days: {len(df):,} samples (filtered from full set)")
            else:
                logger.info(f"   Using full dataset ({len(df):,} samples, 90d window too small)")
        
        if len(df) < NeuralCorrector.MIN_TRAINING_SAMPLES:
            logger.info(f"Not enough data ({len(df)} rows), need {NeuralCorrector.MIN_TRAINING_SAMPLES}")
            return None
        
        # Build features
        X, y = NeuralCorrector._build_features(df, brain_memory)
        
        if X is None or len(X) < NeuralCorrector.MIN_TRAINING_SAMPLES:
            return None
        
        # Scale features
        scaler = StandardScaler()  # type: ignore[reportPossiblyUnboundVariable]
        X_scaled = scaler.fit_transform(X)
        
        # [FIX #5] Use TIME-BASED split (80% oldest / 20% newest) instead of random
        # Prevents data leakage: future data leaking into training set
        split_idx = int(len(X_scaled) * 0.80)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train MLP — wider network for 128k+ rows
        mlp = MLPRegressor(  # type: ignore[reportPossiblyUnboundVariable]
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            max_iter=800,
            early_stopping=True,
            validation_fraction=0.15,
            learning_rate='adaptive',
            random_state=42,
            verbose=False,
        )
        
        mlp.fit(X_train, y_train)
        
        # Evaluate
        train_score = mlp.score(X_train, y_train)
        test_score = mlp.score(X_test, y_test)
        
        # Also compute MAE for interpretability
        from sklearn.metrics import mean_absolute_error
        y_pred_test = mlp.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        logger.info(f"   Neural Corrector R²: Train={train_score:.3f}, Test={test_score:.3f}, Test MAE={test_mae:.2f}")
        
        # [FIX #2] Save only if R² >= 0.10 (was > 0.0 — too permissive)
        # R² < 0.10 means model explains <10% variance → adds noise, not value
        MIN_R2_THRESHOLD = 0.10
        if test_score >= MIN_R2_THRESHOLD:
            model_data = {
                'model': mlp,
                'scaler': scaler,
                'train_score': train_score,
                'test_score': test_score,
                'test_mae': test_mae,
                'n_samples': len(X),
                'created_at': datetime.datetime.now().isoformat(),
            }
            
            with open(NeuralCorrector.MODEL_FILE, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"   ✅ Neural Corrector saved ({len(X)} samples, R²={test_score:.3f}, MAE={test_mae:.2f})")
            return model_data
        else:
            logger.info(f"   ⚠️ Neural Corrector R² too low ({test_score:.3f} < {MIN_R2_THRESHOLD}) — model not saved to avoid adding noise")
            # Remove stale model file if it exists (prevents using an old bad model)
            if NeuralCorrector.MODEL_FILE.exists():
                NeuralCorrector.MODEL_FILE.unlink()
                logger.info(f"   🗑️ Removed stale neural corrector model")
            return None

    @staticmethod
    def _build_features(df, brain_memory=None):
        """
        Build feature matrix for neural corrector.
        Uses vectorized pandas operations instead of iterrows() for 128k+ rows.
        """
        if brain_memory is None:
            try:
                brain_file = PROJECT_ROOT / 'brain_memory.json'
                if brain_file.exists():
                    with open(brain_file, 'r', encoding='utf-8') as f:
                        brain_memory = json.load(f)
            except Exception:
                brain_memory = {}
        
        restaurants_mem = brain_memory.get('restaurants', {})  # type: ignore[reportOptionalMemberAccess]
        
        try:
            # Vectorized feature engineering
            predicted = df['Final_Predicted_Guests'].values.astype(float)
            actual = df['Actual_Guest'].values.astype(float)
            
            # Weekday: map string names to numbers
            # Use is_string_dtype to catch both 'object' and pandas StringDtype
            weekday_col = df['Weekday']
            if pd.api.types.is_string_dtype(weekday_col) or weekday_col.dtype == object:
                weekday_num = weekday_col.map(NeuralCorrector.WEEKDAY_MAP).fillna(0).astype(int).values
            elif weekday_col.dropna().astype(str).str.isalpha().any():
                # Fallback: if values look like strings even with non-string dtype
                weekday_num = weekday_col.astype(str).map(NeuralCorrector.WEEKDAY_MAP).fillna(0).astype(int).values
            else:
                weekday_num = weekday_col.fillna(0).astype(int).values
            
            hour = df['Hour'].fillna(12).astype(int).values if 'Hour' in df.columns else np.full(len(df), 12)
            
            is_holiday = np.where(
                df['Is_Holiday'].fillna(False) if 'Is_Holiday' in df.columns else pd.Series(False, index=df.index),
                1, 0
            )
            is_weekend = np.where(weekday_num >= 5, 1, 0)
            
            holiday_impact = (
                df['Holiday_Impact'].fillna(1.0).values.astype(float)
                if 'Holiday_Impact' in df.columns
                else np.ones(len(df))
            )
            
            # Per-restaurant brain features (vectorized lookup)
            res_codes = df['Restaurant_Code'].astype(str).values
            overall_bias = np.array([restaurants_mem.get(rc, {}).get('overall_bias', 0) for rc in res_codes], dtype=float)
            correction_factor = np.array([restaurants_mem.get(rc, {}).get('correction_factor', 1.0) for rc in res_codes], dtype=float)
            last_mape = np.array([restaurants_mem.get(rc, {}).get('last_mape', 0) or 0 for rc in res_codes], dtype=float)
            category = np.array([
                NeuralCorrector.CATEGORY_MAP.get(restaurants_mem.get(rc, {}).get('category', 'STANDARD'), 0)
                for rc in res_codes
            ], dtype=float)
            avg_daily = np.array([
                max(restaurants_mem.get(rc, {}).get('avg_daily', p), 1)
                for rc, p in zip(res_codes, predicted)
            ], dtype=float)
            relative_pred = predicted / avg_daily
            
            # Stack features
            X = np.column_stack([
                predicted,
                weekday_num,
                hour,
                is_holiday,
                is_weekend,
                holiday_impact,
                overall_bias,
                correction_factor,
                last_mape,
                category,
                relative_pred,
            ])
            
            y = actual - predicted  # Error to learn
            
            # Remove NaN rows
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            logger.info(f"   Neural features built: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.warning(f"Feature building failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    @staticmethod
    def apply_corrections(predictions: List[Dict], res_code: str) -> List[Dict]:
        """
        Apply neural corrections to predictions.
        Falls back gracefully if model not available.
        """
        if not HAS_SKLEARN_NN or not NeuralCorrector.MODEL_FILE.exists():
            return predictions
        
        try:
            with open(NeuralCorrector.MODEL_FILE, 'rb') as f:
                model_data = pickle.load(f)
            
            mlp = model_data['model']
            scaler = model_data['scaler']
            
            # Load brain memory for this restaurant
            brain_memory = {}
            brain_file = PROJECT_ROOT / 'brain_memory.json'
            if brain_file.exists():
                with open(brain_file, 'r', encoding='utf-8') as f:
                    brain_memory = json.load(f)
            
            res_mem = brain_memory.get('restaurants', {}).get(str(res_code), {})
            
            corrected = []
            n_corrected = 0
            
            for pred in predictions:
                p = pred.copy()
                forecast = float(p.get('forecast', 0))
                
                if forecast <= 0:
                    corrected.append(p)
                    continue
                
                # Build feature vector
                feat = np.array([[
                    forecast,
                    int(p.get('weekday_num', 0)),
                    int(p.get('hour', 12)),
                    1 if p.get('is_holiday', False) else 0,
                    1 if int(p.get('weekday_num', 0)) >= 5 else 0,
                    float(p.get('holiday_impact', 1.0)),
                    res_mem.get('overall_bias', 0),
                    res_mem.get('correction_factor', 1.0),
                    res_mem.get('last_mape', 0),
                    NeuralCorrector.CATEGORY_MAP.get(
                        res_mem.get('category', 'STANDARD'), 0
                    ),
                    forecast / max(res_mem.get('avg_daily', forecast), 1),
                ]])
                
                feat_scaled = scaler.transform(feat)
                correction = mlp.predict(feat_scaled)[0]
                
                # Clamp correction to ±60% of original (match Brain's MAX_CORRECTION)
                max_corr = forecast * 0.6
                correction = max(-max_corr, min(max_corr, correction))
                
                adjusted = max(0, int(round(forecast + correction)))
                
                if adjusted != int(forecast):
                    n_corrected += 1
                
                p['forecast'] = adjusted
                p['neural_correction'] = round(correction, 1)
                corrected.append(p)
            
            if n_corrected > 0:
                logger.debug(
                    f"🧠 Neural: {res_code} corrected {n_corrected}/{len(predictions)}"
                )
            
            return corrected
            
        except Exception as e:
            logger.debug(f"Neural correction failed for {res_code}: {e}")
            return predictions
