"""
==============================================
ISHUSHI BRAIN - SELF-LEARNING FROM ERRORS
==============================================
Bộ não tự học cho hệ thống forecast Ishushi.

Core loop:
    1. Forecast → 2. Thực tế → 3. So sánh →
    4. Học bias patterns → 5. Sửa forecast tương lai

Memory structure (ishushi_brain_memory.json):
    {
        "version": 1,
        "last_updated": "2026-04-08",
        "global_patterns": {
            "weekend_bias": 3.2,
            "weekday_biases": {"Monday": -1.5, ...},
        },
        "restaurants": {
            "R001": {
                "overall_bias": +3.2,
                "correction_factor": 0.95,
                "weekday_bias": {"Monday": -2.1, ...},
                "mape_history": [45, 38, 32],
                "last_mape": 32.0,
            }
        },
        "items": {
            "R001": {
                "60011044": {
                    "overall_bias": +1.5,
                    "correction_factor": 0.97,
                    "mape_history": [20, 18],
                }
            }
        },
        "learning_log": [...]
    }
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

from forecast_system.utils.logger import get_logger
from forecast_system.ishushi.config import ISHUSHI_CONFIG

logger = get_logger('ishushi_brain')

# Brain file path
BRAIN_FILE = os.path.join(ISHUSHI_CONFIG['output_dir'], 'ishushi_brain_memory.json')


class IshushiBrain:
    """
    Bộ não tự học cho Ishushi forecast.
    
    Workflow:
    1. LEARN: brain.learn_from_errors(df_guest_master, df_item_master)
       → Phân tích errors, ghi nhớ bias patterns
       
    2. CORRECT: corrected = brain.apply_guest_corrections(predictions, res_code)
       → Trừ bias, điều chỉnh theo weekday patterns
       
    3. REPORT: brain.print_insights()
       → Xem brain đã học được gì
    """
    
    # Thresholds
    SIGNIFICANT_BIAS = 2.0         # Guests/items: bias > 2 mới sửa
    MAX_CORRECTION_PCT = 0.30      # Max ±30% correction
    MIN_SAMPLES_LEARN = 5          # Tối thiểu 5 samples để học
    BIAS_SMOOTHING = 0.3           # Exponential smoothing α
    
    # ==========================================
    # MEMORY MANAGEMENT
    # ==========================================
    
    @staticmethod
    def load_memory() -> Dict:
        """Load brain memory từ JSON file."""
        if os.path.exists(BRAIN_FILE):
            try:
                with open(BRAIN_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load brain memory: {e}")
        
        return IshushiBrain._create_empty_memory()
    
    @staticmethod
    def save_memory(memory: Dict):
        """Save brain memory to JSON file."""
        memory['last_updated'] = str(datetime.date.today())
        memory['version'] = 1
        
        os.makedirs(os.path.dirname(BRAIN_FILE), exist_ok=True)
        
        try:
            # Atomic write: tmp file + os.replace so a crash mid-write
            # cannot corrupt/erase the learned memory.
            tmp_file = BRAIN_FILE + '.tmp'
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(memory, f, indent=2, default=str)
            os.replace(tmp_file, BRAIN_FILE)
            n_restaurants = len(memory.get('restaurants', {}))
            n_items = sum(
                len(v) for v in memory.get('items', {}).values()
            )
            logger.info(
                f"🧠 Brain saved: {n_restaurants} restaurants, "
                f"{n_items} item series"
            )
        except Exception as e:
            logger.error(f"Failed to save brain: {e}")
    
    @staticmethod
    def _create_empty_memory() -> Dict:
        return {
            'version': 1,
            'last_updated': str(datetime.date.today()),
            'global_patterns': {
                'weekend_bias': 0,
                'weekday_biases': {},
                'overall_mape_trend': [],
            },
            'restaurants': {},
            'items': {},
            'learning_log': [],
        }
    
    @staticmethod
    def _get_restaurant_mem(memory: Dict, res_code: str) -> Dict:
        """Get or create restaurant-specific memory."""
        if res_code not in memory['restaurants']:
            memory['restaurants'][res_code] = {
                'overall_bias': 0.0,
                'correction_factor': 1.0,
                'weekday_bias': {},
                'mape_history': [],
                'last_mape': None,
                'learned_at': None,
            }
        return memory['restaurants'][res_code]
    
    @staticmethod
    def _get_item_mem(memory: Dict, res_code: str, sap_code: str) -> Dict:
        """Get or create item-specific memory."""
        if res_code not in memory['items']:
            memory['items'][res_code] = {}
        
        if sap_code not in memory['items'][res_code]:
            memory['items'][res_code][sap_code] = {
                'overall_bias': 0.0,
                'correction_factor': 1.0,
                'mape_history': [],
                'last_mape': None,
            }
        return memory['items'][res_code][sap_code]
    
    # ==========================================
    # 1. LEARN FROM GUEST ERRORS
    # ==========================================
    
    @staticmethod
    def learn_from_guest_errors(df_master: pd.DataFrame) -> Dict:
        """
        Phân tích guest forecast errors và ghi nhớ patterns.
        
        Học:
        - Per-restaurant bias (over/under predict)
        - Weekday-specific bias
        - Correction factor
        
        Args:
            df_master: Guest master DataFrame (phải có Predicted_Guests, Actual_Guests)
        
        Returns:
            Dict: learning summary
        """
        memory = IshushiBrain.load_memory()
        
        if df_master.empty:
            return {'status': 'no_data'}
        
        # Filter rows with both predicted and actual
        mask = (
            pd.notna(df_master['Predicted_Guests']) &
            pd.notna(df_master['Actual_Guests']) &
            (df_master['Predicted_Guests'] >= 0) &
            (df_master['Actual_Guests'] >= 0)
        )
        df = df_master[mask].copy()
        
        if len(df) < IshushiBrain.MIN_SAMPLES_LEARN:
            return {'status': 'insufficient_data', 'samples': len(df)}
        
        # Calculate errors
        df['error'] = df['Predicted_Guests'] - df['Actual_Guests']
        df['abs_error'] = df['error'].abs()  # type: ignore[reportAttributeAccessIssue]
        df['pct_error'] = np.where(
            df['Actual_Guests'] > 0,
            (df['abs_error'] / df['Actual_Guests']) * 100,
            np.nan
        )
        
        summary = {
            'status': 'learned',
            'total_samples': len(df),
            'restaurants_learned': 0,
            'issues_found': 0,
        }
        
        # --- Learn Global Patterns ---
        IshushiBrain._learn_global_guest_patterns(memory, df)  # type: ignore[reportArgumentType]
        
        # --- Learn Per-Restaurant ---
        for res_code, df_res in df.groupby('Restaurant_Code'):
            if len(df_res) < IshushiBrain.MIN_SAMPLES_LEARN:
                continue
            
            res_mem = IshushiBrain._get_restaurant_mem(memory, str(res_code))
            
            # Overall bias (exponential smoothing)
            new_bias = float(df_res['error'].mean())
            old_bias = res_mem.get('overall_bias', 0)
            alpha = IshushiBrain.BIAS_SMOOTHING
            res_mem['overall_bias'] = round(
                alpha * new_bias + (1 - alpha) * old_bias, 2
            )
            
            # Weekday bias
            if 'Weekday' in df_res.columns:
                old_wd = res_mem.get('weekday_bias', {})
                new_wd = {}
                for wd, grp in df_res.groupby('Weekday'):
                    if len(grp) < 3:
                        continue
                    new_val = float(grp['error'].mean())
                    old_val = old_wd.get(str(wd), 0)
                    new_wd[str(wd)] = round(
                        alpha * new_val + (1 - alpha) * old_val, 2
                    )
                # Keep old weekdays not in new data
                for wd, val in old_wd.items():
                    if wd not in new_wd:
                        new_wd[wd] = val
                res_mem['weekday_bias'] = new_wd
            
            # Correction factor (actual/predicted ratio)
            valid_pairs = df_res[
                (df_res['Predicted_Guests'] > 0) &
                (df_res['Actual_Guests'] > 0)
            ]
            if len(valid_pairs) >= 3:
                ratio = (
                    valid_pairs['Actual_Guests'].sum() /
                    valid_pairs['Predicted_Guests'].sum()
                )
                old_cf = res_mem.get('correction_factor', 1.0)
                new_cf = alpha * float(ratio) + (1 - alpha) * old_cf
                # Clamp to [0.7, 1.3]
                new_cf = max(0.7, min(1.3, new_cf))
                res_mem['correction_factor'] = round(new_cf, 4)
            
            # MAPE history
            mape = df_res['pct_error'].dropna().mean()  # type: ignore[reportAttributeAccessIssue]
            if pd.notna(mape):  # type: ignore[reportGeneralTypeIssues]
                res_mem['last_mape'] = round(mape, 1)
                hist = res_mem.get('mape_history', [])
                hist.append(round(mape, 1))
                res_mem['mape_history'] = hist[-30:]
            
            # Detect issues (high error days)
            high_error_days = df_res[df_res['pct_error'] > 50]
            summary['issues_found'] += len(high_error_days)
            
            res_mem['learned_at'] = datetime.datetime.now().isoformat()
            summary['restaurants_learned'] += 1
        
        # Save learning log
        memory['learning_log'].append({
            'date': str(datetime.date.today()),
            'type': 'guest_learning',
            'restaurants': summary['restaurants_learned'],
            'samples': summary['total_samples'],
            'issues': summary['issues_found'],
        })
        memory['learning_log'] = memory['learning_log'][-60:]
        
        IshushiBrain.save_memory(memory)
        
        logger.info(
            f"🧠 Guest learning: {summary['restaurants_learned']} restaurants, "
            f"{summary['total_samples']} samples, "
            f"{summary['issues_found']} issues"
        )
        
        return summary
    
    @staticmethod
    def _learn_global_guest_patterns(memory: Dict, df: pd.DataFrame):
        """Học bias patterns chung toàn hệ thống."""
        gp = memory['global_patterns']
        
        if 'Weekday' in df.columns:
            # Weekend bias
            weekend = df[df['Weekday'].isin(['Saturday', 'Sunday'])]
            if len(weekend) > 0:
                gp['weekend_bias'] = round(float(weekend['error'].mean()), 2)
            
            # Per-weekday bias
            wd_bias = df.groupby('Weekday')['error'].mean()
            gp['weekday_biases'] = {
                str(k): round(float(v), 2) for k, v in wd_bias.items()
            }
        
        # MAPE trend
        overall_mape = df['pct_error'].dropna().mean()
        if pd.notna(overall_mape):  # type: ignore[reportGeneralTypeIssues]
            trend = gp.get('overall_mape_trend', [])
            trend.append({
                'date': str(datetime.date.today()),
                'mape': round(float(overall_mape), 1),
            })
            gp['overall_mape_trend'] = trend[-30:]
    
    # ==========================================
    # 2. LEARN FROM ITEM ERRORS
    # ==========================================
    
    @staticmethod
    def learn_from_item_errors(df_master: pd.DataFrame) -> Dict:
        """
        Phân tích item forecast errors.
        Học bias per (restaurant, sap_code).
        
        Args:
            df_master: Item master DataFrame
        
        Returns:
            Dict: learning summary
        """
        memory = IshushiBrain.load_memory()
        
        if df_master.empty:
            return {'status': 'no_data'}
        
        mask = (
            pd.notna(df_master['Predicted_Quantity']) &
            pd.notna(df_master['Actual_Quantity']) &
            (df_master['Predicted_Quantity'] >= 0) &
            (df_master['Actual_Quantity'] >= 0)
        )
        df = df_master[mask].copy()
        
        if len(df) < IshushiBrain.MIN_SAMPLES_LEARN:
            return {'status': 'insufficient_data', 'samples': len(df)}
        
        df['error'] = df['Predicted_Quantity'] - df['Actual_Quantity']
        df['abs_error'] = df['error'].abs()  # type: ignore[reportAttributeAccessIssue]
        df['pct_error'] = np.where(
            df['Actual_Quantity'] > 0,
            (df['abs_error'] / df['Actual_Quantity']) * 100,
            np.nan
        )
        
        summary = {
            'status': 'learned',
            'total_samples': len(df),
            'items_learned': 0,
        }
        
        alpha = IshushiBrain.BIAS_SMOOTHING
        
        for (res_code, sap_code), grp in df.groupby(['Restaurant_Code', 'SAP_Code']):  # type: ignore[reportGeneralTypeIssues]
            if len(grp) < 3:
                continue
            
            item_mem = IshushiBrain._get_item_mem(
                memory, str(res_code), str(sap_code)
            )
            
            # Bias
            new_bias = float(grp['error'].mean())
            old_bias = item_mem.get('overall_bias', 0)
            item_mem['overall_bias'] = round(
                alpha * new_bias + (1 - alpha) * old_bias, 2
            )
            
            # Correction factor
            valid = grp[(grp['Predicted_Quantity'] > 0) & (grp['Actual_Quantity'] > 0)]
            if len(valid) >= 3:
                ratio = valid['Actual_Quantity'].sum() / valid['Predicted_Quantity'].sum()
                old_cf = item_mem.get('correction_factor', 1.0)
                new_cf = alpha * float(ratio) + (1 - alpha) * old_cf
                new_cf = max(0.5, min(1.5, new_cf))
                item_mem['correction_factor'] = round(new_cf, 4)
            
            # MAPE
            mape = grp['pct_error'].dropna().mean()  # type: ignore[reportAttributeAccessIssue]
            if pd.notna(mape):  # type: ignore[reportGeneralTypeIssues]
                item_mem['last_mape'] = round(mape, 1)
                hist = item_mem.get('mape_history', [])
                hist.append(round(mape, 1))
                item_mem['mape_history'] = hist[-20:]
            
            summary['items_learned'] += 1
        
        # Log
        memory['learning_log'].append({
            'date': str(datetime.date.today()),
            'type': 'item_learning',
            'items': summary['items_learned'],
            'samples': summary['total_samples'],
        })
        memory['learning_log'] = memory['learning_log'][-60:]
        
        IshushiBrain.save_memory(memory)
        
        logger.info(
            f"🧠 Item learning: {summary['items_learned']} item series, "
            f"{summary['total_samples']} samples"
        )
        
        return summary
    
    # ==========================================
    # 3. APPLY CORRECTIONS
    # ==========================================
    
    @staticmethod
    def apply_guest_corrections(
        df_forecast: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Áp dụng brain corrections vào guest forecast.
        
        Logic:
        1. Lấy correction_factor per restaurant
        2. Trừ weekday bias
        3. Clamp correction trong ±30%
        
        Args:
            df_forecast: Guest forecast DataFrame 
                         (restaurant_code, date, predicted_guests, weekday)
        
        Returns:
            DataFrame with corrected predictions
        """
        memory = IshushiBrain.load_memory()
        restaurants = memory.get('restaurants', {})
        
        if not restaurants or df_forecast.empty:
            return df_forecast
        
        df = df_forecast.copy()
        original_total = df['predicted_guests'].sum()
        
        corrections_applied = 0
        
        for idx, row in df.iterrows():
            res_code = str(row['restaurant_code'])
            res_mem = restaurants.get(res_code)
            
            if not res_mem:
                continue
            
            original = float(row['predicted_guests'])
            if original <= 0:
                continue
            
            # 1. Apply correction factor
            cf = res_mem.get('correction_factor', 1.0)
            corrected = original * cf
            
            # 2. Subtract weekday bias
            weekday = str(row.get('weekday', ''))
            wd_bias = res_mem.get('weekday_bias', {}).get(weekday, 0)
            
            if abs(wd_bias) > IshushiBrain.SIGNIFICANT_BIAS:
                corrected -= wd_bias * 0.5  # Apply 50% of bias
            
            # 3. Clamp correction within ±30%
            max_val = original * (1 + IshushiBrain.MAX_CORRECTION_PCT)
            min_val = original * (1 - IshushiBrain.MAX_CORRECTION_PCT)
            corrected = max(min_val, min(max_val, corrected))
            corrected = max(0, round(corrected))
            
            if int(corrected) != int(original):
                corrections_applied += 1
            
            df.at[idx, 'predicted_guests'] = int(corrected)
        
        if corrections_applied > 0:
            new_total = df['predicted_guests'].sum()
            logger.info(
                f"🧠 Guest corrections: {corrections_applied}/{len(df)} adjusted, "
                f"total {original_total:.0f} → {new_total:.0f}"
            )
        
        return df
    
    @staticmethod
    def apply_item_corrections(
        df_forecast: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Áp dụng brain corrections vào item forecast.
        
        Args:
            df_forecast: Item forecast DataFrame
        
        Returns:
            Corrected DataFrame
        """
        memory = IshushiBrain.load_memory()
        items_mem = memory.get('items', {})
        
        if not items_mem or df_forecast.empty:
            return df_forecast
        
        df = df_forecast.copy()
        corrections_applied = 0
        
        for idx, row in df.iterrows():
            res_code = str(row['restaurant_code'])
            sap_code = str(row.get('sap_code', ''))
            
            item_mem = items_mem.get(res_code, {}).get(sap_code)
            if not item_mem:
                continue
            
            original = float(row['predicted_quantity'])
            if original <= 0:
                continue
            
            # Apply correction factor
            cf = item_mem.get('correction_factor', 1.0)
            corrected = original * cf
            
            # Clamp
            max_val = original * (1 + IshushiBrain.MAX_CORRECTION_PCT)
            min_val = original * (1 - IshushiBrain.MAX_CORRECTION_PCT)
            corrected = max(min_val, min(max_val, corrected))
            corrected = max(0, round(corrected))
            
            if int(corrected) != int(original):
                corrections_applied += 1
            
            df.at[idx, 'predicted_quantity'] = int(corrected)
        
        if corrections_applied > 0:
            logger.info(
                f"🧠 Item corrections: {corrections_applied}/{len(df)} adjusted"
            )
        
        return df
    
    # ==========================================
    # 4. INSIGHTS & REPORTING
    # ==========================================
    
    @staticmethod
    def print_insights(logger_func=None):
        """Print brain insights summary."""
        if logger_func is None:
            logger_func = logger.info
        
        memory = IshushiBrain.load_memory()
        restaurants = memory.get('restaurants', {})
        items_mem = memory.get('items', {})
        gp = memory.get('global_patterns', {})
        
        logger_func("\n" + "=" * 55)
        logger_func("🧠 ISHUSHI BRAIN INSIGHTS")
        logger_func("=" * 55)
        logger_func(f"  Last updated: {memory.get('last_updated', 'Never')}")
        logger_func(f"  Restaurants tracked: {len(restaurants)}")
        logger_func(f"  Item series tracked: {sum(len(v) for v in items_mem.values())}")
        
        # Global patterns
        logger_func(f"\n  📊 Global Patterns:")
        logger_func(f"     Weekend bias: {gp.get('weekend_bias', 0):+.1f} guests")
        
        wd_biases = gp.get('weekday_biases', {})
        if wd_biases:
            logger_func("     Weekday biases:")
            for wd in ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                       'Friday', 'Saturday', 'Sunday']:
                bias = wd_biases.get(wd, 0)
                if abs(bias) > 0.5:
                    direction = "▲ over" if bias > 0 else "▼ under"
                    logger_func(f"       {wd:>10}: {bias:+.1f} ({direction})")
        
        # MAPE trend
        trend = gp.get('overall_mape_trend', [])
        if len(trend) >= 2:
            first_mape = trend[0].get('mape', 0)
            last_mape = trend[-1].get('mape', 0)
            direction = "📉 improving" if last_mape < first_mape else "📈 worsening"
            logger_func(f"\n  📈 MAPE Trend: {first_mape}% → {last_mape}% ({direction})")
        
        # Per-restaurant summary
        if restaurants:
            logger_func(f"\n  🏪 Restaurant Corrections:")
            for rc, mem in sorted(restaurants.items()):
                cf = mem.get('correction_factor', 1.0)
                bias = mem.get('overall_bias', 0)
                mape = mem.get('last_mape', None)
                
                cf_label = f"CF={cf:.3f}"
                bias_label = f"bias={bias:+.1f}"
                mape_label = f"MAPE={mape:.1f}%" if mape else "no MAPE"
                
                logger_func(f"     {rc}: {cf_label}, {bias_label}, {mape_label}")
        
        logger_func("=" * 55)
    
    @staticmethod
    def get_correction_summary() -> pd.DataFrame:
        """
        Get correction summary as DataFrame.
        Useful for dashboard display.
        
        Returns:
            DataFrame: Restaurant_Code, Correction_Factor, Overall_Bias,
                      Last_MAPE, MAPE_Trend
        """
        memory = IshushiBrain.load_memory()
        restaurants = memory.get('restaurants', {})
        
        if not restaurants:
            return pd.DataFrame()
        
        rows = []
        for rc, mem in restaurants.items():
            mape_hist = mem.get('mape_history', [])
            
            # Determine trend
            if len(mape_hist) >= 3:
                recent = np.mean(mape_hist[-3:])
                older = np.mean(mape_hist[:3]) if len(mape_hist) >= 6 else np.mean(mape_hist[:len(mape_hist)//2+1])
                trend = '📉 Improving' if recent < older * 0.95 else '📈 Worsening' if recent > older * 1.05 else '➡️ Stable'
            else:
                trend = 'N/A'
            
            rows.append({
                'Restaurant_Code': rc,
                'Correction_Factor': mem.get('correction_factor', 1.0),
                'Overall_Bias': mem.get('overall_bias', 0),
                'Last_MAPE': mem.get('last_mape'),
                'MAPE_Trend': trend,
                'Learning_Cycles': len(mape_hist),
                'Last_Learned': mem.get('learned_at', 'Never'),
            })
        
        return pd.DataFrame(rows)
