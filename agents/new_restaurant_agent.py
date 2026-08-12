"""
==============================================
NEW RESTAURANT AGENT
==============================================
Forecast cho nhà hàng MỚI (< 14 ngày data) bằng cách:
1. Tìm nhà hàng cùng chuỗi (chain/brand)
2. Tính pattern trung bình từ chuỗi (daily volume, hourly ratio, weekday)
3. Blend chain-average với data riêng (weighted by active days)

Ví dụ: Nhà hàng Gogi House mới mở 9 ngày 
→ dùng data từ 143 Gogi House khác để hỗ trợ forecast
"""

import pandas as pd
import numpy as np
import datetime
import re
import unicodedata
from typing import Dict, List, Tuple, Optional, Set, Union

from forecast_system.config.settings import CURRENT_DATE, ANALYSIS_CONFIG
from forecast_system.utils.logger import get_logger

logger = get_logger('new_restaurant')


# ==========================================
# NAME NORMALIZATION
# ==========================================

def _normalize_name(name: str) -> str:
    """
    Chuẩn hóa tên nhà hàng để so sánh:
    1. Bỏ dấu tiếng Việt (Phở → Pho, Lách → Lach)
    2. Bỏ dấu gạch ngang (Kichi-Kichi → Kichi Kichi)
    3. Tách các từ viết liền (ThaiExpress → thai express)
    4. Chuẩn hóa khoảng trắng
    5. Chuyển lowercase
    """
    if not name:
        return ''
    
    # Step 1: Strip Vietnamese diacritics
    # NFD decomposes characters, then remove combining marks
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_name = ''.join(
        c for c in nfkd if not unicodedata.combining(c)
    )
    # Handle special Vietnamese chars that NFD doesn't fully decompose
    ascii_name = ascii_name.replace('đ', 'd').replace('Đ', 'D')
    
    # Step 2: Replace hyphens with spaces
    ascii_name = ascii_name.replace('-', ' ')
    
    # Step 3: Insert space before uppercase letters in camelCase
    # "ThaiExpress" → "Thai Express", "KingBBQ" → "King BBQ"
    ascii_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', ascii_name)
    
    # Step 4: Normalize whitespace and lowercase
    ascii_name = re.sub(r'\s+', ' ', ascii_name).strip().lower()
    
    return ascii_name


# ==========================================
# KNOWN CHAIN/BRAND MAPPINGS
# ==========================================
# Mỗi chuỗi chỉ có 1 tên chuẩn (canonical name).
# detect_chain() sẽ normalize cả input và chain name trước khi so sánh,
# nên tự động match các biến thể:
#   "Kichi-Kichi" ↔ "Kichi Kichi"
#   "ThaiExpress"  ↔ "Thai Express"
#   "KingBBQ"      ↔ "King BBQ"
#   "SumoBBQ"      ↔ "Sumo BBQ"
#   "Lách Ca"      ↔ "Lac Ca"
#   "Phở Inn"      ↔ "Pho Inn"
#
# Ordered by specificity (longer names first)
KNOWN_CHAINS = [
    'GoGi Steak',
    'Gogi House',
    'Kichi Kichi',
    'Crystal Jade',
    'Thai Express',
    'Seoul Garden',
    'King BBQ',
    'Sumo BBQ',
    'Cloud Pot',
    'Ba Con Cuu',
    'Chix Max',
    'Union Pizza',
    'Dao Niu',
    'Man Tang',
    'Truly Vegan',
    'Yutang',
    'Sumogoya',
    'Sakura Yakiniku',
    'Mama Bakery',
    'Universal Coffee',
    'Canteen Cafe',
    'Kpub',
    'Shogun',
    'Lac Ca',
    'Manwah',
    'iSushi',
    'Hutong',
    'Ashima',
    'Daruma',
    'Vuvuzela',
    'Bobapop',
    'Jokul',
    'iCook',
    'Buk Buk',
    'Ktop',
    'Pho Inn',
    'Pizza360',       # [FIX #4] Added
    'WangBi',         # [FIX #4] Added
]

# Alias mapping for spelling variants that normalization alone can't handle.
# Key: alternative spelling, Value: canonical chain name from KNOWN_CHAINS
CHAIN_ALIASES = {
    'Lách Ca': 'Lac Ca',       # "ch" vs "c" — different spelling, same chain
}

# [FIX #4] Brand code prefix → chain canonical name
# Source: Open_Close.xlsx column J (Brand), maps abbreviated codes to chain names
# Used for future-opening restaurants that only have brand code, not full name
BRAND_CODE_TO_CHAIN = {
    'SS':     'Seoul Garden',    # Seoul Garden (Sài Gòn / Seoul)
    'WY':     'Manwah',          # Wan Yi / Wang Yi — Manwah brand
    'TCH':    'Thai Express',    # Thai Chi Hoi — Thai Express brand
    'FC.TCH': 'Thai Express',    # Food Court Thai Chi Hoi
    'FC':     None,              # Generic Food Court — no single chain
    'GG':     'Gogi House',      # GoGi
    'KK':     'Kichi Kichi',     # Kichi Kichi
    'MW':     'Manwah',          # Manwah abbrev
    'CK':     'Chix Max',        # Chix Max
    'DN':     'Dao Niu',         # Dao Niu Guo
    'GS':     'GoGi Steak',      # Gogi Steak
    'PZ':     'Pizza360',        # Pizza 360
    'WB':     'WangBi',          # Wang Bi
    'SG':     'Seoul Garden',    # Seoul Garden alias
    'MT':     'Man Tang',        # Man Tang Guo
    'CP':     'Cloud Pot',       # Cloud Pot
    'SJ':     'Shogun',          # Shogun Japanese
    'AS':     'Ashima',          # Ashima
    'VV':     'Vuvuzela',        # Vuvuzela
    'LC':     'Lac Ca',          # Lac Ca
}

# Pre-compute normalized chain names for fast lookup
# Includes both canonical names and aliases
_CHAIN_NORMALIZED = [(chain, _normalize_name(chain)) for chain in KNOWN_CHAINS]
for alias, canonical in CHAIN_ALIASES.items():
    _CHAIN_NORMALIZED.append((canonical, _normalize_name(alias)))


class NewRestaurantAgent:
    """
    Agent xử lý forecast cho nhà hàng mới.
    Dùng chain-average pattern để bổ sung dự đoán.
    """
    
    # ==========================================
    # CHAIN DETECTION
    # ==========================================
    
    @staticmethod
    def detect_chain(
        restaurant_name: str,
        brand_code: Optional[str] = None   # [FIX #4] Brand code from Open_Close.xlsx
    ) -> Optional[str]:
        """
        Phát hiện chuỗi/brand từ tên nhà hàng.
        
        Priority:
        1. brand_code (viết tắt từ Open_Close.xlsx) — exact match
        2. Tên full — normalized substring match
        
        Args:
            restaurant_name: Tên nhà hàng (VD: "Gogi House Hanoi Center")
            brand_code: Mã viết tắt từ Open_Close.xlsx (VD: "SS", "TCH")
            
        Returns:
            Tên chuỗi canonical (VD: "Seoul Garden") hoặc None
        """
        if not restaurant_name or pd.isna(restaurant_name):
            restaurant_name = ''
        
        # [FIX #4] Priority 1: Brand code exact match (short code from Open_Close.xlsx)
        if brand_code and isinstance(brand_code, str):
            bc = brand_code.strip().upper()
            if bc in BRAND_CODE_TO_CHAIN:
                chain = BRAND_CODE_TO_CHAIN[bc]
                if chain:
                    logger.debug(f"detect_chain: brand_code='{bc}' → '{chain}'")
                    return chain
        
        # Priority 2: Full name normalized substring match
        if restaurant_name:
            name_norm = _normalize_name(restaurant_name)
            for canonical_name, chain_norm in _CHAIN_NORMALIZED:
                if chain_norm in name_norm:
                    return canonical_name
        
        return None
    
    @staticmethod
    def find_chain_siblings(
        target_code: str,
        target_name: str,
        df_info: pd.DataFrame,
        df_train: pd.DataFrame,
        min_sibling_days: int = 30,
        brand_code: Optional[str] = None,     # [FIX #4] Brand code from Open_Close.xlsx
    ) -> Tuple[Optional[str], List[str]]:
        """
        Tìm các nhà hàng "anh em" cùng chuỗi.
        
        Args:
            target_code: Restaurant code của nhà hàng mới
            target_name: Tên nhà hàng mới
            df_info: DataFrame restaurant info
            df_train: Training data
            min_sibling_days: Số ngày active tối thiểu của sibling
            brand_code: Mã viết tắt từ Open_Close.xlsx (e.g. 'SS', 'TCH')
            
        Returns:
            (chain_name, [sibling_codes])
        """
        # [FIX #4] Use brand_code for priority-1 detection
        chain = NewRestaurantAgent.detect_chain(target_name, brand_code=brand_code)
        
        if chain is None:
            return None, []
        
        # Find all restaurants with same chain
        siblings = []
        
        # From df_info
        if not df_info.empty:
            name_col = 'restaurant_name' if 'restaurant_name' in df_info.columns else None
            # Use merge_key (normalized) if available, otherwise Restaurant_Code
            code_col = None
            for c in ['merge_key', 'Restaurant_Code', 'restaurant_code']:
                if c in df_info.columns:
                    code_col = c
                    break
            
            if name_col and code_col:
                for _, row in df_info.iterrows():
                    rname = str(row.get(name_col, ''))
                    rcode = str(row.get(code_col, ''))
                    
                    if rcode == str(target_code):
                        continue  # Skip self
                    
                    if NewRestaurantAgent.detect_chain(rname) == chain:
                        siblings.append(rcode)
        
        # Filter siblings by active days in training data
        if siblings and not df_train.empty:
            df_train_codes = df_train.copy()
            df_train_codes['restaurant_code'] = df_train_codes['restaurant_code'].astype(str).str.strip()
            
            active_siblings = []
            for sib_code in siblings:
                sib_data = df_train_codes[df_train_codes['restaurant_code'] == sib_code]
                if not sib_data.empty:
                    active_days = sib_data['date'].nunique()  # type: ignore[reportAttributeAccessIssue]
                    if active_days >= min_sibling_days:
                        active_siblings.append(sib_code)
            
            siblings = active_siblings
        
        logger.debug(
            f"Chain '{chain}': found {len(siblings)} active siblings "
            f"for restaurant {target_code}"
        )
        
        return chain, siblings
    
    # ==========================================
    # CHAIN-AVERAGE PATTERNS
    # ==========================================
    
    @staticmethod
    def calculate_chain_patterns(
        df_train: pd.DataFrame,
        sibling_codes: List[str],
    ) -> Dict:
        """
        Tính patterns trung bình từ các nhà hàng cùng chuỗi.
        
        Returns:
            Dict: {
                'avg_daily_guests': float,
                'weekday_pattern': {weekday: avg_guests},
                'hourly_ratios': {hour: ratio},
                'weekend_hourly_ratios': {hour: ratio},
                'n_siblings_used': int,
            }
        """
        if not sibling_codes:
            return {}
        
        df = df_train[df_train['restaurant_code'].isin(sibling_codes)].copy()
        
        if df.empty:
            return {}
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['weekday'] = df['date'].dt.day_name()  # type: ignore[reportAttributeAccessIssue]
        df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])  # type: ignore[reportAttributeAccessIssue]
        
        # 1. Average daily guests per restaurant, then average across restaurants
        daily_per_res = df.groupby(
            ['restaurant_code', df['date'].dt.date]  # type: ignore[reportAttributeAccessIssue]
        )['guest_count'].sum().reset_index()
        daily_per_res.columns = ['restaurant_code', 'date', 'daily_total']
        
        avg_per_res = daily_per_res.groupby('restaurant_code')['daily_total'].mean()
        avg_daily = avg_per_res.median()  # Median to avoid outlier chains
        
        # 2. Weekday pattern (average guests per weekday, normalized)
        weekday_avg = daily_per_res.copy()
        weekday_avg['weekday'] = pd.to_datetime(weekday_avg['date']).dt.day_name()
        weekday_pattern = weekday_avg.groupby('weekday')['daily_total'].median().to_dict()
        
        # Normalize to ratios
        total_week = sum(weekday_pattern.values())
        if total_week > 0:
            weekday_pattern = {k: v / total_week * 7 for k, v in weekday_pattern.items()}
        
        # 3. Hourly distribution (weekday vs weekend)
        def calc_hourly_ratios(sub_df):
            hourly = sub_df.groupby('hour')['guest_count'].sum()
            total = hourly.sum()
            if total <= 0:
                return {}
            ratios = {int(h): c / total for h, c in hourly.items() if c / total > 0.005}
            s = sum(ratios.values())
            return {h: v / s for h, v in ratios.items()} if s > 0 else {}
        
        weekday_hourly = calc_hourly_ratios(df[~df['is_weekend']])
        weekend_hourly = calc_hourly_ratios(df[df['is_weekend']])
        
        if not weekday_hourly:
            weekday_hourly = weekend_hourly
        if not weekend_hourly:
            weekend_hourly = weekday_hourly
        
        patterns = {
            'avg_daily_guests': round(float(avg_daily), 1),
            'weekday_pattern': weekday_pattern,
            'hourly_ratios': weekday_hourly,
            'weekend_hourly_ratios': weekend_hourly,
            'n_siblings_used': int(avg_per_res.count()),
            'median_daily_per_restaurant': round(float(avg_daily), 1),
        }
        
        return patterns
    
    # ==========================================
    # BLENDED FORECAST
    # ==========================================
    
    @staticmethod
    def blend_forecast(
        own_daily_avg: float,
        own_active_days: int,
        chain_daily_avg: float,
        chain_weekday_pattern: Dict[str, float],
        target_weekday: str,
        blend_threshold: Optional[int] = None,
    ) -> float:
        """
        Blend dự đoán giữa data riêng và chain-average.
        
        Weight = own_active_days / blend_threshold
        - 1 ngày data  → 93% chain, 7% own
        - 7 ngày data  → 50% chain, 50% own
        - 14 ngày data → 0% chain, 100% own
        
        Args:
            own_daily_avg: Trung bình daily guests từ data riêng
            own_active_days: Số ngày data riêng
            chain_daily_avg: Trung bình daily guests từ chuỗi
            chain_weekday_pattern: {weekday: multiplier} VD: {'Monday': 0.85, 'Saturday': 1.3}
            target_weekday: Ngày trong tuần cần forecast
            blend_threshold: Ngày data tối đa để dừng blend (default: config)
            
        Returns:
            float: Blended daily forecast
        """
        if blend_threshold is None:
            blend_threshold = int(ANALYSIS_CONFIG.get('new_restaurant_forecast_days', 14))
        
        # Calculate weight: 0.0 (full chain) → 1.0 (full own)
        own_weight = min(1.0, own_active_days / blend_threshold)
        chain_weight = 1.0 - own_weight
        
        # Chain daily adjusted for weekday
        weekday_multiplier = chain_weekday_pattern.get(target_weekday, 1.0)
        chain_forecast = chain_daily_avg * weekday_multiplier
        
        # Blend
        blended = own_weight * own_daily_avg + chain_weight * chain_forecast
        
        return max(0, round(blended, 1))
    
    @staticmethod
    def generate_new_restaurant_forecast(
        res_code: str,
        res_name: str,
        df_res: pd.DataFrame,
        df_train: pd.DataFrame,
        df_info: pd.DataFrame,
        next_days: List[Dict],
        vn_holidays=None,
        brand_code: Optional[str] = None,   # [FIX #4] Brand code from Open_Close.xlsx
    ) -> Tuple[List[Dict], Dict]:
        """
        Generate forecast cho nhà hàng mới sử dụng chain-average blending.
        
        Returns:
            (predictions, metadata): List dự đoán theo giờ + metadata
        """
        from forecast_system.agents.data_agent import DataAgent
        
        own_active_days = df_res['date'].nunique() if not df_res.empty else 0
        
        # Calculate own averages
        if not df_res.empty:
            own_daily = df_res.groupby('date')['guest_count'].sum()
            own_daily_avg = own_daily.mean()
            own_wd_ratios, own_we_ratios = DataAgent.get_hourly_ratios(df_res)
        else:
            own_daily_avg = 0
            own_wd_ratios, own_we_ratios = {}, {}
        
        # Find chain siblings — pass brand_code for priority lookup
        chain_name, siblings = NewRestaurantAgent.find_chain_siblings(
            res_code, res_name, df_info, df_train,
            brand_code=brand_code,  # [FIX #4]
        )
        
        # Calculate chain patterns
        chain_patterns = {}
        if siblings:
            chain_patterns = NewRestaurantAgent.calculate_chain_patterns(
                df_train, siblings
            )
        
        metadata = {
            'chain': chain_name,
            'siblings_found': len(siblings),
            'siblings_used': chain_patterns.get('n_siblings_used', 0),
            'own_active_days': own_active_days,
            'own_daily_avg': round(own_daily_avg, 1),
            'chain_daily_avg': chain_patterns.get('avg_daily_guests', 0),
            'method': 'CHAIN_BLEND' if chain_patterns else 'OWN_DATA_ONLY',
        }
        
        # Determine hourly ratios to use
        if chain_patterns and own_active_days < ANALYSIS_CONFIG.get('new_restaurant_forecast_days', 14):
            # Blend hourly ratios too
            chain_wd = chain_patterns.get('hourly_ratios', {})
            chain_we = chain_patterns.get('weekend_hourly_ratios', {})
            
            own_weight = min(1.0, own_active_days / ANALYSIS_CONFIG.get('new_restaurant_forecast_days', 14))
            chain_weight = 1.0 - own_weight
            
            # Merge hourly ratios
            all_hours = sorted(set(list(chain_wd.keys()) + list(own_wd_ratios.keys())))
            blended_wd = {}
            blended_we = {}
            
            for h in all_hours:
                own_v = own_wd_ratios.get(h, 0)
                chain_v = chain_wd.get(h, 0)
                blended_wd[h] = own_weight * own_v + chain_weight * chain_v
                
                own_v_we = own_we_ratios.get(h, 0)
                chain_v_we = chain_we.get(h, 0)
                blended_we[h] = own_weight * own_v_we + chain_weight * chain_v_we
            
            # Normalize
            s_wd = sum(blended_wd.values())
            s_we = sum(blended_we.values())
            if s_wd > 0:
                blended_wd = {h: v / s_wd for h, v in blended_wd.items()}
            if s_we > 0:
                blended_we = {h: v / s_we for h, v in blended_we.items()}
            
            use_wd_ratios = blended_wd
            use_we_ratios = blended_we
        else:
            use_wd_ratios = own_wd_ratios
            use_we_ratios = own_we_ratios
        
        # Generate predictions
        predictions = []
        
        for day_info in next_days:
            target_date = day_info['date']
            target_weekday = day_info['weekday']
            is_weekend = target_weekday in ('Saturday', 'Sunday')
            
            # Calculate blended daily total
            if chain_patterns:
                daily_total = NewRestaurantAgent.blend_forecast(
                    own_daily_avg=own_daily_avg,  # type: ignore[reportArgumentType]
                    own_active_days=own_active_days,  # type: ignore[reportArgumentType]
                    chain_daily_avg=chain_patterns['avg_daily_guests'],
                    chain_weekday_pattern=chain_patterns.get('weekday_pattern', {}),
                    target_weekday=target_weekday,
                )
            else:
                daily_total = own_daily_avg
            
            # Holiday adjustment
            if day_info.get('is_holiday'):
                impact = day_info.get('holiday_impact', 0.3)
                daily_total *= impact
            
            # Distribute to hours
            ratios = use_we_ratios if is_weekend else use_wd_ratios
            
            if not ratios:
                # Fallback: standard distribution
                ratios = {h: 1.0/15 for h in range(8, 23)}
            
            for hour in range(8, 23):
                ratio = ratios.get(hour, 0)
                hourly_guests = max(0, round(daily_total * ratio))
                
                predictions.append({
                    'date': target_date,
                    'weekday': target_weekday,
                    'hour': hour,
                    'predicted_guests': hourly_guests,
                    'method': metadata['method'],
                    'chain': chain_name,
                    'blend_weight_own': round(
                        min(1.0, own_active_days / ANALYSIS_CONFIG.get('new_restaurant_forecast_days', 14)),
                        2
                    ),
                })
        
        if predictions:
            total_forecast = sum(p['predicted_guests'] for p in predictions)
            n_days = len(next_days)
            logger.info(
                f"🆕 New restaurant {res_code} ({chain_name or 'unknown chain'}): "
                f"{own_active_days}d data → "
                f"{'chain-blend' if chain_patterns else 'own-data'} forecast | "
                f"{total_forecast} guests over {n_days} days | "
                f"siblings={len(siblings)}"
            )
        
        return predictions, metadata
