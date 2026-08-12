"""
==============================================
AI FORECAST AGENT (LM Studio Integration)
==============================================
Trách nhiệm:
- Tạo prompt có context analysis report
- Gọi LM Studio local model (openai/gpt-oss-20b)
- Parse JSON response
- Retry logic khi LM Studio timeout/fail

Refactored từ forecast_fb.py AIForecastAgent class.
Fixes: 
- Thêm analysis context vào prompt
- Retry logic
- Better error handling
- Structured prompt format
"""

import json
import re
import time
import datetime
import traceback
from typing import Optional, List, Dict

from openai import OpenAI

from pathlib import Path
from forecast_system.config.settings import LM_STUDIO_CONFIG, CURRENT_DATE, PROJECT_ROOT
from forecast_system.agents.data_agent import DataAgent
from forecast_system.utils.logger import get_logger

logger = get_logger('ai_forecast_agent')

# ==========================================
# LM STUDIO CLIENT
# ==========================================
_client = None

def get_ai_client():
    """Lazy-init LM Studio client"""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=LM_STUDIO_CONFIG['base_url'],
            api_key=LM_STUDIO_CONFIG['api_key'],
            timeout=float(LM_STUDIO_CONFIG['timeout']),
        )
    return _client


class AIForecastAgent:
    """
    Agent sử dụng LLM (LM Studio local) để dự đoán guest count.
    Enhanced với analysis context từ AnalysisAgent.
    """
    
    # ==========================================
    # PROMPT PREPARATION
    # ==========================================
    
    @staticmethod
    def prepare_prompt_data(df_res, vn_holidays):
        """
        Chuẩn bị history text cơ bản (21 ngày gần nhất).
        """
        daily = df_res.groupby('date')['guest_count'].sum().reset_index().sort_values('date')
        recent_21 = daily.tail(21).copy()
        
        history_text = "HISTORY (Date | Weekday | Guest Count):\n"
        for _, row in recent_21.iterrows():
            d = row['date']
            wd = d.strftime('%A')
            cnt = int(row['guest_count'])
            is_hol = "HOLIDAY" if d in vn_holidays else ""
            history_text += f"- {d} ({wd}): {cnt} {is_hol}\n"
        return history_text
    
    @staticmethod
    def prepare_weighted_prompt_data(df_res, target_date, vn_holidays):
        """
        Chuẩn bị data với weighted insights cho AI.
        Gồm 30/60/90-day window statistics.
        """
        history_text = "WEIGHTED ANALYSIS:\n\n"
        
        # Show 30-day window stats (70% weight)
        history_text += "📊 RECENT 30 DAYS (70% importance):\n"
        for wd_name in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            avg_30 = DataAgent.get_window_statistics(df_res, 30, wd_name, target_date)
            if avg_30:
                history_text += f"  {wd_name}: ~{int(avg_30)} guests\n"
        
        # Show 60-day trends (10% weight)
        history_text += "\n📈 60-DAY TREND (10% importance):\n"
        for wd_name in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            avg_60 = DataAgent.get_window_statistics(df_res, 60, wd_name, target_date)
            if avg_60:
                history_text += f"  {wd_name}: ~{int(avg_60)} guests\n"
        
        # Show 90-day seasonal (10% weight)
        history_text += "\n🗓️ 90-DAY SEASONAL (10% importance):\n"
        for wd_name in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            avg_90 = DataAgent.get_window_statistics(df_res, 90, wd_name, target_date)
            if avg_90:
                history_text += f"  {wd_name}: ~{int(avg_90)} guests\n"
        
        # Recent 7 days
        daily = df_res.groupby('date')['guest_count'].sum().reset_index().sort_values('date')
        recent_7 = daily.tail(7)
        history_text += "\n📅 LAST 7 DAYS:\n"
        for _, row in recent_7.iterrows():
            d = row['date']
            wd = d.strftime('%a')
            cnt = int(row['guest_count'])
            history_text += f"  {d} ({wd}): {cnt}\n"
        
        return history_text
    
    @staticmethod
    def prepare_enhanced_prompt(df_res, target_date, vn_holidays, analysis_report=None):
        """
        [MỚI] Tạo prompt ENRICHED với analysis report.
        Đây là phần cải tiến chính - cho LLM biết context về 
        trend, growth, gaps trước khi dự đoán.
        
        Args:
            df_res: Transaction data
            target_date: Reference date
            vn_holidays: Vietnam holidays
            analysis_report: Dict từ AnalysisAgent.generate_restaurant_report()
        
        Returns:
            str: Complete prompt text
        """
        # 1. Basic weighted data
        prompt = AIForecastAgent.prepare_weighted_prompt_data(
            df_res, target_date, vn_holidays
        )
        
        # 2. Inject Analysis Report (NẾU CÓ)
        if analysis_report:
            from forecast_system.agents.analysis_agent import AnalysisAgent
            report_text = AnalysisAgent.format_report_for_prompt(analysis_report)
            prompt = report_text + "\n\n" + prompt
        
        return prompt
    
    # ==========================================
    # FORECAST GENERATION (with retry)
    # ==========================================
    
    @staticmethod
    def generate_forecast(
        res_code: str,
        history_text: str,
        next_days_info: List[Dict],
        analysis_report: Dict = None  # type: ignore[reportArgumentType]
    ) -> Optional[str]:
        """
        Gọi LM Studio để generate forecast.
        Enhanced với In-Context Learning (ICL) + RAG từ brain_memory.
        """
        client = get_ai_client()
        max_retries = LM_STUDIO_CONFIG['max_retries']
        
        # Build future targets text
        future_text = "FORECAST TARGETS (Date | Weekday | Event):\n"
        for d in next_days_info:
            evt = []
            h_type = d.get('holiday_type')
            if h_type:
                evt.append(f"HOLIDAY:{h_type}")
            elif d.get('is_pre_holiday'):
                evt.append(f"PRE_HOLIDAY:{d.get('pre_post_type', 'PRE_HOLIDAY')}")
            elif d.get('is_post_holiday'):
                evt.append(f"POST_HOLIDAY:{d.get('pre_post_type', 'POST_HOLIDAY')}")
            elif d['is_holiday']:
                evt.append("HOLIDAY")
            if d.get('is_veg', False):
                evt.append("VEG_DAY")
            if d['weekday'] in ['Saturday', 'Sunday']:
                evt.append("WEEKEND")
            if d.get('closed_likely'):
                evt.append("⚠️LIKELY_CLOSED")
            # [ICL] Show calibrated impact if available
            impact_src = d.get('holiday_impact_source', '')
            impact_val = d.get('holiday_impact', 1.0)
            if impact_val != 1.0:
                src_tag = '📐DATA' if impact_src == 'calibrated' else 'est'
                evt.append(f"impact:{impact_val:.2f}({src_tag})")
            evt_str = " | ".join(evt) if evt else "Normal"
            future_text += f"- {d['date']} ({d['weekday']}) : {evt_str}\n"
        
        # [RAG] Retrieve brain memory context for this restaurant
        brain_context = AIForecastAgent._build_brain_context(res_code)
        
        # Build system prompt (with brain context injected)
        system_prompt = AIForecastAgent._build_system_prompt(
            analysis_report, brain_context
        )
        user_prompt = (
            f"Restaurant: {res_code}\n\n"
            f"{history_text}\n\n"
            f"{future_text}\n\n"
            f"Generate JSON forecast:"
        )
        
        # Retry loop
        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=LM_STUDIO_CONFIG['model'],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=2000,
                )
                result = response.choices[0].message.content
                
                if result:
                    return result
                else:
                    logger.warning(f"Empty response for {res_code} (attempt {attempt})")
                    
            except Exception as e:
                logger.warning(
                    f"LM Studio call failed for {res_code} "
                    f"(attempt {attempt}/{max_retries}): {e}"
                )
                if attempt < max_retries:
                    wait = attempt * 2  # Exponential backoff
                    time.sleep(wait)
        
        logger.error(f"All {max_retries} LM Studio attempts failed for {res_code}")
        return None
    
    # ==========================================
    # RAG: BRAIN MEMORY RETRIEVAL
    # ==========================================
    
    @staticmethod
    def _build_brain_context(res_code: str) -> str:
        """
        [RAG + ICL] Retrieve brain memory for this restaurant
        and format as prompt context.
        
        This allows the LLM to "learn" from past prediction errors
        without retraining weights.
        """
        try:
            brain_file = PROJECT_ROOT / 'brain_memory.json'
            if not brain_file.exists():
                return ""
            
            with open(brain_file, 'r', encoding='utf-8') as f:
                memory = json.load(f)
            
            res_mem = memory.get('restaurants', {}).get(str(res_code), {})
            if not res_mem:
                return ""
            
            global_patterns = memory.get('global_patterns', {})
            lines = ["\n🧠 BRAIN MEMORY (learned from past errors):"]
            
            # Overall bias
            bias = res_mem.get('overall_bias', 0)
            if abs(bias) >= 2.0:
                direction = "OVER" if bias > 0 else "UNDER"
                lines.append(
                    f"  ⚠️ You tend to {direction}-predict by ~{abs(bias):.0f} guests/hour. "
                    f"ADJUST {'DOWN' if bias > 0 else 'UP'} accordingly."
                )
            
            # Weekday bias
            wd_bias = res_mem.get('weekday_bias', {})
            bad_days = [(d, b) for d, b in wd_bias.items() if abs(b) >= 3.0]
            if bad_days:
                lines.append("  📅 Weekday corrections needed:")
                wd_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                for day_idx, b in sorted(bad_days, key=lambda x: abs(x[1]), reverse=True)[:4]:
                    idx = int(day_idx) if day_idx.isdigit() else 0
                    name = wd_names[idx] if idx < 7 else day_idx
                    lines.append(
                        f"    {name}: {'reduce' if b > 0 else 'increase'} by ~{abs(b):.0f}"
                    )
            
            # Holiday bias
            hol_bias = res_mem.get('holiday_bias', 0)
            if abs(hol_bias) >= 3.0:
                lines.append(
                    f"  🎌 Holiday prediction error: {'over' if hol_bias > 0 else 'under'} "
                    f"by ~{abs(hol_bias):.0f}. Adjust holiday forecasts."
                )
            
            # Last MAPE
            last_mape = res_mem.get('last_mape', 0)
            if last_mape > 0:
                quality = "good" if last_mape < 15 else "moderate" if last_mape < 30 else "poor"
                lines.append(f"  📊 Recent accuracy: MAPE={last_mape:.1f}% ({quality})")
            
            # Issues
            issues = res_mem.get('issues', [])
            active = [i for i in issues if i.get('status') == 'active'][:3]
            if active:
                lines.append("  🔴 Active issues:")
                for iss in active:
                    lines.append(f"    - {iss.get('type', '?')}: {iss.get('description', '')}")
            
            # Correction factor
            cf = res_mem.get('correction_factor', 1.0)
            if abs(cf - 1.0) >= 0.05:
                pct = (cf - 1.0) * 100
                lines.append(
                    f"  🔧 Apply correction: multiply your forecast by {cf:.2f} "
                    f"({'↑' if pct > 0 else '↓'}{abs(pct):.0f}%)"
                )
            
            # Global patterns (chain-wide insights)
            gp = global_patterns
            if gp:
                g_lines = []
                hol_over = gp.get('holiday_overpredict_pct', 0)
                if abs(hol_over) > 5:
                    g_lines.append(f"Chain holidays: {'over' if hol_over > 0 else 'under'} by {abs(hol_over):.0f}%")
                wk_bias = gp.get('weekend_bias', 0)
                if abs(wk_bias) > 3:
                    g_lines.append(f"Chain weekends: {'over' if wk_bias > 0 else 'under'} by {abs(wk_bias):.0f}%")
                if g_lines:
                    lines.append("  🌐 Chain-wide patterns: " + "; ".join(g_lines))
            
            return "\n".join(lines) if len(lines) > 1 else ""
            
        except Exception as e:
            logger.debug(f"Brain context load failed for {res_code}: {e}")
            return ""
    
    # ==========================================
    # SYSTEM PROMPT BUILDER
    # ==========================================
    
    @staticmethod
    def _build_system_prompt(analysis_report: Dict = None, brain_context: str = "") -> str:  # type: ignore[reportArgumentType]
        """Build system prompt với ICL (brain context) + analysis report"""
        
        # Load calibrated holiday impacts if available
        holiday_guide = AIForecastAgent._get_calibrated_holiday_guide()
        
        base_prompt = f"""
You are a Demand Planner for a restaurant chain in Vietnam.
Predict `total_guests` for the forecast period based on HISTORY, ANALYSIS, and BRAIN MEMORY.

LOGIC:
1. Identify WEEKLY PATTERN from History (each weekday has its own baseline).
2. Detect TREND (Growth/Decline) from Analysis Report and apply to baseline.
3. EVENTS impact (VERY IMPORTANT - different holidays have VERY different impacts):
{holiday_guide}
4. Apply GROWTH/DECLINE trend to adjust daily forecast.
5. CHECK BRAIN MEMORY carefully — it tells you past prediction errors. Adjust accordingly!
6. Round to nearest integer. Never forecast negative.
"""
        # [ICL] Inject brain memory context
        if brain_context:
            base_prompt += brain_context + "\n"
        
        if analysis_report:
            trend = analysis_report.get('trend', 'STABLE')
            score = analysis_report.get('trend_score', 0)
            category = analysis_report.get('category', 'STANDARD')
            
            base_prompt += f"""
RESTAURANT CONTEXT:
- Category: {category}
- Trend: {trend} (score: {score})
- Confidence: {analysis_report.get('confidence', 'N/A')}
- Apply trend adjustment: {"increase" if score > 0 else "decrease" if score < 0 else "maintain"} baseline by ~{abs(score) / 10:.1f}% per week
"""
        
        base_prompt += """
OUTPUT JSON ONLY (no explanation):
[
    {"date": "YYYY-MM-DD", "forecast": 150},
    ...
]
"""
        return base_prompt
    
    @staticmethod
    def _get_calibrated_holiday_guide() -> str:
        """
        [RAG] Load calibrated holiday impacts from JSON and format as guide.
        Falls back to default guide if no calibration available.
        """
        default_guide = """   - HOLIDAY:TET_NGUYEN_DAN: -80% to -100% (most restaurants CLOSED!)
   - HOLIDAY:NATIONAL_DAY: +10% to +30% (people celebrate, eat out)
   - HOLIDAY:HUNG_KINGS: +5% to +20% (short holiday)
   - HOLIDAY:LIBERATION_DAY: +10% to +25% (30/4, people travel)
   - HOLIDAY:LABOR_DAY: +10% to +25% (1/5)
   - HOLIDAY:TET_DUONG_LICH: +5% to +15% (New Year)
   - PRE_HOLIDAY:PRE_TET: +20% to +30% (tất niên dinners)
   - POST_HOLIDAY:POST_TET: -20% to -40% (people on vacation)
   - PRE_HOLIDAY: +5% to +15%
   - POST_HOLIDAY: -5% to -15%
   - VEG_DAY: -5% to -15%
   - WEEKEND: Use historical pattern
   - ⚠️LIKELY_CLOSED: Forecast ZERO!"""
        
        try:
            cal_file = PROJECT_ROOT / 'holiday_calibration.json'
            if not cal_file.exists():
                return default_guide
            
            with open(cal_file, 'r', encoding='utf-8') as f:
                cal = json.load(f)
            
            tet = cal.get('holiday_types', {}).get('TET_NGUYEN_DAN', {})
            agg = tet.get('aggregate', {})
            if not agg:
                return default_guide
            
            lines = []
            lines.append("   - HOLIDAY:TET_NGUYEN_DAN: -80% to -100% (CLOSED, 📐data-calibrated)")
            
            # Pre-Tet calibrated
            pre = agg.get('pre', {})
            if pre:
                impacts = [v if isinstance(v, (int, float)) else v.get('impact', 1.0) for v in pre.values()]
                if impacts:
                    min_i, max_i = min(impacts), max(impacts)
                    lines.append(
                        f"   - PRE_TET: {(min_i-1)*100:+.0f}% to {(max_i-1)*100:+.0f}% "
                        f"(📐calibrated from data, {len(impacts)} offset days)"
                    )
            
            # Post-Tet calibrated
            post = agg.get('post', {})
            if post:
                impacts = [v if isinstance(v, (int, float)) else v.get('impact', 1.0) for v in post.values()]
                if impacts:
                    min_i, max_i = min(impacts), max(impacts)
                    lines.append(
                        f"   - POST_TET: {(min_i-1)*100:+.0f}% to {(max_i-1)*100:+.0f}% "
                        f"(📐calibrated from data, gradual recovery)"
                    )
            
            # Other holidays (keep defaults)
            lines.extend([
                "   - HOLIDAY:NATIONAL_DAY: +10% to +30%",
                "   - HOLIDAY:HUNG_KINGS: +5% to +20%",
                "   - HOLIDAY:LIBERATION_DAY: +10% to +25%",
                "   - HOLIDAY:LABOR_DAY: +10% to +25%",
                "   - HOLIDAY:TET_DUONG_LICH: +5% to +15%",
                "   - VEG_DAY: -5% to -15%",
                "   - WEEKEND: Use historical weekend pattern",
                "   - ⚠️LIKELY_CLOSED: Forecast ZERO!",
            ])
            
            return "\n".join(lines)
            
        except Exception:
            return default_guide
    
    # ==========================================
    # RESPONSE PARSING
    # ==========================================
    
    @staticmethod
    def parse_response(text: str) -> Optional[List[Dict]]:
        """
        Parse JSON array từ LLM response.
        Handles các trường hợp response có text extra ngoài JSON.
        
        Returns:
            List[dict] hoặc None
        """
        if not text:
            return None
        
        try:
            # Try direct parse first
            data = json.loads(text.strip())
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Extract JSON array from response text
        try:
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return data
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try to find individual JSON objects
        try:
            objects = re.findall(r'\{[^{}]+\}', text)
            if objects:
                parsed = []
                for obj_str in objects:
                    try:
                        obj = json.loads(obj_str)
                        if 'date' in obj and 'forecast' in obj:
                            parsed.append(obj)
                    except (json.JSONDecodeError, ValueError):
                        continue
                if parsed:
                    return parsed
        except Exception:
            pass
        
        logger.warning(f"Failed to parse AI response: {text[:200]}...")
        return None
