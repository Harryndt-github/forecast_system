"""
==============================================
CORRECTION VALIDATOR - Safe Self-Learning
==============================================
Đảm bảo mỗi correction mới được đo lường trước và sau khi áp dụng.
Nếu correction làm tệ hơn → tự động rollback.

Rule Validation Cycle:
    1. SNAPSHOT: Ghi lại MAPE / hit-rate TRƯỚC khi correction mới
    2. APPLY: ForecastBrain áp dụng correction bình thường
    3. VALIDATE (sau 7 ngày): So sánh MAPE / hit-rate AFTER vs BEFORE
    4. DECIDE:
        - Nếu tốt hơn ≥ 3%  → CONFIRM correction, tăng confidence
        - Nếu tương đương     → HOLD (không rollback, theo dõi tiếp)
        - Nếu tệ hơn ≥ 5%   → ROLLBACK correction về giá trị cũ

Hit-rate logic (theo yêu cầu):
    - CAP > 100 guests: hit nếu |predicted - actual| ≤ 10 guests
    - CAP ≤ 100 guests: hit nếu |predicted - actual| / actual ≤ 10%
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from forecast_system.config.settings import CURRENT_DATE, PROJECT_ROOT
from forecast_system.utils.logger import get_logger

logger = get_logger('correction_validator')

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
VALIDATOR_FILE = str(PROJECT_ROOT / "correction_validations.json")
VALIDATION_WINDOW_DAYS = 7        # Validate sau 7 ngày
MIN_SAMPLES_VALIDATE   = 5        # Cần ít nhất 5 samples mới để validate
IMPROVE_THRESHOLD      = 0.03     # >3% MAPE giảm → CONFIRM
DEGRADE_THRESHOLD      = 0.05     # >5% MAPE tăng → ROLLBACK
HIGH_VOLUME_CAP        = 100
HIT_ABS_THRESHOLD      = 10
HIT_PCT_THRESHOLD      = 0.10
HIT_RATE_IMPROVE_PP    = 0.03
HIT_RATE_DEGRADE_PP    = 0.03


# ──────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────

def _compute_hit_rate(df: pd.DataFrame, avg_daily: float = 0.0) -> float:
    """
    Strict operational hit-rate:
      - Actual < 100: hit if absolute error <= 10 guests
      - Actual >= 100: hit if percentage error <= 10%
    """
    if df.empty:
        return 0.0
    valid = df[df['Actual_Guest'] > 0].copy()
    if valid.empty:
        return 0.0

    valid['abs_error'] = (valid['Final_Predicted_Guests'] - valid['Actual_Guest']).abs()
    valid['pct_error'] = valid['abs_error'] / valid['Actual_Guest']
    small_mask = valid['Actual_Guest'] < HIGH_VOLUME_CAP
    hits = (
        (small_mask & (valid['abs_error'] <= HIT_ABS_THRESHOLD)) |
        (~small_mask & (valid['pct_error'] <= HIT_PCT_THRESHOLD))
    ).sum()
    return round(float(hits) / len(valid), 4)


def _compute_mape(df: pd.DataFrame) -> Optional[float]:
    """MAPE tính ở cấp ngày (sum per day trước)."""
    if df.empty:
        return None
    valid = df[df['Actual_Guest'] > 0].copy()
    if valid.empty:
        return None
    daily = valid.groupby('Date').agg(
        Predicted=('Final_Predicted_Guests', 'sum'),
        Actual=('Actual_Guest', 'sum'),
    ).reset_index()
    daily = daily[daily['Actual'] > 0]
    if daily.empty:
        return None
    mape = ((daily['Predicted'] - daily['Actual']).abs() / daily['Actual'] * 100).mean()
    return round(float(mape), 2) if pd.notna(mape) else None


def _normalize_master(df_master: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy with Date, error, hit and segment helper columns."""
    if df_master.empty or 'Date' not in df_master.columns:
        return df_master.copy()

    df = df_master.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
    mask = (
        pd.notna(df.get('Final_Predicted_Guests')) &
        pd.notna(df.get('Actual_Guest')) &
        (df.get('Actual_Guest') >= 0) &
        (df.get('Final_Predicted_Guests') >= 0)
    )
    df = df[mask].copy()
    if df.empty:
        return df

    df['error'] = df['Final_Predicted_Guests'] - df['Actual_Guest']
    df['abs_error'] = df['error'].abs()
    df['pct_error'] = np.where(
        df['Actual_Guest'] > 0,
        df['abs_error'] / df['Actual_Guest'] * 100,
        np.nan,
    )
    df['_size_group'] = np.where(df['Actual_Guest'] >= HIGH_VOLUME_CAP, 'LARGE', 'SMALL')

    if 'Is_Holiday' in df.columns:
        is_holiday = df['Is_Holiday'].fillna(False).astype(bool)
    else:
        is_holiday = pd.Series(False, index=df.index)

    if 'Weekday' in df.columns:
        weekday = df['Weekday'].astype(str)
    else:
        weekday = pd.to_datetime(df['Date'], errors='coerce').dt.day_name()

    df['_day_type'] = np.where(
        is_holiday,
        'HOLIDAY',
        np.where(weekday.isin(['Saturday', 'Sunday']), 'WEEKEND', 'WEEKDAY'),
    )
    df['_shift'] = df['Shift'].astype(str) if 'Shift' in df.columns else 'DAILY'
    df['_segment'] = df['_shift'] + '|' + df['_day_type'] + '|' + df['_size_group']
    return df


def _compute_metrics(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {'samples': 0, 'mae': None, 'mape': None, 'bias': None, 'hit_rate': 0.0}
    valid = df[df['Actual_Guest'] > 0].copy()
    if valid.empty:
        return {'samples': 0, 'mae': None, 'mape': None, 'bias': None, 'hit_rate': 0.0}
    mae = float((valid['Final_Predicted_Guests'] - valid['Actual_Guest']).abs().mean())
    bias = float((valid['Final_Predicted_Guests'] - valid['Actual_Guest']).mean())
    return {
        'samples': int(len(valid)),
        'mae': round(mae, 2),
        'mape': _compute_mape(valid),
        'bias': round(bias, 2),
        'hit_rate': _compute_hit_rate(valid),
    }


# ──────────────────────────────────────────
# MAIN CLASS
# ──────────────────────────────────────────

class CorrectionValidator:
    """
    Tracks correction checkpoints and validates them after VALIDATION_WINDOW_DAYS.

    Usage (in ForecastBrain.learn_from_errors):
        # After learning new corrections:
        CorrectionValidator.record_checkpoint(res_code, df_master, brain_memory)

    Usage (in main pipeline, daily):
        rollbacks = CorrectionValidator.validate_all_pending(df_master, brain_memory)
        if rollbacks:
            ForecastBrain.save_memory(brain_memory)
    """

    # ──────────────────────────────────────
    # PERSISTENCE
    # ──────────────────────────────────────

    @staticmethod
    def _load() -> Dict:
        if os.path.exists(VALIDATOR_FILE):
            try:
                with open(VALIDATOR_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {'pending': {}, 'history': []}

    @staticmethod
    def _save(data: Dict):
        try:
            with open(VALIDATOR_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"CorrectionValidator save failed: {e}")

    @staticmethod
    def _pending_key(res_code: str, segment: str = 'ALL') -> str:
        return f"{str(res_code)}|{segment}"

    # ──────────────────────────────────────
    # STEP 1: SNAPSHOT (before correction)
    # ──────────────────────────────────────

    @staticmethod
    def record_checkpoint(
        res_code: str,
        df_master: pd.DataFrame,
        res_mem: Dict,
        avg_daily: float = 0.0,
    ):
        """
        Ghi snapshot TRƯỚC khi correction mới được áp dụng.
        Gọi ngay trong ForecastBrain.learn_from_errors() TRƯỚC khi save.

        Args:
            res_code: Restaurant code
            df_master: Master file (có cả Predicted + Actual)
            res_mem: restaurant memory dict (trạng thái MỚI vừa học)
            avg_daily: Average daily guests (cho hit-rate rule)
        """
        data = CorrectionValidator._load()

        # Filter master for this restaurant — recent 30 days
        cutoff = CURRENT_DATE - datetime.timedelta(days=30)
        mask = (
            (df_master['Restaurant_Code'].astype(str) == str(res_code)) &
            pd.notna(df_master['Final_Predicted_Guests']) &
            pd.notna(df_master['Actual_Guest']) &
            (df_master['Actual_Guest'] >= 0)
        )
        df_res = df_master[mask].copy()
        if 'Date' in df_res.columns:
            df_res['Date'] = pd.to_datetime(df_res['Date'], errors='coerce').dt.date
            df_res = df_res[df_res['Date'] >= cutoff]

        mape_before = _compute_mape(df_res)
        hit_before  = _compute_hit_rate(df_res, avg_daily)

        checkpoint = {
            'res_code':         str(res_code),
            'recorded_at':      str(CURRENT_DATE),
            'validate_after':   str(CURRENT_DATE + datetime.timedelta(days=VALIDATION_WINDOW_DAYS)),
            'avg_daily':        avg_daily,
            'mape_before':      mape_before,
            'hit_rate_before':  hit_before,
            'cf_before':        res_mem.get('correction_factor', 1.0),
            'bias_before':      res_mem.get('overall_bias', 0.0),
            'ctx_bias_before':  dict(res_mem.get('contextual_bias', {})),
            'status':           'PENDING',
        }

        data['pending'][str(res_code)] = checkpoint
        CorrectionValidator._save(data)
        logger.debug(
            f"📌 Checkpoint recorded {res_code}: "
            f"MAPE_before={mape_before}, HR_before={hit_before:.2%}"
        )

    @staticmethod
    def record_segment_checkpoint(
        res_code: str,
        segment: str,
        df_master: pd.DataFrame,
        before_state: Dict,
        after_state: Dict,
        reason: str = '',
    ) -> bool:
        """
        Record a pending validation for a changed restaurant segment.

        The checkpoint stores both the old and new learned correction state. It
        never overwrites an existing pending correction for the same segment,
        which prevents daily runs from moving the validation target forever.
        """
        data = CorrectionValidator._load()
        key = CorrectionValidator._pending_key(res_code, segment)
        if key in data.get('pending', {}):
            return False

        df = _normalize_master(df_master)
        if df.empty or 'Restaurant_Code' not in df.columns:
            return False

        cutoff = CURRENT_DATE - datetime.timedelta(days=30)
        mask = (
            (df['Restaurant_Code'].astype(str) == str(res_code)) &
            (df['Date'] >= cutoff)
        )
        if segment != 'ALL':
            mask = mask & (df['_segment'] == segment)
        df_scope = df[mask].copy()
        metrics_before = _compute_metrics(df_scope)
        if metrics_before['samples'] < MIN_SAMPLES_VALIDATE:
            return False

        checkpoint = {
            'key': key,
            'res_code': str(res_code),
            'segment': segment,
            'recorded_at': str(CURRENT_DATE),
            'validate_after': str(CURRENT_DATE + datetime.timedelta(days=VALIDATION_WINDOW_DAYS)),
            'reason': reason,
            'baseline': metrics_before,
            'before_state': before_state,
            'after_state': after_state,
            'status': 'PENDING',
        }
        data.setdefault('pending', {})[key] = checkpoint
        CorrectionValidator._save(data)
        return True

    # ──────────────────────────────────────
    # STEP 2: VALIDATE (after window)
    # ──────────────────────────────────────

    @staticmethod
    def validate_all_pending(
        df_master: pd.DataFrame,
        brain_memory: Dict,
    ) -> List[str]:
        """
        Kiểm tra tất cả corrections đang pending.
        Gọi MỖI NGÀY trong pipeline (sau khi update actuals).

        Returns:
            List[str]: danh sách res_codes bị ROLLBACK
        """
        data = CorrectionValidator._load()
        pending = data.get('pending', {})
        history = data.get('history', [])
        rollbacks = []
        df_norm = _normalize_master(df_master)

        for pending_key, ckpt in list(pending.items()):
            res_code = str(ckpt.get('res_code') or str(pending_key).split('|', 1)[0])
            validate_date_str = ckpt.get('validate_after', '')
            try:
                validate_date = datetime.date.fromisoformat(validate_date_str)
            except (ValueError, TypeError):
                continue

            if CURRENT_DATE < validate_date:
                continue  # Not yet time to validate

            segment = ckpt.get('segment', 'ALL')
            cutoff = CURRENT_DATE - datetime.timedelta(days=VALIDATION_WINDOW_DAYS)
            if df_norm.empty:
                continue
            mask = (
                (df_norm['Restaurant_Code'].astype(str) == str(res_code)) &
                (df_norm['Date'] >= cutoff)
            )
            if segment != 'ALL' and '_segment' in df_norm.columns:
                mask = mask & (df_norm['_segment'] == segment)
            df_res = df_norm[mask].copy()

            if len(df_res) < MIN_SAMPLES_VALIDATE:
                logger.debug(f"⏳ {res_code}: not enough new samples to validate yet")
                continue

            after_metrics = _compute_metrics(df_res)
            mape_after = after_metrics.get('mape')
            mae_after = after_metrics.get('mae')
            hit_after = after_metrics.get('hit_rate', 0.0)
            baseline = ckpt.get('baseline') or {}
            mape_before = baseline.get('mape', ckpt.get('mape_before'))
            mae_before = baseline.get('mae', ckpt.get('mae_before'))
            hit_before = baseline.get('hit_rate', ckpt.get('hit_rate_before'))

            decision = 'HOLD'
            mape_delta_pct = None
            mae_delta_pct = None
            hit_delta = None
            if mape_before is not None and mape_after is not None:
                mape_delta_pct = (mape_after - mape_before) / max(mape_before, 1)
            if mae_before is not None and mae_after is not None:
                mae_delta_pct = (mae_after - mae_before) / max(mae_before, 1)
            if hit_before is not None and hit_after is not None:
                hit_delta = hit_after - hit_before

            if mae_delta_pct is not None or hit_delta is not None:
                improved = (
                    (mae_delta_pct is not None and mae_delta_pct <= -IMPROVE_THRESHOLD) or
                    (hit_delta is not None and hit_delta >= HIT_RATE_IMPROVE_PP)
                )
                degraded = (
                    (mae_delta_pct is not None and mae_delta_pct >= DEGRADE_THRESHOLD) or
                    (hit_delta is not None and hit_delta <= -HIT_RATE_DEGRADE_PP)
                )
                if improved and not degraded:
                    decision = 'CONFIRM'
                elif degraded and not improved:
                    decision = 'ROLLBACK'
                    rollbacks.append(res_code)
            elif mape_delta_pct is not None:
                if mape_delta_pct <= -IMPROVE_THRESHOLD:
                    decision = 'CONFIRM'
                elif mape_delta_pct >= DEGRADE_THRESHOLD:
                    decision = 'ROLLBACK'
                    rollbacks.append(res_code)

            ckpt.update({
                'after':           after_metrics,
                'mape_after':      mape_after,
                'mae_after':       mae_after,
                'hit_rate_after':  hit_after,
                'validated_at':    str(CURRENT_DATE),
                'decision':        decision,
                'status':          'DONE',
            })

            # ── Act on decision ──
            res_mem = brain_memory.get('restaurants', {}).get(str(res_code))
            if res_mem is not None:
                if decision == 'ROLLBACK':
                    logger.warning(
                        f"⚠️ ROLLBACK {res_code} {segment}: MAPE {mape_before} → {mape_after}, "
                        f"Hit {hit_before} → {hit_after}. Restoring previous correction."
                    )
                    before_state = ckpt.get('before_state')
                    if segment != 'ALL' and before_state is not None:
                        res_mem.setdefault('segment_corrections', {})[segment] = before_state
                    else:
                        cf_before = ckpt.get('cf_before', 1.0)
                        bias_before = ckpt.get('bias_before', 0.0)
                        ctx_bias_before = ckpt.get('ctx_bias_before', {})
                        res_mem['correction_factor'] = cf_before
                        res_mem['overall_bias']      = bias_before
                        if ctx_bias_before:
                            res_mem['contextual_bias'] = ctx_bias_before
                    # Mark for manual review
                    res_mem.setdefault('issues', []).append({
                        'date':       str(CURRENT_DATE),
                        'type':       'CORRECTION_ROLLBACK',
                        'segment':    segment,
                        'cause':      f'Correction degraded metrics: MAPE {mape_before}->{mape_after}, Hit {hit_before}->{hit_after}',
                        'error_pct':  round(float(mape_after), 1) if mape_after is not None else None,
                    })
                    res_mem['issues'] = res_mem['issues'][-30:]

                elif decision == 'CONFIRM':
                    logger.info(
                        f"✅ CONFIRM {res_code} {segment}: MAPE {mape_before} → {mape_after}, "
                        f"Hit {hit_before} → {hit_after}. Correction locked."
                    )
                    if segment != 'ALL':
                        seg_mem = res_mem.setdefault('segment_corrections', {}).setdefault(segment, {})
                        seg_mem['validated'] = True
                        seg_mem['confirmed_at'] = str(CURRENT_DATE)
                        seg_mem['confidence'] = min(1.0, float(seg_mem.get('confidence', 0.5)) + 0.1)
                    else:
                        res_mem['correction_confirmed'] = True
                        res_mem['correction_confirmed_at'] = str(CURRENT_DATE)

                else:  # HOLD
                    logger.debug(
                        f"⏸ HOLD {res_code} {segment}: MAPE {mape_before} → "
                        f"{mape_after}, Hit {hit_before} → {hit_after}"
                    )

            history.append(ckpt)
            del pending[pending_key]

        # Trim history (keep last 500 entries)
        data['history'] = history[-500:]
        data['pending'] = pending
        CorrectionValidator._save(data)

        if rollbacks:
            logger.warning(f"🔄 Rolled back {len(rollbacks)} corrections: {rollbacks}")

        return rollbacks

    # ──────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────

    @staticmethod
    def get_stats() -> Dict:
        """Summary of validation history."""
        data = CorrectionValidator._load()
        history = data.get('history', [])
        pending = data.get('pending', {})

        decisions = [h.get('decision', 'UNKNOWN') for h in history]
        return {
            'pending':   len(pending),
            'confirmed': decisions.count('CONFIRM'),
            'held':      decisions.count('HOLD'),
            'rolled_back': decisions.count('ROLLBACK'),
            'total':     len(history),
        }

    @staticmethod
    def generate_daily_learning_audit(
        df_master: pd.DataFrame,
        output_file: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a compact daily audit report for validated learning.

        The report compares latest 7 days vs previous 7 days and lists weak
        segments. This is the primary evidence for whether daily learning is
        improving the model under the strict operational KPI.
        """
        df = _normalize_master(df_master)
        if df.empty or 'Date' not in df.columns:
            return None

        max_date = max(df['Date'])
        latest_start = max_date - datetime.timedelta(days=6)
        prev_start = max_date - datetime.timedelta(days=13)
        prev_end = max_date - datetime.timedelta(days=7)

        def window_metrics(name: str, start: datetime.date, end: datetime.date) -> Dict:
            scoped = df[(df['Date'] >= start) & (df['Date'] <= end)]
            metrics = _compute_metrics(scoped)
            metrics.update({
                'section': 'WINDOW',
                'name': name,
                'start_date': str(start),
                'end_date': str(end),
            })
            return metrics

        rows = [
            window_metrics('previous_7_days', prev_start, prev_end),
            window_metrics('latest_7_days', latest_start, max_date),
        ]

        prev = rows[0]
        latest = rows[1]
        rows.append({
            'section': 'DELTA',
            'name': 'latest_vs_previous',
            'start_date': str(latest_start),
            'end_date': str(max_date),
            'samples': latest.get('samples', 0),
            'mae': (
                round(latest['mae'] - prev['mae'], 2)
                if latest.get('mae') is not None and prev.get('mae') is not None
                else None
            ),
            'mape': (
                round(latest['mape'] - prev['mape'], 2)
                if latest.get('mape') is not None and prev.get('mape') is not None
                else None
            ),
            'bias': (
                round(latest['bias'] - prev['bias'], 2)
                if latest.get('bias') is not None and prev.get('bias') is not None
                else None
            ),
            'hit_rate': round(latest.get('hit_rate', 0) - prev.get('hit_rate', 0), 4),
        })

        recent = df[(df['Date'] >= latest_start) & (df['Date'] <= max_date)]
        if not recent.empty:
            for segment, grp in recent.groupby('_segment'):
                metrics = _compute_metrics(grp)
                if metrics['samples'] < MIN_SAMPLES_VALIDATE:
                    continue
                metrics.update({
                    'section': 'SEGMENT_LATEST_7',
                    'name': segment,
                    'start_date': str(latest_start),
                    'end_date': str(max_date),
                })
                rows.append(metrics)

        out = output_file or str(PROJECT_ROOT / "daily_learning_audit.csv")
        pd.DataFrame(rows).to_csv(out, index=False, encoding='utf-8-sig')
        logger.info(f"📊 Daily learning audit saved: {out}")
        return out

    @staticmethod
    def clear_pending(res_code: str):
        """Remove a pending checkpoint (e.g., after hard reset)."""
        data = CorrectionValidator._load()
        data['pending'].pop(str(res_code), None)
        CorrectionValidator._save(data)
