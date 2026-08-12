"""
=============================================================
HOLIDAY FORECAST PATCH v2 — Correct Logic
=============================================================
Phương pháp:
  Per restaurant:
  1. Tính baseline ngày thường cùng weekday (3 tuần gần nhất)
  2. Multiply bằng calibrated_factor (1.766 cho 30/4, 1.766 cho 1/5)
  3. Replace Final_Predicted_Guests cho ngày 30/4 và 1/5

Đây là cùng logic ensemble_agent đáng lẽ phải làm nhưng bị bug.
=============================================================
"""
import sys, json, datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Workspace root = parent of the forecast_system repo folder (works on Windows/macOS)
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
MASTER_FILE   = PROJECT_ROOT / 'Master_Forecast_Tracking.xlsx'
CAL_FILE      = PROJECT_ROOT / 'holiday_calibration.json'

# ── Calibrated factors (từ 2025 data) ─────────────────────
# Từ holiday_calibration.json sau khi fix:
# LIBERATION_DAY: 1.766 (30/4)
# LABOR_DAY: 1.766 (1/5) — same period, alias LIBERATION_DAY
HOLIDAY_FACTORS = {
    datetime.date(2026, 4, 30): 1.766,   # LIBERATION_DAY
    datetime.date(2026, 5,  1): 1.766,   # LABOR_DAY (same period)
}

# Verify from JSON
try:
    with open(CAL_FILE, encoding='utf-8') as f:
        cal = json.load(f)
    lib_factor = cal['holiday_types']['LIBERATION_DAY']['aggregate']['holiday']
    if isinstance(lib_factor, (int, float)) and 1.0 < lib_factor < 5.0:
        HOLIDAY_FACTORS[datetime.date(2026, 4, 30)] = round(lib_factor, 4)
        HOLIDAY_FACTORS[datetime.date(2026, 5,  1)] = round(lib_factor, 4)
        print(f"📐 Loaded calibrated factor from JSON: {lib_factor:.4f}")
    else:
        print(f"⚠️  Calibrated factor {lib_factor} looks wrong, using 1.766")
except Exception as e:
    print(f"⚠️  Could not load calibration: {e}, using 1.766")

print("=" * 60)
print("🔧 HOLIDAY FORECAST PATCH v2")
print("=" * 60)
for d, f in HOLIDAY_FACTORS.items():
    print(f"   {d}: factor = {f:.4f}")

# ── Load Excel ─────────────────────────────────────────────
print(f"\n📂 Loading {MASTER_FILE.name}...")
df = pd.read_excel(MASTER_FILE, sheet_name='Forecast')
original_len = len(df)
print(f"   Rows: {original_len:,}")

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Forecast_Run_Date'] = pd.to_datetime(df['Forecast_Run_Date'], errors='coerce')

# ── Get latest run ─────────────────────────────────────────
latest_run = df['Forecast_Run_Date'].dropna().dt.date.max()
mask_run = df['Forecast_Run_Date'].dt.date == latest_run
print(f"   Latest run: {latest_run} ({mask_run.sum():,} rows)")

# ── Per-restaurant correction ──────────────────────────────
# For each restaurant, find its baseline forecast on nearby normal weekdays,
# then set holiday forecast = baseline × holiday_factor
print(f"\n⚙️  Computing per-restaurant corrections...")

# Normal weekdays in the same run (27/4=Mon, 28/4=Tue, 29/4=Wed, 4/5=Mon, 5/5=Tue, 6/5=Wed)
# 30/4 is Thursday → use 23/4 (Thu), 7/5 (Thu) as baseline
# 1/5 is Friday → use 24/4 (Fri), 8/5 (Fri) as baseline
BASELINE_DATES = {
    datetime.date(2026, 4, 30): [  # Thursday
        datetime.date(2026, 4, 23),   # Thu -1 week
        datetime.date(2026, 5,  7),   # Thu +1 week
        datetime.date(2026, 4, 16),   # Thu -2 weeks
    ],
    datetime.date(2026, 5,  1): [  # Friday
        datetime.date(2026, 4, 24),   # Fri -1 week
        datetime.date(2026, 5,  8),   # Fri +1 week
        datetime.date(2026, 4, 17),   # Fri -2 weeks
    ],
}

stats = {'patched': 0, 'skipped_no_baseline': 0, 'skipped_zero': 0}
total_before = {d: 0.0 for d in HOLIDAY_FACTORS}
total_after  = {d: 0.0 for d in HOLIDAY_FACTORS}

# Get all restaurants in latest run
restaurants = df.loc[mask_run, 'Restaurant_Code'].astype(str).unique()
print(f"   Restaurants in latest run: {len(restaurants):,}")

for rest in restaurants:
    mask_rest = df['Restaurant_Code'].astype(str) == rest
    df_r = df.loc[mask_run & mask_rest].copy()
    
    for holiday_date, factor in HOLIDAY_FACTORS.items():
        baseline_dates = BASELINE_DATES.get(holiday_date, [])
        mask_hol = df_r['Date'].dt.date == holiday_date
        hol_rows  = df_r[mask_hol]
        
        if hol_rows.empty:
            continue
        
        current_total = hol_rows['Final_Predicted_Guests'].sum()
        total_before[holiday_date] += current_total
        
        if current_total <= 0:
            stats['skipped_zero'] += 1
            total_after[holiday_date] += current_total
            continue
        
        # Find baseline from nearby normal weekday
        baseline_total = None
        for bd in baseline_dates:
            mask_bd = df_r['Date'].dt.date == bd
            bd_rows = df_r[mask_bd]
            if not bd_rows.empty:
                bd_total = bd_rows['Final_Predicted_Guests'].sum()
                if bd_total > 0:
                    baseline_total = bd_total
                    break
        
        if baseline_total is None:
            # Fallback: use current value and just apply correction ratio
            # Current is wrong (had 0.883 applied + other effects)
            # Best estimate: divide by 0.883 (undo calibration error) then multiply by 1.766
            correction = factor / 0.883   # undo wrong factor, apply right one
            new_total = current_total * correction
            stats['skipped_no_baseline'] += 1
        else:
            # Ideal: baseline × holiday_factor
            new_total = baseline_total * factor
        
        # Apply correction per row proportionally
        if current_total > 0:
            row_factor = new_total / current_total
            idx_hol = df.index[(mask_run & mask_rest) & (df['Date'].dt.date == holiday_date)]
            df.loc[idx_hol, 'Final_Predicted_Guests'] = (
                df.loc[idx_hol, 'Final_Predicted_Guests'] * row_factor
            ).round().astype(int)
            stats['patched'] += 1
        
        total_after[holiday_date] += new_total

print(f"   ✅ Patched: {stats['patched']:,} restaurant-day pairs")
print(f"   ⚠️  No baseline (fallback used): {stats['skipped_no_baseline']:,}")
print(f"   ⏭️  Skipped (zero forecast): {stats['skipped_zero']:,}")

# ── Summary ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("📊 CORRECTION RESULTS")
print("=" * 60)
for d in sorted(HOLIDAY_FACTORS):
    b = total_before[d]
    a = total_after[d]
    f = HOLIDAY_FACTORS[d]
    pct = ((a/b)-1)*100 if b > 0 else 0
    print(f"\n  📅 {d}")
    print(f"     Factor applied : ×{f:.3f}")
    print(f"     Trước patch    : {b:>12,.0f} khách")
    print(f"     Sau patch      : {a:>12,.0f} khách")
    print(f"     Thay đổi       : +{a-b:>10,.0f} (+{pct:.1f}%)")
print()

# ── Save Excel ─────────────────────────────────────────────
print(f"💾 Saving to {MASTER_FILE.name}...")

# Check for multiple sheets
try:
    all_sheets = pd.ExcelFile(MASTER_FILE).sheet_names
except Exception:
    all_sheets = ['Forecast']

other_sheets = {}
if len(all_sheets) > 1:
    with pd.ExcelFile(MASTER_FILE) as xls:
        for sheet in all_sheets:
            if sheet != 'Forecast':
                other_sheets[sheet] = pd.read_excel(xls, sheet_name=sheet)

with pd.ExcelWriter(MASTER_FILE, engine='openpyxl', mode='w') as writer:
    df.to_excel(writer, sheet_name='Forecast', index=False)
    for sheet, sdf in other_sheets.items():
        sdf.to_excel(writer, sheet_name=str(sheet), index=False)

print("✅ Master_Forecast_Tracking.xlsx patched successfully!")
print()
print("⚡ Kết quả kỳ vọng:")
print(f"   30/4/2026: ~{total_after[datetime.date(2026,4,30)]:,.0f} khách")
print(f"   01/5/2026: ~{total_after[datetime.date(2026,5,1)]:,.0f} khách")
print()
print("📌 Lưu ý: Đây là patch nhanh. Để có kết quả chính xác nhất,")
print("   hãy chạy lại full pipeline với code đã fix.")
