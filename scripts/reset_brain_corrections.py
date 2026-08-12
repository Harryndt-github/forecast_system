"""
==============================================
RESET BRAIN CORRECTIONS (Fix A3)
==============================================
Reset correction factors in brain_memory.json mà đã bị
tích luỹ qua nhiều lần chạy và tạo feedback loop.

Giữ nguyên:
- Weekday bias (learned patterns)
- Hourly bias (learned patterns)
- MAPE history (tracking)
- Issues log (diagnostic)

Reset:
- correction_factor → 1.0
- actual_ratio_anchor → 1.0
- escalation_level → 0
- mape_history → keep last 5 only (fresh start)
"""

import json
import datetime
import shutil
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
BRAIN_FILE = PROJECT_ROOT / "brain_memory.json"
BACKUP_FILE = PROJECT_ROOT / f"brain_memory_backup_{datetime.date.today().strftime('%Y%m%d')}.json"


def main():
    if not BRAIN_FILE.exists():
        print(f"❌ Brain file not found: {BRAIN_FILE}")
        return
    
    # Backup first
    shutil.copy2(BRAIN_FILE, BACKUP_FILE)
    print(f"📦 Backup saved: {BACKUP_FILE}")
    
    with open(BRAIN_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    restaurants = data.get('restaurants', {})
    n_reset = 0
    
    for res_code, res_mem in restaurants.items():
        old_cf = res_mem.get('correction_factor', 1.0)
        old_anchor = res_mem.get('actual_ratio_anchor', 1.0)
        old_esc = res_mem.get('escalation_level', 0)
        
        # Reset correction factor to neutral
        res_mem['correction_factor'] = 1.0
        res_mem['actual_ratio_anchor'] = 1.0
        res_mem['escalation_level'] = 0
        
        # Keep only last 5 MAPE history entries for fresh learning
        mape_hist = res_mem.get('mape_history', [])
        if len(mape_hist) > 5:
            res_mem['mape_history'] = mape_hist[-5:]
        
        # Reset needs_retune flag
        res_mem.pop('needs_retune', None)
        
        if old_cf != 1.0 or old_anchor != 1.0 or old_esc > 0:
            n_reset += 1
    
    # Add reset event to learning log
    data.get('learning_log', []).append({
        'date': str(datetime.date.today()),
        'timestamp': datetime.datetime.now().isoformat(),
        'action': 'RESET_CORRECTIONS',
        'restaurants_reset': n_reset,
        'reason': 'Fix A3: Break correction factor feedback loop after model fix',
    })
    
    with open(BRAIN_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ Reset {n_reset}/{len(restaurants)} restaurants")
    print(f"   - correction_factor → 1.0")
    print(f"   - actual_ratio_anchor → 1.0")
    print(f"   - escalation_level → 0")
    print(f"   - mape_history → trimmed to last 5")
    print(f"\n🧠 Brain will re-learn corrections based on new model predictions.")


if __name__ == '__main__':
    main()
