"""
==============================================================================
DEPRECATED — forecast_fb.py (legacy monolith, v1)
==============================================================================

File này đã được THAY THẾ hoàn toàn bởi package `forecast_system`:

    forecast_fb.py (800 dòng)  ->  forecast_system/
                                     ├── agents/data_agent.py
                                     ├── agents/ml_forecast_agent.py
                                     ├── agents/ensemble_agent.py
                                     └── pipeline/orchestrator.py

LÝ DO GIỮ LẠI FILE STUB THAY VÌ XOÁ HẲN
---------------------------------------
Để bất kỳ script/cron job cũ nào còn trỏ tới đường dẫn này sẽ báo lỗi rõ ràng
thay vì im lặng chạy code cũ với credential hardcode.

⚠️  CẢNH BÁO BẢO MẬT (lịch sử)
------------------------------
Phiên bản trước của file này chứa username/password của datamart dưới dạng
plaintext và đã từng nằm trong một public repository.

Quy trình khắc phục bắt buộc: xem SECURITY.md § "Credential Leak Runbook".
Tóm tắt: (1) xoay vòng mật khẩu, (2) audit access log, (3) purge git history.

CÁCH CHẠY MỚI
-------------
    # Cấu hình một lần
    cp .env.example config/.env && ${EDITOR:-vi} config/.env

    # Chạy pipeline
    python -m forecast_system.pipeline.orchestrator --mode daily
    # hoặc:  make run
"""

from __future__ import annotations

import sys

_MESSAGE = """
============================================================
  forecast_fb.py đã DEPRECATED và không còn chức năng.
============================================================

Dùng lệnh sau thay thế:

    python -m forecast_system.pipeline.orchestrator --mode daily

Cấu hình kết nối lấy từ biến môi trường (config/.env).
Xem .env.example để biết danh sách biến bắt buộc.
"""


def main() -> int:
    print(_MESSAGE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
