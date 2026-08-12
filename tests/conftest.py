"""
Pytest fixtures dùng chung.

Nguyên tắc: unit test KHÔNG chạm DB, KHÔNG gọi LLM, KHÔNG đọc file thật.
Mọi phụ thuộc ngoài đều được mock hoặc thay bằng fixture in-memory.
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """
    Cô lập biến môi trường cho MỌI test.

    Ngăn test vô tình đọc config/.env thật và kết nối vào datamart production.
    """
    for key in ("DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME", "DB_PORT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "false")


@pytest.fixture
def fake_db_env(monkeypatch):
    """Biến môi trường DB hợp lệ nhưng hoàn toàn giả."""
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "3306")
    monkeypatch.setenv("DB_NAME", "test_db")
    monkeypatch.setenv("DB_USER", "test_user")
    monkeypatch.setenv("DB_PASSWORD", "p@ss:word/with#special")


@pytest.fixture
def sample_history() -> pd.DataFrame:
    """90 ngày lịch sử tổng hợp cho 2 nhà hàng, 2 ca."""
    start = datetime.date(2025, 1, 1)
    rows = []
    for offset in range(90):
        day = start + datetime.timedelta(days=offset)
        weekend_boost = 1.6 if day.weekday() >= 5 else 1.0
        for code, base in (("R001", 100), ("R002", 30)):
            for shift, ratio in (("MORNING", 0.4), ("EVENING", 0.6)):
                rows.append({
                    "restaurant_code": code,
                    "date": day,
                    "shift": shift,
                    "guest_count": round(base * ratio * weekend_boost),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def new_restaurant_history() -> pd.DataFrame:
    """Chỉ 5 ngày dữ liệu — kích hoạt nhánh InsufficientDataError."""
    start = datetime.date(2025, 6, 1)
    return pd.DataFrame([
        {
            "restaurant_code": "R999",
            "date": start + datetime.timedelta(days=i),
            "shift": "EVENING",
            "guest_count": 20 + i,
        }
        for i in range(5)
    ])
