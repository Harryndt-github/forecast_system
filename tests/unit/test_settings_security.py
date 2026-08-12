"""
Test cho tầng cấu hình bảo mật.

Đây là những test QUAN TRỌNG NHẤT của repo: chúng ngăn việc credential
hardcode quay lại, và ngăn hệ thống âm thầm kết nối vào host mặc định sai.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _reload_settings():
    """Reload module settings để bắt lại biến môi trường hiện tại."""
    from forecast_system.config import settings

    return importlib.reload(settings)


# ==============================================================================
# FAIL-FAST: thiếu cấu hình phải dừng ngay, không dùng default
# ==============================================================================
@pytest.mark.unit
def test_thieu_db_config_thi_raise_configuration_error():
    """Thiếu biến DB bắt buộc -> ConfigurationError, KHÔNG fallback default."""
    settings = _reload_settings()

    with pytest.raises(settings.ConfigurationError) as exc_info:
        settings.get_connection_string()

    assert "DB_USER" in str(exc_info.value) or "DB_HOST" in str(exc_info.value)


@pytest.mark.unit
def test_khong_con_host_mac_dinh_hardcode():
    """
    Regression test: phiên bản cũ có `os.getenv('DB_HOST', '192.168.221.200')`.
    Default này làm lộ topology mạng nội bộ và khiến hệ thống kết nối sai host
    một cách im lặng. Phải không còn IP nào hardcode trong settings.
    """
    source = (REPO_ROOT / "forecast_system" / "config" / "settings.py").read_text(
        encoding="utf-8"
    )
    private_ip = re.compile(r"\b(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
    found = [
        ip for ip in private_ip.findall(source)
        if not ip.startswith("127.")
    ]
    assert not found, f"IP nội bộ bị hardcode trong settings.py: {found}"


# ==============================================================================
# CONNECTION STRING: escape đúng, che credential khi log
# ==============================================================================
@pytest.mark.unit
def test_connection_string_url_encode_ky_tu_dac_biet(fake_db_env):
    """Password chứa `:`, `/`, `#` phải được URL-encode, không làm vỡ URI."""
    settings = _reload_settings()
    conn = settings.get_connection_string()

    assert "p%40ss%3Aword%2Fwith%23special" in conn
    assert "p@ss:word/with#special" not in conn


@pytest.mark.unit
def test_safe_connection_string_khong_lo_credential(fake_db_env):
    """Phiên bản dùng cho log phải che user và password."""
    settings = _reload_settings()
    safe = settings.get_safe_connection_string()

    assert "test_user" not in safe
    assert "p@ss" not in safe
    assert "***" in safe
    assert "localhost" in safe          # host vẫn hiện để debug được


# ==============================================================================
# PRODUCTION GUARDRAILS
# ==============================================================================
@pytest.mark.unit
def test_production_cam_bat_debug(monkeypatch, fake_db_env):
    """APP_ENV=production + DASHBOARD_DEBUG=true -> phải chặn (Werkzeug = RCE)."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_DEBUG", "true")
    monkeypatch.setenv("DASHBOARD_API_KEY", "x" * 32)
    settings = _reload_settings()

    with pytest.raises(settings.ConfigurationError, match="DEBUG"):
        settings.validate_startup_config()


@pytest.mark.unit
def test_production_cam_tat_auth(monkeypatch, fake_db_env):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "false")
    settings = _reload_settings()

    with pytest.raises(settings.ConfigurationError, match="auth"):
        settings.validate_startup_config()


@pytest.mark.unit
def test_bat_auth_ma_thieu_api_key_thi_raise(monkeypatch, fake_db_env):
    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "true")
    monkeypatch.delenv("DASHBOARD_API_KEY", raising=False)
    settings = _reload_settings()

    with pytest.raises(settings.ConfigurationError, match="DASHBOARD_API_KEY"):
        settings.validate_startup_config()


@pytest.mark.unit
def test_dashboard_mac_dinh_bind_loopback(fake_db_env):
    """Mặc định phải là 127.0.0.1, không phải 0.0.0.0."""
    settings = _reload_settings()
    assert settings.DASHBOARD_HOST == "127.0.0.1"
    assert settings.DASHBOARD_DEBUG is False


# ==============================================================================
# ANTI-REGRESSION: quét secret trong toàn bộ source
# ==============================================================================
@pytest.mark.unit
def test_khong_co_secret_hardcode_trong_source():
    """
    Chặn credential quay lại repo. Đây là lớp phòng thủ cuối cùng sau
    gitleaks ở pre-commit và CI.
    """
    patterns = [
        re.compile(r"""password\s*[:=]\s*['"][^'"{}$<]{6,}['"]""", re.I),
        re.compile(r"""api[_-]?key\s*[:=]\s*['"][^'"{}$<]{12,}['"]""", re.I),
        re.compile(r"""secret\s*[:=]\s*['"][^'"{}$<]{8,}['"]""", re.I),
    ]
    allowed = ("os.getenv", "os.environ", "_require_env", "example", "placeholder",
               "<", "test_", "dummy", "fake")

    offenders: list[str] = []
    for path in (REPO_ROOT / "forecast_system").rglob("*.py"):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
        ):
            if any(token in line for token in allowed):
                continue
            if any(p.search(line) for p in patterns):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")

    assert not offenders, "Phát hiện secret hardcode:\n" + "\n".join(offenders)
