# Changelog

Định dạng theo [Keep a Changelog](https://keepachangelog.com/),
phiên bản theo [Semantic Versioning](https://semver.org/).

## [9.0.0] — Hardening & Restructure

### 🔐 Security — BREAKING

- **Loại bỏ toàn bộ credential hardcode** khỏi source. `forecast_fb.py` (chứa
  username/password datamart dạng plaintext) được thay bằng stub deprecation.
  ⚠️ Yêu cầu xoay vòng mật khẩu và purge lịch sử git — xem `SECURITY.md`.
- **Loại bỏ default hardcode cho hạ tầng.** `DB_HOST` không còn mặc định về IP
  nội bộ; thiếu biến bắt buộc → `ConfigurationError` ngay lúc khởi động.
- **Thêm xác thực API cho dashboard.** 13 endpoint `/api/*` yêu cầu `X-API-Key`,
  so sánh constant-time chống timing attack.
- Dashboard mặc định bind `127.0.0.1` thay vì `0.0.0.0`.
- Chặn `DASHBOARD_DEBUG=true` khi `APP_ENV=production`.
- Thêm security header theo OWASP; error handler không trả traceback ra client.
- Password được URL-encode; bổ sung `get_safe_connection_string()` cho logging.
- Thêm `gitleaks` + `bandit` vào pre-commit và CI.

### ✨ Added

- `core/exceptions.py` — phân cấp exception nghiệp vụ + `QualityFlag`.
- `pipeline/stages.py`, `pipeline/orchestrator.py` — kiến trúc pipeline test được.
- Bộ test đầu tiên: 19 unit test, gồm regression chặn secret hardcode.
- CI GitHub Actions: security scan → lint → type check → test (3 phiên bản Python) → build.
- `Dockerfile` multi-stage chạy non-root, `docker-compose.yml`.
- `Makefile` chuẩn hoá lệnh phát triển.
- Tài liệu: `README.md`, `ARCHITECTURE.md`, `SECURITY.md`, `CONTRIBUTING.md`, `.env.example`.

### ♻️ Changed

- `pyproject.toml` trở thành package chuẩn (bỏ đường dẫn macOS hardcode),
  cấu hình ruff + mypy + pytest + coverage + bandit.
- Dependency pin theo dải tương thích, tách nhóm optional `[ml]` `[llm]` `[dev]` `[serve]`.
- Entry point mới: `forecast-run` thay cho `python main.py`.
- `.gitignore` mở rộng: secret, ML artifact, dữ liệu kinh doanh.

### 🗑️ Removed

- `Finance/` — project Xcode/SwiftUI không liên quan.
- `geminiservice.ts` — file TypeScript mồ côi.
- `fix_pyright.py`, `fix_pyright2.py` — script sửa lỗi tạm.
- `test_merge.py` — script print thủ công, thay bằng test thật.
- `sys.path.insert()` ở 10 file — thay bằng `pip install -e .`.

### 📋 Còn tồn đọng

Xem `ARCHITECTURE.md` §4 để biết lộ trình:
- Tách `main()` (2.007 dòng, complexity 355) thành các stage — Giai đoạn 2.
- Thay 219 `except Exception` trần — Giai đoạn 3.
- Thay 284 `print()` bằng logger — Giai đoạn 3.
- Hợp nhất nhánh trùng lặp `ishushi/` — Giai đoạn 3.
