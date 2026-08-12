# Security Policy

## 1. Nguyên tắc bảo mật của dự án

| Nguyên tắc | Áp dụng trong repo |
|---|---|
| **No secrets in code** | Mọi credential đọc từ biến môi trường. `config/.env` nằm trong `.gitignore`. |
| **Fail-fast on misconfiguration** | `validate_startup_config()` chạy đầu pipeline, thiếu biến bắt buộc → dừng ngay. |
| **Least privilege** | Service account DB chỉ cần quyền `SELECT` trên schema datamart. |
| **Defense in depth** | API key + security headers + bind loopback + reverse proxy TLS. |
| **No sensitive data in logs** | Dùng `get_safe_connection_string()`; error handler không trả traceback ra client. |
| **Automated detection** | `gitleaks` chạy ở pre-commit và trong CI, chặn secret trước khi vào lịch sử. |

---

## 2. Credential Leak Runbook

> Áp dụng khi phát hiện secret đã bị commit. **Xoá file không đủ** — Git lưu toàn bộ lịch sử,
> và các bot quét GitHub thường lấy được secret trong vòng vài phút sau khi push.

### Bước 1 — Xoay vòng credential (ưu tiên tuyệt đối)

Đây là bước **duy nhất** thực sự vô hiệu hoá secret đã lộ.

```sql
-- Trên DB server, đổi mật khẩu service account đã lộ
ALTER USER 'service_account'@'%' IDENTIFIED BY '<mật-khẩu-mới-mạnh>';
FLUSH PRIVILEGES;
```

Song song: thu hồi mọi API key / token khác xuất hiện trong cùng commit.

### Bước 2 — Hạn chế phơi nhiễm

```bash
# Chuyển repo về private ngay trong lúc xử lý
gh repo edit <owner>/<repo> --visibility private --accept-visibility-change-consequences
```

### Bước 3 — Audit truy cập

Rà soát access log của database từ thời điểm commit chứa secret:

- Kết nối từ IP lạ / ngoài dải nội bộ
- Query bất thường (`SELECT *` toàn bảng, `INFORMATION_SCHEMA` dump)
- Đăng nhập ngoài giờ làm việc

Báo cáo cho bộ phận IT Security / Data Protection theo quy trình nội bộ.

### Bước 4 — Purge lịch sử Git

```bash
# Cài công cụ
pip install git-filter-repo

# Sao lưu trước khi thao tác
git clone --mirror <repo-url> backup-repo.git

# Xoá file chứa secret khỏi TOÀN BỘ lịch sử
git filter-repo --invert-paths --path forecast_fb.py --force

# Hoặc thay thế chuỗi secret, giữ lại file
printf 'literal:<CHUỖI-SECRET>==>***REMOVED***\n' > replacements.txt
git filter-repo --replace-text replacements.txt --force

# Đẩy lại
git remote add origin <repo-url>
git push --force --all
git push --force --tags
```

> ⚠️ Force-push viết lại toàn bộ SHA. Thông báo cho mọi thành viên để họ re-clone.
> Mọi fork/clone đã tồn tại **vẫn giữ bản cũ** — đây là lý do Bước 1 không thể bỏ qua.

### Bước 5 — Ngăn tái diễn

```bash
pre-commit install          # bật gitleaks + detect-secrets tại chỗ
gh secret-scanning enable   # bật GitHub secret scanning + push protection
```

---

## 3. Checklist trước khi deploy production

- [ ] `APP_ENV=production` và `DASHBOARD_DEBUG=false`
- [ ] `DASHBOARD_API_KEY` sinh ngẫu nhiên ≥ 32 byte, lưu trong secret manager
- [ ] `DASHBOARD_HOST=127.0.0.1`, expose qua nginx/traefik có TLS
- [ ] Chạy bằng `gunicorn`, **không** dùng Flask dev server
- [ ] Service account DB chỉ có quyền `SELECT`
- [ ] `pip-audit` và `bandit` pass trong CI (không có finding HIGH/CRITICAL)
- [ ] Log không chứa credential — kiểm tra bằng `grep -riE "password|api_key" logs/`
- [ ] Container chạy bằng non-root user
- [ ] Backup `config/.env` lưu ở nơi được mã hoá, không ở repo

---

## 4. Báo cáo lỗ hổng

Phát hiện vấn đề bảo mật, vui lòng **không** mở public issue.
Gửi email tới `<security-contact@example.com>` kèm mô tả và bước tái hiện.
Cam kết phản hồi trong 48 giờ làm việc.
