# Contributing Guide

## Thiết lập môi trường

```bash
git clone https://github.com/<owner>/forecast_system.git
cd forecast_system
make setup          # venv + dependency + pre-commit hook
cp .env.example config/.env
```

## Quy trình làm việc

```bash
git checkout develop && git pull
git checkout -b feat/ten-tinh-nang

# ... code ...

make check          # PHẢI pass trước khi mở PR
git commit -m "feat(scope): mô tả ngắn"
git push -u origin feat/ten-tinh-nang
```

Branch: `main` (production) ← `develop` (tích hợp) ← `feat/*` `fix/*` `refactor/*`

## Tiêu chuẩn code

**Bắt buộc**

| Quy tắc | Lý do |
|---|---|
| Không hardcode secret | Xem `SECURITY.md` |
| Không `print()` — dùng `logger` | Log có cấu trúc, có level, ghi được ra file |
| Không `except Exception` trần | Dùng exception trong `core/exceptions.py` |
| Mọi hàm public có type hint | mypy cưỡng chế ở `config/`, `utils/`, `core/` |
| Hàm ≤ 50 dòng, complexity ≤ 12 | ruff C901 cưỡng chế |
| Test cho mọi logic mới | Coverage không được giảm |

**Xử lý lỗi đúng cách**

```python
# ❌ SAI — nuốt lỗi, trả số sai mà không ai biết
try:
    result = train_model(df)
except Exception as e:
    logger.warning(f"failed: {e}")
    result = fallback()

# ✅ ĐÚNG — phân loại lỗi, đánh dấu chất lượng
try:
    result = train_model(df)
except InsufficientDataError as exc:
    logger.warning("Không đủ dữ liệu cho %s: %s", exc.restaurant_code, exc)
    result = baseline_forecast(df)
    ctx.mark_degraded(code, QualityFlag.DEGRADED_FALLBACK)
except ModelTrainingError:
    logger.exception("Training thất bại cho %s", code)
    raise
```

**Logging đúng cách**

```python
# ❌ f-string: format ngay cả khi level bị tắt, không có structured field
logger.info(f"Đang xử lý {code} với {n} bản ghi")

# ✅ lazy formatting
logger.info("Đang xử lý %s với %d bản ghi", code, n)

# ❌ TUYỆT ĐỐI KHÔNG log credential
logger.info("Kết nối: %s", get_connection_string())
# ✅
logger.info("Kết nối: %s", get_safe_connection_string())
```

## Quy ước commit

[Conventional Commits](https://www.conventionalcommits.org/), cưỡng chế bằng hook.

```
feat(ensemble): thêm trọng số động theo phân khúc volume
fix(holiday): sửa lệch ngày âm lịch năm nhuận
refactor(pipeline): tách STEP 6 thành EnsembleForecastStage
test(config): thêm test regression chặn secret hardcode
docs(readme): bổ sung hướng dẫn triển khai
chore(deps): nâng lightgbm lên 4.5
```

Type hợp lệ: `feat` `fix` `docs` `refactor` `perf` `test` `build` `ci` `chore` `revert`

## Checklist Pull Request

- [ ] `make check` pass tại máy
- [ ] Đã thêm test cho thay đổi
- [ ] Coverage không giảm
- [ ] Không có secret/credential trong diff
- [ ] Đã cập nhật docstring và tài liệu liên quan
- [ ] Commit theo Conventional Commits
- [ ] PR mô tả rõ *vì sao*, không chỉ *cái gì*

## Đóng góp vào việc refactor

Đang trong Giai đoạn 2 (tách pipeline) — xem `ARCHITECTURE.md` §4.

Khi chuyển một `STEP` từ `main.py` thành `Stage`:

1. Copy **nguyên văn** khối code vào `execute()`. Chưa sửa logic.
2. Thay biến cục bộ bằng trường của `ctx`.
3. Viết test cho stage.
4. Xoá khối cũ khỏi `main()`, gọi stage thay thế.
5. Xác nhận kết quả forecast **không đổi** (so sánh output trước/sau).
6. Mở PR riêng cho từng stage — dễ review, dễ revert.
