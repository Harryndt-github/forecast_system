# AI Forecast System

> Hệ thống dự báo lượng khách nhà hàng đa tầng — kết hợp ensemble machine learning,
> mô hình chuỗi thời gian và LLM có tăng cường tri thức (RAG) để sinh dự báo theo
> ngày và theo ca làm việc cho toàn bộ chuỗi nhà hàng.

[![CI](https://github.com/Harryndt-github/forecast_system/actions/workflows/ci.yml/badge.svg)](https://github.com/Harryndt-github/forecast_system/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-2a6db2)](https://mypy-lang.org/)
[![Security: bandit](https://img.shields.io/badge/security-bandit%20%7C%20gitleaks-yellow)](https://github.com/PyCQA/bandit)
[![License](https://img.shields.io/badge/license-Proprietary-red)](LICENSE)

---

## Trạng thái dự án

Hệ thống đang **vận hành**, đồng thời đang trong quá trình tái cấu trúc theo lộ
trình tại [`ARCHITECTURE.md`](ARCHITECTURE.md). Bảng dưới phản ánh đúng hiện
trạng — không có hạng mục nào được mô tả tốt hơn thực tế.

| Hạng mục | Trạng thái |
|---|---|
| Engine dự báo (ensemble ML + Prophet + LLM/RAG) | ✅ Hoạt động |
| Hardening bảo mật | ✅ Hoàn tất |
| CI/CD, containerization, pre-commit | ✅ Hoàn tất |
| Khung kiến trúc pipeline | 🚧 Khung đã dựng & test; đang migrate từng stage từ `main.py` |
| Chuẩn hoá xử lý lỗi (`except Exception` → exception cụ thể) | 📋 Giai đoạn 3 |
| Chuẩn hoá logging (`print()` → `logger`) | 📋 Giai đoạn 3 |
| Hợp nhất nhánh trùng lặp `ishushi/` | 📋 Giai đoạn 3 |
| Test coverage | 🚧 19 test cho tầng nền; mục tiêu ≥ 60% |

**Về cổng chất lượng:** ruff/mypy được bật **đầy đủ** cho tầng nền
(`core/`, `pipeline/`, `config/`) và mọi code mới; module legacy được miễn trừ
tạm thời và gỡ dần từng dòng trong `pyproject.toml` khi refactor xong. Cách này
giữ CI luôn xanh và có ý nghĩa, thay vì bật toàn bộ rồi phải bỏ qua vì quá nhiều lỗi.

Số liệu đo được và thứ tự ưu tiên xử lý nợ kỹ thuật:
[`ARCHITECTURE.md §1`](ARCHITECTURE.md#1-hiện-trạng--số-liệu-đo-được).

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Tính năng chính](#tính-năng-chính)
- [Kiến trúc](#kiến-trúc)
- [Bắt đầu nhanh](#bắt-đầu-nhanh)
- [Cấu hình](#cấu-hình)
- [Sử dụng](#sử-dụng)
- [Dashboard API](#dashboard-api)
- [Phát triển](#phát-triển)
- [Kiểm thử](#kiểm-thử)
- [Bảo mật](#bảo-mật)
- [Triển khai](#triển-khai)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Hiệu năng & độ chính xác](#hiệu-năng--độ-chính-xác)
- [Xử lý sự cố](#xử-lý-sự-cố)
- [Đóng góp](#đóng-góp)

---

## Tổng quan

Hệ thống giải quyết bài toán **dự báo nhu cầu (demand forecasting)** cho chuỗi F&B:
mỗi nhà hàng cần biết trước lượng khách theo từng ca để xếp lịch nhân sự, đặt nguyên
liệu và lên kế hoạch vận hành.

Thách thức đặc thù khiến một mô hình đơn lẻ không đủ:

| Thách thức | Cách hệ thống xử lý |
|---|---|
| Nhà hàng có quy mô rất khác nhau (20 → 500 khách/ngày) | Phân khúc theo volume, mỗi khúc dùng chiến lược mô hình riêng |
| Lễ Tết âm lịch làm sai lệch mạnh chuỗi thời gian | Hiệu chỉnh theo dữ liệu (`HolidayCalibrator`) + đường cong tác động theo ngày |
| Nhà hàng mới không có lịch sử | Transfer learning từ nhà hàng tương đồng |
| Sai số hệ thống lặp lại theo nhà hàng | `ForecastBrain` học correction từ kết quả thực tế |
| Ca sáng/tối có hành vi khác nhau | Dự báo tách theo ca với alpha-blend thích ứng |

**Đầu vào:** dữ liệu giao dịch lịch sử (StarRocks/MySQL datamart), dữ liệu đặt bàn, lịch nghỉ lễ.
**Đầu ra:** dự báo theo `nhà hàng × ngày × ca`, kèm cờ chất lượng, xuất ra Excel master + REST API.

---

## Tính năng chính

**Tầng mô hình**
- Ensemble stacking (XGBoost, LightGBM, CatBoost) với meta-learner LightGBM
- Prophet + NeuralProphet global model cho thành phần mùa vụ
- LLM (local qua LM Studio) tăng cường bằng RAG cho tri thức ngữ cảnh
- Hợp nhất theo trọng số động, thích ứng theo phân khúc volume

**Tầng nghiệp vụ**
- Hiệu chỉnh lễ Tết dựa trên dữ liệu, hỗ trợ âm lịch Việt Nam
- Phát hiện đóng cửa theo 4 mức ưu tiên (lịch thực tế → mẫu lịch sử → tỷ lệ đóng cửa)
- Tích hợp dữ liệu đặt bàn để ghi đè dự báo khi tín hiệu đủ mạnh
- Onboarding nhà hàng mới bằng transfer learning

**Tầng vận hành**
- Theo dõi độ chính xác + phát hiện drift, có báo cáo Excel
- Vòng phản hồi tự học: kết quả thực tế → correction cho lần chạy sau
- Dashboard REST API có xác thực
- Pipeline chạy song song theo nhà hàng

---

## Kiến trúc

```
┌──────────────────────────────────────────────────────────────────────┐
│                          NGUỒN DỮ LIỆU                               │
│      StarRocks Datamart  │  Hệ thống đặt bàn  │  Lịch nghỉ lễ        │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────┐
│                    PIPELINE ORCHESTRATOR                             │
│   Chuỗi stage tuần tự, mỗi stage cô lập & test được độc lập          │
├──────────────────────────────────────────────────────────────────────┤
│  connect_db → load_data → analysis → holiday_calibration →           │
│  event_calibration → train_global_model → ensemble_forecast →        │
│  persist_results → monitoring → brain_insights                       │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────┐      ┌──────────────────┐     ┌──────────────────┐
│  TẦNG ML      │      │  TẦNG TIME-SERIES│     │  TẦNG LLM/RAG    │
│  XGBoost      │      │  Prophet         │     │  LM Studio       │
│  LightGBM     │      │  NeuralProphet   │     │  ChromaDB        │
│  CatBoost     │      │  (global model)  │     │  KnowledgeStore  │
└───────┬───────┘      └────────┬─────────┘     └────────┬─────────┘
        └───────────────────────┼────────────────────────┘
                                ▼
                   ┌────────────────────────┐
                   │    ENSEMBLE AGENT      │
                   │  Meta-learner + trọng  │
                   │  số động theo phân khúc│
                   └───────────┬────────────┘
                               ▼
                   ┌────────────────────────┐
                   │    FORECAST BRAIN      │
                   │  Correction đã học     │
                   │  + kiểm định           │
                   └───────────┬────────────┘
                               ▼
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
   Excel Master           REST API              Báo cáo giám sát
```

**Nguyên tắc thiết kế**

1. **Suy giảm có kiểm soát (graceful degradation)** — LLM và NeuralProphet là lớp
   *tăng cường*, không phải phụ thuộc bắt buộc. Chúng lỗi thì ensemble vẫn chạy
   100% bằng ML, và kết quả được đánh dấu `DEGRADED` để downstream biết.
2. **Fail-fast khi sai cấu hình** — kiểm tra toàn bộ biến bắt buộc trước khi tốn
   20 phút huấn luyện.
3. **Pipeline là dữ liệu, không phải code** — mỗi stage là một object; thứ tự,
   bỏ qua, và resume đều cấu hình được.
4. **Quan sát được theo mặc định** — mỗi stage tự ghi thời gian; mọi dự báo đều
   mang cờ chất lượng.

Chi tiết: [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`SYSTEM_DOCUMENTATION.md`](SYSTEM_DOCUMENTATION.md)

---

## Bắt đầu nhanh

### Yêu cầu

- Python 3.10 – 3.12
- Quyền truy cập datamart (tài khoản read-only)
- *(Tuỳ chọn)* LM Studio đang chạy, cho tầng LLM
- *(Tuỳ chọn)* Docker 24+

### Cài đặt

```bash
git clone https://github.com/Harryndt-github/forecast_system.git
cd forecast_system

# Tạo venv, cài dependency, bật pre-commit hook
make setup

# Điền cấu hình
cp .env.example config/.env
${EDITOR:-vi} config/.env        # xem bảng biến môi trường bên dưới

# Kiểm tra mọi thứ hoạt động
make check
```

### Chạy lần đầu

```bash
make run                      # dự báo 30 ngày
# hoặc: forecast-run --mode daily
```

---

## Cấu hình

Toàn bộ cấu hình đến từ **biến môi trường**. Không có credential nào nằm trong code.
Sao chép [`.env.example`](.env.example) thành `config/.env` và điền giá trị.

### Biến bắt buộc

| Biến | Mô tả | Ví dụ |
|---|---|---|
| `DB_HOST` | Host datamart | `datamart.internal.example.com` |
| `DB_PORT` | Cổng | `3306` |
| `DB_NAME` | Tên database | `forecast_datamart` |
| `DB_USER` | Service account (read-only) | `svc_forecast_ro` |
| `DB_PASSWORD` | Mật khẩu | *(từ secret manager)* |
| `DASHBOARD_API_KEY` | API key cho endpoint `/api/*` | *(sinh ngẫu nhiên)* |

```bash
# Sinh API key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Biến tuỳ chọn thường dùng

| Biến | Mặc định | Mô tả |
|---|---|---|
| `APP_ENV` | `dev` | `dev` / `staging` / `production` |
| `DASHBOARD_HOST` | `127.0.0.1` | ⚠️ Chỉ đổi khi đã có reverse proxy |
| `DASHBOARD_AUTH_ENABLED` | `true` | Chỉ tắt ở máy dev |
| `DAILY_FORECAST_DAYS` | `30` | Horizon cho mode `daily` |
| `FORECAST_MONTHS_AHEAD` | `3` | Horizon cho mode `full` |
| `AI_INFERENCE_THRESHOLD` | `0.50` | Bỏ qua LLM nếu trọng số AI dưới ngưỡng |
| `META_LEARNER_ENABLED` | `true` | LightGBM meta-learner vs trung bình có trọng số |

Danh sách đầy đủ: [`.env.example`](.env.example).

> **Ở production**, inject biến từ secret manager (Vault, AWS Secrets Manager,
> Kubernetes Secret) thay vì để file `.env` trên đĩa.

---

## Sử dụng

### Giao diện dòng lệnh

```bash
forecast-run --mode daily                    # 30 ngày, nhanh (~15 phút)
forecast-run --mode full                     # 3 tháng, đầy đủ (~60 phút)
forecast-run --log-level DEBUG               # log chi tiết

# Chạy lại từ stage bị lỗi, không phải chạy lại từ đầu
forecast-run --only ensemble_forecast persist_results
```

### Dùng như thư viện

```python
from forecast_system.pipeline.orchestrator import run_pipeline

ctx = run_pipeline(mode="daily")

print(f"Đã dự báo {len(ctx.active_restaurants)} nhà hàng")
print(f"Kết quả cần review: {ctx.degraded_count}")
print(f"Stage chậm nhất: {max(ctx.stage_durations, key=ctx.stage_durations.get)}")
```

### Lịch chạy định kỳ

```cron
# Dự báo hằng ngày, 02:00
0 2 * * * cd /opt/forecast_system && .venv/bin/forecast-run --mode daily >> logs/cron.log 2>&1

# Dự báo đầy đủ, ngày 1 hằng tháng
0 3 1 * * cd /opt/forecast_system && .venv/bin/forecast-run --mode full >> logs/cron.log 2>&1
```

---

## Dashboard API

Toàn bộ endpoint `/api/*` **yêu cầu xác thực**.

```bash
make dashboard                # dev server tại http://127.0.0.1:5050
```

```bash
curl -H "X-API-Key: $DASHBOARD_API_KEY" http://127.0.0.1:5050/api/overview
# hoặc:
curl -H "Authorization: Bearer $DASHBOARD_API_KEY" http://127.0.0.1:5050/api/overview
```

| Endpoint | Method | Mô tả |
|---|---|---|
| `/healthz` | GET | Health check (công khai) |
| `/api/overview` | GET | Tổng quan độ chính xác toàn hệ thống |
| `/api/restaurants` | GET | Chỉ số theo từng nhà hàng |
| `/api/daily` | GET | Chuỗi độ chính xác theo ngày |
| `/api/weekday` | GET | Phân rã theo thứ trong tuần |
| `/api/hourly` | GET | Phân rã theo giờ |
| `/api/drift` | GET | Cảnh báo drift mô hình |
| `/api/problems` | GET | Nhà hàng có sai số vượt ngưỡng |
| `/api/forecast/<code>` | GET | Dự báo của một nhà hàng |
| `/api/forecast/upcoming` | GET | Dự báo sắp tới toàn chuỗi |

**Phản hồi lỗi** cố tình chỉ trả thông tin tối thiểu (`{"error": "..."}`);
chi tiết chỉ ghi vào log phía server để tránh lộ thông tin nội bộ.

---

## Phát triển

```bash
make help          # danh sách lệnh
make format        # tự động format + fix lint
make lint          # ruff
make typecheck     # mypy
make security      # bandit + gitleaks + pip-audit
make test          # pytest + coverage
make complexity    # liệt kê hàm quá phức tạp
make check         # TẤT CẢ cổng chất lượng (giống CI)
```

### Tiêu chuẩn chất lượng

| Cổng | Công cụ | Ngưỡng |
|---|---|---|
| Lint & format | `ruff` | 0 lỗi |
| Type checking | `mypy` | strict cho `config/`, `utils/`, `core/` |
| Độ phức tạp | `ruff` C901 | ≤ 12 mỗi hàm |
| Test coverage | `pytest-cov` | ≥ 40%, nâng dần mỗi sprint |
| Lỗ hổng code | `bandit` | 0 finding HIGH |
| Secret | `gitleaks` | 0 phát hiện |
| CVE dependency | `pip-audit` | 0 CRITICAL |

Các cổng này chạy tại pre-commit (cục bộ) và trong CI (mọi PR).

### Quy ước commit

Theo [Conventional Commits](https://www.conventionalcommits.org/), được cưỡng chế bằng hook:

```
feat(ensemble): thêm trọng số động theo phân khúc volume
fix(holiday): sửa lệch ngày âm lịch cho năm nhuận
refactor(pipeline): tách STEP 6 thành EnsembleForecastStage
test(config): thêm test regression cho secret hardcode
```

---

## Kiểm thử

```bash
make test-fast     # chỉ unit test (nhanh)
make test          # đầy đủ + coverage
make coverage      # báo cáo HTML
pytest -m unit     # theo marker
```

```
tests/
├── conftest.py            # fixture dùng chung; cô lập biến môi trường
├── unit/                  # thuần, không I/O
│   ├── test_settings_security.py   # ⭐ chặn credential quay lại repo
│   ├── test_exceptions.py
│   └── test_pipeline_stages.py
└── integration/           # cần DB — đánh dấu @pytest.mark.integration
```

Fixture `_isolated_env` chạy tự động cho **mọi** test, xoá biến môi trường DB để
đảm bảo test không bao giờ vô tình kết nối vào datamart production.

---

## Bảo mật

Chính sách đầy đủ, kèm runbook xử lý sự cố lộ credential: [`SECURITY.md`](SECURITY.md).

**Các biện pháp đã áp dụng**

- Không có secret trong source; mọi credential đọc từ biến môi trường
- Thiếu cấu hình bắt buộc → `ConfigurationError` ngay lúc khởi động
- Password được URL-encode; log dùng phiên bản đã che (`get_safe_connection_string()`)
- API key cho toàn bộ `/api/*`, so sánh constant-time chống timing attack
- Dashboard mặc định bind `127.0.0.1`; production bắt buộc reverse proxy + TLS
- Security header theo OWASP; error handler không trả traceback ra client
- Chặn `DASHBOARD_DEBUG=true` ở production (Werkzeug debugger = RCE)
- `gitleaks` + `bandit` chạy ở pre-commit và CI
- Container chạy non-root, filesystem read-only
- Test regression tự động quét secret hardcode trong toàn bộ source

**Báo cáo lỗ hổng:** không mở public issue — xem [`SECURITY.md`](SECURITY.md).

---

## Triển khai

### Docker

```bash
make docker-build
make docker-up                 # API tại 127.0.0.1:5050
make docker-batch              # chạy một lần batch forecast
```

### Production

```bash
gunicorn --workers 4 --bind 127.0.0.1:5050 --timeout 300 \
         forecast_system.dashboard.app:app
```

Đặt sau nginx/traefik có TLS. **Không bao giờ** dùng Flask dev server ở production —
`run_dashboard()` sẽ raise nếu `APP_ENV=production`.

Checklist trước khi lên production: [`SECURITY.md`](SECURITY.md#3-checklist-trước-khi-deploy-production).

---

## Cấu trúc thư mục

```
forecast_system/
├── config/
│   └── settings.py          # cấu hình tập trung, fail-fast, không secret
├── core/
│   ├── exceptions.py        # phân cấp exception nghiệp vụ
│   └── logging.py           # logging có cấu trúc
├── pipeline/
│   ├── stages.py            # từng bước pipeline, độc lập & test được
│   └── orchestrator.py      # entrypoint
├── agents/                  # tầng nghiệp vụ
│   ├── data_agent.py
│   ├── ml_forecast_agent.py
│   ├── ensemble_agent.py
│   ├── forecast_brain.py
│   ├── holiday_calibrator.py
│   └── ...
├── dashboard/
│   ├── app.py               # REST API (đã có auth)
│   └── static/
├── utils/
│   ├── date_utils.py        # lịch âm, lễ Tết Việt Nam
│   └── logger.py
tests/
docs/
```

---

## Hiệu năng & độ chính xác

| Chỉ số | Giá trị |
|---|---|
| Mục tiêu MAPE (nhà hàng volume cao) | 10 – 15% |
| Mục tiêu SMAPE (nhà hàng volume thấp) | 20 – 25% |
| Thời gian chạy mode `daily` | ~15 phút |
| Thời gian chạy mode `full` | ~60 phút |
| Horizon dự báo theo ca | 30 ngày |
| Horizon dự báo theo ngày | 3 tháng |

Độ chính xác thực tế theo dõi trong `/api/overview` và báo cáo Excel hằng ngày.

---

## Xử lý sự cố

<details>
<summary><b>ConfigurationError: Thiếu biến môi trường bắt buộc: DB_USER</b></summary>

Chưa tạo `config/.env` hoặc thiếu biến. Chạy:
```bash
cp .env.example config/.env && ${EDITOR:-vi} config/.env
```
Đây là hành vi **cố ý** — hệ thống từ chối chạy với cấu hình mặc định thay vì âm thầm
kết nối sai host.
</details>

<details>
<summary><b>API trả 401 unauthorized</b></summary>

Thiếu hoặc sai `X-API-Key`. Kiểm tra `DASHBOARD_API_KEY` trong `config/.env` khớp
với header đang gửi. Ở máy dev có thể đặt `DASHBOARD_AUTH_ENABLED=false`
(sẽ bị chặn nếu `APP_ENV=production`).
</details>

<details>
<summary><b>Pipeline báo nhiều nhà hàng DEGRADED</b></summary>

Xem log warning để biết stage nào suy giảm. Nguyên nhân thường gặp:
LM Studio không chạy (LLM timeout), hoặc nhà hàng mới chưa đủ lịch sử.
Dự báo vẫn hợp lệ nhưng do model fallback sinh ra — nên review thủ công.
</details>

<details>
<summary><b>Import lỗi khi chạy script trong thư mục con</b></summary>

Package phải được cài dạng editable:
```bash
pip install -e .
```
Phiên bản cũ dùng `sys.path.insert()`; cách đó đã bị loại bỏ.
</details>

---

## Đóng góp

Xem [`CONTRIBUTING.md`](CONTRIBUTING.md). Tóm tắt:

1. Tạo branch từ `develop`
2. `make setup` (bật pre-commit hook)
3. Viết test cho thay đổi
4. `make check` phải pass trước khi mở PR
5. Commit theo Conventional Commits

---

## Giấy phép

Phần mềm độc quyền. Xem [`LICENSE`](LICENSE).
