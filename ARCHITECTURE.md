# Architecture & Refactoring Plan

Tài liệu này mô tả kiến trúc đích, lý do đằng sau các quyết định, và **lộ trình
chuyển đổi tăng dần** từ cấu trúc hiện tại. Mọi con số đo bằng `radon` và `ruff`.

---

## 1. Hiện trạng — số liệu đo được

| Chỉ số | Giá trị đo | Ngưỡng chuẩn | Trạng thái |
|---|---|---|---|
| Complexity `main.py::main()` | **355** | ≤ 10 | 🔴 |
| Độ dài `main()` | **2.007 dòng** | ≤ 50 | 🔴 |
| Số hàm complexity ≥ D (>20) | **26** | 0 | 🔴 |
| File > 1.000 dòng | **10** | 0 | 🔴 |
| `except Exception` trần | **219** | ~0 | 🔴 |
| Lời gọi `print()` | **284** | 0 | 🔴 |
| Độ phủ type hint | **48%** | ≥ 90% | 🟡 |
| `sys.path.insert()` hack | **10 file** | 0 | 🔴 |
| Test coverage | **0%** | ≥ 40% | 🔴 |
| Code trùng lặp (`ishushi/` ↔ `agents/`) | ~2.000 dòng | 0 | 🔴 |

---

## 2. Bốn vấn đề kiến trúc gốc

### 2.1 God Function

`main()` đảm nhiệm cả 10 trách nhiệm: kết nối DB, nạp dữ liệu, phân tích, hiệu
chỉnh lễ, huấn luyện, dự báo, lưu trữ, báo cáo, giám sát, học lại.

Hệ quả cụ thể: muốn kiểm thử logic hiệu chỉnh lễ Tết, phải chạy kết nối datamart
thật và huấn luyện toàn bộ mô hình. Trên thực tế điều đó có nghĩa là **logic này
chưa bao giờ được kiểm thử**.

**Giải pháp:** mẫu Pipeline/Stage — xem [`pipeline/stages.py`](forecast_system/pipeline/stages.py).
Bản thân `main()` đã có sẵn mốc `# STEP N:`, nên việc tách là cơ học, rủi ro thấp.

### 2.2 Nuốt lỗi (Silent Failure)

```python
except Exception as e:
    logger.warning(f"failed: {e}")
    result = fallback()          # trả số SAI, không ai biết
```

219 vị trí như thế này. Với hệ thống phục vụ xếp ca và đặt hàng nguyên liệu,
một con số sai được trình bày như con số đúng là **rủi ro nghiệp vụ**, không chỉ
là nợ kỹ thuật.

**Giải pháp:** phân cấp exception + `QualityFlag` — xem [`core/exceptions.py`](forecast_system/core/exceptions.py).
Mọi fallback đều phải đánh dấu kết quả là `DEGRADED`, và cờ này chảy tới tận
dashboard.

### 2.3 Không có ranh giới package

10 file dùng `sys.path.insert()` để import lẫn nhau. Package chỉ chạy được khi
nằm đúng một vị trí trên đĩa — `pyproject.toml` cũ thậm chí hardcode
`/Users/harryng/Desktop/Coding/.venv/bin/python3`.

**Giải pháp:** package chuẩn cài bằng `pip install -e .`, khai báo entry point
trong `[project.scripts]`.

### 2.4 Nhánh code song song

`ishushi/` là bản sao gần giống của `agents/` (`data_agent.py`: 769 vs 1.041 dòng).
Hai bản sẽ phân kỳ theo thời gian, và bug sửa ở một bên không sang bên kia.

**Giải pháp:** hợp nhất thành một codebase, khác biệt đưa vào cấu hình
(`BrandConfig`) thay vì nhân bản code.

---

## 3. Kiến trúc đích

```
┌────────────────────────────────────────────────────────────┐
│  TẦNG TRÌNH DIỄN                                           │
│  dashboard/ (REST API)  ·  CLI (orchestrator)              │
└──────────────────────────┬─────────────────────────────────┘
                           │ chỉ phụ thuộc xuống dưới
┌──────────────────────────▼─────────────────────────────────┐
│  TẦNG ĐIỀU PHỐI                                            │
│  pipeline/orchestrator.py  ·  pipeline/stages.py           │
│  → Điều phối thứ tự. KHÔNG chứa logic nghiệp vụ.           │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│  TẦNG NGHIỆP VỤ                                            │
│  agents/  → mô hình, hiệu chỉnh, brain                     │
│  → Thuần logic. KHÔNG biết mình được gọi từ CLI hay API.   │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│  TẦNG HẠ TẦNG                                              │
│  config/ · core/ · utils/ · repository truy cập dữ liệu    │
└────────────────────────────────────────────────────────────┘
```

**Quy tắc phụ thuộc:** mũi tên chỉ đi xuống. Tầng nghiệp vụ không được import
tầng trình diễn. Kiểm tra tự động bằng `ruff` (`TID252`) hoặc `import-linter`.

---

## 4. Lộ trình chuyển đổi

Mỗi giai đoạn là một PR độc lập, **hệ thống luôn ở trạng thái chạy được**.
Không big-bang rewrite.

### Giai đoạn 0 — Cầm máu (1–2 ngày) 🔴 ưu tiên tuyệt đối

| Việc | Kết quả |
|---|---|
| Xoay vòng credential DB đã lộ | Secret cũ vô hiệu |
| Repo về private, purge lịch sử git | Giảm phơi nhiễm |
| Bật `pre-commit` + `gitleaks` | Chặn tái diễn |
| Áp `config/settings.py` đã vá | Không còn default hardcode |
| Áp `dashboard/app.py` đã vá | 13 endpoint được bảo vệ |
| Thay `forecast_fb.py` bằng stub | Xoá nguồn rò rỉ |

### Giai đoạn 1 — Nền tảng (tuần 1)

| Việc | Kết quả đo được |
|---|---|
| `pyproject.toml` chuẩn + `pip install -e .` | Xoá 10 `sys.path.insert()` |
| Xoá `Finance/`, `geminiservice.ts`, `fix_pyright*.py` | −8 file rác |
| Thêm CI (lint → type → test) | Mọi PR có cổng kiểm tra |
| Thêm `core/exceptions.py` + test | 19 test pass |
| Pin dependency, thêm `Dockerfile` | Build tái lập được |

### Giai đoạn 2 — Tách pipeline (tuần 2–4)

Chuyển từng `# STEP` thành `Stage`. Thứ tự đề xuất — **dễ nhất trước** để
đội quen với mẫu thiết kế:

| Thứ tự | Stage | Nguồn (main.py) | Ước lượng |
|---|---|---|---|
| 1 | `ConnectDatabaseStage` | STEP 1 | ✅ đã có mẫu |
| 2 | `LoadDataStage` | dòng 461–514 | 0,5 ngày |
| 3 | `AnalysisStage` | dòng 515–533 | 0,5 ngày |
| 4 | `HolidayCalibrationStage` | dòng 626–876 | 1,5 ngày |
| 5 | `EventCalibrationStage` | dòng 877–918 | 0,5 ngày |
| 6 | `TrainGlobalModelStage` | dòng 919–953 | 0,5 ngày |
| 7 | `PersistResultsStage` | dòng 1861–2102 | 1 ngày |
| 8 | `MonitoringStage` | dòng 2165–2268 | 1 ngày |
| 9 | `BrainInsightsStage` | dòng 2269–2317 | 0,5 ngày |
| 10 | `EnsembleForecastStage` | dòng 997–1860 | **3 ngày** ⚠️ |

Stage 10 là khối lớn nhất (~860 dòng). Tách tiếp thành:
- `_forecast_one_restaurant(ctx, code)` — thân vòng lặp
- `_apply_closure_rules(...)` — 4 mức priority hiện đang lồng nhau
- `_split_by_shift(...)` — phân bổ ca

**Định nghĩa hoàn thành:** `main()` chỉ còn ≈ 20 dòng gọi orchestrator;
mọi stage có complexity < 12 và ít nhất một unit test.

### Giai đoạn 3 — Chất lượng code (tuần 5–8)

| Việc | Từ | Đến |
|---|---|---|
| Thay `except Exception` bằng exception cụ thể | 219 | < 20 |
| Thay `print()` bằng `logger` | 284 | 0 |
| Bổ sung type hint | 48% | ≥ 90% |
| Test coverage | 0% | ≥ 60% |
| Hàm complexity ≥ D | 26 | 0 |
| Hợp nhất `ishushi/` vào `agents/` | 2 nhánh | 1 |

Cách làm cho `except Exception`: bật quy tắc `BLE001` của ruff ở chế độ cảnh báo,
xử lý theo từng module, rồi chuyển sang chế độ lỗi cho module đã xong.

### Giai đoạn 4 — Trưởng thành vận hành (quý 2)

- Repository pattern cho tầng truy cập dữ liệu (mock được trong test)
- Model registry (MLflow) thay vì pickle rời rạc
- Kiểm định dữ liệu đầu vào (Pandera / Great Expectations)
- Metric Prometheus + cảnh báo drift
- Blue-green deploy cho phiên bản mô hình

---

## 5. Quyết định kiến trúc (ADR tóm tắt)

### ADR-001: Pipeline/Stage thay vì hàm tuần tự
**Bối cảnh:** `main()` complexity 355, không test được.
**Quyết định:** mỗi bước là một class có contract rõ ràng.
**Đánh đổi:** thêm một lớp gián tiếp; đổi lại được khả năng test, resume, và
đo thời gian từng bước.

### ADR-002: LLM là lớp tăng cường, không phải phụ thuộc
**Bối cảnh:** LM Studio chạy cục bộ, có thể không khả dụng; inference tốn ~175s/nhà hàng.
**Quyết định:** mọi lỗi LLM đều `RecoverableError`; ensemble chạy được 100% bằng ML.
**Hệ quả:** hệ thống không bao giờ dừng vì LLM; kết quả bị đánh dấu `DEGRADED`.

### ADR-003: Fail-fast khi sai cấu hình
**Bối cảnh:** default hardcode khiến hệ thống kết nối sai host một cách im lặng.
**Quyết định:** `validate_startup_config()` chạy trước mọi thứ; thiếu biến → dừng.
**Đánh đổi:** cần setup rõ ràng hơn; đổi lại loại bỏ cả một lớp lỗi âm thầm.

### ADR-004: QualityFlag trên mọi output
**Bối cảnh:** fallback trước đây không phân biệt được với kết quả bình thường.
**Quyết định:** mọi dự báo mang cờ `OK` / `DEGRADED_*` / `UNRELIABLE`.
**Hệ quả:** người vận hành biết con số nào cần review thủ công.

---

## 6. Chỉ số theo dõi tiến độ

Chạy `make complexity` và `make test` hằng tuần, ghi vào bảng:

| Tuần | Complexity max | `except Exception` | Coverage | Hàm ≥ D |
|---|---|---|---|---|
| Gốc | 355 | 219 | 0% | 26 |
| T1 | | | | |
| T2 | | | | |
| … | | | | |
| Mục tiêu | **< 12** | **< 20** | **≥ 60%** | **0** |
