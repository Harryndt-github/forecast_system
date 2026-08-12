# 🚀 AI FORECAST SYSTEM - TÀI LIỆU HỆ THỐNG

> **Phiên bản:** Phase 8 (Shift-Based Forecast MORNING/EVENING + Dual-Horizon + Ensemble + Self-Correction + Auto-Tuner + Dashboard + Holiday Calibrator + Event Calibrator + RAG/ICL + Neural Corrector + Transfer Learning + Parallel Processing + Booking Data + NeuralProphet Global + YoY Features)  
> **Ngày cập nhật:** 2026-03-28  
> **Mục đích:** Dự đoán lượt khách (guest count) cho 500+ nhà hàng trên toàn quốc, theo 2 chế độ: chi tiết theo ca (MORNING/EVENING) và tổng quan (ngày)  

---

## 📋 MỤC LỤC

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Kiến trúc & Cấu trúc thư mục](#2-kiến-trúc--cấu-trúc-thư-mục)
3. [Pipeline chính (Main Flow)](#3-pipeline-chính)
4. [Các Agent chi tiết](#4-các-agent-chi-tiết)
5. [Models & Algorithms](#5-models--algorithms)
6. [Feature Engineering](#6-feature-engineering)
7. [Ensemble Strategy](#7-ensemble-strategy)
8. [Self-Correction (ForecastBrain)](#8-self-correction-forecastbrain)
9. [Monitoring & Accuracy](#9-monitoring--accuracy)
10. [Baseline Configuration](#10-baseline-configuration)
    - 10.5 [Holiday Impact Calibrator (AUTO-CALIBRATION)](#105-holiday-impact-calibrator-auto-calibration)
11. [Outputs & Deliverables](#11-outputs--deliverables)
12. [Limitations & Hướng phát triển](#12-limitations--hướng-phát-triển)
    - 12.4 [Phase 8: Shift-Based Forecast](#124-phase-8-shift-based-forecast-morningevening)
13. [Changelog & Bugfixes](#13-changelog--bugfixes)

---

## 1. TỔNG QUAN HỆ THỐNG

### 1.1 Mục tiêu
Dự đoán **số lượng khách (guest count)** cho từng nhà hàng, **theo từng giờ**, trong khoảng **31 ngày tới**. Kết quả được sử dụng để lên lịch ca làm việc (shift planning) và quản lý nhân sự.

### 1.2 Quy mô
- **500+ nhà hàng** (nhiều thương hiệu/chuỗi)
- **Dual-Horizon Forecast (Phase 8 — Shift-Based):**
  - **SHORT-TERM (30 ngày đầu):** Forecast theo **2 ca/ngày** — MORNING (8h-15h30) + EVENING (15h30-23h) → **2 rows/ngày/nhà hàng**
  - **LONG-TERM (3 tháng tiếp theo):** Forecast chỉ theo nhà hàng + ngày (daily total) → **1 row/ngày/nhà hàng**
- **Tần suất chạy:** Hàng ngày (có thể cấu hình)
- **Data lookback:** 400 ngày gần nhất

### 1.3 Phân loại hệ thống
> **"AI-Augmented Ensemble Forecasting System with Statistical Self-Correction"**

Hệ thống **KHÔNG phải** là một LLM tự học. Nó là hệ thống lai (hybrid) gồm:
- **ML Models** (XGBoost, CatBoost, LightGBM, RandomForest) → train lại mỗi lần chạy
- **Prophet** → daily trend decomposition
- **LLM** (GPT-oss-20b qua LM Studio) → prompt-based prediction, weights cố định
- **ForecastBrain** → rule-based statistical correction, lưu bias vào JSON

### 1.4 Triết lý cốt lõi
```
"Model dự đoán sai → Brain ghi nhớ sai ở đâu → Lần sau tự sửa"
```

---

## 2. KIẾN TRÚC & CẤU TRÚC THƯ MỤC

### 2.1 Kiến trúc tổng thể

```
┌──────────────────────────────────────────────────────────┐
│                    SCHEDULER (scheduler.py)               │
│            Chạy tự động hàng ngày / theo interval         │
├──────────────────────────────────────────────────────────┤
│                    MAIN ORCHESTRATOR (main.py)             │
│         Điều phối toàn bộ pipeline từ đầu đến cuối        │
├────────┬────────┬─────────┬──────────┬──────────┬────────┤
│  Data  │Analysis│   ML    │    AI    │ Ensemble │ Master │
│ Agent  │ Agent  │Forecast │Forecast  │ Forecast │  File  │
│        │        │ Agent   │  Agent   │  Agent   │ Agent  │
├────────┴────────┴─────────┴──────────┴──────────┴────────┤
│                    FORECAST BRAIN                         │
│         Self-Correction / Strategy Optimizer              │
├──────────────────────────────────────────────────────────┤
│     Auto-Tuner  │  Monitoring Agent  │  New Restaurant    │
├──────────────────────────────────────────────────────────┤
│     Parallel Engine  │  Dashboard (Web UI)                │
├──────────────────────────────────────────────────────────┤
│                    UTILITIES                              │
│     db_utils │ date_utils │ logger                        │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Cấu trúc thư mục

```
forecast_system/
├── main.py                    # Pipeline orchestrator chính
├── scheduler.py               # Chạy tự động theo lịch
├── parallel_engine.py         # Song song hóa forecast
├── requirements.txt           # Dependencies
│
├── config/
│   ├── settings.py            # Tất cả cấu hình tập trung
│   └── .env                   # Credentials (không commit)
│
├── agents/
│   ├── data_agent.py          # Load & clean data từ DB + shift mapping
│   ├── analysis_agent.py      # Phân tích trends, gaps, outliers
│   ├── ml_forecast_agent.py   # Feature engineering cho ML (shift-based)
│   ├── ai_forecast_agent.py   # LLM prediction (LM Studio) + RAG/ICL
│   ├── ensemble_agent.py      # Kết hợp ML + Prophet + NeuralProphet + AI
│   ├── forecast_brain.py      # Self-correction engine
│   ├── holiday_calibrator.py  # Auto-calibrate holiday impact (chain + per-restaurant)
│   ├── event_calibrator.py    # Auto-calibrate special event impact (Phase 8)
│   ├── booking_agent.py       # Load & tổng hợp dữ liệu booking (Phase 8)
│   ├── neuralprophet_agent.py # NeuralProphet global model (Phase 8)
│   ├── rag_forecast_agent.py  # RAG Agent + Knowledge Store (Phase 6+)
│   ├── knowledge_store.py     # Vector knowledge store cho RAG
│   ├── master_file_agent.py   # Quản lý file Excel kết quả (shift-aware)
│   ├── monitoring_agent.py    # Đo lường accuracy
│   ├── auto_tuner.py          # Tối ưu hyperparameters
│   ├── neural_corrector.py    # Neural network corrections (MLP)
│   ├── transfer_learning.py   # Transfer learning between similar restaurants
│   ├── llm_finetuner.py       # LLM Fine-Tuning (QLoRA) — Phase 7
│   └── new_restaurant_agent.py # Xử lý nhà hàng mới
│
├── utils/
│   ├── db_utils.py            # Database connection & queries
│   ├── date_utils.py          # Calendar, holidays, lunar, special events
│   └── logger.py              # Logging system
│
└── dashboard/
    ├── app.py                 # Flask web server (API)
    └── static/
        ├── index.html         # Dashboard UI
        ├── dashboard.js       # Frontend logic
        └── styles.css         # Styling
```

---

## 3. PIPELINE CHÍNH

Mỗi lần chạy `main.py`, hệ thống thực hiện **10+ bước tuần tự**:

```
STEP 1: Database Connection
  └─ Kết nối MySQL (retry 3 lần, timeout 60s)

STEP 2: Load Data
  └─ Load 400 ngày transaction data từ 2 bảng DB
  └─ Load restaurant info (tên, SAP code)

STEP 3: Analysis
  ├─ Xác định nhà hàng active (có data trong 30 ngày gần nhất)
  ├─ Phân tích growth/decline trends (30/60/90 ngày)
  ├─ Phát hiện activity gaps (ngắt quãng hoạt động)
  ├─ Phát hiện outliers (IQR method per weekday)
  ├─ Phân loại nhà hàng (NEW/YOUNG/VOLATILE/HIGH_VOLUME/STANDARD)
  └─ Chọn forecast strategy per restaurant

STEP 4: Update Master File
  └─ Cập nhật Actual_Guest từ DB vào file kết quả cũ (shift-aware)

STEP 4.5: Brain Learning ← CLOSED-LOOP
  ├─ Phân tích errors từ dữ liệu lịch sử
  ├─ Học bias patterns (overall, weekday, shift, holiday)
  ├─ Detect issues (MAPE > 25%)
  ├─ Tính correction factors
  └─ Recommend strategy overrides per restaurant

STEP 5: Prepare Forecast
  ├─ Build danh sách ngày cần forecast
  ├─ Xác định holidays, pre/post holiday, lunar calendar
  └─ Detect holiday closures (pattern-based + actual schedule)

STEP 5.6: Load Lunar NY Closure Schedule
  └─ Override pattern-based detection với closure data từ operations

STEP 5.7: Holiday Impact Calibration (DATA-DRIVEN)
  ├─ Load data lịch sử dịp lễ (VD: Tết 2025)
  ├─ Tính baseline (median cùng thứ, non-holiday) per restaurant
  ├─ Tính ratio actual/baseline cho mỗi ngày pre/post
  ├─ Aggregate bằng median qua 500+ nhà hàng
  └─ Update next_days với calibrated impacts (thay hardcoded)

STEP 5.8: Special Event Calibration (DATA-DRIVEN)
  ├─ Tính impact thực tế cho Valentine, Women's Day, Christmas, v.v.
  ├─ Per-restaurant calibration (ưu tiên) + Aggregate (fallback)
  └─ Override estimated event impacts bằng data thực

STEP 5.9: NeuralProphet Global Model (Train Once)
  └─ Train 1 model cho TẤT CẢ nhà hàng (thay per-restaurant → tránh hang)

STEP 6: Ensemble Forecast Loop (per restaurant)
  ├─ 6.0   Holiday Closure Check → forecast = 0 nếu đóng cửa
  ├─ 6.0.5 New Restaurant → chain-average blend (nếu < 14 ngày data)
  ├─ 6.1   Outlier Cleaning (IQR per weekday)
  ├─ 6.2   Feature Engineering (shift-based, 17+ features)
  ├─ 6.3   AI Forecast (RAG Agent / LM Studio → daily total)
  ├─ 6.4A  SHORT-TERM: Shift-Based Ensemble (ML + Prophet + NeuralProphet + AI → 2 rows/ngày)
  ├─ 6.4B  LONG-TERM: Daily-Only Ensemble (ML + Prophet + NeuralProphet + AI → 1 row/ngày)
  ├─ 6.5   Brain Correction + Neural Corrector + Transfer Learning
  ├─ 6.55  Final Smart Max Cap (safety net per ngày thường/lễ)
  └─ 6.6   Collect Results (shift rows + daily-only rows)

STEP 6.5: Load Booking Data
  └─ Load khách đặt bàn trước → sheet riêng "Booking_Guests"

STEP 7: Save Results
  ├─ Protect historical data (preserve previous run dates)
  ├─ Dedup với shift-aware key: (Restaurant_Code, Date, Hour, Shift)
  ├─ Merge restaurant info (sap_code, restaurant_name)
  ├─ Merge booking daily totals
  └─ Save Master File (Excel multi-sheet + CSV backup)

STEP 8: Summary Report
  └─ In thống kê: success rate, model usage, category breakdown
```

---

## 4. CÁC AGENT CHI TIẾT

### 4.1 DataAgent (`data_agent.py`)
**Trách nhiệm:** Load & clean dữ liệu từ Database

| Chức năng | Mô tả |
|---|---|
| `load_recent_data()` | Load 120 ngày từ 2 bảng: `payment_hub` + `rk_dc`, merge & clean |
| `load_restaurant_info()` | Load tên, SAP code từ `v_dim_restaurant_address` |
| `load_date_range()` | Load data cho khoảng ngày cụ thể (VD: Tết năm trước) |
| `get_hourly_ratios()` | Tính tỷ lệ phân bổ khách theo giờ (weekday vs weekend) |
| `get_weekday_hourly_ratios()` | Tỷ lệ phân bổ theo giờ CHO TỪNG ngày trong tuần |
| `aggregate_shifts()` | Gom kết quả theo ca: Open (8-9), Lunch (10-13), Before Dinner (14-16), Dinner (17-20), Close (21-22) |
| `load_lunar_ny_closures()` | Load lịch đóng cửa Tết từ file Excel |
| `get_active_restaurants()` | Lọc nhà hàng có hoạt động trong N ngày gần nhất |

**Nguồn dữ liệu:**
- `v_fact_db_payment_hub_transactions` → restaurant_code, transaction_id, guest_count, open_time, date
- `v_fact_db_rk_dc_transactions` → tương tự
- `v_dim_restaurant_address` → restaurant info

### 4.2 AnalysisAgent (`analysis_agent.py`)
**Trách nhiệm:** Phân tích dữ liệu lịch sử, phân loại nhà hàng

| Chức năng | Mô tả |
|---|---|
| `calculate_growth_rate()` | Tính tỷ lệ tăng trưởng (so sánh 2 nửa của time window) |
| `detect_activity_gaps()` | Phát hiện ngắt quãng hoạt động (gap ≥ 7 ngày) |
| `should_exclude_restaurant()` | Quyết định loại bỏ (gap > 50% hoặc active < 7 ngày) |
| `detect_outliers()` | IQR method per weekday, threshold = 2.5×IQR |
| `classify_restaurant()` | Phân loại → chọn strategy phù hợp |
| `generate_restaurant_report()` | Báo cáo tổng hợp per restaurant |
| `format_report_for_prompt()` | Format report thành text cho AI prompt |

**Phân loại nhà hàng:**

| Category | Điều kiện | Strategy |
|---|---|---|
| **NEW** | < 14 ngày data | `AI_ONLY` |
| **YOUNG** | 14-45 ngày | `AI_PRIMARY_ML_SECONDARY` |
| **VOLATILE** | CV > 0.5 | `AI_PRIMARY_ML_SECONDARY` |
| **HIGH_VOLUME** | Avg daily > 200 guests | `ML_PRIMARY_AI_VALIDATE` |
| **STANDARD** | Còn lại | `ENSEMBLE_EQUAL` |

**Phân loại xu hướng:**

| Trend | Score |
|---|---|
| STRONG_GROWTH | > +10% |
| MILD_GROWTH | +3% đến +10% |
| STABLE | -3% đến +3% |
| MILD_DECLINE | -10% đến -3% |
| STRONG_DECLINE | < -10% |

### 4.3 MLForecastAgent (`ml_forecast_agent.py`)
**Trách nhiệm:** Feature engineering từ transaction data

| Chức năng | Mô tả |
|---|---|
| `prepare_data()` | Chuyển transaction → hourly time series + 17 features |
| `get_feature_columns()` | Trả về danh sách 17 features |
| `train_and_predict()` | Train single best ML model + predict |
| `save_model_cache()` | Cache model đã train (24h) |
| `load_model_cache()` | Load cached model nếu còn hạn |

### 4.4 AIForecastAgent (`ai_forecast_agent.py`)
**Trách nhiệm:** Dự đoán dùng LLM (GPT-oss-20b qua LM Studio)

| Chức năng | Mô tả |
|---|---|
| `prepare_prompt_data()` | Chuẩn bị 21 ngày history gần nhất |
| `prepare_weighted_prompt_data()` | Statistics 30/60/90 ngày |
| `prepare_enhanced_prompt()` | Prompt enriched với analysis report |
| `generate_forecast()` | Gọi LM Studio API, có retry logic |
| `_build_system_prompt()` | Build system prompt với holiday impact guide |
| `parse_response()` | Extract JSON array từ LLM response |

**Cấu hình LLM (Phase 6 - RAG Agent):**
- **Primary:** Qwen2.5-7B-Instruct (GGUF Q4_K_M, ~4.4GB, local via llama-cpp-python)
- **Fallback 1:** Qwen2.5-1.5B-Instruct (GGUF Q4_K_M, ~1.1GB, local)
- **Fallback 2:** LM Studio server (`http://127.0.0.1:1234/v1`, model: `openai/gpt-oss-20b`)
- Context window: 8192 tokens
- Metal acceleration (Apple Silicon)
- Max retries: 2

**System Prompt bao gồm:**
- Holiday impact guide (Tết: -80~-100%, Quốc Khánh: +10~+30%, v.v.)
- Restaurant context (category, trend, confidence)
- Output format: JSON array `[{"date": "YYYY-MM-DD", "forecast": N}, ...]`

### 4.5 EnsembleForecastAgent (`ensemble_agent.py`)
**Trách nhiệm:** Kết hợp ML Stacking + Prophet + NeuralProphet + AI, phân phối theo ca

**Bao gồm 4 sub-components:**

#### EnsembleMLAgent (ML Stacking)
- **Level 0 (Base Models):** XGBoost (35%), CatBoost (30%), LightGBM (25%), RandomForest (10%)
- **Level 1 (Meta-learner):** Ridge Regression
- **Fallback:** Nếu data quá ít → train chỉ model tốt nhất

#### ProphetDailyAgent
- Dự đoán daily total guests
- Tự phát hiện weekly/yearly seasonality
- Tích hợp Vietnam holidays

#### NeuralProphetGlobalAgent (`neuralprophet_agent.py`) — Phase 8
- **Global model:** Train 1 lần cho TẤT CẢ nhà hàng (dùng restaurant_code làm feature)
- Forecast daily total → dùng cho ensemble
- Tránh hang/timeout khi train per-restaurant

#### EnsembleForecastAgent (Main)
- Kết hợp 4 nguồn: ML + Prophet + NeuralProphet + AI
- **2 chế độ output (Phase 8):**
  - `run_ensemble_forecast()` → **2 rows/ngày** (MORNING + EVENING) cho short-term
  - `run_ensemble_forecast_daily_only()` → **1 row/ngày** cho long-term
- **Shift Distribution (short-term):** Blended ratio = 60% historical weekday pattern + 40% ML predicted ratio
- **Smart Max Cap:** Historical max cap (ngày thường vs holiday/event-specific)
- Tính confidence score (0.0 - 1.0)

### 4.6 ForecastBrain (`forecast_brain.py`)
**Trách nhiệm:** Self-correction engine. Chi tiết tại [Mục 8](#8-self-correction-forecastbrain).

### 4.7 MasterFileAgent (`master_file_agent.py`)
**Trách nhiệm:** Quản lý file Excel kết quả

| Chức năng | Mô tả |
|---|---|
| `load_or_create()` | Load Excel → fallback CSV → tạo mới |
| `update_actuals()` | Cập nhật Actual_Guest từ DB data |
| `save_excel_safely()` | Lưu an toàn: CSV backup → temp file → backup old → move |

**Columns Master File (Phase 8):**
`Forecast_Run_Date, Restaurant_Code, sap_code, restaurant_name, Date, Weekday, Hour, Shift, Final_Predicted_Guests, Actual_Guest, Diff_Guest, Error_%, Is_Holiday, Is_Veg, AI_Raw_Daily_Forecast, Booking_Guests, Forecast_Mode`

> **Lưu ý Phase 8:**
> - Cột `Shift` = `'MORNING'` hoặc `'EVENING'` (short-term) hoặc `NULL` (daily-only long-term)
> - Cột `Hour` = `NULL` cho cả shift-based và daily-only rows (chỉ còn dùng cho legacy hourly rows)
> - Cột `Forecast_Mode` = `'shift'` (short-term 2 ca/ngày) hoặc `'daily_only'` (long-term 1 row/ngày)
> - Cột `Booking_Guests` = số khách đặt bàn trước cho ngày đó
> 
> **Shift Definitions:**
> | Shift | Giờ | Hours |
> |---|---|---|
> | **MORNING** (Ca Sáng) | 8h – 15h30 | 8, 9, 10, 11, 12, 13, 14, 15 |
> | **EVENING** (Ca Tối) | 15h30 – 23h | 16, 17, 18, 19, 20, 21, 22, 23 |

### 4.8 EventCalibrator (`event_calibrator.py`) — Phase 8
**Trách nhiệm:** Tự động tính impact factors cho SPECIAL EVENTS từ data thực tế

| Chức năng | Mô tả |
|---|---|
| `calibrate()` | Tính aggregate + per-restaurant impact cho tất cả events |
| `apply_calibration_to_forecast_days()` | Override default event impacts với calibrated factors |
| `get_calibrated_factor()` | Lấy factor cho event+restaurant cụ thể |

**Priority:** Per-restaurant calibrated > Aggregate calibrated > Default factor

**Events hỗ trợ:** Valentine, Women's Day, Christmas, Children's Day, Teacher's Day, Father's Day, v.v.

### 4.9 BookingAgent (`booking_agent.py`) — Phase 8
**Trách nhiệm:** Load và tổng hợp dữ liệu đặt bàn trước

| Chức năng | Mô tả |
|---|---|
| `load_booking_data()` | Load booking từ DB, loại trừ cancelled bookings |
| `aggregate_booking_summary()` | Tổng hợp theo Restaurant + Date + Hour |
| `get_daily_booking_totals()` | Tóm tắt daily totals để merge vào Master file |

**Nguồn dữ liệu:** `v_fact_db_booking_booking_info`
**Output:** Sheet `Booking_Guests` trong Master file + cột `Booking_Guests` trong Forecast sheet

### 4.10 MonitoringAgent (`monitoring_agent.py`)
**Trách nhiệm:** Đo lường và giám sát accuracy

| Chức năng | Mô tả |
|---|---|
| `calculate_metrics()` | MAE, MAPE, RMSE, Bias, Hit_Rate |
| `calculate_daily_accuracy()` | Accuracy theo ngày |
| `calculate_restaurant_accuracy()` | Accuracy per restaurant |
| `calculate_weekday_accuracy()` | Accuracy per weekday |
| `calculate_hourly_accuracy()` | Accuracy per hour |
| `calculate_run_performance()` | Performance mỗi lần chạy model |
| `calculate_rolling_accuracy()` | Rolling 7d/14d/30d accuracy |
| `compare_ml_vs_ai()` | So sánh ML vs AI accuracy |
| `detect_drift()` | Phát hiện suy giảm accuracy |
| `get_problem_restaurants()` | Tìm nhà hàng accuracy tệ nhất |
| `save_accuracy_snapshot()` | Lưu snapshot hàng ngày (JSON) |
| `generate_full_report()` | Comprehensive accuracy report |
| `save_report_excel()` | Export báo cáo Excel (6 sheets) |

### 4.9 AutoTuner (`auto_tuner.py`)
**Trách nhiệm:** Tự động tối ưu hyperparameters

| Chức năng | Mô tả |
|---|---|
| `walk_forward_cv()` | Walk-forward cross-validation (time-series aware) |
| `tune_with_optuna()` | Bayesian optimization (nếu có Optuna) |
| `tune_random_search()` | Random search (fallback) |
| `tune_all_models()` | Tune tất cả ML models |
| `tune_ensemble_weights()` | Tối ưu ML vs AI weights |

**Strategies:** QUICK (20 trials), STANDARD (50), THOROUGH (100)

### 4.10 NewRestaurantAgent (`new_restaurant_agent.py`)
**Trách nhiệm:** Forecast cho nhà hàng mới (< 14 ngày data)

**Logic:**
1. Detect chain/brand từ tên (VD: "Gogi House Hanoi" → chain "Gogi House")
2. Tìm nhà hàng "anh em" cùng chuỗi
3. Tính chain-average patterns (daily, weekday, hourly)
4. Blend: `weight = own_days / 14` → 1 ngày = 7% own + 93% chain, 14 ngày = 100% own

---

## 5. MODELS & ALGORITHMS

### 5.1 ML Models (Level 0 - Base Models)

| Model | Library | Default Hyperparameters | Weight |
|---|---|---|---|
| **XGBoost** | `xgboost` | n_estimators=150, lr=0.08, max_depth=6, subsample=0.8 | 35% |
| **CatBoost** | `catboost` | iterations=150, lr=0.08, depth=6, l2_leaf_reg=3.0 | 30% |
| **LightGBM** | `lightgbm` | n_estimators=150, lr=0.08, max_depth=6, num_leaves=31 | 25% |
| **RandomForest** | `sklearn` | n_estimators=100, max_depth=10, min_samples_leaf=3 | 10% |

### 5.2 Meta-Learner (Level 1)
- **Ridge Regression** (sklearn)
- Input: predictions từ 4 base models
- Output: final combined prediction
- Fallback: weighted average nếu meta-learner fails

### 5.3 Prophet
- **Facebook Prophet** cho daily trend forecasting
- Seasonality: weekly + yearly
- Holiday effects: Vietnam holidays
- Không predict hourly – chỉ daily total

### 5.4 LLM (AI)
- **Model:** `openai/gpt-oss-20b`
- **Server:** LM Studio local (http://127.0.0.1:1234/v1)
- **Role:** Demand Planner – predict daily total guests
- **Input:** 21-30 ngày history + analysis report + holiday guide
- **Output:** JSON array `[{"date": "...", "forecast": N}]`
- **Lưu ý:** Model weights KHÔNG thay đổi. Chỉ là inference.

---

## 6. FEATURE ENGINEERING

### 6.1 Danh sách 17 Features

| # | Feature | Loại | Mô tả |
|---|---|---|---|
| 1 | `hour` | Time | Giờ trong ngày (8-22) |
| 2 | `weekday` | Time | Ngày trong tuần (0=Mon, 6=Sun) |
| 3 | `month` | Time | Tháng (1-12) |
| 4 | `day_of_month` | Time | Ngày trong tháng (1-31) |
| 5 | `is_weekend` | Time | Cuối tuần (Sat/Sun = 1) |
| 6 | `is_holiday` | Calendar | Ngày lễ VN (0/1) |
| 7 | `is_tet` | Calendar | Tết Nguyên Đán (0/1) |
| 8 | `is_pre_holiday` | Calendar | Trước ngày lễ (0/1) |
| 9 | `is_post_holiday` | Calendar | Sau ngày lễ (0/1) |
| 10 | `holiday_impact` | Calendar | Hệ số ảnh hưởng ngày lễ (float) |
| 11 | `lunar_day` | Lunar | Ngày âm lịch (1-30) |
| 12 | `lunar_month` | Lunar | Tháng âm lịch (1-12) |
| 13 | `is_veg` | Lunar | Ngày ăn chay - Mùng 1, 15 ÂL (0/1) |
| 14 | `lag_7d` | Lag | Guest count cùng giờ, 7 ngày trước |
| 15 | `lag_14d` | Lag | Guest count cùng giờ, 14 ngày trước |
| 16 | `lag_28d` | Lag | Guest count cùng giờ, 28 ngày trước |
| 17 | `rolling_7d_mean` | Rolling | Trung bình daily total 7 ngày gần nhất |

### 6.2 Data Preparation Flow
```
Raw Transactions → Group by (date, hour) → Sum guest_count
→ Add Time Features (hour, weekday, month, is_weekend)
→ Add Holiday Features (is_holiday, is_tet, pre/post, impact)
→ Add Lunar Features (lunar_day, lunar_month, is_veg)
→ Add Lag Features (7d, 14d, 28d same-hour lookback)
→ Add Rolling Mean (7-day daily average)
→ Ready for ML training
```

---

## 7. ENSEMBLE STRATEGY

### 7.1 Strategies & Weights

| Strategy | ML Weight | AI Weight | ML Split | Khi nào dùng |
|---|---|---|---|---|
| `AI_ONLY` | 0% | 100% | – | Data rất ít, ML không train được |
| `AI_PRIMARY_ML_SECONDARY` | 30% | 70% | 18% ML + 12% Prophet | Nhà hàng mới hoặc volatile |
| `ENSEMBLE_EQUAL` | 50% | 50% | 30% ML + 20% Prophet | Mặc định |
| `ENSEMBLE_WEIGHTED` | 50% | 50% | 30% ML + 20% Prophet | ML nhỉnh hơn AI |
| `ML_PRIMARY_AI_VALIDATE` | 70% | 30% | 42% ML + 28% Prophet | Nhiều data, ổn định |

### 7.2 Weight Distribution
```
ML Weight ──┬── 60% → ML Stacking (XGB + CAT + LGBM + RF)
            └── 40% → Prophet Daily

AI Weight ──── 100% → LLM (GPT-oss-20b)
```

### 7.3 Combination Logic (Phase 8)
```python
# 4 nguồn: ML Stacking + Prophet + NeuralProphet + AI
combined = (ml_total × ml_share + prophet_total × prophet_share 
            + neuralprophet_total × np_share + ai_total × ai_share) / total_weight
```
Nếu bất kỳ nguồn nào missing → redistribute weight cho các nguồn còn lại.

### 7.4 Shift Distribution Algorithm (Phase 8 — Short-Term Only)
```
Daily Total (from ensemble)
  │
  ├─ Tính Historical Shift Ratio (60% weight):
  │   ratio = historical shift guests / historical daily total (cùng weekday)
  │
  ├─ Tính ML Predicted Ratio (40% weight):
  │   ratio = ML shift prediction / ML daily total
  │
  ├─ Blended Ratio = 0.6 × historical + 0.4 × ML predicted
  │
  ├─ MORNING guests = Daily Total × Blended MORNING Ratio
  ├─ EVENING guests = Daily Total × Blended EVENING Ratio
  │
  └─ Consistency Check: |MORNING + EVENING - Daily Total| ≤ 10% (target)
```

### 7.5 Smart Max Cap
```
Forecast per shift:
  ├─ Normal days: cap = historical max (cùng weekday + cùng shift, p95)
  └─ Holiday/Events: cap = historical max × event_factor
```

---

## 8. SELF-CORRECTION (ForecastBrain)

### 8.1 Workflow

```
1. LEARN   → Phân tích errors, ghi nhớ bias patterns
2. CORRECT → Trừ bias, apply correction factor vào predictions mới
3. RECOMMEND → So sánh ML vs AI → recommend strategy tối ưu
4. DIAGNOSE → Debug tại sao model sai (khi cần)
```

### 8.2 Memory Structure (`brain_memory.json`)
```json
{
  "version": 2,
  "global_patterns": {
    "holiday_bias_pct": 12.5,
    "weekend_bias": -3.2,
    "weekday_biases": {"Monday": -2.1, "Friday": +1.8},
    "hourly_biases": {"12": +1.5, "19": -0.8}
  },
  "restaurants": {
    "R001": {
      "overall_bias": +3.2,
      "correction_factor": 0.92,
      "weekday_bias": {"Monday": -2.1},
      "hourly_bias": {"10": +1.5},
      "holiday_bias": -5.0,
      "mape_history": [45, 38, 32, 28],
      "best_strategy": "ML_PRIMARY_AI_VALIDATE",
      "ml_mape": 30, "ai_mape": 45,
      "issues": [{"date": "2026-01-20", "type": "HOLIDAY_SPIKE", "error_pct": 35}]
    }
  }
}
```

### 8.3 Correction Pipeline (5 bước)
```
Original Prediction
  │
  ├─ Step 1: × correction_factor (scale ratio, VD: ×0.92)
  ├─ Step 2: − overall_bias × 50% (partial correction)
  ├─ Step 3: − weekday_bias × 40%
  ├─ Step 4: − hourly_bias × 30%
  └─ Step 5: − holiday_bias × 40% (nếu ngày lễ)
  │
  └─ Clamp: ±35% max correction (bình thường)
             ±60% nếu MAPE > 100%
             ±80% nếu MAPE > 200%
  │
  └─ Floor at 0
```

### 8.4 Bias Learning (Exponential Smoothing)
```
new_bias = α × observed_bias + (1-α) × old_bias
α = 0.3 (bình thường)
α = 0.4 (MAPE > 100%)
α = 0.6 (MAPE > 200%)
```

### 8.5 Strategy Recommendation Logic
```
if ml_mape < ai_mape × 0.7  → ML_PRIMARY_AI_VALIDATE
if ai_mape < ml_mape × 0.7  → AI_PRIMARY_ML_SECONDARY
if ml_mape < ai_mape        → ENSEMBLE_WEIGHTED
else                         → ENSEMBLE_EQUAL
```

---

## 9. MONITORING & ACCURACY

### 9.1 Metrics đo lường

| Metric | Công thức | Mục tiêu |
|---|---|---|
| **MAE** | Mean Absolute Error | Càng thấp càng tốt |
| **MAPE** | Mean Absolute Percentage Error | < 25% |
| **RMSE** | Root Mean Squared Error | Càng thấp càng tốt |
| **Bias** | Mean (Predicted - Actual) | Gần 0 |
| **Hit Rate** | % predictions within ±20% or ±3 guests | Càng cao càng tốt |

### 9.2 Drift Detection
- **Alert nếu:** MAPE tăng > 15%, MAE tăng > 30%, Hit Rate giảm > 10 points
- **Retune trigger:** MAPE > 40%
- **So sánh:** Tuần này vs tuần trước

### 9.3 Accuracy Report (Excel – 6 sheets)
1. Overall Summary
2. Per Restaurant
3. Daily Accuracy
4. Weekday Analysis
5. Hourly Analysis
6. Problem Restaurants

---

## 10. BASELINE CONFIGURATION

### 10.1 Forecast Parameters

| Parameter | Value | Mô tả |
|---|---|---|
| `FORECAST_HORIZON` | 31 ngày | (Legacy) Số ngày dự đoán fallback |
| `FORECAST_MONTHS_AHEAD` | 3 tháng | Forecast đến cuối tháng thứ N |
| `SHORT_HORIZON_DAYS` | 30 ngày | Ranh giới Short-term vs Long-term |
| `DATA_LOOKBACK_DAYS` | 400 ngày | Cửa sổ data lịch sử (bao gồm cùng kỳ năm trước) |
| `ROLLING_WINDOW_WEEKS` | 4 tuần | Window cho rolling stats |
| `INACTIVE_THRESHOLD` | 30 ngày | Ngưỡng inactive |
| `START_DATE_DATA` | 2024-01-01 | Ngày bắt đầu data |

**Dual-Horizon Mode (Phase 8 — Shift-Based):**
```
┌────────────────────────────────────────────────────────────┐
│    DUAL-HORIZON FORECAST MODE (Phase 8: Shift-Based)        │
├────────────────────────────────────────────────────────────┤
│ SHORT-TERM (30 ngày đầu):                                  │
│   → Forecast theo 2 CA/NGÀY: MORNING (8-15h30) + EVENING   │
│   → Output: 2 rows/ngày/nhà hàng (1 row per shift)         │
│   → Shift = MORNING hoặc EVENING, Hour = NULL               │
│   → Forecast_Mode = 'shift'                                 │
│   → Dùng cho: Shift planning, nhân sự ngắn hạn             │
├────────────────────────────────────────────────────────────┤
│ LONG-TERM (3 tháng tiếp theo):                              │
│   → Forecast CHỈ theo NHÀ HÀNG + NGÀY (daily total)        │
│   → Output: 1 row/ngày/nhà hàng                            │
│   → Shift = NULL, Hour = NULL                               │
│   → Forecast_Mode = 'daily_only'                            │
│   → Dùng cho: Budget planning, chiến lược dài hạn          │
└────────────────────────────────────────────────────────────┘
```

### 10.2 Brain Thresholds

| Parameter | Value | Mô tả |
|---|---|---|
| `MAPE_TARGET` | 25% | Target MAPE |
| `SIGNIFICANT_BIAS` | 3 guests | Ngưỡng bias cần sửa |
| `MIN_SAMPLES_LEARN` | 10 | Min samples để học |
| `MAX_CORRECTION` | ±35% | Max correction bình thường |
| `BIAS_SMOOTHING` | α = 0.3 | Exponential smoothing |
| `ISSUE_RETENTION_DAYS` | 60 ngày | Giữ issues |

### 10.3 Analysis Thresholds

| Parameter | Value |
|---|---|
| Min gap days | 7 ngày |
| Max gap ratio | 50% |
| Min active days | 7 ngày |
| Outlier IQR threshold | 2.5× |
| New restaurant threshold | < 14 ngày |
| Young restaurant threshold | < 45 ngày |
| Volatile CV threshold | > 0.5 |
| High volume threshold | > 200 guests/day |

### 10.4 Holiday Impact Factors

| Holiday | Default Factor | Closed Likely? |
|---|---|---|
| Tết Nguyên Đán | 0.05 (-95%) | ✅ Yes |
| Quốc Khánh (2/9) | 1.20 (+20%) | ❌ No |
| Giỗ Tổ Hùng Vương | 1.15 (+15%) | ❌ No |
| 30/4 Giải Phóng | 1.20 (+20%) | ❌ No |
| 1/5 Quốc tế Lao Động | 1.20 (+20%) | ❌ No |
| Tết Dương Lịch | 1.10 (+10%) | ❌ No |
| Pre-Tết (Tất niên) | 1.25 (+25%) | ❌ No |
| Post-Tết | 0.70 (-30%) | ❌ No |
| Pre-Holiday chung | 1.15 (+15%) | ❌ No |
| Post-Holiday chung | 0.90 (-10%) | ❌ No |

> **Lưu ý:** Các factor trên là **default fallback**. Kể từ Phase 5.1, hệ thống
> sử dụng **HolidayCalibrator** (`agents/holiday_calibrator.py`) để tự động
> tính impact factor từ data thực tế. Calibrated factors được lưu tại
> `holiday_calibration.json` và override các default trên.

### 10.5 Holiday Impact Calibrator (AUTO-CALIBRATION)

**File:** `agents/holiday_calibrator.py`
**Output:** `holiday_calibration.json`

**Methodology:**
1. Load data lịch sử quanh các dịp lễ (VD: Tết 2025)
2. Với mỗi nhà hàng: tính baseline (median cùng thứ, non-holiday) vs actual
3. Ratio = actual / baseline → aggregate bằng **median** (robust)
4. Lưu calibrated factors vào JSON → `date_utils.py` tự động lookup
5. Fallback: nếu chưa calibrate → dùng default factors

**Pipeline tích hợp:** Step 5.7 trong main.py (sau closure detection, trước ensemble)

**Ưu điểm so với hardcoded:**
- Dựa trên data thực tế, không phải "educated guesses"
- Per-offset granularity (mỗi ngày pre/post có ratio riêng)
- Tự động tính lại khi có data mới (mỗi lần Tết qua)
- Phân biệt open vs closed restaurants

---

## 11. OUTPUTS & DELIVERABLES

### 11.1 Files đầu ra

| File | Mô tả |
|---|---|
| `Master_Forecast_Tracking.xlsx` | File chính: forecast + actuals per restaurant/date/shift (2 sheets: Forecast + Booking_Guests) |
| `Accuracy_Report.xlsx` | Báo cáo accuracy (6 sheets) |
| `brain_memory.json` | Bộ nhớ Brain (bias, corrections, issues) |
| `holiday_calibration.json` | Calibrated holiday impact factors (data-driven) |
| `accuracy_history.json` | Lịch sử accuracy snapshots |
| `logs/forecast_system_YYYY-MM-DD.log` | Log file hàng ngày |

### 11.2 Dashboard (Web UI)
- **URL:** `http://localhost:5050`
- **Tabs:** Overview, Weekday, Hourly, Restaurants, Trends, Models, Alerts, Forecast Viewer
- **Auto-refresh:** Mỗi 5 phút

---

## 12. LIMITATIONS & HƯỚNG PHÁT TRIỂN

### 12.1 Limitations hiện tại
1. ~~**LLM không tự học**~~ → ✅ **Đã giải quyết** (RAG + In-Context Learning, Phase 6)
2. ~~**ML retrain từ scratch**~~ → ✅ **Đã giải quyết** (Online Learning / warm-start, Phase 6)
3. ~~**Brain correction là rule-based**~~ → ✅ **Đã giải quyết** (Neural Corrector MLP, Phase 6)
4. ~~**Không có transfer learning**~~ → ✅ **Đã giải quyết** (Cluster-based Transfer Learning, Phase 6)
5. **Phụ thuộc LM Studio** – cần server local chạy liên tục (có RAG fallback)
6. ~~**Sequential processing**~~ → ✅ **Đã giải quyết** (Parallel Processing tích hợp, Phase 6)
7. ~~**Holiday calibration cần ≥1 năm data**~~ → ✅ **Cải thiện** (Per-restaurant calibration + Transfer Learning hỗ trợ NH mới)

### 12.2 Features mới (Phase 6)

| Feature | Module | Mô tả | Trạng thái |
|---|---|---|---|
| **RAG (Retrieval-Augmented Generation)** | `ai_forecast_agent.py` | LLM đọc brain_memory → tự điều chỉnh prediction | ✅ Đã triển khai |
| **In-Context Learning** | `ai_forecast_agent.py` | Inject bias history vào prompt | ✅ Đã triển khai |
| **Online Learning** | `ml_forecast_agent.py` | Warm-start ML models (XGB/LGBM/RF) thay vì retrain | ✅ Đã triển khai |
| **Neural Corrector** | `neural_corrector.py` | MLP neural network thay thế rule-based corrections | ✅ Đã triển khai |
| **Transfer Learning** | `transfer_learning.py` | Cluster restaurants → share corrections giữa siblings | ✅ Đã triển khai |
| **Parallel Processing** | `main.py` + `parallel_engine.py` | ThreadPool concurrent processing cho forecast loop | ✅ Đã triển khai |
| **Per-Restaurant Holiday Calibration** | `holiday_calibrator.py` | Calibrate holiday impact riêng từng restaurant | ✅ Đã triển khai |
| ~~**Holiday Calibration**~~ | `holiday_calibrator.py` | ~~Auto-calibrate impact factors từ data~~ | ✅ **Đã triển khai (Phase 5.1)** |
| **LLM Fine-Tuning (QLoRA)** | `llm_finetuner.py` | Fine-tune local LLM trên forecast data bằng LoRA adapters | ✅ **Đã triển khai (Phase 7)** |

### 12.3 Phase 7: LLM Fine-Tuning (Domain Adaptation)

**Mục tiêu:** Fine-tune local LLM (Qwen2.5) trên dữ liệu forecast thực tế để model "học" patterns riêng của hệ thống nhà hàng Việt Nam.

**Architecture:**
```
┌─────────────────────────────────────────────────┐
│           LLM Fine-Tuning Pipeline              │
│                                                 │
│   Master File → Training Data Generator         │
│   (actual vs predicted) → ChatML format         │
│                                                 │
│   Base Model (Qwen2.5-1.5B) + QLoRA            │
│   → LoRA Adapter (~50MB)                        │
│   → Fine-tuned inference (priority in fallback) │
│                                                 │
│   Self-Retraining: weekly via auto_retrain      │
└─────────────────────────────────────────────────┘
```

**Modules:**
| File | Mô tả |
|---|---|
| `llm_finetuner.py` | Core fine-tuning module (TrainingDataGenerator, LLMFineTuner, FineTunedModelLoader, FineTuneValidator) |
| `rag_forecast_agent.py` | Updated LocalLLM with fine-tuned model priority in fallback chain |
| `settings.py` | FINETUNE_CONFIG + USE_FINETUNED_LLM toggle |
| `main.py` | Step 4.8 integration |

**CLI Usage:**
```bash
# Check requirements
python -m forecast_system.agents.llm_finetuner check

# Generate training data only
python -m forecast_system.agents.llm_finetuner generate --lookback 90

# Fine-tune model
python -m forecast_system.agents.llm_finetuner train --epochs 5 --lora-r 16

# Check adapter status
python -m forecast_system.agents.llm_finetuner status

# Validate fine-tuned model
python -m forecast_system.agents.llm_finetuner validate
```

**Fallback Chain (updated):**
```
Fine-tuned (LoRA) → GGUF 7B → GGUF 1.5B → LM Studio → Transformers
```

**Dependencies (optional):**
```
pip install peft trl datasets accelerate bitsandbytes
```

### 12.4 Phase 8: Shift-Based Forecast (MORNING/EVENING)

**Mục tiêu:** Chuyển từ hourly forecast (15 rows/ngày) sang shift-based forecast (2 rows/ngày: MORNING + EVENING) để giảm noise, tăng accuracy target 85-90%.

| Feature | Module | Mô tả | Trạng thái |
|---|---|---|---|
| **Shift-Based Forecasting** | `ensemble_agent.py` | Forecast theo 2 ca (MORNING 8-15h30, EVENING 15h30-23h) thay vì hourly | ✅ Đã triển khai |
| **Shift Distribution Algorithm** | `ensemble_agent.py` | Blended ratio (60% historical + 40% ML) phân phối daily → shifts | ✅ Đã triển khai |
| **NeuralProphet Global Model** | `neuralprophet_agent.py` | Train 1 model global cho tất cả nhà hàng (thay per-restaurant) | ✅ Đã triển khai |
| **EventCalibrator** | `event_calibrator.py` | Auto-calibrate special event impacts (Valentine, Women's Day, v.v.) | ✅ Đã triển khai |
| **BookingAgent** | `booking_agent.py` | Load & tổng hợp dữ liệu đặt bàn từ DB | ✅ Đã triển khai |
| **Multi-Sheet Master File** | `master_file_agent.py` | Export Forecast + Booking_Guests sheets | ✅ Đã triển khai |
| **Shift-Aware Actuals** | `master_file_agent.py` | Cập nhật Actual_Guest theo shift (MORNING/EVENING) | ✅ Đã triển khai |
| **Smart Max Cap** | `ensemble_agent.py` | Historical max cap riêng cho ngày thường vs holiday/event | ✅ Đã triển khai |
| **YoY Historical Features** | `data_agent.py` | Lag 364/365 ngày cho cùng kỳ năm trước | ✅ Đã triển khai |

### 12.5 Hướng phát triển tiềm năng

| Hướng | Mô tả | Ưu tiên |
|---|---|---|
| **Ensemble Neural Corrector** | Dùng ensemble of MLP thay 1 model | ⭐ Thấp |
| **Real-time Dashboard** | WebSocket cho live updates | ⭐ Thấp |
| **Multi-target Forecasting** | Predict revenue + guests cùng lúc | ⭐⭐ Trung bình |
| **Distributed Fine-Tuning** | Multi-GPU training cho larger models | ⭐ Thấp |

---

## 13. CHANGELOG & BUGFIXES

### 2026-03-28: Shift Dedup Bug Fix

**Bug:** Output chỉ có 1 ca/ngày (MORNING) thay vì 2 ca (MORNING + EVENING).

**Nguyên nhân:** Bước deduplication trong `main.py` Step 7 dùng key `(Restaurant_Code, Date, Hour)`. Shift-based rows cả MORNING lẫn EVENING đều có `Hour=None` → cùng dedup key `(Res, Date, -999)` → `drop_duplicates` xóa mất 1 row.

**Fix:** Thêm cột `Shift` vào dedup key → `(Restaurant_Code, Date, Hour, Shift)` → cả 2 ca đều được giữ lại.

**File sửa:** `main.py` dòng 1399-1425

### 2026-06-12: Windows Portability & Reliability Hardening

**Bối cảnh:** Hệ thống chạy chính trên PC Windows (clone repo vào thư mục tên
`forecast_system`). Windows mặc định đọc/ghi file text bằng cp1252 thay vì
UTF-8, gây rủi ro lỗi encoding với tiếng Việt.

**Các thay đổi (không ảnh hưởng giá trị forecast):**

1. **Encoding:** Thêm `encoding='utf-8'` vào toàn bộ 31 lệnh `open()` text-mode
   còn thiếu (forecast_brain, correction_validator, auto_tuner,
   shift_residual_corrector, transfer_learning, llm_finetuner, neural_corrector,
   monitoring_agent, ishushi/brain, main.py lock file, scripts, backtest).
   Trước đây nếu đọc lỗi, `except` sẽ âm thầm reset brain memory về rỗng →
   mất toàn bộ kết quả học.
2. **Atomic write brain memory:** `ForecastBrain.save_memory()` và
   `IshushiBrain.save_memory()` ghi ra file `.tmp` rồi `os.replace()` —
   crash giữa chừng không thể làm hỏng `brain_memory.json` /
   `ishushi_brain_memory.json`.
3. **Log level:** Lỗi `CorrectionValidator` (main.py) và segment validation
   checkpoint (forecast_brain.py) nâng từ DEBUG lên WARNING để không hỏng
   âm thầm.
4. **Path:** `scripts/patch_holiday_forecast.py` bỏ đường dẫn hardcode
   `/Users/mdluffy/...`, dùng `Path(__file__).resolve().parents[2]`.
5. **Cấu trúc repo:** Xóa bản copy cũ lồng trong `forecast_system/` (bản macOS,
   dừng cập nhật từ 2026-05-01). Repo root là bản duy nhất. **Bắt buộc clone
   repo vào thư mục tên `forecast_system`** để import
   `from forecast_system....` hoạt động đúng.

**Khuyến nghị môi trường Windows:** đặt biến môi trường `PYTHONUTF8=1`
(system-wide hoặc trong Task Scheduler) để console/redirect không crash vì
emoji trong log (`UnicodeEncodeError` với cp1252).

**Lưu ý đánh giá:** `main.py` (Master_Forecast_Tracking) và `ishushi/`
(Ishushi_Master_Guests/Items) là **2 model độc lập** — luôn đánh giá metrics
riêng từng model, không gộp chung. 25/29 nhà hàng Isushi cũng xuất hiện trong
`Master_Forecast_Tracking`; khi tính metrics cho main model cần loại các mã
Isushi (`ishushi/config.py: ISHUSHI_SAP_CODES`) hoặc tách slice riêng.

---

> **Tài liệu này mô tả hệ thống tại thời điểm Phase 8 (2026-03-28). Mọi thay đổi kiến trúc hoặc model cần được cập nhật tại đây.**
