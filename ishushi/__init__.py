"""
==============================================
ISHUSHI FORECAST MODULE (v2)
==============================================
Module dự đoán cho chuỗi nhà hàng Ishushi:
1. Số khách (guest count) theo ngày
2. Số lượng món ăn theo sap_code
3. ⭐ Feedback loop: cập nhật actual → brain tự học → sửa forecast

Components:
- data_agent.py: Load & transform data từ DB
- forecast_model.py: ML ensemble (XGBoost + LightGBM + CatBoost + Prophet)
- master_file_agent.py: Lưu forecast + actual, tính error
- brain.py: Self-learning từ errors, bias correction
- config.py: SAP codes, parameters
- main.py: Pipeline orchestrator
"""
