"""
Dashboard API Server cho Forecast System.
Dùng Flask cung cấp REST API + serve static HTML dashboard.
"""

import os
import hmac
import json
import datetime
import functools
import traceback

from flask import Flask, jsonify, send_from_directory, request
import pandas as pd
import numpy as np

# Package đã cài bằng `pip install -e .` -> không cần sys.path hack nữa.
from forecast_system.config.settings import (
    CURRENT_DATE, MASTER_FILE_NAME, PROJECT_ROOT,
    ACCURACY_REPORT_FILE, ACCURACY_HISTORY_FILE,
    DASHBOARD_HOST, DASHBOARD_PORT, DASHBOARD_DEBUG,
    DASHBOARD_API_KEY, DASHBOARD_AUTH_ENABLED, DASHBOARD_CORS_ORIGINS,
    IS_PRODUCTION,
)
from forecast_system.utils.logger import get_logger

logger = get_logger('dashboard')

app = Flask(__name__, static_folder='static')

# Giới hạn kích thước request để chặn DoS đơn giản
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1 MB


# ==========================================
# SECURITY LAYER
# ==========================================

def require_api_key(view):
    """
    Bảo vệ endpoint bằng API key (header `X-API-Key` hoặc `Authorization: Bearer <key>`).

    Dùng hmac.compare_digest để so sánh constant-time, tránh timing attack.
    Có thể tắt ở môi trường dev qua DASHBOARD_AUTH_ENABLED=false.
    """
    @functools.wraps(view)
    def wrapper(*args, **kwargs):
        if not DASHBOARD_AUTH_ENABLED:
            return view(*args, **kwargs)

        supplied = request.headers.get('X-API-Key', '')
        if not supplied:
            auth = request.headers.get('Authorization', '')
            if auth.startswith('Bearer '):
                supplied = auth[7:]

        if not DASHBOARD_API_KEY or not hmac.compare_digest(supplied, DASHBOARD_API_KEY):
            logger.warning(
                "Unauthorized API access | path=%s | ip=%s",
                request.path, request.remote_addr,
            )
            return jsonify({'error': 'unauthorized'}), 401

        return view(*args, **kwargs)

    return wrapper


@app.after_request
def _set_security_headers(response):
    """Security headers tối thiểu theo khuyến nghị OWASP."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Cache-Control'] = 'no-store'
    if IS_PRODUCTION:
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

    origin = request.headers.get('Origin')
    if origin and origin in DASHBOARD_CORS_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Vary'] = 'Origin'
    return response


@app.errorhandler(Exception)
def _handle_unexpected(exc):
    """
    SECURITY: không bao giờ trả traceback/thông điệp lỗi gốc ra client.
    Chi tiết chỉ ghi vào log phía server.
    """
    logger.exception("Unhandled error on %s: %s", request.path, exc)
    return jsonify({'error': 'internal server error'}), 500


@app.route('/healthz')
def healthz():
    """Health check công khai - không lộ dữ liệu nghiệp vụ."""
    return jsonify({'status': 'ok', 'date': str(CURRENT_DATE)})


# ==========================================
# HELPERS
# ==========================================

def _load_master_safe():
    """Load master file hoặc return empty."""
    try:
        from forecast_system.agents.master_file_agent import MasterFileAgent
        df = MasterFileAgent.load_or_create(MASTER_FILE_NAME)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        return df
    except Exception as e:
        logger.warning(f"Failed to load master file: {e}")
        return pd.DataFrame()


def _safe_json(obj):
    """Convert numpy/pandas types to JSON-safe types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return round(float(obj), 2) if not np.isnan(obj) else None
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime.date, datetime.datetime)):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    elif pd.isna(obj):
        return None
    return obj


# ==========================================
# STATIC FILES (Dashboard UI)
# ==========================================

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)


# ==========================================
# API ENDPOINTS
# ==========================================

@app.route('/api/overview')
@require_api_key
def api_overview():
    """Tổng quan hệ thống."""
    try:
        from forecast_system.agents.monitoring_agent import MonitoringAgent
        
        df = _load_master_safe()
        if df.empty:
            return jsonify({'error': 'No data available'}), 404
        
        overall = MonitoringAgent.calculate_metrics(df)
        rolling = MonitoringAgent.calculate_rolling_accuracy(df)
        
        data = {
            'date': str(CURRENT_DATE),
            'total_rows': len(df),
            'restaurants': df['Restaurant_Code'].nunique() if 'Restaurant_Code' in df.columns else 0,
            'overall': overall.to_dict('records')[0] if not overall.empty else {},
            'rolling': {
                w: m.to_dict('records')[0] if not m.empty else {}
                for w, m in rolling.items()
            },
        }
        return jsonify(_safe_json(data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/restaurants')
@require_api_key
def api_restaurants():
    """Accuracy per restaurant."""
    try:
        from forecast_system.agents.monitoring_agent import MonitoringAgent
        
        df = _load_master_safe()
        if df.empty:
            return jsonify([])
        
        metrics = MonitoringAgent.calculate_restaurant_accuracy(df)
        return jsonify(_safe_json(metrics.to_dict('records')))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/daily')
@require_api_key
def api_daily():
    """Daily accuracy trend."""
    try:
        from forecast_system.agents.monitoring_agent import MonitoringAgent
        
        df = _load_master_safe()
        if df.empty:
            return jsonify([])
        
        daily = MonitoringAgent.calculate_daily_accuracy(df)
        return jsonify(_safe_json(daily.to_dict('records')))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/weekday')
@require_api_key
def api_weekday():
    """Weekday accuracy analysis."""
    try:
        from forecast_system.agents.monitoring_agent import MonitoringAgent
        
        df = _load_master_safe()
        if df.empty:
            return jsonify([])
        
        wd = MonitoringAgent.calculate_weekday_accuracy(df)
        return jsonify(_safe_json(wd.to_dict('records')))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/hourly')
@require_api_key
def api_hourly():
    """Hourly accuracy analysis."""
    try:
        from forecast_system.agents.monitoring_agent import MonitoringAgent
        
        df = _load_master_safe()
        if df.empty:
            return jsonify([])
        
        hourly = MonitoringAgent.calculate_hourly_accuracy(df)
        return jsonify(_safe_json(hourly.to_dict('records')))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/drift')
@require_api_key
def api_drift():
    """Drift detection results."""
    try:
        from forecast_system.agents.monitoring_agent import MonitoringAgent
        
        df = _load_master_safe()
        if df.empty:
            return jsonify({})
        
        drift = MonitoringAgent.detect_drift(df)
        return jsonify(_safe_json(drift))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/problems')
@require_api_key
def api_problems():
    """Problem restaurants."""
    try:
        from forecast_system.agents.monitoring_agent import MonitoringAgent
        
        df = _load_master_safe()
        if df.empty:
            return jsonify({})
        
        problems = MonitoringAgent.get_problem_restaurants(df)
        return jsonify(_safe_json(problems))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/comparison')
@require_api_key
def api_comparison():
    """ML vs AI model comparison."""
    try:
        from forecast_system.agents.monitoring_agent import MonitoringAgent
        
        df = _load_master_safe()
        if df.empty:
            return jsonify({})
        
        comp = MonitoringAgent.compare_ml_vs_ai(df)
        return jsonify(_safe_json(comp))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history')
@require_api_key
def api_history():
    """Accuracy history over time."""
    try:
        if os.path.exists(ACCURACY_HISTORY_FILE):
            with open(ACCURACY_HISTORY_FILE, 'r') as f:
                history = json.load(f)
            return jsonify(_safe_json(history))
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast/<restaurant_code>')
@require_api_key
def api_forecast_detail(restaurant_code):
    """Forecast detail cho 1 nhà hàng cụ thể."""
    try:
        df = _load_master_safe()
        if df.empty:
            return jsonify({'error': 'No data'}), 404
        
        df_res = df[df['Restaurant_Code'] == restaurant_code.upper()].copy()
        if df_res.empty:
            return jsonify({'error': f'Restaurant {restaurant_code} not found'}), 404
        
        # Recent forecasts vs actuals
        recent = df_res.sort_values('Date', ascending=False).head(200)
        
        records = []
        for _, row in recent.iterrows():
            records.append({
                'date': str(row.get('Date', '')),
                'hour': int(row.get('Hour', 0)),
                'predicted': row.get('Final_Predicted_Guests'),
                'actual': row.get('Actual_Guest'),
                'error_pct': row.get('Error_%'),
                'weekday': row.get('Weekday', ''),
            })
        
        return jsonify(_safe_json(records))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast/upcoming')
@require_api_key
def api_forecast_upcoming():
    """
    Forecast tương lai — số khách dự đoán cho các ngày sắp tới.
    Query params: ?restaurant=R001 (optional, nếu không có → tất cả)
    """
    try:
        df = _load_master_safe()
        if df.empty:
            return jsonify({'error': 'No forecast data available'}), 404
        
        today = CURRENT_DATE
        
        # Filter future dates (or today onwards)
        if 'Date' in df.columns:
            df['_date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
            df_future = df[df['_date'] >= today].copy()
        else:
            return jsonify({'error': 'No Date column'}), 404
        
        # Optional: filter by restaurant
        res_filter = request.args.get('restaurant', '').strip().upper()
        if res_filter:
            df_future = df_future[df_future['Restaurant_Code'] == res_filter]
        
        if df_future.empty:
            return jsonify([])
        
        # Sort by date, hour
        df_future = df_future.sort_values(['Restaurant_Code', 'Date', 'Hour'])
        
        records = []
        for _, row in df_future.iterrows():
            records.append({
                'restaurant': row.get('Restaurant_Code', ''),
                'date': str(row.get('Date', '')),
                'weekday': row.get('Weekday', ''),
                'hour': int(row.get('Hour', 0)),
                'predicted': row.get('Final_Predicted_Guests'),
                'confidence': row.get('Confidence'),
                'strategy': row.get('Strategy', ''),
                'is_holiday': bool(row.get('Is_Holiday', False)),
            })
        
        return jsonify(_safe_json(records))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast/summary')
@require_api_key
def api_forecast_summary():
    """
    Tổng hợp forecast theo ngày cho từng nhà hàng.
    Trả về tổng predicted guests/ngày — dễ xem hơn hourly.
    """
    try:
        df = _load_master_safe()
        if df.empty:
            return jsonify([])
        
        if 'Date' not in df.columns or 'Final_Predicted_Guests' not in df.columns:
            return jsonify([])
        
        df['_date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        
        # Optional: filter future only
        scope = request.args.get('scope', 'all')  # 'future' or 'all'
        if scope == 'future':
            df = df[df['_date'] >= CURRENT_DATE]
        
        # Optional: filter by restaurant
        res_filter = request.args.get('restaurant', '').strip().upper()
        if res_filter:
            df = df[df['Restaurant_Code'] == res_filter]
        
        if df.empty:
            return jsonify([])
        
        # Group by restaurant + date
        agg_dict = {
            'Final_Predicted_Guests': ('Final_Predicted_Guests', 'sum'),
            'hours_count': ('Hour', 'count'),
        }
        if 'Actual_Guest' in df.columns:
            agg_dict['total_actual'] = ('Actual_Guest', 'sum')
        if 'Confidence' in df.columns:
            agg_dict['avg_confidence'] = ('Confidence', 'mean')
        
        grouped = df.groupby(['Restaurant_Code', '_date']).agg(**agg_dict).reset_index()
        grouped.rename(columns={'Final_Predicted_Guests': 'total_predicted'}, inplace=True)
        
        grouped = grouped.sort_values(['Restaurant_Code', '_date'])
        
        records = []
        for _, row in grouped.iterrows():
            actual = row.get('total_actual', None)
            predicted = row.get('total_predicted', 0)
            has_actual = pd.notna(actual) and actual > 0
            conf = row.get('avg_confidence', None)
            
            rec = {
                'restaurant': row['Restaurant_Code'],
                'date': str(row['_date']),
                'predicted_total': int(round(predicted)) if pd.notna(predicted) else 0,
                'actual_total': int(round(actual)) if has_actual else None,
                'confidence': round(conf, 2) if pd.notna(conf) else None,
                'hours': int(row.get('hours_count', 0)),
            }
            
            if has_actual and predicted > 0:
                rec['error_pct'] = round(abs(predicted - actual) / actual * 100, 1)
                rec['accuracy'] = round(max(0, 100 - rec['error_pct']), 1)
            
            records.append(rec)
        
        return jsonify(_safe_json(records))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/restaurants/list')
@require_api_key
def api_restaurant_list():
    """Danh sách restaurant codes (cho dropdown / search)."""
    try:
        df = _load_master_safe()
        if df.empty or 'Restaurant_Code' not in df.columns:
            return jsonify([])
        
        codes = sorted(df['Restaurant_Code'].dropna().unique().tolist())
        return jsonify(codes)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



def run_dashboard(host=None, port=None, debug=None):
    """
    Khởi động dashboard (DEV ONLY).

    Ở production dùng WSGI server thật:
        gunicorn -w 4 -b 127.0.0.1:5050 forecast_system.dashboard.app:app
    và đặt sau nginx/traefik có TLS. Flask dev server không chịu được tải
    và không an toàn khi expose trực tiếp.
    """
    host = host or DASHBOARD_HOST
    port = port or DASHBOARD_PORT
    debug = DASHBOARD_DEBUG if debug is None else debug

    if IS_PRODUCTION:
        raise RuntimeError(
            "Không chạy Flask dev server ở production. Dùng gunicorn/uwsgi."
        )
    if not DASHBOARD_AUTH_ENABLED:
        logger.warning("⚠️  API authentication ĐANG TẮT - chỉ dùng ở máy dev.")

    logger.info("🌐 Dashboard starting at http://%s:%s (auth=%s)",
                host, port, DASHBOARD_AUTH_ENABLED)
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Forecast Dashboard (dev server)')
    parser.add_argument('--port', type=int, default=None,
                        help='Default: DASHBOARD_PORT env (5050)')
    parser.add_argument('--host', default=None,
                        help='Default: DASHBOARD_HOST env (127.0.0.1). '
                             'KHÔNG dùng 0.0.0.0 nếu chưa có reverse proxy.')
    parser.add_argument('--debug', action='store_true', default=None)
    args = parser.parse_args()

    run_dashboard(host=args.host, port=args.port, debug=args.debug)
