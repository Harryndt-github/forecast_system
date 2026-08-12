# ==============================================================================
# FORECAST SYSTEM — Multi-stage build
# Mục tiêu: image nhỏ, chạy non-root, không chứa build toolchain ở runtime.
# ==============================================================================

# ------------------------------------------------------------------ Stage 1
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Thư viện build cho prophet/lightgbm/catboost
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy manifest trước để tận dụng layer cache
COPY pyproject.toml README.md ./
COPY forecast_system/__init__.py forecast_system/
RUN pip install --upgrade pip && pip install ".[ml,serve]"

# ------------------------------------------------------------------ Stage 2
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="forecast-system" \
      org.opencontainers.image.description="Hệ thống dự báo lượng khách nhà hàng" \
      org.opencontainers.image.source="https://github.com/Harryndt-github/forecast_system"

# Chỉ runtime lib, không có compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production

WORKDIR /app
COPY --chown=appuser:appuser forecast_system/ ./forecast_system/
COPY --chown=appuser:appuser pyproject.toml README.md ./
RUN pip install --no-deps -e . && \
    mkdir -p /app/logs /app/output && chown -R appuser:appuser /app

# SECURITY: không bao giờ chạy container bằng root
USER appuser

EXPOSE 5050

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:5050/healthz || exit 1

# Production dùng gunicorn, KHÔNG dùng Flask dev server
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5050", \
     "--timeout", "300", "--access-logfile", "-", \
     "forecast_system.dashboard.app:app"]
