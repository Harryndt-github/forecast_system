# ==============================================================================
# FORECAST SYSTEM — lệnh phát triển chuẩn hoá
# Chạy `make` hoặc `make help` để xem danh sách.
# ==============================================================================
.DEFAULT_GOAL := help
.PHONY: help setup install lint format typecheck security test test-fast \
        coverage check run dashboard docker-build docker-up clean audit

PYTHON  := python3
PKG     := forecast_system

help:  ## Hiện danh sách lệnh
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- Thiết lập
setup:  ## Tạo venv, cài dependency dev, bật pre-commit
	$(PYTHON) -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -e ".[dev,ml,llm]"
	./.venv/bin/pre-commit install
	./.venv/bin/pre-commit install --hook-type commit-msg
	@test -f config/.env || cp .env.example config/.env
	@echo "✅ Xong. Điền giá trị vào config/.env rồi chạy: make check"

install:  ## Cài package ở chế độ editable
	pip install -e ".[dev]"

# ------------------------------------------------------------ Chất lượng code
lint:  ## Ruff lint
	ruff check $(PKG) tests

format:  ## Ruff format (tự sửa)
	ruff format $(PKG) tests
	ruff check --fix $(PKG) tests

typecheck:  ## Mypy
	mypy $(PKG)

complexity:  ## Liệt kê hàm quá phức tạp
	radon cc $(PKG) -s -n C

# ------------------------------------------------------------------ Bảo mật
security:  ## Bandit + gitleaks + pip-audit
	bandit -r $(PKG) -c pyproject.toml -ll
	gitleaks detect --source . --verbose --redact
	pip-audit --desc

audit: security  ## Alias của security

# -------------------------------------------------------------------- Test
test:  ## Toàn bộ test + coverage
	pytest

test-fast:  ## Chỉ unit test, bỏ qua test chậm
	pytest -m "unit and not slow" --no-cov -q

coverage:  ## Báo cáo coverage dạng HTML
	pytest --cov-report=html
	@echo "Mở htmlcov/index.html"

# ------------------------------------------------------- Cổng kiểm tra tổng
check: lint typecheck security test  ## Chạy TOÀN BỘ cổng chất lượng (giống CI)
	@echo "✅ Tất cả cổng chất lượng đã pass"

# --------------------------------------------------------------------- Chạy
run:  ## Chạy pipeline forecast (mode=daily)
	forecast-run --mode daily

run-full:  ## Chạy pipeline forecast (mode=full, 3 tháng)
	forecast-run --mode full

dashboard:  ## Chạy dashboard dev server (loopback)
	$(PYTHON) -m $(PKG).dashboard.app

# ------------------------------------------------------------------- Docker
docker-build:  ## Build image
	docker build -t forecast-system:local .

docker-up:  ## Chạy stack bằng docker compose
	docker compose up -d

docker-batch:  ## Chạy batch forecast một lần trong container
	docker compose --profile batch run --rm forecast-batch

# -------------------------------------------------------------------- Dọn
clean:  ## Xoá cache và file tạm
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml \
	       build dist *.egg-info
