PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install install-dev install-all test test-slow test-cov test-fast lint type-check format serve train index-lore eval render-ft-results docker-up docker-down clean

# ─── Installation ───────────────────────────────────────────────
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

install-all:
	$(PIP) install -e ".[dev,eval,train]"

# ─── Code Quality ───────────────────────────────────────────────
lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

type-check:
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

# ─── Testing ────────────────────────────────────────────────────
# `make test` skips slow tests by default (intent classifier model download,
# LoRA merge round-trip). Use `make test-slow` to include them.
test:
	pytest tests/ -v --tb=short -m "not slow"

test-slow:
	pytest tests/ -v --tb=short -m "slow"

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-fast:
	pytest tests/ -v --tb=short -m "not slow and not integration"

# ─── Application ────────────────────────────────────────────────
serve:
	uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

# ─── ML Pipeline ────────────────────────────────────────────────
index-lore:
	$(PYTHON) scripts/index_lore.py

train:
	$(PYTHON) src/training/train_lora.py

eval:
	$(PYTHON) scripts/run_evaluation.py

render-ft-results:
	$(PYTHON) scripts/render_eval_comparison.py --ft results/eval_report_ft.json --output results/eval_comparison.md --update-readme

generate-data:
	$(PYTHON) scripts/generate_training_data.py

export-model:
	$(PYTHON) scripts/export_model.py

# ─── Docker ─────────────────────────────────────────────────────
docker-up:
	docker compose -f docker/docker-compose.yml up -d --build

docker-down:
	docker compose -f docker/docker-compose.yml down

# ─── Cleanup ────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .mypy_cache dist build
