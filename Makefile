.PHONY: install install-dev install-all test lint format serve train index-lore eval docker-up docker-down clean

# ─── Installation ───────────────────────────────────────────────
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[dev,eval,train]"

# ─── Code Quality ───────────────────────────────────────────────
lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# ─── Testing ────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-fast:
	pytest tests/ -v --tb=short -m "not slow and not integration"

# ─── Application ────────────────────────────────────────────────
serve:
	uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

# ─── ML Pipeline ────────────────────────────────────────────────
index-lore:
	python scripts/index_lore.py

train:
	python src/training/train_lora.py

eval:
	python scripts/run_evaluation.py

generate-data:
	python scripts/generate_training_data.py

export-model:
	python scripts/export_model.py

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
