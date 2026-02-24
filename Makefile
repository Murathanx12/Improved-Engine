.PHONY: install test run lint clean setup check

# First-time setup: install package + verify
setup: install check

# Install package in editable mode with dev deps
install:
	pip install -e ".[dev]"

# Run the full engine pipeline
run:
	python -m finpredict.main

# Run tests
test:
	pytest tests/ -v --tb=short

# Run tests with coverage
coverage:
	pytest tests/ -v --cov=finpredict --cov-report=term-missing

# Lint with ruff
lint:
	ruff check src/ tests/

# Verify all imports and API connections work
check:
	python test_setup.py

# Clean build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache *.egg-info
	rm -rf data_cache/*.parquet

# Force re-fetch all data (ignore cache)
refresh:
	python -c "from finpredict.data import cached_fetch_all_data; cached_fetch_all_data(force_refresh=True)"
