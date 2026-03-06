.PHONY: install test run lint clean setup check coverage refresh

# First-time setup: install package + verify
setup: install check

# Install package in editable mode with dev deps
install:
	pip install -e ".[dev]"

# Run the full engine pipeline
# Uses PYTHONPATH so it works even without 'pip install -e .'
run:
	PYTHONPATH=src python -m finpredict.main

# Run tests
test:
	PYTHONPATH=src pytest tests/ -v --tb=short

# Run tests with coverage
coverage:
	PYTHONPATH=src pytest tests/ -v --cov=finpredict --cov-report=term-missing

# Lint with ruff
lint:
	ruff check src/ tests/

# Verify all imports and API connections work
check:
	PYTHONPATH=src python test_setup.py

# Clean build artifacts and cached data
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache *.egg-info build dist
	rm -rf data_cache/*.parquet

# Force re-fetch all data (ignore cache)
refresh:
	PYTHONPATH=src python -c "from finpredict.data import cached_fetch_all_data; cached_fetch_all_data(force_refresh=True)"
