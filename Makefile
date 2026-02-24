.PHONY: install test run lint clean

install:
	pip install -e ".[dev]" --break-system-packages

test:
	pytest tests/ -v --tb=short

run:
	python -m finpredict.main

lint:
	ruff check src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache