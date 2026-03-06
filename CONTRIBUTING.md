# Contributing to FinPredict

Thanks for your interest in contributing! This document explains how to set up your development environment, follow coding standards, and submit changes.

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment and install dev dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

3. Copy `.env.example` to `.env` and fill in your API keys
4. Verify setup: `python test_setup.py`

## Coding Standards

- **Python version**: 3.11+ (targeting 3.12)
- **Line length**: 100 characters max
- **Linter**: [ruff](https://docs.astral.sh/ruff/) -- run `make lint` before committing
- **Tests**: pytest -- run `make test` to execute the test suite
- **Type hints**: Encouraged but not strictly enforced

## Project Conventions

- Configuration goes in `engine_config.yaml`, never hardcoded in source files
- API keys go in `.env` only -- never commit secrets
- All data fetching lives in `src/finpredict/data/`
- ML models live in `src/finpredict/ml/` and must implement `is_trained`, `predict_proba` or `predict` methods
- Use the caching system (`data/cache.py`) for any external API calls

## Making Changes

1. Create a feature branch from `master`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the coding standards above

3. Run tests and linting:
   ```bash
   make lint
   make test
   ```

4. Commit with a clear, descriptive message:
   ```bash
   git commit -m "Add: description of what you added"
   ```

   Commit message prefixes:
   - `Add:` -- new feature or file
   - `Fix:` -- bug fix
   - `Update:` -- enhancement to existing feature
   - `Refactor:` -- code restructuring without behavior change
   - `Docs:` -- documentation only
   - `Test:` -- test additions or changes

5. Push and open a Pull Request against `master`

## Testing

- Unit tests go in `tests/unit/`
- Integration tests go in `tests/integration/`
- Name test files `test_<module>.py`
- Use the fixtures in `tests/conftest.py` for shared setup

Run specific tests:
```bash
pytest tests/unit/test_engine.py -v          # Single file
pytest tests/unit/ -v -k "test_garch"        # By name pattern
pytest tests/ -v --cov=finpredict            # With coverage
```

## Questions?

Open an issue on GitHub if you have questions about the codebase or need help getting started.
