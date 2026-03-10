.PHONY: install lint typecheck test format all

# Use only real executables from PATH (avoid shell functions/aliases that `command -v` can return).
PYRIGHT := $(shell if [ -x .venv/bin/pyright ]; then echo .venv/bin/pyright; else p=$$(command -v pyright 2>/dev/null); echo "$$p" | grep -E '^/' || true; fi)
PYTEST := $(shell if [ -x .venv/bin/pytest ]; then echo .venv/bin/pytest; else p=$$(command -v pytest 2>/dev/null); echo "$$p" | grep -E '^/' || true; fi)

install:
	pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	@if [ -z "$(PYRIGHT)" ]; then \
		echo "pyright not found. Run: pip install -e \".[dev]\" (or activate .venv)"; \
		exit 1; \
	fi
	$(PYRIGHT)

test:
	@if [ -z "$(PYTEST)" ]; then \
		echo "pytest not found. Run: pip install -e \".[dev]\" (or activate .venv)"; \
		exit 1; \
	fi
	PYTHONPATH=src $(PYTEST) -v --tb=short

all: lint typecheck test
