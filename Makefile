.PHONY: install lint format test test-all precommit clean

VENV_PY := .venv/Scripts/python

install:
	$(VENV_PY) -m pip install -e ".[dev]"
	$(VENV_PY) -m pre_commit install

lint:
	$(VENV_PY) -m ruff check .

format:
	$(VENV_PY) -m ruff format .

test:
	$(VENV_PY) -m pytest -m "not slow"

test-all:
	$(VENV_PY) -m pytest

precommit:
	$(VENV_PY) -m pre_commit run --all-files

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
