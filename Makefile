.PHONY: install install-repnet lint format test test-all coverage precommit clean

VENV_PY := .venv/Scripts/python
VENV_TF_PY := .venv-tf/Scripts/python

install:
	$(VENV_PY) -m pip install -e ".[dev]"
	$(VENV_PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(VENV_PY) -m pre_commit install

# RepNet (Phase 10) runs in its own isolated TensorFlow environment, invoked
# via subprocess from the main venv -- see PROGRESS.md Phase 10 / ADR 0003.
install-repnet:
	python -m venv .venv-tf
	$(VENV_TF_PY) -m pip install --upgrade pip
	$(VENV_TF_PY) -m pip install tensorflow opencv-python-headless scipy

lint:
	$(VENV_PY) -m ruff check .

format:
	$(VENV_PY) -m ruff format .

test:
	$(VENV_PY) -m pytest -m "not slow"

test-all:
	$(VENV_PY) -m pytest

coverage:
	$(VENV_PY) -m pytest -m "not slow" --cov=src/hybrid --cov-report=term-missing

precommit:
	$(VENV_PY) -m pre_commit run --all-files

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
