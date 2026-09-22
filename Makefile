# Convenience targets. Works with GNU make on Linux/macOS and Git Bash on Windows.
PY ?= .venv/bin/python
ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
endif
SURVPATH := data/survpath_repo

.PHONY: help venv install survpath dummy-data test test-all smoke tables lint clean

help:
	@echo "make venv        create .venv with Python 3.12 (uv)"
	@echo "make install     install requirements (CPU torch; install a CUDA wheel yourself for GPUs)"
	@echo "make survpath    clone the SurvPath repo (splits, metadata, RNA, pathway compositions)"
	@echo "make dummy-data  random BLCA features so tests and --smoke run without TCGA data"
	@echo "make test        fast unit tests (~20 s)"
	@echo "make test-all    + end-to-end trainer tests on the dummy data (~10 min CPU)"
	@echo "make smoke       one-epoch smoke run of the full pipeline on the dummy data"
	@echo "make tables      regenerate every table in REPORT.md from the archived results -> results/final/"

venv:
	uv venv --python 3.12 .venv

install:
	uv pip install --python $(PY) torch --index-url https://download.pytorch.org/whl/cpu
	uv pip install --python $(PY) -r requirements.txt einops

survpath: $(SURVPATH)/splits

$(SURVPATH)/splits:
	git clone --depth 1 https://github.com/mahmoodlab/SurvPath $(SURVPATH)

dummy-data: survpath
	$(PY) scripts/make_dummy_embeddings.py --cancer blca

test:
	$(PY) -m pytest -m "not slow"

test-all:
	$(PY) -m pytest

smoke:
	$(PY) -m src.training.train --config configs/blca_hybrid_v2.yaml --smoke

tables:
	bash scripts/reproduce_tables.sh

lint:
	$(PY) -m ruff check .

clean:
	rm -rf outputs_smoke .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
