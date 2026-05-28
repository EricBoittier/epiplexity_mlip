PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

.PHONY: snakemake-setup snakemake-dryrun snakemake-run

snakemake-setup:
	@if command -v uv >/dev/null 2>&1; then \
		echo "Using uv to create and populate .venv"; \
		uv venv $(VENV_DIR); \
		uv pip install --python $(VENV_PYTHON) -r requirements-snakemake.txt; \
	else \
		echo "uv not found; falling back to python venv + pip"; \
		$(PYTHON) -m venv $(VENV_DIR); \
		$(VENV_PIP) install --upgrade pip; \
		$(VENV_PIP) install -r requirements-snakemake.txt; \
	fi

snakemake-dryrun:
	$(VENV_DIR)/bin/snakemake -n

snakemake-run:
	$(VENV_DIR)/bin/snakemake -j 2
