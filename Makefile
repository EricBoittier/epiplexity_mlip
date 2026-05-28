PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

.PHONY: snakemake-setup snakemake-dryrun snakemake-run plot-setup plot-results

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
	@mkdir -p logs/slurm

snakemake-dryrun:
	$(VENV_DIR)/bin/snakemake -n

snakemake-run:
	$(VENV_DIR)/bin/snakemake -j 2

# Use active interpreter by default (e.g. mmml on SciCORE). Override: make plot-results PLOT_PYTHON=python3
PLOT_PYTHON ?= python3

plot-setup:
	$(PLOT_PYTHON) -m pip install -r requirements-plot.txt

plot-results: plot-setup
	$(PLOT_PYTHON) -m src.plot_experiment_results
