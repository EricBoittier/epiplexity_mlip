PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

.PHONY: snakemake-setup snakemake-dryrun snakemake-run plot-setup plot-results plot-dataset

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

# Official-split dataset KL / pair-distance / gzip analysis (login node; uses mmml)
DATASET_PLOTS_DIR ?= $(HOME)/epiplexity_storage/rmd17_aspirin_dataset_plots
DATASET_DATA_PATH ?= /scicore/home/meuwly/boitti0000/data/rmd17/npz_data/rmd17_aspirin.npz
DATASET_SPLITS_DIR ?=

plot-dataset: plot-setup
	$(PLOT_PYTHON) -m src.plot_dataset_analysis \
		--data-path $(DATASET_DATA_PATH) \
		--splits-dir "$(DATASET_SPLITS_DIR)" \
		--split-ids 1 2 3 4 5 \
		--output-dir $(DATASET_PLOTS_DIR)
