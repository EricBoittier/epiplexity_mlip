# epiplexity_mlip

## Run experiments with Snakemake

1. Edit matrix/settings in `config/experiments.yaml`.
2. Dry-run the DAG:
   - `snakemake -n`
3. Run locally (example with 2 parallel jobs):
   - `snakemake -j 2`

Artifacts:
- Per-selection results: `checkpoints/rmd17_aspirin/experiment_metadata/<run_name>/result_summary.json`
- Aggregated results: `checkpoints/rmd17_aspirin/experiment_results.json`

Useful commands:
- Run one target only (example):
  - `snakemake checkpoints/rmd17_aspirin/experiment_results.json -j 1`
- Force re-run aggregation:
  - `snakemake -R aggregate -j 1`

## Reproducible setup (recommended)

Bootstrap with `uv` if available (automatic pip fallback):

- `make snakemake-setup`

Then run via the project virtualenv:

- `make snakemake-dryrun`
- `make snakemake-run`

Manual equivalent:

- `python3 -m venv .venv`
- `.venv/bin/pip install -r requirements-snakemake.txt`
- `.venv/bin/snakemake -n`
- `.venv/bin/snakemake -j 2`
