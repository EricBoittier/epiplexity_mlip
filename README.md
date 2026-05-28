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

When running on cluster, workflow jobs use `execution.python_bin` from `config/experiments.yaml` (default: `.venv/bin/python`) to ensure the compute nodes use the same environment.

## Run on SciCORE Slurm

This repository includes a Snakemake profile at `profiles/scicore/` with Slurm headers matching your template:

- `--job-name=GPU_JOB`
- `--time=01:00:00`
- `--qos=rtx4090-6hours`
- `--mem-per-cpu=20G`
- `--ntasks=1`
- `--cpus-per-task=2`
- `--partition=rtx4090`
- `--gres=gpu:1`

Run:

- `.venv/bin/snakemake --profile profiles/scicore`

Notes:

- Snakemake 8+ requires executor plugins; this setup uses `cluster-generic` (sbatch/scancel).
- If you updated from an older setup, rerun `make snakemake-setup` to install/update the plugin.
- This profile uses partition-specific QoS for RTX4090: `rtx4090-6hours`.
- The `#SBATCH --array=0-999` line is intentionally not included in the jobscript. Snakemake already submits one Slurm job per workflow job; combining that with a fixed array would create unintended duplicate tasks.
- Logs go to `logs/slurm/`.
- If Snakemake says `Params have changed since last execution`, that is expected after config changes; it will rerun affected jobs.
