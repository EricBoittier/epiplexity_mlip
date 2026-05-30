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

When `teacher_noise.enabled: true` in `config/experiments.yaml`, Snakemake runs an additional
variant for every selection with run names suffixed by `teacher_noise` (configurable). Before
teacher training, Gaussian noise is added to train/valid energies and forces with
`std_noise = scale * std(labels)` computed separately for each split and quantity.
Set `teacher_noise.enabled: false` to run only the baseline matrix.

Plot aggregated results (after `experiment_results.json` exists):

- Local (after `make plot-setup` or a working project `.venv`): `make plot-results`
- **SciCORE / cluster:** use the same Python as training (`mmml`), not `epiplexity/.venv` (that venv is often broken on login nodes if it was created elsewhere):

```bash
# (mmml) env active, from repo root:
python -m pip install -r requirements-plot.txt   # once, if needed
python -m src.plot_experiment_results \
  --results-json checkpoints/rmd17_aspirin/experiment_results.json \
  --output-dir checkpoints/rmd17_aspirin/plots
```

Or explicitly: `/scicore/home/meuwly/boitti0000/mmml/.venv/bin/python -m src.plot_experiment_results ...`

Figures are written to `checkpoints/rmd17_aspirin/plots/` (force RMSE, KL divergences, sample force histograms, plus a CSV summary).

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

To run splits 1–5 in one workflow:

- `snakemake --profile profiles/scicore --configfile config/experiments_splits1_5.yaml`

(Uses checkpoint dir from that config file.)

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

Run from `~/epiplexity` with the **mmml** environment (do not use `epiplexity/.venv` on the cluster — it is often broken on login nodes):

```bash
conda activate mmml   # or: source /scicore/home/meuwly/boitti0000/mmml/.venv/bin/activate
cd ~/epiplexity

# optional: install snakemake into mmml once
python -m pip install -r requirements-snakemake.txt

snakemake --profile profiles/scicore -n
snakemake --profile profiles/scicore -j 40
```

Another config (example):

```bash
snakemake --profile profiles/scicore --configfile config/experiments_splits1_5.yaml
```

Use `--configfile`, not `--config`.

### Storage on SciCORE (use group share, not `$HOME`)

Home (`/scicore/home/meuwly/boitti0000/`) has a small quota. Checkpoints fill it quickly with `save_every_epoch: true`.

Use `config/experiments_splits1_5_scicore.yaml` for splits 1–5: heavy Orbax checkpoints on node `/tmp`, Snakemake outputs on shared storage (`outputs.shared_ckpt_root`). Each job syncs `experiment_metadata/<run_name>/` to shared storage when it finishes (and pulls it to `/tmp` on resume).

```bash
mkdir -p /tmp/epiplexity_mlip/rmd17_aspirin_splits1_5/checkpoints
mkdir -p ~/epiplexity_storage/rmd17_aspirin_splits1_5/checkpoints
cd ~/epiplexity
export TMPDIR=/tmp
snakemake --profile profiles/scicore --configfile config/experiments_splits1_5_scicore.yaml -j 40
```

Legacy lab-storage helper (symlinks `.snakemake/` only):

```bash
bash scripts/scicore_find_lab_root.sh
export SCICORE_LAB_ROOT=/scicore/home/meuwly/boitti0000/epiplexity_storage/rmd17_aspirin_splits1_5
bash scripts/scicore_use_lab_storage.sh
source .scicore_lab_env
snakemake --profile profiles/scicore --configfile "${GENERATED_CONFIG}"
```

Snakemake still writes temp job scripts to `.snakemake/` in the repo; that directory must not live on a full `$HOME`. The script above symlinks it to lab storage.

If home is already full, free a little space first:

```bash
rm -rf ~/epiplexity/.snakemake ~/epiplexity/checkpoints
df -h ~
```

Keep the repo clone in `~/epiplexity`; only checkpoints and aggregates go to the lab path. Optionally delete the old home copy after jobs finish:

```bash
rm -rf ~/epiplexity/checkpoints/rmd17_aspirin_splits2_5
```

Compute jobs already use `execution.python_bin` from the YAML (mmml’s Python on SciCORE).

Notes:

- Snakemake 8+ requires executor plugins; this setup uses `cluster-generic` (sbatch/scancel).
- If you updated from an older setup, rerun `make snakemake-setup` to install/update the plugin.
- This profile uses partition-specific QoS for RTX4090: `rtx4090-6hours`.
- The `#SBATCH --array=0-999` line is intentionally not included in the jobscript. Snakemake already submits one Slurm job per workflow job; combining that with a fixed array would create unintended duplicate tasks.
- Logs go to `logs/slurm/`.
- If Snakemake says `Params have changed since last execution`, that is expected after config changes; it will rerun affected jobs.
- `execution.resume: true` (default) makes `run-selection` reuse an existing `result_summary.json` for the same run name, so forced reruns do not retrain completed runs.
