#!/bin/bash
#SBATCH --job-name=GPU_JOB
#SBATCH --time=01:00:00
#SBATCH --qos=gpu6hours
#SBATCH --mem-per-cpu=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

hostname
which python

cd /scicore/home/meuwly/boitti0000/epiplexity && /scicore/home/meuwly/boitti0000/epiplexity/.venv/bin/python -m snakemake --snakefile '/scicore/home/meuwly/boitti0000/epiplexity/Snakefile' --target-jobs 'run_selection:run_name=rmd17_aspirin_split01_gzip_bytes_cart_ws10_st10_seed43' --allowed-rules run_selection --cores 'all' --attempt 1 --force-use-threads  --wait-for-files '/scicore/home/meuwly/boitti0000/epiplexity/.snakemake/tmp.fblvc168' --force --target-files-omit-workdir-adjustment --max-inventory-time 0 --retries 0 --nocolor --no-hooks --nolock --ignore-incomplete --rerun-triggers input code software-env params mtime --conda-frontend 'conda' --shared-fs-usage input-output storage-local-copies persistence source-cache software-deployment software-deployment-cache sources --printshellcmds  --latency-wait 60 --scheduler 'ilp' --local-storage-prefix base64//LnNuYWtlbWFrZS9zdG9yYWdl --scheduler-solver-path '/scicore/home/meuwly/boitti0000/epiplexity/.venv/bin' --runtime-source-cache-path '/scicore/home/meuwly/boitti0000/.cache/snakemake/snakemake/source-cache/snakemake-runtime-cache/tmp_shn8g4f' --default-resources base64//dG1wZGlyPXN5c3RlbV90bXBkaXI= --mode 'remote' && touch '/scicore/home/meuwly/boitti0000/epiplexity/.snakemake/tmp.fblvc168/8.jobfinished' || (touch '/scicore/home/meuwly/boitti0000/epiplexity/.snakemake/tmp.fblvc168/8.jobfailed'; exit 1)

