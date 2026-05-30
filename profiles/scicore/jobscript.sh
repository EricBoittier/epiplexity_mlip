#!/bin/bash
#SBATCH --job-name=GPU_JOB
#SBATCH --time=06:00:00
#SBATCH --qos=rtx4090-6hours
#SBATCH --mem-per-cpu=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

# Avoid brace characters in this file; Snakemake expands it with Python str.format.
export TMPDIR=/tmp
_slurm_tmp="$(printenv SLURM_TMPDIR 2>/dev/null || true)"
if [ -n "$_slurm_tmp" ]; then
  export TMPDIR="$_slurm_tmp"
fi
unset _slurm_tmp
mkdir -p "$TMPDIR"

hostname
which python

{exec_job}
