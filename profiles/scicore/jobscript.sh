#!/bin/bash
#SBATCH --job-name=GPU_JOB
#SBATCH --time=03:00:00
#SBATCH --qos=rtx4090-6hours
#SBATCH --mem-per-cpu=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

# Double braces: Snakemake treats {{ as literal { in the submitted script.
export TMPDIR="${{TMPDIR:-/tmp}}"
mkdir -p "${{TMPDIR}}"

hostname
which python

{exec_job}
