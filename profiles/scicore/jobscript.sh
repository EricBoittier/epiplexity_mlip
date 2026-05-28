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

{exec_job}
