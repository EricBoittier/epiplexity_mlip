#!/usr/bin/env bash
# Point Snakemake metadata and temp files at lab storage (not $HOME).
# Usage (on login12, from repo root):
#   source scripts/scicore_use_lab_storage.sh
#   # optional: LAB_ROOT=/scicore/home/meuwly/YOUR_GROUP/epiplexity
#
# Default lab root — change GROUP if needed:
LAB_ROOT="${LAB_ROOT:-/scicore/home/meuwly/meuwly/epiplexity/rmd17_aspirin_splits1_5}"

mkdir -p "${LAB_ROOT}/snakemake" "${LAB_ROOT}/logs/slurm"

# Snakemake temp + metadata (fixes "No space left" on ~/.snakemake)
if [[ -L .snakemake ]]; then
  rm .snakemake
elif [[ -d .snakemake ]]; then
  echo "Move or remove existing .snakemake on home first, e.g.:"
  echo "  rm -rf .snakemake"
  return 1 2>/dev/null || exit 1
fi
ln -sfn "${LAB_ROOT}/snakemake/.snakemake" .snakemake
mkdir -p "${LAB_ROOT}/snakemake/.snakemake"

# Slurm logs from profiles/scicore/jobscript.sh
mkdir -p logs
if [[ -L logs/slurm ]]; then
  :
elif [[ -d logs/slurm ]]; then
  echo "Move or remove logs/slurm on home first"
  return 1 2>/dev/null || exit 1
else
  mkdir -p logs
  ln -sfn "${LAB_ROOT}/logs/slurm" logs/slurm
fi
mkdir -p "${LAB_ROOT}/logs/slurm"

export TMPDIR="${LAB_ROOT}/tmp"
mkdir -p "${TMPDIR}"

echo "LAB_ROOT=${LAB_ROOT}"
echo ".snakemake -> $(readlink -f .snakemake 2>/dev/null || readlink .snakemake)"
echo "logs/slurm -> $(readlink -f logs/slurm 2>/dev/null || readlink logs/slurm)"
echo "TMPDIR=${TMPDIR}"
