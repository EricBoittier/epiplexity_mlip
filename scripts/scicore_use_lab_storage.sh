#!/usr/bin/env bash
# Point Snakemake metadata and temp files at lab storage (not $HOME).
#
# Usage (from repo root):
#   bash scripts/scicore_find_lab_root.sh          # discover a writable path
#   export SCICORE_LAB_ROOT=/writable/path/epiplexity/rmd17_aspirin_splits1_5
#   source scripts/scicore_use_lab_storage.sh
#
# /scicore/home/meuwly/meuwly is often PI-only — use a path marked WRITABLE above.

set -euo pipefail

_resolve_lab_root() {
  if [[ -n "${SCICORE_LAB_ROOT:-}" ]]; then
    echo "${SCICORE_LAB_ROOT}"
    return 0
  fi
  if [[ -n "${LAB_ROOT:-}" ]]; then
    echo "${LAB_ROOT}"
    return 0
  fi
  local candidates=(
    "/scicore/home/meuwly/${USER}/epiplexity_storage/rmd17_aspirin_splits1_5"
    "/scicore/home/meuwly/${USER}/GROUP/epiplexity/rmd17_aspirin_splits1_5"
    "/scicore/home/meuwly/${USER}/group/epiplexity/rmd17_aspirin_splits1_5"
  )
  local c
  for c in "${candidates[@]}"; do
    if mkdir -p "${c}/.write_test" 2>/dev/null; then
      rmdir "${c}/.write_test"
      echo "$c"
      return 0
    fi
  done
  return 1
}

if ! LAB_ROOT="$(_resolve_lab_root)"; then
  echo "ERROR: No writable SCICORE_LAB_ROOT set and no default path worked." >&2
  echo "Run:  bash scripts/scicore_find_lab_root.sh" >&2
  echo "Then: export SCICORE_LAB_ROOT=/writable/.../epiplexity/rmd17_aspirin_splits1_5" >&2
  return 1 2>/dev/null || exit 1
fi

export SCICORE_LAB_ROOT="${LAB_ROOT}"
mkdir -p "${LAB_ROOT}/snakemake" "${LAB_ROOT}/logs/slurm" "${LAB_ROOT}/tmp"

if [[ -d .snakemake && ! -L .snakemake ]]; then
  echo "Remove home .snakemake first:  rm -rf .snakemake" >&2
  return 1 2>/dev/null || exit 1
fi
rm -f .snakemake
ln -sfn "${LAB_ROOT}/snakemake/.snakemake" .snakemake
mkdir -p "${LAB_ROOT}/snakemake/.snakemake"

mkdir -p logs
if [[ -d logs/slurm && ! -L logs/slurm ]]; then
  echo "Remove home logs/slurm first:  rm -rf logs/slurm" >&2
  return 1 2>/dev/null || exit 1
fi
rm -f logs/slurm
ln -sfn "${LAB_ROOT}/logs/slurm" logs/slurm
mkdir -p "${LAB_ROOT}/logs/slurm"

export TMPDIR="${LAB_ROOT}/tmp"

GENERATED_CONFIG="config/experiments_splits1_5_scicore.generated.yaml"
sed "s|__SCICORE_LAB_ROOT__|${SCICORE_LAB_ROOT}|g" \
  config/experiments_splits1_5_scicore.yaml > "${GENERATED_CONFIG}"

echo "SCICORE_LAB_ROOT=${SCICORE_LAB_ROOT}"
echo ".snakemake -> $(readlink .snakemake)"
echo "logs/slurm -> $(readlink logs/slurm)"
echo "TMPDIR=${TMPDIR}"
echo "Wrote ${GENERATED_CONFIG}"
echo ""
echo "Run:"
echo "  snakemake --profile profiles/scicore --configfile ${GENERATED_CONFIG}"
