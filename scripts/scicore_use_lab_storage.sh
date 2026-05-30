#!/usr/bin/env bash
# Point Snakemake metadata and temp files at lab storage or /tmp (not $HOME).
#
# Checkpoints on node /tmp (see config/experiments_splits1_5_scicore.yaml):
#   export TMPDIR=/tmp
#   snakemake --profile profiles/scicore --configfile config/experiments_splits1_5_scicore.yaml
#
# Lab storage + generated config (legacy):
#   export SCICORE_LAB_ROOT=/scicore/home/meuwly/boitti0000/epiplexity_storage/rmd17_aspirin_s1to5_ws_metrics_noise
#   bash scripts/scicore_use_lab_storage.sh
#   source .scicore_lab_env
#
# Or: set +e && source scripts/scicore_use_lab_storage.sh

_relocate_to_symlink() {
  local path="$1"
  local target="$2"
  local parent
  parent="$(dirname "$path")"
  mkdir -p "$parent" "$(dirname "$target")" "$target"

  if [[ -L "$path" ]]; then
    rm -f "$path"
  elif [[ -d "$path" ]]; then
    local bak="${path}.bak.$(date +%Y%m%d_%H%M%S)"
    echo "Moving ${path} -> ${bak}"
    if ! mv "$path" "$bak" 2>/dev/null; then
      echo "WARN: could not move ${path} (NFS busy?). Try: scancel -u \$USER; wait; rm -rf ${path}" >&2
      return 1
    fi
  elif [[ -e "$path" ]]; then
    rm -f "$path"
  fi

  ln -sfn "$target" "$path"
  return 0
}

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
    "/tmp/epiplexity_mlip/rmd17_aspirin_s1to5_ws_metrics_noise"
    "/scicore/home/meuwly/${USER}/epiplexity_storage/rmd17_aspirin_s1to5_ws_metrics_noise"
    "/scicore/home/meuwly/${USER}/GROUP/epiplexity/rmd17_aspirin_s1to5_ws_metrics_noise"
    "/scicore/home/meuwly/${USER}/group/epiplexity/rmd17_aspirin_s1to5_ws_metrics_noise"
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

_main() {
  local rc=0
  if ! LAB_ROOT="$(_resolve_lab_root)"; then
    echo "ERROR: Set SCICORE_LAB_ROOT or run: bash scripts/scicore_find_lab_root.sh" >&2
    return 1
  fi

  export SCICORE_LAB_ROOT="${LAB_ROOT}"
  mkdir -p "${LAB_ROOT}/snakemake/.snakemake" "${LAB_ROOT}/logs/slurm" "${LAB_ROOT}/tmp" \
    "${LAB_ROOT}/checkpoints"

  _relocate_to_symlink ".snakemake" "${LAB_ROOT}/snakemake/.snakemake" || rc=1
  _relocate_to_symlink "logs/slurm" "${LAB_ROOT}/logs/slurm" || rc=1

  export TMPDIR="/tmp"
  export GENERATED_CONFIG="config/experiments_splits1_5_scicore.generated.yaml"
  sed "s|__SCICORE_LAB_ROOT__|${SCICORE_LAB_ROOT}|g" \
    config/experiments_splits1_5_scicore.yaml > "${GENERATED_CONFIG}"

  cat > .scicore_lab_env <<EOF
export SCICORE_LAB_ROOT='${SCICORE_LAB_ROOT}'
export TMPDIR='${TMPDIR}'
export GENERATED_CONFIG='${GENERATED_CONFIG}'
EOF

  echo "SCICORE_LAB_ROOT=${SCICORE_LAB_ROOT}"
  echo ".snakemake -> $(readlink .snakemake 2>/dev/null || echo missing)"
  echo "logs/slurm -> $(readlink logs/slurm 2>/dev/null || echo missing)"
  echo "TMPDIR=${TMPDIR}"
  echo "Wrote ${GENERATED_CONFIG} and .scicore_lab_env"
  echo ""
  echo "Run:"
  echo "  source .scicore_lab_env"
  echo "  snakemake --profile profiles/scicore --configfile \${GENERATED_CONFIG}"

  return "$rc"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  cd "$(dirname "$0")/.." || exit 1
  _main
  exit $?
fi

# Sourced: do not use 'return 1' with set -e (can disconnect SSH)
set +e
_main
return $?
