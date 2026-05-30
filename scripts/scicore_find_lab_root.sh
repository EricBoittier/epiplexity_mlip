#!/usr/bin/env bash
# Print sciCORE paths where the current user can create directories.
# Usage: bash scripts/scicore_find_lab_root.sh

set -euo pipefail

test_writable() {
  local d="$1"
  [[ -d "$d" ]] || return 1
  if mkdir -p "${d}/.epiplexity_write_test" 2>/dev/null; then
    rmdir "${d}/.epiplexity_write_test" 2>/dev/null || true
    echo "WRITABLE  $d"
    return 0
  fi
  echo "readonly  $d"
  return 1
}

echo "=== Quota (home) ==="
df -h "/scicore/home/meuwly/${USER}" 2>/dev/null || df -h ~

echo ""
echo "=== Under /scicore/home/meuwly/ ==="
for d in /scicore/home/meuwly/*/; do
  test_writable "$d" || true
done

echo ""
echo "=== Under your home ==="
for d in "/scicore/home/meuwly/${USER}/"* "/scicore/home/meuwly/${USER}/GROUP" "/scicore/home/meuwly/${USER}/group"; do
  [[ -e "$d" ]] || continue
  test_writable "$d" || true
done

echo ""
echo "=== /scicore/projects (if present) ==="
if [[ -d /scicore/projects ]]; then
  for d in /scicore/projects/*/; do
    test_writable "$d" || true
  done
else
  echo "(no /scicore/projects)"
fi

echo ""
echo "Pick a WRITABLE path, then:"
echo "  export SCICORE_LAB_ROOT=/path/you/chose/epiplexity/rmd17_aspirin_s1to5_ws_metrics_noise"
echo "  source scripts/scicore_use_lab_storage.sh"
echo "  # edit config/experiments_splits1_5_scicore.yaml ckpt_root to match"
