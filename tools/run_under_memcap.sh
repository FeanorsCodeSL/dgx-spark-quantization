#!/usr/bin/env bash
set -euo pipefail

MEMORY_MAX="${MEMORY_MAX:-112G}"
MEMORY_HIGH="${MEMORY_HIGH:-100G}"
MEMORY_SWAP_MAX="${MEMORY_SWAP_MAX:-0}"
TASKS_MAX="${TASKS_MAX:-infinity}"

if [[ $# -eq 0 ]]; then
  echo "Usage: MEMORY_MAX=112G MEMORY_HIGH=100G $0 <command> [args...]" >&2
  exit 2
fi

if ! command -v systemd-run >/dev/null 2>&1; then
  echo "systemd-run not found; refusing to run without cgroup guard on DGX Spark." >&2
  exit 1
fi

COMMON_PROPS=(
  -p "MemoryAccounting=yes"
  -p "MemoryMax=${MEMORY_MAX}"
  -p "MemoryHigh=${MEMORY_HIGH}"
  -p "MemorySwapMax=${MEMORY_SWAP_MAX}"
  -p "TasksMax=${TASKS_MAX}"
)

if systemctl --user show >/dev/null 2>&1; then
  exec systemd-run --user --scope --collect "${COMMON_PROPS[@]}" "$@"
else
  exec sudo systemd-run --scope --collect "${COMMON_PROPS[@]}" "$@"
fi
