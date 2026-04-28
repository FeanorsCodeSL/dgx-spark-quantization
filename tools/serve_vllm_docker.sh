#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model-path-or-hf-id> [vllm args...]" >&2
  exit 2
fi

MODEL="$1"
shift

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$SCRIPT_DIR")}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf-cache}"
VLLM_IMAGE="${VLLM_IMAGE:-ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260415}"

docker run --rm \
  --gpus all \
  --ipc=host \
  --network host \
  --mount "type=bind,source=${PROJECT_ROOT},target=${PROJECT_ROOT}" \
  -w "$PROJECT_ROOT" \
  -e "HF_HOME=${HF_HOME}" \
  -e "TRANSFORMERS_CACHE=${HF_HOME}" \
  -e "HF_HUB_CACHE=${HF_HOME}/hub" \
  "$VLLM_IMAGE" \
  vllm serve "$MODEL" \
    --trust-remote-code \
    "$@"
