#!/usr/bin/env bash
# Full-set quality eval against a running vLLM endpoint.
#
# Usage:
#   ./run_eval_full.sh <served_model_name> <tokenizer_dir> <out_dir>
#
#   <served_model_name>  --served-model-name vLLM was started with
#   <tokenizer_dir>      local dir containing the tokenizer files
#   <out_dir>            where results go. Either:
#                          - an absolute path (recommended), or
#                          - a relative path resolved from the current cwd
#
# Examples:
#   ./run_eval_full.sh qwen3.6-fp8 \
#     /path/to/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic \
#     /path/to/runs/qwen3.6-35b-distill/results/fp8_full
#
# Runs:
#   - GSM8K  (full 1,319, 5-shot CoT, chat-templated)        via /v1/chat/completions
#   - MMLU   (full ~14,042 across 57 subtasks, raw mc)        via /v1/completions
#   - ARC-C  (full 1,172, raw mc)                             via /v1/completions
#
# Optional env vars:
#   LM_EVAL          path to lm_eval binary (default: $PROJECT_DIR/.venv/bin/lm_eval)
#   PROJECT_DIR      project root (default: parent of this script's dir)
#   VLLM_URL         vLLM endpoint root (default: http://localhost:8000)
#   NUM_CONCURRENT   per-request concurrency (default: 4)
#   MAX_LENGTH       prompt+gen ceiling passed to lm-eval (default: 4096)

set -uo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <served_model_name> <tokenizer_dir> <out_dir>" >&2
  exit 2
fi

SERVED="$1"
TOKENIZER="$2"
OUT_DIR="$3"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
LM_EVAL="${LM_EVAL:-${PROJECT_DIR}/.venv/bin/lm_eval}"
VLLM_URL="${VLLM_URL:-http://localhost:8000}"
NUM_CONCURRENT="${NUM_CONCURRENT:-4}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
LOG="${OUT_DIR}/run.log"

if [[ ! -x "$LM_EVAL" ]]; then
  echo "lm_eval not found at $LM_EVAL" >&2
  echo "  install with: pip install lm-eval==0.4.11" >&2
  echo "  or set LM_EVAL=/path/to/lm_eval" >&2
  exit 3
fi

if [[ ! -d "$TOKENIZER" ]]; then
  echo "tokenizer dir not found: $TOKENIZER" >&2
  exit 4
fi

mkdir -p "$OUT_DIR"/{gsm8k,mmlu,arc_challenge}

# Sanity: confirm the named model is actually being served.
served_check=$(curl -s "${VLLM_URL}/v1/models" | python3 -c \
  "import json,sys;print(json.load(sys.stdin).get('data',[{}])[0].get('id',''))" 2>/dev/null)
if [[ "$served_check" != "$SERVED" ]]; then
  echo "vLLM at ${VLLM_URL} reports served model '${served_check}', expected '${SERVED}'" >&2
  echo "Aborting before generating misleading results." >&2
  exit 5
fi

# Common args.
COMMON_COMPL="model=${SERVED},base_url=${VLLM_URL}/v1/completions,tokenizer_backend=huggingface,tokenizer=${TOKENIZER},num_concurrent=${NUM_CONCURRENT},tokenized_requests=False,max_length=${MAX_LENGTH},trust_remote_code=True"
COMMON_CHAT="model=${SERVED},base_url=${VLLM_URL}/v1/chat/completions,tokenizer_backend=huggingface,tokenizer=${TOKENIZER},num_concurrent=${NUM_CONCURRENT},tokenized_requests=False,max_length=${MAX_LENGTH},trust_remote_code=True"

run_task() {
  local label="$1"; shift
  echo
  echo "=========================================================="
  echo "[$(date -Is)] ${label}  (served=${SERVED}, out=${OUT_DIR})"
  echo "=========================================================="
  "$LM_EVAL" run "$@"
  local rc=$?
  echo "[$(date -Is)] ${label} rc=${rc}"
  return $rc
}

{
  echo "[$(date -Is)] full-eval start: served=${SERVED} tokenizer=${TOKENIZER} out=${OUT_DIR}"

  run_task "gsm8k (full, chat+CoT)" \
    --model local-chat-completions \
    --model_args "$COMMON_CHAT" \
    --tasks gsm8k \
    --apply_chat_template \
    --gen_kwargs max_gen_toks=1024,temperature=0,top_p=1 \
    --output_path "${OUT_DIR}/gsm8k" \
    --log_samples
  rc_gsm=$?

  run_task "mmlu (full, raw mc)" \
    --model local-completions \
    --model_args "$COMMON_COMPL" \
    --tasks mmlu \
    --output_path "${OUT_DIR}/mmlu" \
    --log_samples
  rc_mmlu=$?

  run_task "arc_challenge (full, raw mc)" \
    --model local-completions \
    --model_args "$COMMON_COMPL" \
    --tasks arc_challenge \
    --output_path "${OUT_DIR}/arc_challenge" \
    --log_samples
  rc_arc=$?

  echo
  echo "[$(date -Is)] full-eval done: served=${SERVED}"
  echo "  gsm8k rc=${rc_gsm}"
  echo "  mmlu  rc=${rc_mmlu}"
  echo "  arc-c rc=${rc_arc}"
} 2>&1 | tee -a "$LOG"
