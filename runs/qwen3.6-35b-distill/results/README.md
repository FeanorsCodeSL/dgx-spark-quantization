# qwen3.6-35b-distill — eval results

Three full evaluation passes (GSM8K + MMLU + ARC-Challenge) of the two
quantized builds and the bf16 baseline of
`lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled`, all run on the
same DGX Spark vLLM endpoint with `lm-evaluation-harness 0.4.11`.

## Layout

```
results/
├── awq_full/                       AWQ-INT4 GEMM eval (1 h 33 min)
│   ├── run.log                     full lm-eval transcript
│   ├── gsm8k/qwen3.6-awq-smoke/results_*.json
│   ├── mmlu/qwen3.6-awq-smoke/results_*.json
│   └── arc_challenge/qwen3.6-awq-smoke/results_*.json
├── fp8_full/                       FP8 W8A8 dynamic eval (1 h 59 min)
│   └── ... (same shape, served-name qwen3.6-fp8)
└── bf16_full/                      bf16 baseline eval (2 h 45 min)
    └── ... (same shape, served-name qwen3.6-bf16)
```

The per-sample `samples_*.jsonl` files (~60 MB per run, several hundred MB
total) are not committed — they're reproducible from `results_*.json` +
`run.log` + the same lm-eval seed. If you need them, re-run
`tools/run_eval_full.sh`.

## Headline numbers

See [`../REPORT.md`](../REPORT.md) — the "Three-way head-to-head: bf16 vs
FP8 vs AWQ" section pulls these JSONs into a single delta table.

## Reproducing

```bash
# Start vLLM serving the artifact you want to eval.
tools/serve_vllm_docker.sh /path/to/<artifact-dir> \
  --served-model-name <name> \
  --max-model-len 4096 \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.7

# In another terminal, run the eval battery.
tools/run_eval_full.sh \
  <name> \
  /path/to/<artifact-dir> \
  "$PWD/runs/qwen3.6-35b-distill/results/<subdir>"
```
