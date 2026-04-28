# <BASE_MODEL_SLUG> — quantization run

> Copy this template to `runs/<slug>/` and fill in. See
> [`docs/adding-a-run.md`](../../docs/adding-a-run.md) for the step-by-step.

**Base model**: [`<org>/<name>`](https://huggingface.co/<org>/<name>)
*(short description — N B parameters, architecture, lineage)*

**Hardware**: *(e.g. NVIDIA DGX Spark, GB10 / SM121a, 128 GiB unified memory)*

**Status**: *(in-progress / done) — last updated YYYY-MM-DD*

---

## TL;DR

| build | bits | disk | MMLU | GSM8K | ARC-C | Δ MMLU vs bf16 |
|---|---|---|---|---|---|---|
| bf16 baseline | 16 | ~? GiB | ? | ? | ? | — |
| FP8 W8A8 dynamic | 8 | ~? GiB | ? | ? | ? | ? |
| AWQ-INT4 GEMM | 4 | ~? GiB | ? | ? | ? | ? |

*One-line headline interpretation goes here.*

For full numbers, settings, and the three-way head-to-head, see
[`REPORT.md`](./REPORT.md).

---

## Schemes used

- *(link to `docs/schemes/<scheme>.md` for each — describe the
  model-specific deviations from the generic recipe)*

## What's in this run

| path | content |
|---|---|
| [`REPORT.md`](./REPORT.md) | full quant report with deltas |
| [`recipes/`](./recipes/) | model-specific quantizer drivers |
| [`results/`](./results/) | `results_*.json` + `run.log` per full eval |

## Reproducing

```bash
# Quantize (per scheme). Output goes under artifacts/ (gitignored).
export MODEL_ID="<org>/<name>"        # auto-downloaded into ~/.cache/huggingface
export SAVE_DIR="$PWD/artifacts/<artifact-name>"
tools/run_under_memcap.sh python runs/<slug>/recipes/<scheme>.py

# Serve (per artifact)
tools/serve_vllm_docker.sh "$PWD/artifacts/<artifact-name>" \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --served-model-name <served-name>

# Eval (per artifact)
tools/run_eval_full.sh \
  <served-name> \
  "$PWD/artifacts/<artifact-name>" \
  "$PWD/runs/<slug>/results/<scheme>_full"
```

## Publishing

The artifact directories are uploaded to Hugging Face per
[`HUGGINGFACE_PUBLISHING.md`](../../HUGGINGFACE_PUBLISHING.md).
