# qwen3.6-35b-distill — quantization run

**Base model**: [`lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled`](https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled)
35 B parameters / ~3 B active (top-8 of 256 experts), Qwen3.5-MoE multimodal
architecture (`Qwen3_5MoeForConditionalGeneration`), reasoning-distilled from
Claude 4.7 Opus.

**Hardware**: NVIDIA DGX Spark, GB10 / SM121a, Ubuntu 24.04 aarch64,
128 GiB unified memory.

**Status**: complete (full eval done 2026-04-27).

---

## TL;DR

| build | bits | disk | MMLU | GSM8K (strict) | ARC-C (acc_norm) | Δ MMLU vs bf16 |
|---|---|---|---|---|---|---|
| **bf16 baseline** | 16 | ~67 GiB | 0.8341 | 0.9447 | 0.5648 | — |
| **FP8 W8A8 dynamic** (text-only) | 8 | ~35 GiB | 0.8332 | 0.9447 | 0.5708 | **−0.09 pp** |
| **AWQ-INT4 GEMM** (multimodal) | 4 | ~24 GiB | 0.8068 | 0.9386 | 0.5606 | **−2.73 pp** |

FP8 is statistically indistinguishable from bf16 on this battery. AWQ-INT4
loses ~2.7 pp on MMLU but holds GSM8K within ~1 σ and keeps the vision tower
at fp16. Pick FP8 for headline quality, AWQ for disk + multimodal.

For full numbers, settings, and the three-way head-to-head, see
[`REPORT.md`](./REPORT.md).

---

## Schemes used

- [**FP8 W8A8 dynamic**](../../docs/schemes/fp8-dynamic.md) — text-only build
  (vision tower + MTP head stripped before save to fit Spark's 121 GiB
  ceiling end-to-end). Custom in-place quantizer to dodge `llmcompressor`'s
  expert-unfusing memory blow-up. → [`recipes/fp8_dynamic.py`](./recipes/fp8_dynamic.py)
- [**AWQ-INT4 GEMM**](../../docs/schemes/awq-gemm.md) — data-free RTN variant.
  Vision tower kept at fp16, layer 0 kept dense. Shard-streaming pure-Python
  packer; bit-for-bit compatible with `QuantTrio/Qwen3.6-35B-A3B-AWQ`'s
  layout so vLLM's `moe_wna16` kernel picks it up unchanged. →
  [`recipes/awq_gemm.py`](./recipes/awq_gemm.py)

A *calibrated* AWQ pass (real activation-aware salience over a calibration
corpus) would likely close ~half of the −2.73 pp MMLU gap — open as a
follow-up if needed.

---

## What's in this run

| path | content |
|---|---|
| [`REPORT.md`](./REPORT.md) | full quant report with bf16 / FP8 / AWQ deltas |
| [`PLAN.md`](./PLAN.md) | historical planning blueprint (pre-execution) |
| [`recipes/fp8_dynamic.py`](./recipes/fp8_dynamic.py) | FP8 W8A8 dynamic quantizer (with `--selftest`) |
| [`recipes/awq_gemm.py`](./recipes/awq_gemm.py) | AWQ-INT4 GEMM data-free RTN quantizer |
| [`recipes/inspect_modules.py`](./recipes/inspect_modules.py) | architecture-discovery helper |
| [`results/awq_full/`](./results/awq_full/) | AWQ full-eval results JSONs + run.log |
| [`results/fp8_full/`](./results/fp8_full/) | FP8 full-eval results JSONs + run.log |
| [`results/bf16_full/`](./results/bf16_full/) | bf16 baseline results JSONs + run.log |

The two artifact directories (with the actual safetensors shards) live
under [`artifacts/`](../../artifacts/) and are gitignored — they're for
upload to Hugging Face per
[`HUGGINGFACE_PUBLISHING.md`](../../HUGGINGFACE_PUBLISHING.md). The
canonical model-card sources for each artifact are also tracked here in
this run dir as `HF_PREVIEW_FP8.md` and `HF_PREVIEW_AWQ.md`.

```
artifacts/
├── Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic/    ~35 GiB FP8 artifact (gitignored)
└── Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4/       ~24 GiB AWQ artifact (gitignored)
```

---

## Reproducing

### Quantize (FP8)

```bash
export MODEL_ID="lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled"
# HF_CACHE is optional — defaults to ~/.cache/huggingface. Override only if
# you want a project-local cache (e.g. on shared boxes).
# export HF_HOME="$PWD/hf-cache"
export SAVE_DIR="$PWD/artifacts/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic"

tools/run_under_memcap.sh python runs/qwen3.6-35b-distill/recipes/fp8_dynamic.py
```

Wall-clock: ~30–60 min on Spark. Peak RSS: ~92 GiB. Aborts cleanly above
105 GiB before the kernel SIGKILLs the scope.

Sanity check (synthetic tensors, no model load):

```bash
python runs/qwen3.6-35b-distill/recipes/fp8_dynamic.py --selftest
```

### Quantize (AWQ)

```bash
# Adjust SRC_DIR to wherever your local snapshot lives. With a default HF
# cache, that's ~/.cache/huggingface/hub/models--lordx64--<...>/snapshots/<sha>
export SRC_DIR="<path-to-bf16-snapshot>"
export DST_DIR="$PWD/artifacts/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4"

python runs/qwen3.6-35b-distill/recipes/awq_gemm.py
```

Wall-clock: ~10–20 min. Host RAM peak: ~24 GiB.

### Serve

```bash
# FP8
tools/serve_vllm_docker.sh "$PWD/artifacts/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic" \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8_e4m3 --enforce-eager \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --served-model-name qwen3.6-fp8 --reasoning-parser qwen3

# AWQ
tools/serve_vllm_docker.sh "$PWD/artifacts/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4" \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --served-model-name qwen3.6-awq-smoke --reasoning-parser qwen3

# bf16 baseline (tighter memory; matched KV-cache dtype for parity)
tools/serve_vllm_docker.sh "$MODEL_ID" \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 4096 --gpu-memory-utilization 0.7 \
  --served-model-name qwen3.6-bf16 --reasoning-parser qwen3
```

### Eval

```bash
tools/run_eval_full.sh qwen3.6-fp8 \
  "$PWD/artifacts/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic" \
  "$PWD/runs/qwen3.6-35b-distill/results/fp8_full"

tools/run_eval_full.sh qwen3.6-awq-smoke \
  "$PWD/artifacts/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4" \
  "$PWD/runs/qwen3.6-35b-distill/results/awq_full"

tools/run_eval_full.sh qwen3.6-bf16 \
  "$PWD/<bf16-snapshot-dir>" \
  "$PWD/runs/qwen3.6-35b-distill/results/bf16_full"
```

Each run takes ~1.5–3 hours on Spark at `num_concurrent=4`.

---

## Publishing

The artifact dirs each already carry a populated HF model card with the
right YAML frontmatter (`base_model`, `base_model_relation: quantized`,
`quantized_by`). Follow [`HUGGINGFACE_PUBLISHING.md`](../../HUGGINGFACE_PUBLISHING.md)
for the upload pipeline.
