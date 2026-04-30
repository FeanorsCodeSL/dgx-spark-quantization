# Nemotron-3-Nano-Omni-30B-A3B-Reasoning — Quantization Report

**Status:** in progress. The AWQ artifact is built and smoke-tested; full
AWQ, NVIDIA NVFP4, and bf16 evals are pending.

**Base model**:
[`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)

**Hardware:** NVIDIA DGX Spark, GB10 / SM121a, 128 GiB unified memory.

**Plan:** [`PLAN.md`](./PLAN.md)

---

## TL;DR

| build | bits | disk | GSM8K | MMLU | ARC-C | status |
|---|---:|---:|---:|---:|---:|---|
| AWQ-INT4 W4A16 compressed-tensors | 4 | ~22G | pending | pending | pending | built + vLLM smoke passed |
| NVFP4 (NVIDIA official) | 4 | ~21G | pending | pending | pending | eval pending |
| bf16 baseline | 16 | ~66G | pending | pending | pending | eval pending |

The current usable output is the AWQ compressed-tensors artifact at
`artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4/`. It is not an
AutoAWQ/GEMM artifact.

---

## AWQ Artifact

| key | value |
|---|---|
| Recipe | [`recipes/awq_compressed_tensors.py`](./recipes/awq_compressed_tensors.py) |
| Scheme doc | [`docs/schemes/awq-compressed-tensors.md`](../../docs/schemes/awq-compressed-tensors.md) |
| Format | `compressed-tensors`, `pack-quantized`, W4A16 |
| Group size | 64 |
| Weight quantization | symmetric INT4, routed expert MLP weights only |
| Dense components | Mamba, attention, shared experts, router, layer 0, embeddings, lm_head, norms, vision/RADIO, audio/Parakeet, projectors |
| Output | `artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4/` |
| Size | 6 safetensors shards, ~21.34 GiB payload / ~22G on disk |
| vLLM image | `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260428` |

The recipe uses `llm-compressor` AWQ calibration, then avoids the
Transformers serializer memory spike by compressing the inner LM in memory
and streaming compressed LM tensors plus dense multimodal tensors into
bounded safetensors shards.

Smoke transcript:
[`results/awq_ct_smoke.txt`](./results/awq_ct_smoke.txt)

Smoke result: `/v1/chat/completions` returned `2+2 equals 4.` with
`finish_reason="stop"` when served as `nemotron-omni-awq-ct`.

---

## Pending Evals

The final report should be filled after these runs complete:

1. AWQ artifact eval: `results/awq_full/`
2. NVIDIA NVFP4 eval or loader-failure note: `results/nvfp4_full/` or
   `results/nvfp4_loader_failure.txt`
3. bf16 baseline eval: `results/bf16_full/`

Use the eval order and commands in [`PLAN.md`](./PLAN.md): AWQ first, then
NVFP4, then bf16.

---

## File Index

- [`PLAN.md`](./PLAN.md) — current execution plan and phase checklist.
- [`README.md`](./README.md) — run index, architecture notes, reproduction
  commands.
- [`recipes/awq_compressed_tensors.py`](./recipes/awq_compressed_tensors.py)
  — working AWQ compressed-tensors recipe.
- [`recipes/_classify.py`](./recipes/_classify.py) — shared quantize/skip
  policy.
- [`results/module_inspection.txt`](./results/module_inspection.txt) —
  architecture inspection output.
- [`results/awq_ct_smoke.txt`](./results/awq_ct_smoke.txt) — successful
  vLLM smoke transcript.
