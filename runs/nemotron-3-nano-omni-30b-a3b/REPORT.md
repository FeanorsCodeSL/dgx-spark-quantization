# Nemotron-3-Nano-Omni-30B-A3B-Reasoning — Quantization Report

**Status:** full eval complete. The AWQ artifact is built, smoke-tested,
and evaluated against NVIDIA NVFP4 and bf16 baselines.

**Base model**:
[`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)

**Hardware:** NVIDIA DGX Spark, GB10 / SM121a, 128 GiB unified memory.

**Plan:** [`PLAN.md`](./PLAN.md)

---

## TL;DR

| build | bits | disk | GSM8K | MMLU | ARC-C | Δ MMLU vs bf16 |
|---|---:|---:|---:|---:|---:|---:|
| AWQ-INT4 W4A16 compressed-tensors | 4 | ~22G | 0.7983 strict / 0.8893 flexible | 0.6904 | 0.5247 / 0.5589 norm | -2.46 pp |
| NVFP4 (NVIDIA official) | 4 | ~21G | 0.7589 strict / 0.8992 flexible | 0.7124 | 0.5230 / 0.5401 norm | -0.26 pp |
| bf16 baseline | 16 | ~66G | 0.7900 strict / 0.9090 flexible | 0.7150 | 0.5239 / 0.5631 norm | — |

The current usable output is the AWQ compressed-tensors artifact at
`artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4/`. It is not an
AutoAWQ/GEMM artifact.

Headline result: AWQ is within 2.46 pp MMLU of bf16, is effectively tied on
ARC-C acc, and is +0.83 pp on GSM8K strict while trailing bf16 by 1.97 pp on
GSM8K flexible extraction. NVFP4 is closer on MMLU and flexible GSM8K, but
lower on strict GSM8K and ARC-C normalized accuracy in this text-only eval.

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

### Multimodal Smoke

Smoke script:
[`recipes/multimodal_smoke.py`](./recipes/multimodal_smoke.py)

The script downloads public fixtures, saves local copies under
`results/*/fixtures/`, and calls `/v1/chat/completions` for image, audio,
and video inputs.

| serve environment | image | audio | video | result |
|---|---:|---:|---:|---|
| pinned `:20260428` image as-is | pass | fail | pass | `2/3` |
| pinned `:20260428` image with `av` + `soundfile` installed before `vllm serve` | pass | pass | pass | `3/3` |

Result transcripts:

- [`results/multimodal_smoke/multimodal_smoke.md`](./results/multimodal_smoke/multimodal_smoke.md)
- [`results/multimodal_smoke_audio_extra/multimodal_smoke.md`](./results/multimodal_smoke_audio_extra/multimodal_smoke.md)

The pinned image as-is loaded the vision and sound encoder modules, and
image/video requests worked. Audio failed before model execution because
vLLM could not decode audio (`Please install vllm[audio] for audio support`,
missing `av`). Installing `av` and `soundfile` before server startup made
the downloaded WAV audio smoke pass. This validates the artifact's preserved
audio path, but the serving image should include vLLM audio extras before
advertising turnkey audio support.

---

## AWQ Full Eval

Run command:

```bash
tools/run_eval_full.sh nemotron-omni-awq-ct \
  "$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4" \
  "$PWD/runs/nemotron-3-nano-omni-30b-a3b/results/awq_full"
```

Serve settings: `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260428`,
`--kv-cache-dtype fp8_e4m3`, `--max-model-len 4096`,
`--gpu-memory-utilization 0.55`, `--reasoning-parser nemotron_v3`.

Result files:

- `results/awq_full/gsm8k/nemotron-omni-awq-ct/results_2026-05-01T01-53-59.036217.json`
- `results/awq_full/mmlu/nemotron-omni-awq-ct/results_2026-05-01T02-40-27.040417.json`
- `results/awq_full/arc_challenge/nemotron-omni-awq-ct/results_2026-05-01T02-43-12.473144.json`
- `results/awq_full/run.log`

| task | metric | value | stderr |
|---|---|---:|---:|
| GSM8K | exact_match, strict-match | 0.7983 | 0.0111 |
| GSM8K | exact_match, flexible-extract | 0.8893 | 0.0086 |
| MMLU | acc | 0.6904 | 0.0037 |
| ARC-Challenge | acc | 0.5247 | 0.0146 |
| ARC-Challenge | acc_norm | 0.5589 | 0.0145 |

The full eval completed at `2026-05-01T02:43:13+02:00` with `rc=0` for
GSM8K, MMLU, and ARC-Challenge. GSM8K emitted repeated `API returned null
content` warnings after generation; saved samples should be inspected
before publishing, because Nemotron can place reasoning-only text in the
OpenAI response reasoning field when the final answer is empty.

## NVFP4 Full Eval

Run command:

```bash
tools/run_eval_full.sh nemotron-omni-nvfp4 \
  "$PWD/hf-cache/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4/snapshots/889396e9cebaefdb69a469afc7bd111660f78eff" \
  "$PWD/runs/nemotron-3-nano-omni-30b-a3b/results/nvfp4_full"
```

Serve settings: `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260428`,
`--kv-cache-dtype fp8_e4m3`, `--max-model-len 4096`,
`--gpu-memory-utilization 0.55`, `--reasoning-parser nemotron_v3`.
The model loaded with `quantization=modelopt_mixed`; vLLM selected ModelOpt
FP8/NVFP4 kernels, including `FLASHINFER_CUTLASS` for NVFP4 MoE.

Result files:

- `results/nvfp4_full/gsm8k/nemotron-omni-nvfp4/results_2026-05-01T03-43-58.735164.json`
- `results/nvfp4_full/mmlu/nemotron-omni-nvfp4/results_2026-05-01T04-47-58.908374.json`
- `results/nvfp4_full/arc_challenge/nemotron-omni-nvfp4/results_2026-05-01T04-50-20.617379.json`
- `results/nvfp4_full/run.log`

| task | metric | value | stderr |
|---|---|---:|---:|
| GSM8K | exact_match, strict-match | 0.7589 | 0.0118 |
| GSM8K | exact_match, flexible-extract | 0.8992 | 0.0083 |
| MMLU | acc | 0.7124 | 0.0036 |
| ARC-Challenge | acc | 0.5230 | 0.0146 |
| ARC-Challenge | acc_norm | 0.5401 | 0.0146 |

The full eval completed at `2026-05-01T04:50:21+02:00` with `rc=0` for
GSM8K, MMLU, and ARC-Challenge. The standalone `391` smoke prompt was not
run; the full eval harness did verify the OpenAI `/v1/models` endpoint for
`nemotron-omni-nvfp4` before starting the tasks.

---

## BF16 Full Eval

Run command:

```bash
tools/run_eval_full.sh nemotron-omni-bf16 \
  "$PWD/hf-cache/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/d15962741057ae3a07147df504060e9f0838224e" \
  "$PWD/runs/nemotron-3-nano-omni-30b-a3b/results/bf16_full"
```

Serve settings: `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260428`,
`--kv-cache-dtype fp8_e4m3`, `--max-model-len 4096`,
`--gpu-memory-utilization 0.70`, `--reasoning-parser nemotron_v3`.
The first `--gpu-memory-utilization 0.45` attempt loaded 61.59 GiB of
weights, then failed before serving because vLLM had no available memory
for KV cache blocks.

Result files:

- `results/bf16_full/gsm8k/nemotron-omni-bf16/results_2026-05-01T07-08-15.256862.json`
- `results/bf16_full/mmlu/nemotron-omni-bf16/results_2026-05-01T08-44-32.988561.json`
- `results/bf16_full/arc_challenge/nemotron-omni-bf16/results_2026-05-01T08-49-55.588052.json`
- `results/bf16_full/run.log`

| task | metric | value | stderr |
|---|---|---:|---:|
| GSM8K | exact_match, strict-match | 0.7900 | 0.0112 |
| GSM8K | exact_match, flexible-extract | 0.9090 | 0.0079 |
| MMLU | acc | 0.7150 | 0.0036 |
| ARC-Challenge | acc | 0.5239 | 0.0146 |
| ARC-Challenge | acc_norm | 0.5631 | 0.0145 |

The full eval completed at `2026-05-01T08:49:56+02:00` with `rc=0` for
GSM8K, MMLU, and ARC-Challenge. The original Phase 5 GSM8K strict sanity
floor of 0.85 was too high for this serving/eval setup: bf16 reached 0.7900
strict but 0.9090 flexible extraction.

---

## Comparison

| build | GSM8K strict Δ | GSM8K flexible Δ | MMLU Δ | ARC-C acc Δ | ARC-C acc_norm Δ |
|---|---:|---:|---:|---:|---:|
| AWQ-INT4 vs bf16 | +0.83 pp | -1.97 pp | -2.46 pp | +0.09 pp | -0.43 pp |
| NVFP4 vs bf16 | -3.11 pp | -0.99 pp | -0.26 pp | -0.09 pp | -2.30 pp |

All deltas use the same vLLM image, max model length, FP8 KV cache dtype,
reasoning parser, and local eval harness.

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
- [`results/multimodal_smoke/`](./results/multimodal_smoke/) — image/video
  smoke on the pinned image as-is; audio fails due missing audio decode deps.
- [`results/multimodal_smoke_audio_extra/`](./results/multimodal_smoke_audio_extra/)
  — image/audio/video smoke after installing `av` + `soundfile` before
  server startup.
