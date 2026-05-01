---
license: other
language:
- en
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- nemotron
- nemotron-h
- mixture-of-experts
- moe
- mamba
- multimodal
- vision-language
- audio-language
- reasoning
- awq
- int4
- compressed-tensors
- quantization
- vllm
base_model: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
base_model_relation: quantized
quantized_by: feanors
inference: false
---

# Nemotron-3-Nano-Omni-30B-A3B-Reasoning - AWQ-INT4

Calibrated AWQ-INT4 W4A16 quantization of
[`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16),
a 30B-parameter / 3B-active NemotronH hybrid model with Mamba2,
Transformer attention, MoE routed experts, vision, audio, and reasoning
tuning.

This artifact preserves the full multimodal stack and only packs the routed
expert MLP weights to INT4. It is a `compressed-tensors` `pack-quantized`
artifact, not an AutoAWQ/GEMM artifact.

Quantized by **[FeanorsCode](https://feanorscode.com)** as part of
[`FeanorsCodeSL/dgx-spark-quantization`](https://github.com/FeanorsCodeSL/dgx-spark-quantization).

## Status

Full text eval completed on 2026-05-01 on NVIDIA DGX Spark, GB10 / SM121a,
using the same vLLM image, prompts, max length, FP8 KV cache, and eval
harness for AWQ, NVIDIA NVFP4, and bf16.

| build | disk | GSM8K strict | GSM8K flexible | MMLU | ARC-C acc | ARC-C norm |
|---|---:|---:|---:|---:|---:|---:|
| bf16 baseline | ~66 GiB | 0.7900 | 0.9090 | 0.7150 | 0.5239 | 0.5631 |
| NVIDIA NVFP4 | ~21 GiB | 0.7589 | 0.8992 | 0.7124 | 0.5230 | 0.5401 |
| AWQ-INT4 this build | ~22 GiB | 0.7983 | 0.8893 | 0.6904 | 0.5247 | 0.5589 |

AWQ deltas vs bf16:

| metric | delta |
|---|---:|
| GSM8K strict | +0.83 pp |
| GSM8K flexible | -1.97 pp |
| MMLU | -2.46 pp |
| ARC-C acc | +0.09 pp |
| ARC-C acc_norm | -0.43 pp |

## Model Summary

| key | value |
|---|---|
| Base | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` |
| Architecture | `NemotronH_Nano_Omni_Reasoning_V3` / `NemotronHForCausalLM` |
| Total params | 30B |
| Active params | 3B |
| Layers | 52 hidden layers |
| Hybrid pattern | 23 Mamba2 + 23 MoE MLP + 6 attention |
| Experts | 128 routed experts, top-6 routing, 1 shared expert |
| Vision | RADIO / CRADIO v2-H, kept dense |
| Audio | Parakeet-style sound encoder, kept dense |
| Quantization | AWQ-INT4 W4A16, symmetric, group size 64 |
| Format | `compressed-tensors`, `pack-quantized` |
| Quantized scope | routed expert `up_proj` and `down_proj` weights |
| Kept dense | Mamba, attention, shared experts, routers, layer 0, embeddings, `lm_head`, norms, vision, audio, projectors |
| Disk size | 6 safetensors shards, ~21.34 GiB payload / ~22 GiB on disk |

## Quantization Recipe

The build uses `llm-compressor` AWQ calibration, then streams the packed
weights into bounded safetensors shards to avoid the Transformers serializer
memory spike that caused previous OOM attempts.

Reference implementation:

- Recipe: `runs/nemotron-3-nano-omni-30b-a3b/recipes/awq_compressed_tensors.py`
- Scheme doc: `docs/schemes/awq-compressed-tensors.md`
- Full report: `runs/nemotron-3-nano-omni-30b-a3b/REPORT.md`

The packed artifact carries the base model's custom remote-code files and
requires `trust_remote_code=True`.

## vLLM

Tested with:

```bash
tools/serve_vllm_docker.sh \
  "$PWD/artifacts/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ-INT4" \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.55 \
  --reasoning-parser nemotron_v3 \
  --media-io-kwargs '{"video":{"fps":2,"num_frames":256}}' \
  --video-pruning-rate 0.5 \
  --allowed-local-media-path / \
  --served-model-name nemotron-omni-awq-ct
```

Pinned image:

```text
ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260428
```

Smoke test returned `2+2 equals 4.` through `/v1/chat/completions`.

Multimodal smoke on downloaded public fixtures:

| environment | image | audio | video |
|---|---:|---:|---:|
| pinned `:20260428` image as-is | pass | fail | pass |
| pinned `:20260428` image with `av` + `soundfile` installed before server startup | pass | pass | pass |

The audio failure in the bare pinned image happens during vLLM media decode
before model execution (`vllm[audio]` extras missing). Include `av` and
`soundfile` in the serving image for audio inputs.

## Evaluation

Harness:

- `tools/run_eval_full.sh`
- GSM8K full set, chat-templated CoT, `temperature=0`, `max_gen_toks=1024`
- MMLU full set, raw multiple-choice loglikelihood
- ARC-Challenge full set, raw multiple-choice loglikelihood
- `max_model_len=4096`
- `kv_cache_dtype=fp8_e4m3`
- `reasoning_parser=nemotron_v3`

Result files:

- `runs/nemotron-3-nano-omni-30b-a3b/results/awq_full/`
- `runs/nemotron-3-nano-omni-30b-a3b/results/nvfp4_full/`
- `runs/nemotron-3-nano-omni-30b-a3b/results/bf16_full/`

GSM8K emitted repeated `API returned null content` warnings after generation
for the AWQ run. The eval still completed with `rc=0`; samples should be
inspected before relying on strict-answer behavior in a production card.

## Limitations

- The reported benchmark evals are text-only. Smoke tests confirm image and
  video requests work on the pinned image, and audio works after installing
  vLLM audio decode dependencies; this is not a full multimodal quality eval.
- This is a `compressed-tensors` AWQ artifact. Consumers expecting AutoAWQ
  GEMM metadata will not load it as an AutoAWQ checkpoint.
- The bf16 baseline required `--gpu-memory-utilization 0.70` on DGX Spark.
  A lower `0.45` setting loaded weights but failed KV-cache allocation.

## License & Usage

**Model weights:** governed by the
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/),
inherited from
[`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16).
NVIDIA's model card states that the base model is available for commercial
use.

**Required NVIDIA notice:** this repository includes `NOTICE` with:
`Licensed by NVIDIA Corporation under the NVIDIA Open Model License`.

**Quantization code and public reproduction:** Apache-2.0, published by
[FeanorsCode](https://feanorscode.com) at
[`FeanorsCodeSL/dgx-spark-quantization`](https://github.com/FeanorsCodeSL/dgx-spark-quantization).

Review the NVIDIA Open Model Agreement, NVIDIA Trustworthy AI terms, and
any accompanying third-party component notices before production
redistribution.

## Citation

```bibtex
@misc{feanorscode_nemotron3_nano_omni_awq_int4_2026,
  title        = {Nemotron-3-Nano-Omni-30B-A3B-Reasoning AWQ-INT4},
  author       = {FeanorsCode},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/feanors/nemotron-3-nano-omni-30b-a3b-awq}},
  note         = {AWQ W4A16 compressed-tensors quantization of NVIDIA Nemotron 3 Nano Omni}
}
```

## Credits

### Model Lineage

- Base model by
  [NVIDIA](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16):
  Nemotron 3 Nano Omni, a 30B / 3B-active hybrid Mamba2-Transformer MoE
  with CRADIO vision and Parakeet-style audio encoders.
- NVIDIA's model card says the model was improved using
  Qwen3-VL-30B-A3B-Instruct, Qwen3.5-122B-A10B, Qwen3.5-397B-A17B,
  Qwen2.5-VL-72B-Instruct, and gpt-oss-120b.

### Quantization

- Quantized by [FeanorsCode](https://feanorscode.com).
- AWQ calibration and compressed-tensors packing used
  [`llm-compressor`](https://github.com/vllm-project/llm-compressor) and
  [`compressed-tensors`](https://github.com/neuralmagic/compressed-tensors).
- The artifact preserves Mamba, attention, shared experts, routing,
  embeddings, norms, `lm_head`, vision, audio, and projector modules at
  dense precision, and quantizes routed expert `up_proj` / `down_proj`
  weights to W4A16.

### Serving & Evaluation

- Served with [`vLLM`](https://github.com/vllm-project/vllm) using the
  community DGX Spark image
  `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260428`.
- Evaluated with
  [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
  on GSM8K, MMLU, and ARC-Challenge.
- Multimodal smoke fixtures were downloaded at test time and are not
  redistributed in the public GitHub repo.
