---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen3
- qwen3_5_moe
- mixture-of-experts
- moe
- reasoning
- distillation
- claude-distill
- fp8
- compressed-tensors
- quantization
- vllm
base_model: lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled
base_model_relation: quantized
quantized_by: feanors
inference: false
---

# Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled — FP8 W8A8 Dynamic (text-only)

FP8 W8A8 **dynamic** quantization (compressed-tensors / `float-quantized`) of
[`lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled`](https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled),
a 35B-parameter / 3B-active Qwen3.5-MoE reasoning-distilled fine-tune.

**This is a text-only build.** The vision tower and the multi-token-prediction
(MTP) head were stripped before quantization to fit the DGX Spark's 128 GiB
unified-memory budget end-to-end. If you need the multimodal stack, use the
sibling **AWQ-INT4 build** —
[`feanorscode/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4`](https://huggingface.co/feanorscode/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4) —
that one keeps `visual.*` and `mtp.*` at fp16/bf16.

> **Status — text-only build, full eval done (2026-04-27).**
> GSM8K **0.9447** (strict), MMLU **0.8332**, ARC-C **0.5708** (acc_norm) on
> the full benchmark sets. **Effectively lossless vs the bf16 baseline**:
> −0.09 pp MMLU and 0.00 pp GSM8K strict-match. See
> [Evaluation vs bf16](#evaluation-vs-bf16-baseline) for the full delta table.

---

## Source code & reproduction

Quantized by **[FeanorsCode](https://feanorscode.com)**
([github.com/FeanorsCodeSL](https://github.com/FeanorsCodeSL)) as part of
the public DGX-Spark quantization framework:
**[github.com/FeanorsCodeSL/dgx-spark-quantization](https://github.com/FeanorsCodeSL/dgx-spark-quantization)**.

The framework includes the eval driver, vLLM container launcher, the
recipes for both the FP8 and AWQ-INT4 builds of this base, the full eval
results (GSM8K + MMLU + ARC-Challenge), and the publishing walkthrough
used to produce this artifact. If you want a vLLM-loadable quantization
of a different bf16 / GGUF model, fork it.

- Per-scheme reference: [`docs/schemes/fp8-dynamic.md`](https://github.com/FeanorsCodeSL/dgx-spark-quantization/blob/main/docs/schemes/fp8-dynamic.md)
- Recipe used for this build: [`runs/qwen3.6-35b-distill/recipes/fp8_dynamic.py`](https://github.com/FeanorsCodeSL/dgx-spark-quantization/blob/main/runs/qwen3.6-35b-distill/recipes/fp8_dynamic.py)
- Three-way bf16 / FP8 / AWQ comparison report: [`runs/qwen3.6-35b-distill/REPORT.md`](https://github.com/FeanorsCodeSL/dgx-spark-quantization/blob/main/runs/qwen3.6-35b-distill/REPORT.md)

---

## TL;DR

| | bf16 (base) | **FP8 W8A8 dynamic** |
|---|---|---|
| Bits / weight | 16 | 8 |
| Activation precision | 16 | 8 dynamic per-token |
| Calibration | n/a | none (one-shot) |
| Vision tower preserved | yes | **no** (stripped) |
| MTP head preserved | yes | **no** (stripped) |
| Disk size | ~67 GiB | ~35 GiB |
| Format | safetensors / bf16 | compressed-tensors `float-quantized` |
| GSM8K (1,319, 5-shot CoT, strict) | 0.9447 | **0.9447** |
| MMLU overall (14,042, 5-shot raw MC) | 0.8341 | **0.8332** |
| ARC-Challenge (1,172, raw MC, acc_norm) | 0.5648 | **0.5708** |
| Δ MMLU vs bf16 | — | **−0.09 pp** (within 0.3 σ) |
| Δ GSM8K strict vs bf16 | — | **0.00 pp** (identical) |

Same eval harness (`lm-evaluation-harness 0.4.11`), same prompts, same
`temperature=0`, same KV-cache dtype.

---

## Quick numbers

| | |
|---|---|
| Bits | 8 (E4M3, `torch.float8_e4m3fn`) |
| Activation precision | dynamic FP8 per-token (no on-disk activation scales) |
| Weight strategy | per-output-channel, fp32 scales |
| Symmetric | yes (both weights and activations) |
| Calibration data | **none** (FP8 dynamic is one-shot) |
| Multimodal | **no** — vision tower removed |
| MTP head | **no** — removed |
| On-disk size | ~35 GiB (vs ~67 GiB bf16 source, ~24 GiB AWQ-INT4 sibling) |

---

## Model summary

| | |
|---|---|
| **Base model** | [`lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled`](https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled) |
| **Underlying base** | [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Apache-2.0) |
| **Architecture exposed** | `Qwen3_5MoeForConditionalGeneration` (multimodal class with vision skipped) |
| **Effective stack** | text-only (vision + MTP stripped) |
| **Total params** | ~35 B |
| **Active params / token** | ~3 B (top-8 of 256 experts) |
| **Hidden size** | 2,048 |
| **MoE intermediate** | 512 |
| **Layers** | 40 (12 full-attention + 28 linear-attention, repeating) |
| **Experts** | 256 routed + 1 shared, top-8 routing |
| **Vocab** | 248,320 |
| **Max position** | 262,144 (RoPE θ = 10,000,000) |
| **Quantization format** | compressed-tensors `float-quantized` (vLLM-native) |
| **Stored as** | `torch.float8_e4m3fn` weights + per-channel `weight_scale` (fp32) |
| **Expert layout on disk** | **fused** 3-D tensors (`experts.gate_up_proj`, `experts.down_proj`) under `language_model.` prefix |

---

## Quantization recipe

### Method

Custom in-place per-layer quantizer, **not** `llmcompressor.oneshot`. The
canonical `oneshot` path unfuses Qwen3.5-MoE's 256 experts × 40 layers into
individual `nn.Linear` modules (via `CalibrationQwen3_5MoeSparseMoeBlock`),
which keeps both the unfused parameters AND the parent's reference to the
original fused 3-D tensors alive simultaneously and overshoots the Spark's
121 GiB unified-memory ceiling around layer 34/40.

This script avoids that by:

1. Loading the bf16 model once at `device_map="cpu"` (Spark unified RAM —
   no GPU duplicate).
2. Stripping `visual.*` and `mtp.*` immediately to recover ~10–15 GiB.
3. Walking `model.layers` and quantizing in place:
   - Every targeted `nn.Linear` (`module.weight` → fp8, plus a registered
     `weight_scale` parameter, fp32, shape `[out_features, 1]`).
   - The fused `experts.gate_up_proj` (`[E, 2·I, H]`) and
     `experts.down_proj` (`[E, H, I]`) are quantized **per-expert
     per-output-channel** in a single 3-D pass, not unfused. Result:
     fp8 tensors keep their original shape, with paired
     `gate_up_proj_scale` and `down_proj_scale` of shape `[E, O, 1]`.
4. `gc.collect()` + `torch.cuda.empty_cache()` after every layer.
5. RSS hard-stop at 105 GiB so the script aborts cleanly with a Python
   traceback before the cgroup `MemoryMax=112G` kernel-kills it.

### Math

Per-channel symmetric FP8:

```
fp8_max = torch.finfo(torch.float8_e4m3fn).max   # 448.0
abs_max = |W|.amax(dim=in_features)              # along reduction axis
scale   = max(abs_max / fp8_max, eps)            # fp32, shape [out, 1]  (or [E, out, 1])
W_fp8   = clip(W / scale, ±fp8_max).to(float8_e4m3fn)
```

Activations are computed in fp16/bf16 at runtime and quantized **dynamically
per token** by vLLM's compressed-tensors kernel — no activation scales are
stored on disk.

### Settings

| key | value |
|---|---|
| `quant_method` | `compressed-tensors` |
| `format` | `float-quantized` |
| weight `num_bits` | 8 |
| weight `type` | `float` |
| weight `strategy` | `channel` (one scale per output channel) |
| weight `symmetric` | `true` |
| weight `dynamic` | `false` |
| weight `observer` | `minmax` |
| activation `num_bits` | 8 |
| activation `type` | `float` |
| activation `strategy` | `token` (one scale per token at runtime) |
| activation `symmetric` | `true` |
| activation `dynamic` | `true` |
| activation `observer` | `null` |
| `targets` | `["Linear"]` |
| `kv_cache_scheme` | `null` (vLLM applies `--kv-cache-dtype fp8_e4m3` at serve time if requested) |

### What is **NOT** quantized

The `quantization_config.ignore` list contains 191 module names. Categories:

| module class | reason |
|---|---|
| `lm_head` | Output projection; quantization-sensitive. |
| `*.mlp.gate` (per layer) | MoE router. Quantizing it changes routing decisions across all 256 experts. |
| `*.shared_expert_gate` | Sigmoid gate for the always-active shared expert. |
| `*.linear_attn.*` | Gated DeltaNet (Mamba-style) inner projections. The recurrence is INT8-unfriendly. |
| `model.embed_tokens`, `model.norm` | Embeddings and final norm. |

Note: layer 0 is **not** specially excluded here (unlike the AWQ build) — FP8
dynamic on layer 0 is empirically safe; AWQ-INT4 has tighter headroom and
benefits from keeping it dense.

### Hardware & runtime

- Quantizer runtime: NVIDIA DGX Spark (single node, GB10 / SM121a, 128 GiB
  unified memory), CPU-only quantization path. Roughly 30–60 min including
  the bf16 model download.
- Peak RSS during quantization: ~92 GiB (anon) + page cache, reached around
  layer 39/40.
- Output: 9 shards × ~4 GiB each, total ~35 GiB.

---

## Inference (vLLM)

```bash
vllm serve feanorscode/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8_e4m3 \
  --enforce-eager \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3
```

Notes:

- `--quantization compressed-tensors` selects vLLM's compressed-tensors
  loader, which reads `quantization_config.format = "float-quantized"` from
  `config.json` and builds the FP8 GEMM path.
- `--kv-cache-dtype fp8_e4m3` is optional but stacks well with FP8
  weights/activations; without it the KV cache stays at fp16/bf16.
- `--reasoning-parser qwen3` enables the `<think>...</think>` block parser
  used by the distill output.

### Smoke verification

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-fp8",
    "messages": [{"role":"user","content":"What is the largest prime less than 100? Think step by step."}],
    "max_tokens": 2048
  }'
```

A successful response contains a `<think>...</think>` block followed by `97`
(or whichever final answer the model arrives at). Presence of the think
block confirms the reasoning-distill signal survived quantization.

---

## Evaluation vs bf16 baseline

Full eval (2026-04-27, total wall-clock 1 h 59 min) via
[`lm-evaluation-harness 0.4.11`](https://github.com/EleutherAI/lm-evaluation-harness)
against vLLM `:8000` (`num_concurrent=4`, `max_length=4096`,
`temperature=0`). vLLM was served from the community pre-built DGX-Spark
image `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260415`, built from
[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker)
(MIT) — no upstream vLLM bare-metal build is yet stable on aarch64 /
SM121a. Same harness, same prompts, same `fp8_e4m3` KV-cache dtype as the
bf16 baseline run.

### Headline numbers

| benchmark | n | metric | bf16 | **FP8** | Δ (FP8 − bf16) |
|---|---|---|---|---|---|
| GSM8K (5-shot CoT, chat-templated) | 1,319 | exact_match strict | 0.9447 | **0.9447** | **0.00 pp** (identical) |
| GSM8K (5-shot CoT, chat-templated) | 1,319 | exact_match flexible | 0.9447 | 0.9439 | −0.08 pp (within σ) |
| MMLU overall (5-shot, raw MC) | 14,042 | acc | 0.8341 | **0.8332** | **−0.09 pp** (within 0.3 σ) |
| MMLU humanities | — | acc | 0.7751 | 0.7756 | +0.05 pp |
| MMLU social sciences | — | acc | 0.9074 | 0.9061 | −0.13 pp |
| MMLU STEM | — | acc | 0.8208 | 0.8148 | −0.60 pp |
| MMLU other | — | acc | 0.8642 | 0.8671 | +0.29 pp |
| ARC-Challenge (raw MC) | 1,172 | acc | 0.5478 | 0.5529 | +0.51 pp (within stderr) |
| ARC-Challenge (raw MC) | 1,172 | acc_norm | 0.5648 | 0.5708 | +0.60 pp (within stderr) |

**Effectively lossless.** Every FP8 metric sits inside ±1 σ of the bf16
baseline. GSM8K strict-match is bit-for-bit identical (same final answers
under deterministic greedy decoding). MMLU overall is −0.09 pp on a 0.30 pp
stderr — about as quiet as a quantizer can be.

### MMLU best / worst subtasks (FP8 build)

Best: `high_school_government_and_politics` 0.984, `high_school_microeconomics` 0.962, `high_school_biology` 0.961, `conceptual_physics` 0.945, `marketing` 0.944.

Weakest: `global_facts` 0.580, `virology` 0.584, `high_school_mathematics` 0.589, `moral_scenarios` 0.632.

(These weak spots are largely shared with the bf16 baseline — they're
characteristics of the source model, not quantization damage.)

### Surprise that isn't a surprise

bf16 and FP8 produced **identical** GSM8K strict-match scores (0.9447 /
0.9447). On a deterministic greedy-decode path with a shared `fp8_e4m3` KV
cache, the FP8 weight round-off was small enough that the two models
returned the same final answers across all 1,319 problems.

### Wall-clock notes

| build | total | GSM8K | MMLU | ARC-C |
|---|---|---|---|---|
| bf16 baseline | 2 h 45 min | 66 min | 94 min | 6 min |
| **FP8 (this build)** | **1 h 59 min** | 47 min | 67 min | 4 min |

Same hardware, same concurrency, same `--gpu-memory-utilization`, same
`--max-model-len 4096`. FP8 wins the throughput race against bf16 because
of the smaller weight footprint, but the AWQ-INT4 sibling is even faster
(1 h 33 min) — W4A16 has lower memory-bandwidth pressure than W8A8 in
vLLM's kernels.

---

## Limitations

- **Text-only.** Vision tower and MTP head were stripped to fit memory at
  quantization time. Cannot accept image inputs and cannot use multi-token
  prediction speculative decoding. Use the AWQ-INT4 sibling for multimodal.
- **No calibration.** This is `FP8_DYNAMIC`, not `FP8_STATIC`. The dynamic
  recipe is already statistically indistinguishable from bf16 on this battery
  (MMLU −0.09 pp, GSM8K 0.00 pp), so a calibrated static build offers no
  meaningful headroom on these tasks. Worth revisiting only if a long-context
  or domain-specific eval surfaces a regression.
- **Reasoning-distilled, not RLHF'd.** Inherits all behavior characteristics
  of the base distill, including any quirks of distilling Claude 4.7 Opus
  reasoning traces into a 3 B-active MoE.
- **Anthropic usage-policy obligation propagates.** See [License & usage](#license--usage).

---

## Files

```
config.json                   # arch + compressed-tensors quantization_config
chat_template.jinja           # passed through from base
tokenizer.json                # passed through from base
tokenizer_config.json         # passed through from base
processor_config.json         # passed through from base (multimodal class init)
generation_config.json
model.safetensors.index.json
model-00001-of-00009.safetensors
model-00002-of-00009.safetensors
...
model-00009-of-00009.safetensors
README.md                     # this file
```

---

## License & usage

**Model weights:** Apache-2.0, inherited from the base model
[`lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled`](https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled),
which in turn inherits from
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (also
Apache-2.0).

**Quantization code:** Apache-2.0 — published by
[FeanorsCode](https://feanorscode.com) at
[`FeanorsCodeSL/dgx-spark-quantization`](https://github.com/FeanorsCodeSL/dgx-spark-quantization).

**Distillation-data provenance — read this before deploying.** The base
model's training data was generated using **Anthropic's Claude Opus 4.7
via API**. lordx64's model card explicitly states that *"downstream users
should confirm compliance with Anthropic's [usage policies](https://www.anthropic.com/legal/usage-policy)
for their specific use case."* That obligation propagates to this
quantized derivative. If you intend to use this model in production —
particularly for any commercial or model-training purpose — verify that
your use case is compatible with Anthropic's usage policy as of your
deployment date.

---

## Credits

This build stands on a stack of open-source work. Listed in the order each
piece appears in the pipeline.

### Model lineage

- **[Qwen team](https://huggingface.co/Qwen)** — the Qwen3.5-MoE
  architecture and the
  [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
  base (Apache-2.0).
- **[`lordx64`](https://huggingface.co/lordx64)** — the
  [Claude-4.7-Opus reasoning-distilled base](https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled)
  this build quantizes (Apache-2.0).
- **Anthropic** — the reasoning-trace data source (Claude Opus 4.7 via
  API, used by lordx64 to train the distill). See
  [License & usage](#license--usage) for the propagated policy obligation.

### Quantization (FP8 W8A8 dynamic)

- **[`compressed-tensors`](https://github.com/vllm-project/compressed-tensors)**
  — Apache-2.0, `vllm-project`. The `float-quantized` on-disk format and
  validator this build conforms to exactly.
- **[`llmcompressor`](https://github.com/vllm-project/llm-compressor)** —
  Apache-2.0, `vllm-project`. The canonical `FP8_DYNAMIC` scheme this build
  mirrors. The custom in-place quantizer used here exists purely to bypass
  the `oneshot` MoE-unfusing path on Spark; the on-disk format is identical.

### Serving & evaluation infrastructure

- **[vLLM](https://github.com/vllm-project/vllm)** — Apache-2.0,
  `vllm-project`. Serving stack and the FP8 GEMM kernel path that loads
  this checkpoint.
- **[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker)**
  — MIT, by `eugr`. The community pre-built DGX-Spark vLLM container image
  (`ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260415`) used to serve
  this build during evaluation.
- **[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)**
  — MIT, by EleutherAI. Eval framework and benchmark implementations
  (GSM8K, MMLU, ARC-Challenge) used for the bf16 ↔ FP8 comparison.
- **[Hugging Face](https://huggingface.co)** — `transformers` /
  `safetensors` / the Hub itself (Apache-2.0).
- **[PyTorch](https://pytorch.org)** — BSD-3-Clause. Source of the FP8
  dtype (`torch.float8_e4m3fn`) and the tensor primitives used by the
  quantizer.

### Hardware

- **NVIDIA DGX Spark** (GB10 / SM121a, 128 GiB unified memory) for both
  the quantization and the evaluation runs. This repository is not
  affiliated with NVIDIA.

No method patents are involved. All cited works are open-source under
permissive licenses (Apache-2.0, MIT, BSD-3-Clause). Per-channel symmetric
FP8 with dynamic per-token activation quantization is standard PTQ — see
compressed-tensors' `FP8_DYNAMIC` reference scheme.

---

## Citation

```bibtex
@misc{qwen36_distill_claude47_fp8_dynamic_2026,
  title  = {Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled --- FP8 W8A8 Dynamic (text-only)},
  author = {feanors (FeanorsCode)},
  year   = {2026},
  url    = {https://huggingface.co/feanorscode/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic},
  note   = {compressed-tensors FP8 dynamic quantization of lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled, text-only}
}
```

If you cite this build, please also cite the base model and Qwen3.5-MoE.
