---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- qwen3
- qwen3_5_moe
- mixture-of-experts
- moe
- multimodal
- vision-language
- reasoning
- distillation
- claude-distill
- awq
- int4
- quantization
- vllm
base_model: lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled
base_model_relation: quantized
quantized_by: feanors
inference: false
---

# Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled — AWQ-INT4 (W4A16, multimodal)

INT4 quantization (AutoAWQ GEMM layout, W4A16, group_size 128, asymmetric)
of [`lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled`](https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled),
a 35B-parameter / 3B-active Qwen3.5-MoE multimodal model that has been
reasoning-distilled from Claude 4.7 Opus.

This artifact preserves the **full multimodal stack** (vision tower,
linear-attention layers, shared experts, layer 0, MTP head, router gates) at
fp16/bf16 and only quantizes the **routed expert MLPs** (`gate_proj` /
`up_proj` / `down_proj` per expert × 256 experts × ~39 routed-MoE layers).
The on-disk layout mirrors
[`QuantTrio/Qwen3.6-35B-A3B-AWQ`](https://huggingface.co/QuantTrio/Qwen3.6-35B-A3B-AWQ)
exactly so vLLM auto-detects and runs it with the same `moe_wna16` /
AWQ kernel path. If you don't need multimodal and want maximum quality,
use the sibling **FP8 build** —
[`feanorscode/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic`](https://huggingface.co/feanorscode/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-FP8-Dynamic).

> **Method note — read this before citing as "AWQ".** This build does
> **data-free RTN** quantization packed in the AutoAWQ GEMM **format**.
> It uses the AWQ on-disk layout and the AWQ-aware vLLM kernel path, but
> does **not** run AWQ's defining activation-aware salience pass over a
> calibration corpus. See [Quantization recipe](#quantization-recipe).

> **Status — smoke build, full eval done (2026-04-27).** GSM8K **0.9386**
> (strict), MMLU **0.8068**, ARC-C **0.5648** (acc) on the full benchmark
> sets. Strong enough that this checkpoint already looks production-usable;
> a calibrated re-run is on the roadmap.

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

- Per-scheme reference: [`docs/schemes/awq-gemm.md`](https://github.com/FeanorsCodeSL/dgx-spark-quantization/blob/main/docs/schemes/awq-gemm.md)
- Recipe used for this build: [`runs/qwen3.6-35b-distill/recipes/awq_gemm.py`](https://github.com/FeanorsCodeSL/dgx-spark-quantization/blob/main/runs/qwen3.6-35b-distill/recipes/awq_gemm.py)
- Three-way bf16 / FP8 / AWQ comparison report: [`runs/qwen3.6-35b-distill/REPORT.md`](https://github.com/FeanorsCodeSL/dgx-spark-quantization/blob/main/runs/qwen3.6-35b-distill/REPORT.md)

---

## TL;DR

| | bf16 (base) | **AWQ-INT4 (this build)** |
|---|---|---|
| Bits / weight (routed experts) | 16 | 4 |
| Activation precision | 16 | 16 (W4A16) |
| Calibration | n/a | none (data-free RTN) |
| Vision tower preserved | yes | **yes** (kept fp16) |
| MTP head preserved | yes | **yes** (kept fp16) |
| Layer 0 quantized? | n/a | **no** (kept fully fp16) |
| Disk size | ~67 GiB | **~24 GiB** |
| Format | safetensors / bf16 | AutoAWQ GEMM (`moe_wna16` consumer) |
| GSM8K (1,319, 5-shot CoT, strict) | 0.9447 | **0.9386** |
| MMLU overall (14,042, 5-shot raw MC) | 0.8341 | **0.8068** |
| ARC-Challenge (1,172, raw MC, acc) | 0.5478 | **0.5648** |
| Δ MMLU vs bf16 | — | **−2.73 pp** (~9 σ — real) |
| Δ GSM8K strict vs bf16 | — | **−0.61 pp** (~1 σ) |

Same eval harness (`lm-evaluation-harness 0.4.11`), same prompts, same
`temperature=0`, same KV-cache dtype.

---

## Quick numbers

| benchmark | metric | score | n |
|---|---|---|---|
| GSM8K (5-shot CoT, chat-templated) | exact_match strict | **0.9386 ± 0.0066** | 1,319 (full) |
| GSM8K (5-shot CoT, chat-templated) | exact_match flexible | **0.9416 ± 0.0065** | 1,319 (full) |
| MMLU overall (5-shot, raw multiple-choice loglikelihood) | acc | **0.8068 ± 0.0032** | 14,042 (full, 57 subtasks) |
| ARC-Challenge (raw multiple-choice loglikelihood) | acc | **0.5648 ± 0.0145** | 1,172 (full) |
| ARC-Challenge (raw multiple-choice loglikelihood) | acc_norm | **0.5606 ± 0.0145** | 1,172 (full) |

Full eval, no subsetting. See [Evaluation vs bf16](#evaluation-vs-bf16-baseline)
for methodology and the bf16 deltas.

---

## Model summary

| | |
|---|---|
| **Base model** | [`lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled`](https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled) |
| **Underlying base** | [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Apache-2.0) |
| **Architecture** | `Qwen3_5MoeForConditionalGeneration` (multimodal: text + vision) |
| **Total params** | ~35 B |
| **Active params / token** | ~3 B (top-8 of 256 experts) |
| **Hidden size** | 2,048 |
| **MoE intermediate** | 512 |
| **Layers** | 40 (12 full-attention + 28 linear-attention, repeating) |
| **Experts** | 256 routed + 1 shared, top-8 routing |
| **Vocab** | 248,320 |
| **Max position** | 262,144 (RoPE θ = 10,000,000) |
| **Vision** | 27-block ViT, hidden 1,152, patch 16, m-RoPE — **kept fp16** |
| **MTP head** | 1 layer — **kept fp16** |
| **Quantization** | AWQ-GEMM, W4A16 asymmetric, group_size 128, zero_point true |
| **Quantized scope** | routed expert MLPs only |
| **Kept fp16/bf16** | `visual.*`, `linear_attn.*`, `self_attn.*`, `shared_expert.*`, `mlp.gate` (router), `model.layers.0.*`, `mtp.*`, `lm_head` |
| **On-disk size** | ~24 GiB (vs. ~67 GiB bf16 source, ~35 GiB FP8 sibling) |

---

## Quantization recipe

### Method (precise wording)

Custom shard-streaming **data-free round-to-nearest (RTN)** quantizer,
packed in the **AutoAWQ GEMM layout** so vLLM's existing AWQ / `moe_wna16`
kernels consume it unchanged.

This is **not** the activation-aware salience pass that defines AWQ in the
[Lin et al. 2023 paper](https://arxiv.org/abs/2306.00978). We use:

- AWQ's **on-disk layout** (`qweight` / `qzeros` / `scales` with the
  canonical 8-nibble pack permutation `[0, 4, 1, 5, 2, 6, 3, 7]`).
- AWQ's **runtime kernel path** in vLLM (auto-detected from
  `quantization_config.quant_method = "awq"`).
- **Plain RTN** for the math — per-group asymmetric min/max, no calibration
  corpus, no salience scaling.

A separate calibrated AWQ build (real activation-aware pass over a
small reasoning + chat corpus) is on the [roadmap](#roadmap).

### Math (per in-scope `[out, in]` weight, per expert sub-tensor)

Per-group asymmetric quantization along `in_features` with `group_size = 128`:

```
scale  = (max(W_g) - min(W_g)) / 15        # round-trip via fp16
zp     = round(-min(W_g) / scale)          # int, clamped to [0, 15]
q      = clip(round(W / scale) + zp, 0, 15)
```

Then 8 nibbles along `out_features` are packed into one `int32` using AWQ's
canonical lane permutation `[0, 4, 1, 5, 2, 6, 3, 7]`, producing the exact
qweight / qzeros / scales tensor layout vLLM's `convert_awq_tensor`
(`moe_wna16.py`) consumes.

The fused 3-D Qwen3.5-MoE expert tensors are **unfused** on the way out:

- `experts.gate_up_proj` (`[256, 2·512, 2048]`) is split into 256 ×
  `gate_proj` + 256 × `up_proj` (`[512, 2048]` each).
- `experts.down_proj` (`[256, 2048, 512]`) is split into 256 × `down_proj`
  per layer.

### Settings

| key | value |
|---|---|
| `quant_method` | `awq` |
| `version` | `gemm` |
| `bits` | 4 |
| `group_size` | 128 |
| `zero_point` | `true` |
| asymmetric | yes (per-group min/max) |
| symmetric | no |
| pack order | `[0, 4, 1, 5, 2, 6, 3, 7]` (AWQ canonical) |
| calibration data | **none** (data-free RTN) |
| calibration samples | 0 |
| `modules_to_not_convert` | `["visual", "linear_attn", "self_attn", "shared_expert", "mlp.gate", "model.layers.0.", "mtp"]` |
| activation precision | fp16 (W4A16) |
| storage dtype | `int32` (packed nibbles) + `float16` (scales) |
| MoE expert layout | unfused → per-expert `gate_proj` / `up_proj` / `down_proj` |

### Why these specific exclusions

| module class | reason for keeping fp16 |
|---|---|
| `visual.*` | Multimodal vision tower; vision quality degrades sharply under naive INT4. |
| `linear_attn.*` | Mamba-style state — INT4 destabilizes the recurrence. |
| `self_attn.*` | Q/K/V/O are the largest single bottleneck for downstream quality on attention layers; cheap to keep at fp16 because there are only 12 full-attn layers. |
| `shared_expert.*` | Always-active dense MLP path — kept full precision for stable baseline routing. |
| `mlp.gate` | Router gate (the small `[hidden, num_experts]` projection). Quantizing it would corrupt routing decisions. |
| `model.layers.0.*` | Earliest layer is the most sensitive in MoE models — by convention kept dense. |
| `mtp.*` | Speculative-decoding head; small and quality-critical for accept rates. |
| `lm_head` | Output embedding — standard practice. |

This list matches QuantTrio's reference build of the same architecture so
vLLM's MoE-WNA16 / AWQ kernels recognize the layout without tweaks.

### Hardware & runtime

- Quantizer runtime: NVIDIA DGX Spark (single node, GB10 / SM121a, 128 GiB
  unified memory), CPU-only quantization path — no GPU needed because the
  math is pure RTN over per-group min/max.
- Source streaming: one bf16 safetensors shard at a time (~6 source shards,
  each ≤ 5 GiB), so peak host RAM stayed well under 24 GiB.
- Output: 6 shards × ~4 GiB each, total ~24 GiB.

---

## Inference (vLLM)

vLLM auto-detects `quantization_config.quant_method = "awq"` from
`config.json`. Tested with the official `vllm/vllm-openai` container:

```bash
docker run --gpus all --ipc=host --rm \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model feanorscode/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4 \
  --served-model-name qwen3.6-awq \
  --quantization awq \
  --dtype float16 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --trust-remote-code
```

Then hit `http://localhost:8000/v1/chat/completions` with the OpenAI
chat format. (`/v1/completions` works for raw text too, but multimodal
inputs require the chat endpoint.)

Notes:

- `--quantization awq` selects vLLM's AWQ / `moe_wna16` kernel path.
- `--dtype float16` matches the saved scale dtype.
- `max-model-len` is capped at 4096 in our smoke config for memory headroom
  on consumer-grade GPUs; the underlying model supports 262 K.

---

## Evaluation vs bf16 baseline

Full eval (2026-04-27, total wall-clock 1 h 33 min) via
[`lm-evaluation-harness 0.4.11`](https://github.com/EleutherAI/lm-evaluation-harness)
against the live vLLM endpoint (`local-completions` for raw multiple-choice
tasks, `local-chat-completions` for generation tasks that need the chat
template). vLLM was served from the community pre-built DGX-Spark image
`ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260415`, built from
[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker)
(MIT) — no upstream vLLM bare-metal build is yet stable on aarch64 /
SM121a. Same harness, same prompts, same `fp8_e4m3` KV-cache dtype as the
bf16 baseline run.

### Setup

| | |
|---|---|
| harness | `lm-evaluation-harness==0.4.11` (+ `[api]` extras: `tenacity`) |
| backend | vLLM `--quantization awq --dtype float16` |
| `max_length` | 4,096 |
| `num_concurrent` | 4 |
| sampling (gen tasks) | `temperature=0, top_p=1, max_gen_toks=1024` |
| tokenizer | `huggingface` backend pointed at this artifact's local dir |

### Headline numbers vs bf16

| benchmark | metric | bf16 | **AWQ-INT4** | Δ (AWQ − bf16) |
|---|---|---|---|---|
| GSM8K (5-shot CoT, chat-templated) | exact_match strict | 0.9447 | **0.9386** | −0.61 pp (~1 σ) |
| GSM8K (5-shot CoT, chat-templated) | exact_match flexible | 0.9447 | **0.9416** | −0.31 pp (within σ) |
| MMLU overall (5-shot, raw MC) | acc | 0.8341 | **0.8068** | **−2.73 pp** (~9 σ — real) |
| MMLU humanities | acc | 0.7751 | 0.7458 | −2.93 pp |
| MMLU social sciences | acc | 0.9074 | 0.8902 | −1.72 pp |
| MMLU STEM | acc | 0.8208 | 0.8059 | −1.49 pp |
| MMLU other | acc | 0.8642 | 0.8175 | **−4.67 pp** |
| ARC-Challenge (raw MC) | acc | 0.5478 | **0.5648** | +1.70 pp (within stderr) |
| ARC-Challenge (raw MC) | acc_norm | 0.5648 | 0.5606 | −0.42 pp (within stderr) |

**Reading the deltas.** The headline cost of INT4 routed-experts is on
**MMLU** — ~−2.7 pp overall, with the largest single hit on the **"other"**
cluster (general-knowledge subtasks, −4.67 pp), consistent with INT4
routed-experts losing some long-tail-knowledge precision. STEM and social
sciences hold tighter (~−1.5 pp). **GSM8K barely moves** — reasoning chains
stay coherent at INT4 weights / fp16 activations. **ARC-Challenge is a tie**
(both deltas inside ±1.45 pp stderr).

### MMLU best / worst subtasks (AWQ build)

Best: `high_school_government_and_politics` 0.974, `high_school_microeconomics` 0.958, `conceptual_physics` 0.945.

Weakest: `virology` 0.548, `global_facts` 0.550, `moral_scenarios` 0.590.

(Several weak subtasks are also weak on the unquantized base — they don't
necessarily reflect quantization damage.)

### Quantization-cost benchmark in plain terms

You're paying ~2.7 pp on MMLU to get:

- Disk size from ~67 GiB → **~24 GiB** (~2.8× compression).
- The full multimodal stack preserved (vision + MTP).
- The fastest decode path of the three builds (1 h 33 min for the eval
  battery vs FP8's 1 h 59 min vs bf16's 2 h 45 min on identical hardware).

If MMLU points matter more than disk and multimodal, ship the FP8 sibling
instead.

### Eval limitations

- **Single seed.** No variance over seeds.
- **No vision evals.** The vision tower is unquantized so multimodal
  performance should match the base, but this hasn't been confirmed
  end-to-end on standard multimodal benchmarks.
- **No long-context evals.** Truncated to 4 K context to fit the smoke
  serving config; the model itself supports 262 K.

---

## Limitations

- **Data-free RTN, not calibrated AWQ.** This build does not run an
  activation-aware salience pass over a calibration corpus. The strong
  benchmark numbers indicate the routed-expert weight distribution is
  benign enough that naive RTN is already close to optimal, but on
  out-of-distribution / low-resource inputs you may see more degradation
  than a calibrated AWQ build would.
- **Vision/language drift unmeasured.** The vision tower stays at fp16,
  but interactions between fp16 vision features and INT4 expert MLPs in
  downstream layers haven't been benchmarked on multimodal tasks.
- **Smoke serving config ≠ production config.** Long-context throughput,
  batch-size sweeps, and m-RoPE behavior at full 262 K weren't part of
  this round.
- **Reasoning-distilled, not RLHF'd.** Inherits all the safety/style
  characteristics of the base distill, including any quirks of distilling
  Claude 4.7 Opus reasoning traces into a 3 B-active MoE.
- **Anthropic usage-policy obligation propagates.** See [License & usage](#license--usage).

---

## Roadmap

1. ~~**Full-set re-runs.**~~ Done (2026-04-27).
2. ~~**BF16 baseline diff.**~~ Done (2026-04-27): MMLU −2.73 pp, GSM8K
   strict −0.61 pp, ARC-C tie. See table above.
3. **Calibrated AWQ build.** Re-quantize with an activation-aware salience
   pass (small calibration corpus, e.g. ~256 sequences from a mix of
   reasoning and chat data) — likely to recover the ~2.7 pp MMLU gap.
4. **Code coverage.** HumanEval in a sandboxed container.
5. **Long-context probe.** RULER / NIAH at ≥ 32 K to confirm the linear-
   attention layers tolerate the W4A16 routed experts in long-context
   settings.

---

## Files

```
config.json                   # arch + quantization_config (matches QuantTrio AWQ)
chat_template.jinja           # passed through from base
tokenizer.json                # passed through from base
tokenizer_config.json         # passed through from base
processor_config.json         # passed through from base (multimodal)
model.safetensors.index.json
model-00001-of-00006.safetensors
model-00002-of-00006.safetensors
model-00003-of-00006.safetensors
model-00004-of-00006.safetensors
model-00005-of-00006.safetensors
model-00006-of-00006.safetensors
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

## Citation (this build)

```bibtex
@misc{qwen36_distill_claude47_awq_int4_2026,
  title  = {Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled --- AWQ-INT4 (W4A16, multimodal, RTN-packed-AWQ-format)},
  author = {feanors (FeanorsCode)},
  year   = {2026},
  url    = {https://huggingface.co/feanorscode/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4},
  note   = {Data-free RTN INT4 quantization of lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled, packed in AutoAWQ GEMM format for vLLM compatibility}
}
```

If you cite this build, please also cite the AWQ paper (above), the base
model, and Qwen3.5-MoE.

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

### Quantization (AWQ-INT4 / W4A16)

- **AWQ method** — Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang, Gan, Han,
  *"AWQ: Activation-aware Weight Quantization for LLM Compression and
  Acceleration"*, [arXiv:2306.00978](https://arxiv.org/abs/2306.00978),
  2023. **This build uses the AWQ format and kernel path but does *not* run
  AWQ's defining activation-aware salience pass** — the math is plain RTN.
  See [Method](#method-precise-wording).

  ```bibtex
  @article{Lin2023AWQ,
    title  = {AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
    author = {Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and
              Chen, Wei-Ming and Wang, Wei-Chen and Xiao, Guangxuan and
              Dang, Xingyu and Gan, Chuang and Han, Song},
    journal= {arXiv preprint arXiv:2306.00978},
    year   = {2023}
  }
  ```
- **[`AutoAWQ`](https://github.com/casper-hansen/AutoAWQ)** — MIT, by
  [`casper-hansen`](https://github.com/casper-hansen). Source of the
  on-disk packing layout (`qweight` / `qzeros` / `scales`, canonical
  8-nibble pack permutation `[0, 4, 1, 5, 2, 6, 3, 7]`). The repository
  was archived in May 2025; vLLM's `moe_wna16` kernels still consume the
  format directly.
- **[`QuantTrio/Qwen3.6-35B-A3B-AWQ`](https://huggingface.co/QuantTrio/Qwen3.6-35B-A3B-AWQ)**
  — layout reference. This build's `modules_to_not_convert` list and
  per-expert unfusing scheme match QuantTrio's reference exactly so vLLM
  consumes both checkpoints through the same code path.

### Serving & evaluation infrastructure

- **[vLLM](https://github.com/vllm-project/vllm)** — Apache-2.0,
  `vllm-project`. Serving stack and the `moe_wna16` AWQ kernel path that
  loads this checkpoint.
- **[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker)**
  — MIT, by `eugr`. The community pre-built DGX-Spark vLLM container image
  (`ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:20260415`) used to serve
  this build during evaluation.
- **[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)**
  — MIT, by EleutherAI. Eval framework and benchmark implementations
  (GSM8K, MMLU, ARC-Challenge) used for the bf16 ↔ AWQ comparison.
- **[Hugging Face](https://huggingface.co)** — `transformers` /
  `safetensors` / the Hub itself (Apache-2.0).
- **[PyTorch](https://pytorch.org)** — BSD-3-Clause. Tensor primitives
  used by the streaming quantizer.

### Hardware

- **NVIDIA DGX Spark** (GB10 / SM121a, 128 GiB unified memory) for both
  the quantization and the evaluation runs. This repository is not
  affiliated with NVIDIA.

No method patents are involved. All cited works are open-source under
permissive licenses (Apache-2.0, MIT, BSD-3-Clause).
